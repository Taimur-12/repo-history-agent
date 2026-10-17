import json, os, re
from urllib.parse import quote_plus
from pathlib import Path
from dotenv import load_dotenv
import requests

API = "https://api.github.com"
REPO = "langchain-ai/langgraph"
TIMEOUT = 30

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
CACHE_DIR = ROOT / "data" / "cache"


def headers():
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN is not set")
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

def cache_file(path):
    name = path.strip("/").replace("/", "__").replace("?", "__").replace("&", "__")
    return CACHE_DIR / f"{name}.json"

def fetch(path):
    cached = cache_file(path)

    if cached.exists():
        return json.loads(cached.read_text()), True

    url = f"{API}{path}"
    items=[]

    while url:
        response = requests.get(url, headers=headers(), timeout=TIMEOUT)
        response.raise_for_status()
        page=response.json()

        if not isinstance(page, list):
            items = page
            break

        items.extend(page)

        url = response.links.get("next", {}).get("url")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(items, indent=2))
    return items, False

INTERESTING_EVENTS = ("merged", "closed", "reviewed", "labeled", "renamed", "reopened")


def get_timeline(n):
    events, cache_hit = fetch(f"/repos/{REPO}/issues/{n}/timeline?per_page=100")

    cross_refs = []
    history = []
    for event in events:
        kind = event.get("event")

        if kind == "cross-referenced":
            issue = event["source"]["issue"]
            cross_refs.append({
                "repo": issue["repository"]["full_name"],
                "number": issue["number"],
                "kind": "pr" if "pull_request" in issue else "issue",
                "state": issue["state"],
                "title": issue["title"],
                "referenced_at": event["created_at"],
            })
            continue

        if kind in INTERESTING_EVENTS:
            history.append({
                "event": kind,
                "actor": (event.get("actor") or event.get("user") or {}).get("login"),
                "at": event.get("created_at") or event.get("submitted_at"),
                "detail": event.get("state") or (event.get("label") or {}).get("name") or event.get("commit_id"),
            })

    return {"cross_references": cross_refs, "history": history, "reviewed": any(e["event"] == "reviewed" for e in history)}, cache_hit


def error_message(exc, kind, n):
    status = exc.response.status_code if exc.response is not None else "unknown"
    if status == 404:
        return f"{kind} {n} does not exist in {REPO}"
    if status == 403:
        return f"github refused the request for {kind} {n} (rate limit or permissions)"
    return f"github returned status {status} for {kind} {n}"


def get_pr(n):
    try:
        pr, cache_hit = fetch(f"/repos/{REPO}/pulls/{n}")
    except requests.HTTPError as exc:
        return error_message(exc, "pr", n), False

    return {
        "number": pr["number"],
        "title": pr["title"],
        "body": pr["body"],
        "state": pr["state"],
        "merged": pr["merged"],
        "merged_at": pr["merged_at"],
        "created_at": pr["created_at"],
        "closed_at": pr.get("closed_at"),
        "author": (pr.get("user") or {}).get("login"),
        "base_branch": pr["base"]["ref"],
        "draft": pr.get("draft"),
        "labels": [label["name"] for label in pr.get("labels", [])],
        "comment_count": pr.get("comments"),
        "review_comment_count": pr.get("review_comments"),
        "merge_commit_sha": pr.get("merge_commit_sha"),
    }, cache_hit


def get_issue(n):
    try:
        issue, cache_hit = fetch(f"/repos/{REPO}/issues/{n}")
    except requests.HTTPError as exc:
        return error_message(exc, "issue", n), False

    return {
        "number": issue["number"],
        "title": issue["title"],
        "body": issue["body"],
        "state": issue["state"],
        "state_reason": issue.get("state_reason"),
        "created_at": issue["created_at"],
        "closed_at": issue.get("closed_at"),
        "author": (issue.get("user") or {}).get("login"),
        "labels": [label["name"] for label in issue.get("labels", [])],
        "is_pr": "pull_request" in issue,
    }, cache_hit


def get_comments(n):
    try:
        comments, cache_hit = fetch(f"/repos/{REPO}/issues/{n}/comments?per_page=100")
    except requests.HTTPError as exc:
        return error_message(exc, "issue", n), False

    return [
        {
            "author": (comment.get("user") or {}).get("login"),
            "author_type": (comment.get("user") or {}).get("type"),
            "created_at": comment["created_at"],
            "body": comment["body"],
        }
        for comment in comments
    ], cache_hit


SEARCH_KINDS = {"any": "", "pr": "is:pr ", "issue": "is:issue "}
SNIPPET_CHARS = 200
CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
ITEM_URL = re.compile(rf"https://github\.com/{REPO}/(?:pull|issues)/(\d+)\S*")
OTHER_URL = re.compile(r"https?://\S+")
SKIPPED_LINE = re.compile(r"^\s*(#|- \[[ xX]\]|\*\*|>)")


def snippet(body):
    text = HTML_COMMENT.sub(" ", CODE_BLOCK.sub(" ", body or ""))
    lines = [line for line in text.splitlines() if line.strip() and not SKIPPED_LINE.match(line)]
    text = OTHER_URL.sub("", ITEM_URL.sub(r"#\1", " ".join(lines)))
    text = " ".join(text.split())
    if len(text) <= SNIPPET_CHARS:
        return text
    return text[:SNIPPET_CHARS].rsplit(" ", 1)[0] + "…"


def search(query, kind="any", limit=10):
    if kind not in SEARCH_KINDS:
        return f"unknown kind {kind!r}, use one of: any, pr, issue", False
    encoded = quote_plus(f"repo:{REPO} {SEARCH_KINDS[kind]}{query}")
    try:
        result, cache_hit = fetch(f"/search/issues?q={encoded}&per_page={limit}")
    except requests.HTTPError as exc:
        return error_message(exc, "search", query), False

    return [
        {
            "number": item["number"],
            "kind": "pr" if "pull_request" in item else "issue",
            "state": item["state"],
            "title": item["title"],
            "snippet": snippet(item.get("body")),
        }
        for item in result.get("items", [])
    ], cache_hit


def split_tag(tag):
    if "==" in tag:
        package, version = tag.split("==", 1)
        return package, version
    return "langgraph", tag


PR_LINK = re.compile(r"\(#(\d+)\)|\[#(\d+)\]\(")
REVERT = re.compile(r'Revert "(.+)"', re.IGNORECASE)
TRAILING_LINK = re.compile(r"\s*(\(#\d+\)|\[#\d+\]\([^)]*\))\s*$")


def pr_numbers(text):
    return [int(a or b) for a, b in PR_LINK.findall(text)]


def note_title(text):
    return TRAILING_LINK.sub("", text.strip().lstrip("*-").strip()).strip()


def read_notes(body):
    listed, reverted = {}, {}
    for line in body.splitlines():
        match = REVERT.search(line)
        if not match:
            for number in pr_numbers(line):
                listed.setdefault(number, note_title(line))
            continue
        quoted = match.group(1)
        outside = pr_numbers(line[match.end():])
        by = outside[0] if outside else None
        reverted[note_title(quoted)] = by
        for number in pr_numbers(quoted):
            reverted[number] = by
        for number in outside:
            listed.setdefault(number, note_title(line))
    return listed, reverted


def release_map():
    releases, cache_hit = fetch(f"/repos/{REPO}/releases?per_page=100")

    mapping = {}
    for release in releases:
        package, version = split_tag(release["tag_name"])
        listed, reverted = read_notes(release.get("body") or "")
        for number, title in listed.items():
            entry = {
                "package": package,
                "version": version,
                "tag": release["tag_name"],
                "published_at": (release.get("published_at") or "")[:10],
            }
            for key in (number, title):
                if key in reverted:
                    entry["reverted_in_same_release"] = True
                    if reverted[key]:
                        entry["reverted_by"] = reverted[key]
                    break
            mapping.setdefault(number, []).append(entry)
    return mapping, cache_hit


def version_for_pr(n):
    mapping, cache_hit = release_map()
    releases = mapping.get(n)
    if not releases:
        return f"pr {n} does not appear in any published release body", cache_hit
    return releases, cache_hit


if __name__ == "__main__":
    timeline, hit = get_timeline(7498)
    refs = timeline["cross_references"]
    print(f"cache_hit={hit}, {len(refs)} cross-refs, reviewed={timeline['reviewed']}")
    for event in timeline["history"]:
        print(" ", event["event"], event["actor"], event["at"], event["detail"])
    for ref in refs:
        print(ref["repo"], ref["number"], ref["kind"], ref["state"], ref["title"][:55])