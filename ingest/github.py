import json, os, re, threading, time
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
ARCHIVE_DIR = ROOT / "data" / "cache-archive-20261021"
AS_OF = "2026-10-16T18:34:00Z"
MATCHED_TEXT = True
REVIEW_COMMENTS = True
PR_REVIEWS = True
CODE_CHANGES = True
TEXT_MATCH = "application/vnd.github.text-match+json"


def by_freeze(when):
    return AS_OF is None or (when is not None and when <= AS_OF)


def known(when):
    return when if when and by_freeze(when) else None


def frozen_state(item):
    if AS_OF is None:
        return item["state"]
    return "closed" if known(item.get("closed_at")) else "open"


def not_yet(kind, n):
    return f"{kind} {n} does not exist in {REPO} as of {AS_OF}"


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

SEARCH_INTERVAL = 2.2
RATE_LIMIT_RETRIES = 3
search_lock = threading.Lock()
last_search = [0.0]


def pace_search():
    with search_lock:
        wait = last_search[0] + SEARCH_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        last_search[0] = time.monotonic()


def rate_limit_wait(response):
    if response.status_code not in (403, 429):
        return None
    if response.headers.get("retry-after"):
        return int(response.headers["retry-after"]) + 1
    if response.headers.get("x-ratelimit-remaining") == "0":
        return max(int(response.headers.get("x-ratelimit-reset", time.time())) - time.time(), 0) + 1
    return None


def get(url, accept=None):
    sent = headers()
    if accept:
        sent["Accept"] = accept
    for _ in range(RATE_LIMIT_RETRIES):
        if url.startswith(f"{API}/search/"):
            pace_search()
        response = requests.get(url, headers=sent, timeout=TIMEOUT)
        wait = rate_limit_wait(response)
        if wait is None:
            return response
        time.sleep(min(wait, 120))
    return response


def fetch(path, accept=None, variant=""):
    cached = cache_file(path + variant)

    if cached.exists():
        return json.loads(cached.read_text()), True

    url = f"{API}{path}"
    items=[]

    while url:
        response = get(url, accept)
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
        if not by_freeze(event.get("created_at") or event.get("submitted_at")):
            continue

        if kind == "cross-referenced":
            issue = event["source"]["issue"]
            cross_refs.append({
                "repo": issue["repository"]["full_name"],
                "number": issue["number"],
                "kind": "pr" if "pull_request" in issue else "issue",
                "state": frozen_state(issue),
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


NOT_CODE = re.compile(
    r"(^|/)(tests?|docs?|examples?)/|(^|/)test_[^/]*$|\.(md|mdx|ipynb|lock|txt|toml|cfg|ini|json|ya?ml)$|(^|/)(Makefile|LICENSE)$"
)
MAX_CHANGED_LINES = 40
MAX_LINE_CHARS = 160


def changed_lines(patch):
    return [line[:MAX_LINE_CHARS] for line in patch.splitlines() if line[:1] in "+-" and line[1:].strip()]


def code_changes(n):
    try:
        files, cache_hit = fetch(f"/repos/{REPO}/pulls/{n}/files?per_page=100")
    except requests.HTTPError:
        return None, False

    source, other = [], []
    for file in files:
        lines = changed_lines(file.get("patch") or "")
        if NOT_CODE.search(file["filename"]) or not lines:
            other.append(file["filename"])
        else:
            source.append((file, lines))

    room = {}
    budget = MAX_CHANGED_LINES
    edited = [(file, lines) for file, lines in source if file["status"] != "removed"]
    for i, (file, lines) in enumerate(sorted(edited, key=lambda pair: len(pair[1]))):
        share = budget // (len(edited) - i)
        room[file["filename"]] = min(len(lines), share)
        budget -= room[file["filename"]]

    shown = []
    for file, lines in source:
        entry = {"file": file["filename"], "status": file["status"], "additions": file["additions"], "deletions": file["deletions"]}
        keep = room.get(file["filename"], 0)
        if keep:
            entry["lines"] = lines[:keep]
        if len(lines) > keep and file["status"] != "removed":
            entry["more_lines"] = len(lines) - keep
        shown.append(entry)
    return {"code": shown, "other_files": other}, cache_hit


def get_pr(n):
    try:
        pr, cache_hit = fetch(f"/repos/{REPO}/pulls/{n}")
    except requests.HTTPError as exc:
        return error_message(exc, "pr", n), False
    if not by_freeze(pr["created_at"]):
        return not_yet("pr", n), cache_hit

    result = {
        "number": pr["number"],
        "title": pr["title"],
        "body": pr["body"],
        "state": frozen_state(pr),
        "merged": pr["merged"] if AS_OF is None else known(pr["merged_at"]) is not None,
        "merged_at": known(pr["merged_at"]),
        "created_at": pr["created_at"],
        "closed_at": known(pr.get("closed_at")),
        "author": (pr.get("user") or {}).get("login"),
        "base_branch": pr["base"]["ref"],
        "draft": pr.get("draft"),
        "labels": [label["name"] for label in pr.get("labels", [])],
        "comment_count": pr.get("comments"),
        "review_comment_count": pr.get("review_comments"),
        "merge_commit_sha": pr.get("merge_commit_sha"),
    }
    changes_hit = True
    if CODE_CHANGES and known(pr.get("closed_at")) is None:
        result["code_changes"] = "not shown: the pull request was still open, so its code could still change"
    elif CODE_CHANGES:
        result["code_changes"], changes_hit = code_changes(n)
    if not PR_REVIEWS:
        return result, cache_hit and changes_hit

    try:
        reviews, reviews_hit = review_comments(n)
    except requests.HTTPError:
        reviews, reviews_hit = [], False
    shown, more = first_reviews(sorted(reviews, key=lambda entry: entry["created_at"] or ""))
    if shown:
        result["review_comments"] = shown
    if more:
        result["more_review_comments"] = f"{more} more review entries; get_comments shows all of them"
    return result, cache_hit and reviews_hit and changes_hit


PR_REVIEW_CHARS = 2000


def first_reviews(entries):
    shown, used = [], 0
    for entry in entries:
        size = len(json.dumps(entry))
        if shown and used + size > PR_REVIEW_CHARS:
            break
        shown.append(entry)
        used += size
    return shown, len(entries) - len(shown)


def get_issue(n):
    try:
        issue, cache_hit = fetch(f"/repos/{REPO}/issues/{n}")
    except requests.HTTPError as exc:
        return error_message(exc, "issue", n), False
    if not by_freeze(issue["created_at"]):
        return not_yet("issue", n), cache_hit

    state = frozen_state(issue)
    return {
        "number": issue["number"],
        "title": issue["title"],
        "body": issue["body"],
        "state": state,
        "state_reason": issue.get("state_reason") if state == "closed" else None,
        "created_at": issue["created_at"],
        "closed_at": known(issue.get("closed_at")),
        "author": (issue.get("user") or {}).get("login"),
        "labels": [label["name"] for label in issue.get("labels", [])],
        "is_pr": "pull_request" in issue,
    }, cache_hit


def get_comments(n):
    try:
        comments, cache_hit = fetch(f"/repos/{REPO}/issues/{n}/comments?per_page=100")
    except requests.HTTPError as exc:
        return error_message(exc, "issue", n), False

    discussion = [
        {
            "author": (comment.get("user") or {}).get("login"),
            "author_type": (comment.get("user") or {}).get("type"),
            "created_at": comment["created_at"],
            "body": comment["body"],
        }
        for comment in comments
        if by_freeze(comment["created_at"])
    ]
    if not REVIEW_COMMENTS:
        return discussion, cache_hit

    try:
        reviews, reviews_hit = review_comments(n)
    except requests.HTTPError:
        reviews, reviews_hit = [], False
    for entry in discussion:
        entry["type"] = "discussion"
    return sorted(discussion + reviews, key=lambda entry: entry["created_at"] or ""), cache_hit and reviews_hit


REVIEW_CHARS = 800


def capped(text):
    text = text or ""
    if len(text) <= REVIEW_CHARS:
        return text
    return text[:REVIEW_CHARS].rsplit(" ", 1)[0] + "…"


def is_bot(user):
    return (user or {}).get("type") == "Bot" or (user or {}).get("login", "").endswith("[bot]")


def review_comments(n):
    issue, hit = fetch(f"/repos/{REPO}/issues/{n}")
    if "pull_request" not in issue:
        return [], hit
    inline, inline_hit = fetch(f"/repos/{REPO}/pulls/{n}/comments?per_page=100")
    summaries, summaries_hit = fetch(f"/repos/{REPO}/pulls/{n}/reviews?per_page=100")
    entries = [
        {
            "type": "review",
            "author": (comment.get("user") or {}).get("login"),
            "author_type": (comment.get("user") or {}).get("type"),
            "created_at": comment["created_at"],
            "file": comment.get("path"),
            "body": capped(comment.get("body")),
        }
        for comment in inline
        if by_freeze(comment["created_at"]) and not is_bot(comment.get("user"))
    ]
    entries += [
        {
            "type": "review",
            "author": (review.get("user") or {}).get("login"),
            "author_type": (review.get("user") or {}).get("type"),
            "created_at": review["submitted_at"],
            "review_state": review.get("state"),
            "body": capped(review.get("body")),
        }
        for review in summaries
        if (review.get("body") or "").strip() and by_freeze(review.get("submitted_at"))
        and not is_bot(review.get("user"))
    ]
    return entries, hit and inline_hit and summaries_hit


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


DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def created_range(before, after):
    upper = AS_OF
    if before and (upper is None or before < upper):
        upper = before
    if after:
        return f" created:{after}..{upper or '*'}"
    return f" created:<{upper}" if upper else ""


def search(query, kind="any", merged_only=False, before=None, after=None, oldest_first=False, limit=10):
    if kind not in SEARCH_KINDS:
        return f"unknown kind {kind!r}, use one of: any, pr, issue", False
    before, after = before or None, after or None
    for name, value in (("before", before), ("after", after)):
        if value is not None and not DATE.fullmatch(str(value)):
            return f"{name} must be a date written YYYY-MM-DD, got {value!r}", False
    scope = "is:pr is:merged " if merged_only else SEARCH_KINDS[kind]
    encoded = quote_plus(f"repo:{REPO} {scope}{query}{created_range(before, after)}")
    order = "&sort=created&order=asc" if oldest_first else ""
    try:
        path = f"/search/issues?q={encoded}&per_page={limit}{order}"
        if MATCHED_TEXT:
            result, cache_hit = fetch(path, TEXT_MATCH, "&text_match")
        else:
            result, cache_hit = fetch(path)
    except requests.HTTPError as exc:
        return error_message(exc, "search", query), False

    return [search_result(item) for item in result.get("items", [])], cache_hit


def search_result(item):
    found = {
        "number": item["number"],
        "kind": "pr" if "pull_request" in item else "issue",
        "state": frozen_state(item),
    }
    found["created"] = item["created_at"][:10]
    if "pull_request" in item:
        merged_at = known(item["pull_request"].get("merged_at"))
        found["merged"] = bool(merged_at)
        found["merged_on"] = merged_at[:10] if merged_at else None
    found["title"] = item["title"]
    found["snippet"] = matched_text(item) or snippet(item.get("body"))
    return found


MATCH_CHARS = 400


def matched_text(item):
    fragments = [" ".join(m["fragment"].split()) for m in item.get("text_matches", []) if m.get("property") == "body"]
    text = " … ".join(f for f in fragments if f)
    if len(text) <= MATCH_CHARS:
        return text
    return text[:MATCH_CHARS].rsplit(" ", 1)[0] + "…"


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
        if not by_freeze(release.get("published_at")):
            continue
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