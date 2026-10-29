import sys

from ingest import github


def issue(n):
    return github.get_issue(n)[0]


def comments(n):
    return github.get_comments(n)[0]


def cross_refs(n):
    return [ref["number"] for ref in github.get_timeline(n)[0]["cross_references"]]


def search_numbers(query, kind="any"):
    return [item["number"] for item in github.search(query, kind)[0]]


CHECKS = [
    ("#6574 closed 10-19 as not_planned: shown open", lambda: issue(6574)["state"] == "open"),
    ("#6574: no closed_at or state_reason", lambda: issue(6574)["closed_at"] is None and issue(6574)["state_reason"] is None),
    ("#6574 in search: shown open", lambda: [i["state"] for i in github.search("MALFORMED_FUNCTION_CALL silently terminates")[0]
                                              if i["number"] == 6574] == ["open"]),
    ("#9333 merged 10-18: shown open and unmerged", lambda: github.get_pr(9333)[0]["state"] == "open"
                                                     and github.get_pr(9333)[0]["merged"] is False),
    ("#4940 closed 10-16 17:48, before the freeze: shown closed", lambda: issue(4940)["state"] == "closed"),
    ("#9185 closed 17:52: shown closed", lambda: issue(9185)["state"] == "closed"),
    ("#9185: the closing comment is kept", lambda: any("this is how resume works" in (c["body"] or "") for c in comments(9185))),
    ("#9265 merged 18:33: shown merged", lambda: github.get_pr(9265)[0]["merged"] is True),
    ("#9334 created 19:42: does not exist yet", lambda: "as of" in github.get_pr(9334)[0]),
    ("#9336 created 22:04: hidden from search", lambda: 9336 not in search_numbers("clear_cache no-op", "pr")),
    ("#4956 timeline: cross-reference from #5115 on 10-19 dropped", lambda: 5115 not in cross_refs(4956)),
    ("#7345: the one comment after the freeze dropped", lambda: len(comments(7345)) == 12),
    ("#8619: review from 10-17 dropped, earlier review kept", lambda: [c["created_at"][:10] for c in comments(8619)
                                                                     if c.get("type") == "review"] == ["2026-08-14"]),
    ("#8619 in get_pr: review from 10-17 dropped, earlier review kept", lambda: [c["created_at"][:10] for c in
                                                                     github.get_pr(8619)[0].get("review_comments", [])]
                                                                    == ["2026-08-14"]),
    ("#9333 merged 10-18: no code shown", lambda: isinstance(github.get_pr(9333)[0]["code_changes"], str)),
    ("#6676 merged before the freeze: code shown", lambda: isinstance(github.get_pr(6676)[0]["code_changes"], dict)),
    ("#9333 merged 10-18: not in its file's history", lambda: 9333 not in [c.get("number") for c in
                                                            github.file_history("libs/cli/langgraph_cli/deploy.py")[0]["changes"]]),
    ("file history: no change dated after the freeze", lambda: all(c["date"] <= github.AS_OF[:10] for c in
                                                                github.file_history("libs/cli/langgraph_cli/deploy.py")[0]["changes"]
                                                                if "date" in c)),
    ("#9302 shipped only in 1.2.16 (10-16 22:11): no release yet", lambda: isinstance(github.version_for_pr(9302)[0], str)),
    ("#6497 shipped in 1.0.4 (2025): release kept", lambda: any(r["version"] == "1.0.4" for r in github.version_for_pr(6497)[0])),
]


def main():
    wrong = 0
    for label, check in CHECKS:
        try:
            ok = check()
        except Exception as exc:
            ok = False
            label += f" (raised {type(exc).__name__}: {exc})"
        wrong += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {label}")
    print(f"{len(CHECKS) - wrong}/{len(CHECKS)} freeze checks passed (as of {github.AS_OF})")
    sys.exit(1 if wrong else 0)


if __name__ == "__main__":
    main()
