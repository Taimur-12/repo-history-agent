import sys

import yaml

from harness.score import (GOLDEN, NEGATION, ROOT, score_citations, score_forbidden, score_match_any,
                           score_mentions)

EXAMPLES = ROOT / "evals" / "grading_examples.yaml"
TYPES = {"which-way-current", "why-like-this", "prior-art", "evolution", "ownership", "in-flight", "quantitative", "refuse"}


def problems(case, answer):
    missing, _ = score_citations(case, answer)
    return ([f"cite {n}" for n in missing] + score_mentions(case, answer)
            + score_forbidden(case, answer) + score_match_any(case, answer))


def unflagged_negations(case):
    rules = case.get("should_not", [])
    return [rule for rule in rules if isinstance(rule, str) and NEGATION.search(rule)] + [
        rule["pattern"] for rule in rules
        if isinstance(rule, dict) and (set(rule) != {"pattern", "negative_claim"} or not isinstance(rule["negative_claim"], bool))]


def main():
    cases = {c["id"]: c for c in yaml.safe_load(GOLDEN.read_text())}
    examples = yaml.safe_load(EXAMPLES.read_text())
    total = wrong = 0
    for case_id, groups in examples.items():
        for expected in ("pass", "fail"):
            for answer in groups.get(expected, []):
                total += 1
                found = problems(cases[case_id], answer)
                if (not found) != (expected == "pass"):
                    wrong += 1
                    print(f"WRONG {case_id}: expected {expected}, got {found or 'pass'}\n    {answer[:120]}")
    untested = sorted(set(cases) - set(examples))
    untyped = sorted(c for c, case in cases.items() if case.get("type") not in TYPES)
    unverified = sorted(c for c, case in cases.items() if not case.get("verified_by"))
    unflagged = sorted(c for c, case in cases.items() if unflagged_negations(case))
    print(f"{total - wrong}/{total} examples graded as expected")
    for label, ids in (("cases with no examples", untested), ("cases without a known type", untyped),
                       ("cases without verified_by", unverified),
                       ("should_not rules with a negation word and no negative_claim flag, or malformed", unflagged)):
        if ids:
            print(f"{label}: {', '.join(ids)}")
    sys.exit(1 if wrong or untested or untyped or unverified or unflagged else 0)


if __name__ == "__main__":
    main()
