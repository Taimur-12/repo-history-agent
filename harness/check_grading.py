import sys

import yaml

from harness.score import GOLDEN, ROOT, score_citations, score_forbidden, score_match_any, score_mentions

EXAMPLES = ROOT / "evals" / "grading_examples.yaml"


def problems(case, answer):
    missing, _ = score_citations(case, answer)
    return ([f"cite {n}" for n in missing] + score_mentions(case, answer)
            + score_forbidden(case, answer) + score_match_any(case, answer))


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
    print(f"{total - wrong}/{total} examples graded as expected")
    if untested:
        print(f"cases with no examples: {', '.join(untested)}")
    sys.exit(1 if wrong or untested else 0)


if __name__ == "__main__":
    main()
