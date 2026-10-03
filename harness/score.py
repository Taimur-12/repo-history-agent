import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "evals" / "golden.yaml"
RESULTS_DIR = ROOT / "data" / "results"


def cited_numbers(answer):
    return set(int(n) for n in re.findall(r"#(\d{3,6})", answer or ""))


def score_citations(case, answer):
    found = cited_numbers(answer)
    expected = set(case.get("expect_cites", []))
    return sorted(expected - found), sorted(found & expected)


def call_keys(calls):
    keys = Counter()
    for call in calls:
        keys[call["tool"]] += 1
        if call.get("args"):
            keys[f"{call['tool']}:{call['args'][0].strip(chr(39))}"] += 1
    return keys


def score_tools(case, calls):
    wanted = Counter(str(item) for item in case.get("expect_tools_called", []))
    actual = call_keys(calls)
    return sorted(tool for tool, count in wanted.items() if actual[tool] < count)


def score_mentions(case, answer):
    lowered = (answer or "").lower()
    return [phrase for phrase in case.get("must_mention", []) if phrase.lower() not in lowered]


def score_forbidden(case, answer):
    return [
        pattern
        for pattern in case.get("should_not", [])
        if re.search(pattern, answer or "", re.IGNORECASE)
    ]


def score(case, record):
    answer = record.get("answer")
    missing_cites, hit_cites = score_citations(case, answer)
    missing_tools = score_tools(case, record.get("calls", []))
    missing_mentions = score_mentions(case, answer)
    forbidden = score_forbidden(case, answer)

    return {
        "case_id": case["id"],
        "run_id": record["run_id"],
        "passed": not (missing_cites or missing_tools or missing_mentions or forbidden),
        "missing_cites": missing_cites,
        "found_cites": hit_cites,
        "missing_tools": missing_tools,
        "missing_mentions": missing_mentions,
        "forbidden_hits": forbidden,
        "termination": record["termination"],
        "tool_calls": record["tool_calls"],
        "tools_called": record.get("tools_called", []),
        "latency_ms": record["latency_ms"],
        "tokens_in": record.get("tokens_in", 0),
        "tokens_out": record.get("tokens_out", 0),
        "cache_misses": sum(1 for c in record.get("calls", []) if c.get("cache_hit") is False),
        "tool_errors": sum(1 for c in record.get("calls", []) if c.get("tool_error")),
        "answer": answer,
    }


def summarise(results):
    by_case = {}
    for r in results:
        by_case.setdefault(r["case_id"], []).append(r)

    rows = []
    for case_id, runs in by_case.items():
        n = len(runs)
        missing = Counter(num for r in runs for num in r["missing_cites"])
        unused = Counter(tool for r in runs for tool in r["missing_tools"])
        unsaid = Counter(phrase for r in runs for phrase in r["missing_mentions"])
        banned = Counter(pattern for r in runs for pattern in r["forbidden_hits"])
        rows.append({
            "case_id": case_id,
            "runs": n,
            "passed": sum(r["passed"] for r in runs),
            "avg_calls": sum(r["tool_calls"] for r in runs) / n,
            "avg_tokens_in": sum(r["tokens_in"] for r in runs) / n,
            "avg_tokens_out": sum(r["tokens_out"] for r in runs) / n,
            "avg_seconds": sum(r["latency_ms"] for r in runs) / n / 1000,
            "terminations": dict(Counter(r["termination"] for r in runs)),
            "missing_cites": dict(missing),
            "missing_tools": dict(unused),
            "missing_mentions": dict(unsaid),
            "forbidden_hits": dict(banned),
        })
    return rows


def report(meta, rows):
    print()
    print(f"model={meta['model']}  prompt={meta['prompt_version']}  reasoning={meta['reasoning']}  repeat={meta['repeat']}")
    print(f"{'case':<34} {'pass':>6} {'calls':>6} {'tok_in':>8} {'tok_out':>8} {'sec':>6}  terminations")
    print("-" * 90)
    for row in rows:
        print(f"{row['case_id']:<34} {row['passed']:>2}/{row['runs']:<3} {row['avg_calls']:>6.1f} "
              f"{row['avg_tokens_in']:>8.0f} {row['avg_tokens_out']:>8.0f} {row['avg_seconds']:>6.1f}  {row['terminations']}")
    print("-" * 90)
    total_runs = sum(row["runs"] for row in rows)
    total_passed = sum(row["passed"] for row in rows)
    print(f"{total_passed}/{total_runs} runs passed")

    for row in rows:
        if row["passed"] == row["runs"]:
            continue
        n = row["runs"]
        print(f"\n{row['case_id']}  ({row['passed']}/{n} passed) — how often each problem occurred")
        for label, key in (("missing citation", "missing_cites"), ("tool not called", "missing_tools"),
                           ("never mentioned", "missing_mentions"), ("said banned thing", "forbidden_hits")):
            for item, count in sorted(row[key].items(), key=lambda kv: -kv[1]):
                print(f"  {label:<18} {str(item):<28} {count}/{n}")


def main():
    from agent.graph import MODEL, ask, build

    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="*")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--prompt", default="v0")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--reasoning", default=None, choices=["none", "low", "medium", "high"])
    args = parser.parse_args()

    cases = yaml.safe_load(GOLDEN.read_text())
    if args.cases:
        cases = [c for c in cases if c["id"] in args.cases]

    agent = build(args.prompt, args.reasoning)
    jobs = [case for case in cases for _ in range(args.repeat)]
    print(f"running {len(jobs)} runs ({len(cases)} cases x {args.repeat}) with prompt {args.prompt} ...", flush=True)

    def execute(case):
        record = ask(case["q"], case_id=case["id"], agent=agent, prompt_version=args.prompt, reasoning=args.reasoning)
        result = score(case, record)
        print(f"  {case['id']:<34} {'PASS' if result['passed'] else 'fail'}", flush=True)
        return result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(execute, jobs))

    meta = {
        "model": MODEL,
        "prompt_version": args.prompt,
        "reasoning": args.reasoning or "default",
        "repeat": args.repeat,
        "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    rows = summarise(results)
    report(meta, rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"{stamp}_{args.prompt}_r-{args.reasoning or 'default'}_x{args.repeat}.json"
    out.write_text(json.dumps({"meta": meta, "summary": rows, "runs": results}, indent=2))
    print(f"\nwritten to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
