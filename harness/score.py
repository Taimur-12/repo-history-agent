import argparse
import ast
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
RUNS_FILE = ROOT / "data" / "traces" / "runs.jsonl"

SMALL_MODELS = ("-mini", "-nano")
FREE_TOKENS = {"small": 2_500_000, "large": 250_000}
TOKENS_PER_RUN = 85_000


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


def score_match_any(case, answer):
    patterns = case.get("must_match_any", [])
    if not patterns:
        return []
    if any(re.search(pattern, answer or "", re.IGNORECASE) for pattern in patterns):
        return []
    return ["none of must_match_any"]


QUOTED = re.compile(r'"[^"\n]*"|\u201c[^\u201d\n]*\u201d')


def without_quotes(text):
    return QUOTED.sub(" ", text or "")


CLAUSE_START = re.compile(r"[.!?;\n]")
NEGATION = re.compile(r"\b(no|not|never|nothing|none|neither|nor|without|cannot|isn't|aren't|wasn't|weren't|doesn't|didn't|don't|hasn't|haven't|can't|won't)\b", re.IGNORECASE)


def negated(text, start):
    clause_starts = [m.end() for m in CLAUSE_START.finditer(text, 0, start)]
    clause = text[clause_starts[-1] if clause_starts else 0:start]
    return bool(NEGATION.search(clause))


def claims(pattern, text):
    return any(not negated(text, m.start()) for m in re.finditer(pattern, text, re.IGNORECASE))


def score_forbidden(case, answer):
    claimed = without_quotes(answer)
    return [pattern for pattern in case.get("should_not", []) if claims(pattern, claimed)]


OPENING_TOOLS = ("get_pr", "get_issue", "get_timeline", "get_comments")


def call_args(call):
    values = []
    for text in call.get("args", []):
        try:
            values.append(ast.literal_eval(text))
        except (ValueError, SyntaxError):
            values.append(text)
    return values


def replay(call):
    from ingest import github

    tools = {
        "search": github.search,
        "get_pr": github.get_pr,
        "get_issue": github.get_issue,
        "get_timeline": github.get_timeline,
        "get_comments": github.get_comments,
        "version_for_pr": github.version_for_pr,
    }
    chosen = tools.get(call["tool"])
    if chosen is None or call.get("tool_error") or call.get("failed"):
        return None
    try:
        return chosen(*call_args(call))[0]
    except Exception:
        return None


def target_items(case):
    items = set(case.get("expect_cites", []))
    for entry in case.get("expect_tools_called", []):
        _, _, arg = str(entry).partition(":")
        if arg.isdigit():
            items.add(int(arg))
    return sorted(items)


def score_stages(case, record):
    surfaced, opened, seen = set(), set(), []
    for call in record.get("calls", []):
        result = replay(call)
        if result is None:
            continue
        if call["tool"] == "search" and isinstance(result, list):
            surfaced |= {item["number"] for item in result}
        if call["tool"] in OPENING_TOOLS:
            opened.add(int(call_args(call)[0]))
        seen.append(json.dumps(result, default=str).lower())

    context = " ".join(seen)
    answer = (record.get("answer") or "").lower()
    cited = cited_numbers(record.get("answer"))
    items = {
        str(n): {"surfaced": n in surfaced, "opened": n in opened, "cited": n in cited}
        for n in target_items(case)
    }
    facts = {
        phrase: {"in_context": phrase.lower() in context, "said": phrase.lower() in answer}
        for phrase in case.get("must_mention", [])
    }
    return {"items": items, "facts": facts}


def score(case, record):
    answer = record.get("answer")
    missing_cites, hit_cites = score_citations(case, answer)
    missing_tools = score_tools(case, record.get("calls", []))
    missing_mentions = score_mentions(case, answer)
    forbidden = score_forbidden(case, answer)
    missing_any = score_match_any(case, answer)

    return {
        "case_id": case["id"],
        "run_id": record["run_id"],
        "passed": not (missing_cites or missing_tools or missing_mentions or forbidden or missing_any),
        "missing_cites": missing_cites,
        "found_cites": hit_cites,
        "missing_tools": missing_tools,
        "missing_mentions": missing_mentions,
        "forbidden_hits": forbidden,
        "missing_any": missing_any,
        "termination": record["termination"],
        "tool_calls": record["tool_calls"],
        "tools_called": record.get("tools_called", []),
        "latency_ms": record["latency_ms"],
        "tokens_in": record.get("tokens_in", 0),
        "tokens_out": record.get("tokens_out", 0),
        "cache_misses": sum(1 for c in record.get("calls", []) if c.get("cache_hit") is False),
        "tool_errors": sum(1 for c in record.get("calls", []) if c.get("tool_error")),
        "stages": score_stages(case, record),
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
        unmatched = Counter(item for r in runs for item in r.get("missing_any", []))
        stages = {"items": {}, "facts": {}}
        for r in runs:
            for group in ("items", "facts"):
                for key, flags in r["stages"][group].items():
                    counts = stages[group].setdefault(key, Counter())
                    counts.update(flag for flag, value in flags.items() if value)
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
            "missing_any": dict(unmatched),
            "stages": {group: {key: dict(c) for key, c in found.items()} for group, found in stages.items()},
        })
    return rows


def report_stages(rows):
    print("\nwhere required evidence was lost (runs out of n)")
    print(f"{'case':<34} {'needed':<18} {'surfaced':>9} {'opened':>7} {'cited':>6}   {'in context':>10} {'said':>5}")
    print("-" * 98)
    for row in rows:
        n = row["runs"]
        first = True
        for number, c in row["stages"]["items"].items():
            label = row["case_id"] if first else ""
            print(f"{label:<34} #{number:<17} {c.get('surfaced', 0):>5}/{n:<3} {c.get('opened', 0):>3}/{n:<3} {c.get('cited', 0):>2}/{n:<3}")
            first = False
        for phrase, c in row["stages"]["facts"].items():
            label = row["case_id"] if first else ""
            print(f"{label:<34} {repr(phrase):<18} {'':>9} {'':>7} {'':>6}   {c.get('in_context', 0):>6}/{n:<3} {c.get('said', 0):>2}/{n:<3}")
            first = False


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
    k = min(row["runs"] for row in rows)
    any_pass = sum(row["passed"] > 0 for row in rows)
    all_pass = sum(row["passed"] == row["runs"] for row in rows)
    print(f"pass@{k} (at least one run passed): {any_pass}/{len(rows)} cases   "
          f"pass^{k} (every run passed): {all_pass}/{len(rows)} cases")
    report_stages(rows)

    for row in rows:
        if row["passed"] == row["runs"]:
            continue
        n = row["runs"]
        print(f"\n{row['case_id']}  ({row['passed']}/{n} passed) — how often each problem occurred")
        for label, key in (("missing citation", "missing_cites"), ("tool not called", "missing_tools"),
                           ("never mentioned", "missing_mentions"), ("said banned thing", "forbidden_hits"),
                           ("no required phrase", "missing_any")):
            for item, count in sorted(row[key].items(), key=lambda kv: -kv[1]):
                print(f"  {label:<18} {str(item):<28} {count}/{n}")


def rescore(path):
    results_file = Path(path).resolve()
    runs_file = results_file.parents[1] / "traces" / "runs.jsonl"
    records = {}
    for line in runs_file.read_text().splitlines():
        record = json.loads(line)
        records[record["run_id"]] = record

    cases = {c["id"]: c for c in yaml.safe_load(GOLDEN.read_text())}
    saved = json.loads(results_file.read_text())
    results = [score(cases[r["case_id"]], records[r["run_id"]]) for r in saved["runs"]]

    before = sum(r["passed"] for r in saved["runs"])
    after = sum(r["passed"] for r in results)
    print(f"rescored {results_file.name}: {before}/{len(results)} as saved -> {after}/{len(results)} with current grader")
    report(saved["meta"], summarise(results))


def model_group(model):
    return "small" if any(tag in model for tag in SMALL_MODELS) else "large"


def tokens_used_today(model):
    if not RUNS_FILE.exists():
        return 0
    today = datetime.now(timezone.utc).date().isoformat()
    group = model_group(model)
    used = 0
    for line in RUNS_FILE.read_text().splitlines():
        run = json.loads(line)
        if not run.get("started_at", "").startswith(today):
            continue
        if model_group(run.get("model") or "") != group:
            continue
        used += run.get("tokens_in", 0) + run.get("tokens_out", 0)
    return used


def check_budget(model, jobs, allow_paid):
    limit = FREE_TOKENS[model_group(model)]
    used = tokens_used_today(model)
    needed = jobs * TOKENS_PER_RUN
    print(f"free tokens today (UTC): {used:,} used of {limit:,}; this run needs about {needed:,}")
    if used + needed <= limit or allow_paid:
        return True
    print("this run would go past today's free tokens. Wait for 00:00 UTC, run fewer cases, or pass --allow-paid.")
    return False


def main():
    from agent.graph import MODEL, ask, build

    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="*")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--prompt", default="v0")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--reasoning", default=None, choices=["none", "low", "medium", "high"])
    parser.add_argument("--rescore", default=None, help="re-grade a saved results file without rerunning the agent")
    parser.add_argument("--check", action="store_true", help="add the answer-checking node before the final answer")
    parser.add_argument("--allow-paid", action="store_true", help="run even if it goes past today's free tokens")
    args = parser.parse_args()

    if args.rescore:
        rescore(args.rescore)
        return

    cases = yaml.safe_load(GOLDEN.read_text())
    if args.cases:
        cases = [c for c in cases if c["id"] in args.cases]

    agent = build(args.prompt, args.reasoning, args.check)
    jobs = [case for case in cases for _ in range(args.repeat)]
    if not check_budget(MODEL, len(jobs), args.allow_paid):
        return
    print(f"running {len(jobs)} runs ({len(cases)} cases x {args.repeat}) with prompt {args.prompt} ...", flush=True)

    def execute(case):
        record = ask(case["q"], case_id=case["id"], agent=agent, prompt_version=args.prompt, reasoning=args.reasoning, check=args.check)
        result = score(case, record)
        print(f"  {case['id']:<34} {'PASS' if result['passed'] else 'fail'}", flush=True)
        return result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(execute, jobs))

    meta = {
        "model": MODEL,
        "prompt_version": args.prompt,
        "check": args.check,
        "reasoning": args.reasoning or "default",
        "repeat": args.repeat,
        "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    rows = summarise(results)
    report(meta, rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = f"{args.prompt}_check" if args.check else args.prompt
    out = RESULTS_DIR / f"{stamp}_{label}_r-{args.reasoning or 'default'}_x{args.repeat}.json"
    out.write_text(json.dumps({"meta": meta, "summary": rows, "runs": results}, indent=2))
    print(f"\nwritten to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
