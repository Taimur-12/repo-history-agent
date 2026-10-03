import json, threading, time, uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACE_DIR = ROOT / "data" / "traces"
CALLS_FILE = TRACE_DIR / "calls.jsonl"
RUNS_FILE = TRACE_DIR / "runs.jsonl"
MAX_ARG_CHARS = 200

current_run = ContextVar("current_run", default=None)
write_lock = threading.Lock()


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append(path, record):
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record) + "\n"
    with write_lock, path.open("a") as handle:
        handle.write(line)


@contextmanager
def run(question, case_id=None):
    record = {
        "run_id": uuid.uuid4().hex[:12],
        "case_id": case_id,
        "question": question,
        "started_at": now(),
        "calls": [],
        "termination": "model_stopped",
        "answer": None,
        "tokens_in": 0,
        "tokens_out": 0,
    }
    token = current_run.set(record)
    started = time.perf_counter()
    try:
        yield record
    except Exception as exc:
        record["termination"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        current_run.reset(token)
        record["latency_ms"] = round((time.perf_counter() - started) * 1000)
        record["tool_calls"] = len(record["calls"])
        record["tools_called"] = [call["tool"] for call in record["calls"]]
        append(RUNS_FILE, record)


def summarise(value):
    text = repr(value)
    return text if len(text) <= MAX_ARG_CHARS else text[:MAX_ARG_CHARS] + "..."


def traced(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        started = time.perf_counter()
        record = {
            "run_id": (current_run.get() or {}).get("run_id"),
            "at": now(),
            "tool": fn.__name__,
            "args": [summarise(a) for a in args],
            "kwargs": {k: summarise(v) for k, v in kwargs.items()},
            "cache_hit": None,
            "tool_error": False,
            "failed": False,
        }

        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            record["failed"] = True
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        else:
            data = result
            if isinstance(result, tuple) and len(result) == 2:
                data, record["cache_hit"] = result
            record["tool_error"] = isinstance(data, str)
            record["result_chars"] = len(json.dumps(data, default=str))
            if record["tool_error"]:
                record["error"] = data
            return result
        finally:
            record["latency_ms"] = round((time.perf_counter() - started) * 1000)
            append(CALLS_FILE, record)
            active = current_run.get()
            if active is not None:
                active["calls"].append(record)

    return wrapper
