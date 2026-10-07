import ast
import json
import os
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agent import tools as traced_tools
from agent.prompts import DEFAULT_PROMPT, PROMPTS
from agent.trace import ROOT, current_run, run

load_dotenv(ROOT / ".env")

MODEL = os.environ.get("MODEL", "gpt-5.1")
MAX_STEPS = 12
RECURSION_LIMIT = MAX_STEPS * 3 + 10
OUT_OF_STEPS = """You have used your whole step budget and cannot call any more tools.
Answer now using only what you have already found. Say clearly what you could not check."""

def unwrap(result):
    data, _ = result
    return json.dumps(data, default=str)


SPARSE_RESULTS = 2


def sparse_note(query, count):
    words = len(query.split())
    if words > 1:
        return (f"Only {count} result(s). Every word in the query must appear in a result, "
                f"so each of your {words} words removes matches. Search again with a single "
                f"identifier or fewer words before concluding anything.")
    return (f"Only {count} result(s) for this single word. Try a different identifier, "
            f"or a different kind, before concluding anything.")


SEARCH_STREAK = 3


def earlier_calls():
    active = current_run.get()
    return active["calls"] if active else []


def plain(value):
    try:
        return str(ast.literal_eval(value))
    except (ValueError, SyntaxError):
        return value.strip("'\"")


def search_key(query, kind):
    return frozenset(query.lower().split()), kind


def key_of(call):
    args = [plain(a) for a in call["args"]]
    kind = args[1] if len(args) > 1 else plain(call["kwargs"].get("kind", "'any'"))
    return search_key(args[0], kind)


def loop_note(query, kind):
    calls = earlier_calls()
    if any(c["tool"] == "search" and key_of(c) == search_key(query, kind) for c in calls):
        return ("You already ran this search earlier (same words, same kind), so these are the same results. "
                "Open a promising result with get_pr or get_issue, or search for something different.")
    streak = 0
    for call in reversed(calls):
        if call["tool"] != "search":
            break
        streak += 1
    if streak >= SEARCH_STREAK:
        return (f"This is search number {streak + 1} in a row without opening any result. Titles alone rarely "
                "answer the question: open the most promising result with get_pr or get_issue before searching again.")
    return None


@tool(parse_docstring=True)
def search(query: str, kind: Literal["any", "pr", "issue"] = "any") -> str:
    """Find pull requests and issues in langchain-ai/langgraph whose text contains every word of the query.

    Returns up to 10 matches, each with number, kind (pr or issue), state and title. May add a note about the search itself.

    Args:
        query: Keywords only. Every word must appear in a result, so each extra word removes matches.
        kind: Which items to return. "pr" returns only pull requests, "issue" only issues, "any" both.
    """
    repeat = loop_note(query, kind)
    results, _ = traced_tools.search(query, kind)
    if isinstance(results, str):
        return results
    notes = [n for n in (repeat, sparse_note(query, len(results)) if len(results) <= SPARSE_RESULTS else None) if n]
    if not notes:
        return json.dumps(results, default=str)
    return json.dumps({"results": results, "note": " ".join(notes)}, default=str)


@tool(parse_docstring=True)
def get_pr(number: int) -> str:
    """Get one pull request: title, body, state, merged, dates, author, labels, comment counts and merge commit.

    Args:
        number: The pull request number, without the #.
    """
    return unwrap(traced_tools.get_pr(number))


@tool(parse_docstring=True)
def get_issue(number: int) -> str:
    """Get one issue: title, body, state, state_reason, dates, author and labels.

    Args:
        number: The issue number, without the #.
    """
    return unwrap(traced_tools.get_issue(number))


@tool(parse_docstring=True)
def get_timeline(number: int) -> str:
    """Get what happened to a pull request or issue after it was opened.

    Returns cross_references (every pull request or issue that mentions it, including ones filed later, with their state), history (merged, closed, reopened, reviewed, labeled and renamed events with who did them) and reviewed (whether a human ever reviewed it).

    Args:
        number: The pull request or issue number, without the #.
    """
    return unwrap(traced_tools.get_timeline(number))


@tool(parse_docstring=True)
def get_comments(number: int) -> str:
    """Get the conversation on a pull request or issue: each comment's author, author_type (User or Bot), date and body.

    Args:
        number: The pull request or issue number, without the #.
    """
    return unwrap(traced_tools.get_comments(number))


@tool(parse_docstring=True)
def version_for_pr(number: int) -> str:
    """Get the GitHub releases whose notes list a pull request, with package name, version and publish date.

    Args:
        number: The pull request number, without the #.
    """
    return unwrap(traced_tools.version_for_pr(number))


TOOLBOX = [search, get_pr, get_issue, get_timeline, get_comments, version_for_pr]


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    steps: int
    out_of_steps: bool


def build(prompt_version=DEFAULT_PROMPT, reasoning=None):
    options = {"reasoning_effort": reasoning} if reasoning else {}
    model = ChatOpenAI(model=MODEL, **options)
    model_with_tools = model.bind_tools(TOOLBOX)
    tools_by_name = {t.name: t for t in TOOLBOX}
    system = SystemMessage(PROMPTS[prompt_version])

    def agent_node(state):
        reply = model_with_tools.invoke([system] + state["messages"])
        return {"messages": [reply], "steps": state["steps"] + 1}

    def tools_node(state):
        results = []
        for call in state["messages"][-1].tool_calls:
            chosen = tools_by_name.get(call["name"])
            if chosen is None:
                output = f"unknown tool {call['name']}"
            else:
                try:
                    output = chosen.invoke(call["args"])
                except Exception as exc:
                    output = f"tool {call['name']} failed: {type(exc).__name__}: {exc}"
            results.append(ToolMessage(content=output, tool_call_id=call["id"]))
        return {"messages": results}

    def finalize_node(state):
        skipped = [
            ToolMessage(content="not run: step budget exhausted", tool_call_id=call["id"])
            for call in state["messages"][-1].tool_calls
        ]
        history = [system] + state["messages"] + skipped + [HumanMessage(OUT_OF_STEPS)]
        reply = model.invoke(history)
        return {"messages": skipped + [reply], "out_of_steps": True}

    def route(state):
        if not state["messages"][-1].tool_calls:
            return END
        if state["steps"] >= MAX_STEPS:
            return "finalize"
        return "tools"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("finalize", finalize_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, ["tools", "finalize", END])
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)
    return graph.compile()


def ask(question, case_id=None, agent=None, prompt_version=DEFAULT_PROMPT, reasoning=None):
    agent = agent or build(prompt_version, reasoning)

    with run(question, case_id=case_id) as record:
        record["model"] = MODEL
        record["prompt_version"] = prompt_version
        record["reasoning"] = reasoning or "default"
        try:
            state = agent.invoke(
                {"messages": [HumanMessage(question)], "steps": 0, "out_of_steps": False},
                config={"recursion_limit": RECURSION_LIMIT},
            )
        except GraphRecursionError:
            record["termination"] = "step_limit"
            record["answer"] = None
            return record

        messages = state["messages"]
        record["answer"] = messages[-1].content
        record["steps"] = state["steps"]
        if state["out_of_steps"]:
            record["termination"] = "step_limit"
        for message in messages:
            usage = getattr(message, "usage_metadata", None) or {}
            record["tokens_in"] += usage.get("input_tokens", 0)
            record["tokens_out"] += usage.get("output_tokens", 0)
        return record


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "time travel to a subgraph checkpoint with an interrupt gives me the wrong state - is this fixed"
    result = ask(question)
    print("=" * 70)
    print(result["answer"])
    print("=" * 70)
    print(f"termination={result['termination']} tool_calls={result['tool_calls']} "
          f"latency={result['latency_ms']}ms tokens_in={result['tokens_in']} tokens_out={result['tokens_out']}")
    print("path:", " -> ".join(result["tools_called"]))
