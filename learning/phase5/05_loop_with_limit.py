"""Lesson 5.3 - Loops: an edge that points BACKWARDS, plus a limit so it can't loop forever. (No LLM.)"""
from typing import TypedDict

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

MAX_ATTEMPTS = 3


class State(TypedDict, total=False):
    attempts: int
    succeeded: bool
    log: list[str]


def call_flaky_service(state: State) -> dict:
    """Pretend service that fails on the first 2 calls and works on the 3rd."""
    attempts = state.get("attempts", 0) + 1
    ok = attempts >= 3
    log = state.get("log", []) + [f"attempt {attempts}: {'OK' if ok else 'failed'}"]
    return {"attempts": attempts, "succeeded": ok, "log": log}


def retry_or_stop(state: State) -> str:
    if state["succeeded"]:
        return "done"
    if state["attempts"] < MAX_ATTEMPTS:
        return "call_flaky_service"            # ← the loop: go BACK to the same node
    return "give_up"


def done(state: State) -> dict:
    return {"log": state["log"] + ["finished successfully"]}


def give_up(state: State) -> dict:
    return {"log": state["log"] + [f"gave up after {MAX_ATTEMPTS} attempts"]}


builder = StateGraph(State)
builder.add_node("call_flaky_service", call_flaky_service)
builder.add_node("done", done)
builder.add_node("give_up", give_up)
builder.add_edge(START, "call_flaky_service")
builder.add_conditional_edges("call_flaky_service", retry_or_stop,
                              ["call_flaky_service", "done", "give_up"])
builder.add_edge("done", END)
builder.add_edge("give_up", END)
graph = builder.compile()

for line in graph.invoke({})["log"]:
    print("  ", line)

# Safety net #2: LangGraph's own recursion limit (max number of steps per run).
# Even if YOUR limit had a bug, the graph stops instead of looping forever.
try:
    graph.invoke({}, {"recursion_limit": 2})
except GraphRecursionError as e:
    print("\nRecursion limit hit:", str(e).splitlines()[0])
