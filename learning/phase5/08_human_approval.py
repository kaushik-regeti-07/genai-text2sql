"""Lesson 5.5 - Human-in-the-loop: interrupt() pauses the graph; Command(resume=...) continues it. (No LLM.)"""
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class State(TypedDict, total=False):
    customer: str
    amount: float
    status: str


def check_amount(state: State) -> dict:
    return {}


def ask_manager(state: State) -> dict:
    # The graph STOPS here. The value passed to interrupt() is shown to the human.
    answer = interrupt(f"Approve a ${state['amount']} refund for {state['customer']}? (yes/no)")
    # ...and continues HERE when someone calls graph.invoke(Command(resume=<answer>), config)
    return {"status": "approved by manager" if answer == "yes" else "rejected by manager"}


def auto_approve(state: State) -> dict:
    return {"status": "approved automatically (≤ $20)"}


builder = StateGraph(State)
builder.add_node("check_amount", check_amount)
builder.add_node("ask_manager", ask_manager)
builder.add_node("auto_approve", auto_approve)
builder.add_edge(START, "check_amount")
builder.add_conditional_edges("check_amount",
                              lambda s: "auto_approve" if s["amount"] <= 20 else "ask_manager",
                              ["auto_approve", "ask_manager"])
builder.add_edge("ask_manager", END)
builder.add_edge("auto_approve", END)
graph = builder.compile(checkpointer=InMemorySaver())     # pausing NEEDS a checkpointer

# Small refund: no pause
small = {"configurable": {"thread_id": "refund-1"}}
print(graph.invoke({"customer": "Ravi", "amount": 9.99}, small)["status"])

# Big refund: the graph pauses
big = {"configurable": {"thread_id": "refund-2"}}
out = graph.invoke({"customer": "Priya", "amount": 49.50}, big)
question = out["__interrupt__"][0].value
print("\nPAUSED. Question for the human:", question)
print("Waiting at node:", graph.get_state(big).next)

answer = input("Your decision (yes/no): ").strip().lower() or "yes"
final = graph.invoke(Command(resume=answer), big)         # resume from where it stopped
print("Final status:", final["status"])
