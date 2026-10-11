"""Lesson 5.2 - Conditional edges: the graph DECIDES where to go next. (No LLM.)"""
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict, total=False):
    amount: float
    decision: str


def receive(state: State) -> dict:
    print(f"   received a refund request for ${state['amount']}")
    return {}


def agent_approves(state: State) -> dict:
    return {"decision": "approved by the Sales Support Agent"}


def manager_approves(state: State) -> dict:
    return {"decision": "sent to the Sales Manager"}


# The ROUTING FUNCTION reads the state and returns the NAME of the next node.
# (Rule from the Chinook handbook, page 3: agents may approve refunds up to $20.)
def who_approves(state: State) -> str:
    return "agent_approves" if state["amount"] <= 20 else "manager_approves"


builder = StateGraph(State)
builder.add_node("receive", receive)
builder.add_node("agent_approves", agent_approves)
builder.add_node("manager_approves", manager_approves)
builder.add_edge(START, "receive")
builder.add_conditional_edges("receive", who_approves, ["agent_approves", "manager_approves"])
builder.add_edge("agent_approves", END)
builder.add_edge("manager_approves", END)
graph = builder.compile()

for amount in [9.99, 20, 49.50]:
    print(f"\nRefund ${amount}:")
    print("   →", graph.invoke({"amount": amount})["decision"])

print("\n" + graph.get_graph().draw_mermaid())   # dotted arrows = conditional edges
