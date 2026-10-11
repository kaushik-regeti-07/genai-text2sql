"""Lesson 5.1 - Reducers: does a node's update REPLACE a value or ADD to it? (No LLM.)"""
import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class State(TypedDict):
    current_step: str                                  # no reducer  -> each update REPLACES it
    visited: Annotated[list[str], operator.add]        # operator.add -> each update is APPENDED
    messages: Annotated[list, add_messages]            # add_messages -> chat messages appended


def step_a(state: State) -> dict:
    return {"current_step": "A", "visited": ["A"], "messages": [AIMessage("Hello from A")]}


def step_b(state: State) -> dict:
    return {"current_step": "B", "visited": ["B"], "messages": [AIMessage("Hello from B")]}


builder = StateGraph(State)
builder.add_node("step_a", step_a)
builder.add_node("step_b", step_b)
builder.add_edge(START, "step_a")
builder.add_edge("step_a", "step_b")
builder.add_edge("step_b", END)
graph = builder.compile()

final = graph.invoke({"current_step": "start", "visited": ["start"],
                      "messages": [HumanMessage("Hi graph")]})
print("current_step:", final["current_step"])            # 'B'          (replaced)
print("visited:     ", final["visited"])                 # ['start', 'A', 'B']  (appended)
print("messages:")
for m in final["messages"]:
    print(f"   {type(m).__name__:12} {m.content}")       # all 3 kept, in order
