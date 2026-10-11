"""Lesson 5.1 - Your first graph: State, nodes, edges. (No LLM, no API calls.)"""
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


# 1. STATE: the "folder" that travels through the graph.
class State(TypedDict):
    question: str
    cleaned: str
    word_count: int
    report: str


# 2. NODES: plain Python functions. Each one reads the state and returns ONLY what it changes.
def clean(state: State) -> dict:
    return {"cleaned": state["question"].strip().lower()}


def count_words(state: State) -> dict:
    return {"word_count": len(state["cleaned"].split())}


def make_report(state: State) -> dict:
    return {"report": f"'{state['cleaned']}' has {state['word_count']} words"}


# 3. BUILD: add nodes, connect them with edges, compile.
builder = StateGraph(State)
builder.add_node("clean", clean)
builder.add_node("count_words", count_words)
builder.add_node("make_report", make_report)
builder.add_edge(START, "clean")
builder.add_edge("clean", "count_words")
builder.add_edge("count_words", "make_report")
builder.add_edge("make_report", END)
graph = builder.compile()

# 4. RUN: invoke = run to the end and return the final state
final = graph.invoke({"question": "   Which GENRE sells the most?  "})
print("Final state:", final)

# 5. WATCH: stream(..., stream_mode="updates") shows what each node changed, step by step
print("\nStep by step:")
for step in graph.stream({"question": "How many customers are from Brazil?"}, stream_mode="updates"):
    print("  ", step)

# 6. SEE the graph (paste this text into https://mermaid.live to get a picture)
print("\nMermaid diagram:\n" + graph.get_graph().draw_mermaid())
