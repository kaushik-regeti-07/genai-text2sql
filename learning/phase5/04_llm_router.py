"""Lesson 5.2 - An LLM router: the Lesson 3.4 classifier becomes the first node of a graph."""
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")


class Route(BaseModel):
    category: Literal["database_question", "greeting", "off_topic"] = Field(
        description="database_question = anything about the music store's customers, sales, "
                    "artists, albums, tracks or employees. greeting = hello/thanks/small talk. "
                    "off_topic = anything else.")


classifier = (
    ChatPromptTemplate.from_messages([
        ("system", "Classify the user's message for a music-store database assistant."),
        ("human", "{question}"),
    ])
    | llm.with_structured_output(Route)
)


class State(TypedDict, total=False):
    question: str
    category: str
    reply: str


def classify(state: State) -> dict:                       # LLM node
    return {"category": classifier.invoke({"question": state["question"]}).category}


def greet(state: State) -> dict:                          # plain Python nodes, no LLM needed
    return {"reply": "Hi! Ask me anything about the Chinook music store."}


def database(state: State) -> dict:
    return {"reply": f"(would run Text2SQL for: {state['question']!r})"}


def refuse(state: State) -> dict:
    return {"reply": "Sorry, I only answer questions about the music store."}


builder = StateGraph(State)
for name, fn in [("classify", classify), ("greet", greet), ("database", database), ("refuse", refuse)]:
    builder.add_node(name, fn)
builder.add_edge(START, "classify")
builder.add_conditional_edges(
    "classify",
    lambda s: {"greeting": "greet", "database_question": "database"}.get(s["category"], "refuse"),
    ["greet", "database", "refuse"],
)
for name in ["greet", "database", "refuse"]:
    builder.add_edge(name, END)
graph = builder.compile()

for q in ["Hello!", "Which artist has the most albums?", "What's the weather in Paris?"]:
    out = graph.invoke({"question": q})
    print(f"{q:38} → {out['category']:18} → {out['reply']}")
