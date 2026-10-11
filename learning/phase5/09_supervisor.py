"""Lesson 5.6 - Multi-agent: a SUPERVISOR sends each question to the right specialist.

Specialist 1: the handbook RAG assistant (Phase 4)  - company policies
Specialist 2: Text2SQL (the project)                - data in the database
Run from the Gen_AI folder. Needs learning/phase4/chroma_db from Lesson 4.3 (Script 11).
"""
import sys
from pathlib import Path
from typing import Literal, TypedDict

# This script lives in learning/phase5/, but imports the project package `text2sql`.
# Python only looks for packages next to the script, so add the Gen_AI folder to the search path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from text2sql.app.graph import build_sql_graph

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

# ---------- specialist 1: handbook RAG (Lesson 4.4) ----------
handbook = Chroma(
    collection_name="chinook_handbook",
    embedding_function=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"),
    persist_directory="learning/phase4/chroma_db",
    collection_metadata={"hnsw:space": "cosine"},
).as_retriever(search_kwargs={"k": 3})

handbook_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using ONLY the handbook context. Cite pages like (page 3). If the context "
               "doesn't contain the answer, say \"I don't know based on the handbook.\"\n\n{context}"),
    ("human", "{question}"),
])

# ---------- specialist 2: Text2SQL graph (the project, v0.4) ----------
sql_graph = build_sql_graph()   # a compiled graph used INSIDE another graph's node


# ---------- the supervisor ----------
class Choice(BaseModel):
    specialist: Literal["handbook", "database", "none"] = Field(
        description="handbook = company POLICIES and rules (refunds, prices, support hours, pay, "
                    "vacation, passwords). database = FACTS and numbers about customers, sales, "
                    "invoices, employees, artists, albums, tracks. none = unrelated.")


supervisor_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a supervisor. Choose which specialist should answer."),
        ("human", "{question}"),
    ])
    | llm.with_structured_output(Choice)
)


class State(TypedDict, total=False):
    question: str
    specialist: str
    answer: str


def supervisor(state: State) -> dict:
    return {"specialist": supervisor_chain.invoke({"question": state["question"]}).specialist}


def handbook_agent(state: State) -> dict:
    docs = handbook.invoke(state["question"])
    context = "\n\n".join(f"[page {d.metadata['page_label']}]\n{d.page_content}" for d in docs)
    answer = (handbook_prompt | llm | StrOutputParser()).invoke(
        {"context": context, "question": state["question"]})
    return {"answer": str(answer)}


def database_agent(state: State) -> dict:
    return {"answer": sql_graph.invoke({"question": state["question"]})["answer"]}


def no_agent(state: State) -> dict:
    return {"answer": "Neither specialist can help with that."}


builder = StateGraph(State)
for name, fn in [("supervisor", supervisor), ("handbook_agent", handbook_agent),
                 ("database_agent", database_agent), ("no_agent", no_agent)]:
    builder.add_node(name, fn)
builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    lambda s: {"handbook": "handbook_agent", "database": "database_agent"}.get(s["specialist"], "no_agent"),
    ["handbook_agent", "database_agent", "no_agent"],
)
for name in ["handbook_agent", "database_agent", "no_agent"]:
    builder.add_edge(name, END)
graph = builder.compile()

for q in ["Who has to approve a $50 refund?",
          "How many customers do we have in Canada?",
          "How many vacation days do full-time employees get?",
          "Which sales agent sold the most in 2023?",
          "What's a good pizza place nearby?"]:
    out = graph.invoke({"question": q})
    print(f"\nQ: {q}\n   → {out['specialist']}: {out['answer']}")
