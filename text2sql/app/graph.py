"""Text2SQL as a LangGraph graph.

v0.4  build_sql_graph():  find tables -> write SQL -> validate -> execute -> explain,
                          looping back to write_sql when a check or the database complains.
v0.5  build_chat_graph(): + a router (greeting / off-topic / database question),
                          + memory across turns (checkpointer) with follow-up rewriting,
                          + optional human approval before a query runs (interrupt).
v1.0  + persistent memory (SQLite checkpointer), a results table and a per-turn history
        (`turns`) so a web UI or API can redraw any conversation.
"""
import operator
import sqlite3
from pathlib import Path
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from text2sql.app.config import MAX_ATTEMPTS, TOP_K
from text2sql.app.db import check_sql, check_values, get_db, run_table
from text2sql.app.llm import get_llm
from text2sql.app.prompts import (ANSWER_PROMPT, GREETING_REPLY, OFF_TOPIC_REPLY, RETRY_MESSAGE,
                                  ROUTER_PROMPT, SQL_PROMPT)
from text2sql.app.schema_index import find_tables
from text2sql.app.schemas import Route, SQLQuery


class State(TypedDict, total=False):
    """The 'folder' that travels through the graph. Each node reads it and returns updates."""
    messages: Annotated[list[AnyMessage], add_messages]  # conversation (APPENDED, never replaced)
    route: str                        # database_question / greeting / off_topic
    question: str                     # the standalone question we're answering
    tables: list[str]                 # tables picked by schema RAG
    schema: str                       # their CREATE TABLE text
    sql_query: Optional[SQLQuery]     # the LLM's latest attempt
    attempts: int                     # how many times SQL was written for this question
    error: Optional[str]              # why the latest attempt failed (None = no problem)
    fix_log: list[str]                # every problem that was fixed along the way (for display)
    widened: bool                     # True once we fell back to the full schema
    result: Optional[str]             # rows, or NOT_ANSWERABLE / ERROR / CANCELLED
    columns: list[str]                # v1.0: result table for the UI / API
    rows: list[list]
    answer: Optional[str]             # the final English answer
    turns: Annotated[list[dict], operator.add]   # v1.0: one record per question, APPENDED (history)


# ======================= nodes (each = one job) =======================

def router(state: State) -> dict:
    """v0.5: classify the latest message and rewrite follow-ups into standalone questions.
    Also RESETS the per-question fields, because with memory the old values would still be there."""
    decision = (ROUTER_PROMPT | get_llm().with_structured_output(Route)).invoke(
        {"messages": state["messages"]})
    latest = state["messages"][-1].content
    return {
        "route": decision.category,
        "question": str(decision.standalone_question or latest),
        "tables": [], "schema": "", "sql_query": None, "attempts": 0,
        "error": None, "fix_log": [], "widened": False, "result": None,
        "columns": [], "rows": [], "answer": None,
    }


def _turn(state: State, answer: str, **extra) -> dict:
    """One history record (v1.0): everything a UI needs to redraw this question later."""
    query = state.get("sql_query")
    record = {
        "user": state["messages"][-1].content if state.get("messages") else state.get("question"),
        "route": state.get("route", "database_question"),
        "question": state.get("question"),
        "tables": state.get("tables") or [],
        "sql": query.sql if query is not None else "",
        "fix_log": state.get("fix_log") or [],
        "columns": state.get("columns") or [],
        "rows": state.get("rows") or [],
        "result": state.get("result"),
        "answer": answer,
    }
    record.update(extra)
    return record


def greet(state: State) -> dict:
    return {"answer": GREETING_REPLY, "messages": [AIMessage(GREETING_REPLY)],
            "turns": [_turn(state, GREETING_REPLY)]}


def off_topic(state: State) -> dict:
    return {"answer": OFF_TOPIC_REPLY, "messages": [AIMessage(OFF_TOPIC_REPLY)],
            "turns": [_turn(state, OFF_TOPIC_REPLY)]}


def pick_tables(state: State) -> dict:
    """Schema RAG from v0.3: only the relevant tables' schema goes to the LLM."""
    tables = find_tables(state["question"])
    return {"tables": tables, "schema": get_db().get_table_info(tables)}


def write_sql(state: State) -> dict:
    """Ask the LLM for SQL. On a retry, show it its previous query and what went wrong."""
    db = get_db()
    messages = SQL_PROMPT.invoke({
        "dialect": db.dialect, "top_k": TOP_K,
        "schema": state["schema"], "question": state["question"],
    }).to_messages()
    fix_log = list(state.get("fix_log") or [])
    if state.get("error") and state.get("sql_query"):        # this is a retry
        messages += [
            AIMessage(state["sql_query"].sql),
            HumanMessage(RETRY_MESSAGE.format(sql=state["sql_query"].sql, error=state["error"])),
        ]
        fix_log.append(state["error"])
    query = get_llm().with_structured_output(SQLQuery).invoke(messages)
    return {"sql_query": query, "attempts": state.get("attempts", 0) + 1,
            "error": None, "fix_log": fix_log}


def widen_schema(state: State) -> dict:
    """The LLM said 'not answerable' but only saw SOME tables. Schema RAG may have missed one
    (e.g. Artist for 'tracks by Nirvana'), so try once more with every table."""
    db = get_db()
    tables = db.get_usable_table_names()
    note = (f"Not answerable with tables {', '.join(state['tables'])}; "
            f"retrying with all {len(tables)} tables.")
    return {"tables": tables, "schema": db.get_table_info(tables), "widened": True,
            "fix_log": list(state.get("fix_log") or []) + [note]}


def validate(state: State) -> dict:
    """Check the SQL BEFORE running it: safe? (parsed allowlist) and do its text values exist?"""
    sql = state["sql_query"].sql
    problem = check_sql(sql)
    if problem is None:
        try:
            problem = check_values(sql)
        except Exception:        # the value check is a helper; if it fails, just skip it
            problem = None
    return {"error": problem}


def approve(state: State) -> dict:
    """v0.5: pause the graph and ask a human. The program resumes with Command(resume=...)."""
    decision = interrupt({"question": state["question"], "tables": state["tables"],
                          "sql": state["sql_query"].sql})
    if str(decision).strip().lstrip("﻿").lower() in {"y", "yes"}:   # ﻿: stray marker some terminals add
        return {}
    return {"result": "CANCELLED: the user did not approve running this query."}


def execute(state: State) -> dict:
    ok, as_text, columns, rows = run_table(state["sql_query"].sql)
    if not ok:
        return {"error": as_text}
    return {"result": as_text, "columns": columns, "rows": rows}


def explain(state: State) -> dict:
    """Turn whatever happened into an English answer, and add it to the conversation."""
    query = state.get("sql_query")
    if query is not None and not query.is_answerable:
        result = f"NOT_ANSWERABLE: {query.explanation}"
    elif state.get("error"):                         # ran out of attempts
        result = f"ERROR: {state['error']}"
    else:
        result = state.get("result") or "(no rows)"
    answer = str((ANSWER_PROMPT | get_llm() | StrOutputParser()).invoke(
        {"question": state["question"], "result": result}))
    return {"result": result, "answer": answer, "messages": [AIMessage(answer)],
            "turns": [_turn(state, answer, result=result)]}


# ======================= routing (which node runs next) =======================

def after_router(state: State) -> str:
    return {"greeting": "greet", "off_topic": "off_topic"}.get(state["route"], "pick_tables")


def after_write(state: State) -> str:
    query = state["sql_query"]
    if query.is_answerable and query.sql.strip():
        return "validate"
    # "Not answerable" may just mean schema RAG missed a table: widen once, if attempts remain.
    all_tables = get_db().get_usable_table_names()
    if (not state.get("widened") and len(state.get("tables") or []) < len(all_tables)
            and state["attempts"] < MAX_ATTEMPTS):
        return "widen_schema"
    return "explain"


def retry_or_give_up(state: State) -> str:
    return "write_sql" if state["attempts"] < MAX_ATTEMPTS else "explain"


def after_validate_auto(state: State) -> str:
    return retry_or_give_up(state) if state.get("error") else "execute"


def after_validate_with_approval(state: State) -> str:
    return retry_or_give_up(state) if state.get("error") else "approve"


def after_approve(state: State) -> str:
    return "explain" if (state.get("result") or "").startswith("CANCELLED") else "execute"


def after_execute(state: State) -> str:
    return retry_or_give_up(state) if state.get("error") else "explain"


# ======================= graphs =======================

def make_checkpointer(path: Optional[Path] = None):
    """Memory for build_chat_graph.

    path=None  -> InMemorySaver: forgotten when the program stops (Phase 5).
    path=file  -> SqliteSaver:   saved to a SQLite file, survives restarts (v1.0).

    The state holds our own SQLQuery objects; when memory is loaded back, LangGraph only
    rebuilds custom classes we explicitly allow (a safety feature)."""
    serde = JsonPlusSerializer(allowed_msgpack_modules=[("text2sql.app.schemas", "SQLQuery")])
    if path is None:
        return InMemorySaver(serde=serde)
    path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: web servers (Streamlit, FastAPI) call from different threads
    conn = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(conn, serde=serde)


def list_threads(checkpointer) -> list[str]:
    """All saved conversation ids, most recently used first."""
    seen = []
    for checkpoint in checkpointer.list(None):          # newest checkpoints come first
        thread_id = checkpoint.config["configurable"]["thread_id"]
        if thread_id not in seen:
            seen.append(thread_id)
    return seen



def _add_sql_nodes(builder: StateGraph, approval: bool) -> None:
    """The v0.4 core, shared by both graphs: pick_tables -> write_sql -> validate -> execute -> explain."""
    for name, fn in [("pick_tables", pick_tables), ("write_sql", write_sql),
                     ("widen_schema", widen_schema), ("validate", validate),
                     ("execute", execute), ("explain", explain)]:
        builder.add_node(name, fn)
    builder.add_edge("pick_tables", "write_sql")
    builder.add_conditional_edges("write_sql", after_write, ["validate", "widen_schema", "explain"])
    builder.add_edge("widen_schema", "write_sql")
    if approval:
        builder.add_node("approve", approve)
        builder.add_conditional_edges("validate", after_validate_with_approval,
                                      ["write_sql", "approve", "explain"])
        builder.add_conditional_edges("approve", after_approve, ["execute", "explain"])
    else:
        builder.add_conditional_edges("validate", after_validate_auto,
                                      ["write_sql", "execute", "explain"])
    builder.add_conditional_edges("execute", after_execute, ["write_sql", "explain"])
    builder.add_edge("explain", END)


def build_sql_graph():
    """v0.4: one question in, answer out, with self-correction. Input: {"question": "..."}."""
    builder = StateGraph(State)
    _add_sql_nodes(builder, approval=False)
    builder.add_edge(START, "pick_tables")
    return builder.compile()


def build_chat_graph(checkpointer=None, approval: bool = False):
    """v0.5: a conversation. Input: {"messages": [HumanMessage("...")]} plus a thread_id config."""
    builder = StateGraph(State)
    builder.add_node("router", router)
    builder.add_node("greet", greet)
    builder.add_node("off_topic", off_topic)
    _add_sql_nodes(builder, approval=approval)
    builder.add_edge(START, "router")
    builder.add_conditional_edges("router", after_router, ["greet", "off_topic", "pick_tables"])
    builder.add_edge("greet", END)
    builder.add_edge("off_topic", END)
    return builder.compile(checkpointer=checkpointer)
