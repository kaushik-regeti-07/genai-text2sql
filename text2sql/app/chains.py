"""Connects the pieces into LCEL chains.

v0.1  question -> SQL
v0.2  + run the SQL + answer in English
v0.3  + schema RAG: only the relevant tables' schema goes into the prompt
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough

from text2sql.app.config import TOP_K
from text2sql.app.db import get_db, run_sql
from text2sql.app.llm import get_llm
from text2sql.app.prompts import ANSWER_PROMPT, SQL_PROMPT
from text2sql.app.schema_index import find_tables
from text2sql.app.schemas import SQLQuery


def _pick_tables(schema_rag: bool):
    if schema_rag:
        return lambda x: find_tables(x["question"])
    return lambda x: get_db().get_usable_table_names()  # v0.2 behaviour: every table


def build_sql_chain(schema_rag: bool = True) -> Runnable:
    """{question} -> {question, tables, schema, sql_query}."""
    db = get_db()
    write_sql = SQL_PROMPT.partial(dialect=db.dialect, top_k=TOP_K) | get_llm().with_structured_output(SQLQuery)
    return (
        RunnablePassthrough.assign(tables=_pick_tables(schema_rag))
        .assign(schema=lambda x: db.get_table_info(x["tables"]))
        .assign(sql_query=write_sql)
    )


def _execute(x: dict) -> str:
    """Run the generated SQL, turning every outcome into text the answer step can read."""
    query: SQLQuery = x["sql_query"]
    if not query.is_answerable:
        return f"NOT_ANSWERABLE: {query.explanation}"
    ok, rows = run_sql(query.sql)
    return rows if ok else f"ERROR: {rows}"


def build_text2sql_chain(schema_rag: bool = True) -> Runnable:
    """{question} -> {question, tables, schema, sql_query, result, answer}."""
    write_answer = ANSWER_PROMPT | get_llm() | StrOutputParser()
    return (
        build_sql_chain(schema_rag)
        .assign(result=_execute)
        .assign(answer=write_answer)
    )
