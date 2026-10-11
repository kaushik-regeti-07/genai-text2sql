"""The shapes of data the LLM must return (Pydantic models)."""
from typing import Literal

from pydantic import BaseModel, Field


class Route(BaseModel):
    """How to handle the user's latest message (v0.5)."""

    category: Literal["database_question", "greeting", "off_topic"] = Field(
        description="database_question = anything answerable from the music store database, "
                    "including follow-ups to earlier questions. greeting = hello/thanks/small talk. "
                    "off_topic = anything else."
    )
    standalone_question: str = Field(
        default="",
        description="For database_question: the latest message rewritten so it makes sense without "
                    "the chat history. Empty for other categories.",
    )


class SQLQuery(BaseModel):
    """A SQL query that answers the user's question."""

    is_answerable: bool = Field(
        description="False if the database does not contain the information needed"
    )
    tables_used: list[str] = Field(
        default_factory=list, description="Tables the query reads from"
    )
    explanation: str = Field(
        description="One sentence: how the query answers the question, or why it can't be answered"
    )
    sql: str = Field(
        default="",
        description="One SQLite SELECT query, no markdown. Empty if not answerable.",
    )