from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, ValidationError
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
db = SQLDatabase.from_uri("sqlite:///text2sql/data/chinook.db")

FORBIDDEN = ("insert", "update", "delete", "drop", "alter", "create",
             "replace", "truncate", "attach", "pragma")

# ---------- 1. The shape of a good answer ----------
class SQLQuery(BaseModel):
    """A SQL query that answers the user's question."""
    is_answerable: bool = Field(
        description="False if the database does not contain the information needed")
    tables_used: list[str] = Field(
        default_factory=list, description="Tables the query reads from")
    sql: str = Field(
        default="", description="One SQLite SELECT query, no markdown. Empty if not answerable.")
    explanation: str = Field(
        description="One sentence: what the query does, or why it can't be answered")

    # Runs in CODE every time a SQLQuery is created. The LLM can't skip it.
    @field_validator("sql")
    @classmethod
    def must_be_select(cls, v: str) -> str:
        v = v.strip()
        if not v:
            return v                                   # empty = not answerable
        low = v.lower()
        if not (low.startswith("select") or low.startswith("with")):
            raise ValueError("Query must start with SELECT or WITH")
        if any(f" {word} " in f" {low} " for word in FORBIDDEN):
            raise ValueError("Query contains a forbidden keyword")
        return v

# Prove the guard works, with no LLM involved:
try:
    SQLQuery(is_answerable=True, sql="DELETE FROM Customer", explanation="oops")
except ValidationError as e:
    print("BLOCKED:", e.errors()[0]["msg"])

# ---------- 2. Question → SQLQuery object ----------
SQL_SYSTEM = """You are an expert {dialect} developer.
Given the database schema below, write ONE {dialect} SELECT query that answers the user's question.
- Only use tables and columns that exist in the schema.
- Unless the user asks for more, LIMIT results to {top_k} rows.
- If the schema does not contain the needed information, set is_answerable to false and leave sql empty.

Schema:
{schema}"""

sql_prompt = ChatPromptTemplate.from_messages([
    ("system", SQL_SYSTEM),
    ("human", "{question}"),
]).partial(dialect=db.dialect, schema=db.get_table_info(), top_k=5)

write_sql = sql_prompt | llm.with_structured_output(SQLQuery)
# Notice: no StrOutputParser, no clean_sql(). The object IS the output.

# ---------- 3. Run only if answerable ----------
def run_query(x: dict) -> str:
    q: SQLQuery = x["sql_obj"]
    if not q.is_answerable:
        return "NOT_ANSWERABLE: " + q.explanation
    return db.run(q.sql)

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You answer questions about a music store database. Use ONLY the SQL result. "
               "If it says NOT_ANSWERABLE, politely explain the database doesn't have that "
               "information. Be concise."),
    ("human", "Question: {question}\nSQL result: {result}\n\nAnswer:"),
])

text2sql = (
    RunnablePassthrough.assign(sql_obj=write_sql)
    .assign(result=run_query)
    .assign(answer=answer_prompt | llm | StrOutputParser())
)

# ---------- 4. Try it ----------
for q in ["How many invoices were made in 2022?",
          "Which 3 artists have the most tracks?",
          "What is the favourite colour of each customer?"]:
    out = text2sql.invoke({"question": q})
    s = out["sql_obj"]
    print(f"\nQ: {q}")
    print(f"   answerable: {s.is_answerable} | tables: {s.tables_used}")
    print(f"   SQL:    {s.sql or '(none)'}")
    print(f"   Answer: {out['answer']}")