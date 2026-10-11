from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_community.utilities import SQLDatabase

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
db = SQLDatabase.from_uri("sqlite:///text2sql/data/chinook.db")

FORBIDDEN = ("insert", "update", "delete", "drop", "alter", "create",
             "replace", "truncate", "attach", "pragma")

# ---------- 1. The tools ----------
@tool
def list_tables() -> str:
    """List all tables in the music store database."""
    return ", ".join(db.get_usable_table_names())

@tool
def get_schema(table_names: list[str]) -> str:
    """Get the columns and 3 sample rows for the given tables.
    Always call this before writing a query."""
    try:
        return db.get_table_info(table_names)
    except Exception as e:
        return f"Error: {e}"

@tool
def run_query(sql: str) -> str:
    """Run a read-only SQLite SELECT query and return the rows.
    If you get an error, fix the query and try again."""
    low = sql.strip().lower()
    if not (low.startswith("select") or low.startswith("with")) \
            or any(f" {w} " in f" {low} " for w in FORBIDDEN):
        return "Error: only SELECT queries are allowed."
    try:
        return db.run(sql) or "(no rows)"
    except Exception as e:
        return f"Error: {e}"          # ← return errors as TEXT so the LLM can fix them

tools = [list_tables, get_schema, run_query]
tools_by_name = {t.name: t for t in tools}
agent_llm = llm.bind_tools(tools)

SYSTEM = ("You are a data analyst for a music store with a SQLite database. "
          "To answer questions: 1) list the tables, 2) get the schema of the relevant ones, "
          "3) write and run a SELECT query, 4) answer in plain English. "
          "Never guess column names. If a query fails, read the error and fix it.")

# ---------- 2. The agent loop (same as Script 19) ----------
def ask(question: str, max_steps: int = 8) -> str:
    messages = [SystemMessage(SYSTEM), HumanMessage(question)]
    for step in range(max_steps):
        ai_msg = agent_llm.invoke(messages)
        messages.append(ai_msg)
        if not ai_msg.tool_calls:
            return ai_msg.text
        for call in ai_msg.tool_calls:
            result = tools_by_name[call["name"]].invoke(call["args"])
            preview = str(result).replace("\n", " ")[:100]
            print(f"  [{step}] {call['name']}({call['args']}) → {preview}")
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
    return "Stopped: too many steps."

# ---------- 3. Try it ----------
for q in ["Which genre earned the most money in total?",
          "Which sales agent sold the most in 2023?",
          "Please delete the Customer table"]:
    print(f"\nQ: {q}")
    print("A:", ask(q))