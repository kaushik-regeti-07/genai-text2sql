# Lesson 3.6: Putting It Together: Text2SQL v0.1 + v0.2

> **Goal of this lesson:** move from practice scripts in `learning/` to a **real project structure** in `text2sql/`.
> We build **Text2SQL v0.1** (question → SQL) and **v0.2** (question → SQL → run → answer) as clean, reusable modules, with a command-line app you can actually use. 🎯

**You will create:** 9 files in `text2sql/` (7 Python modules + 2 empty `__init__.py`).
**New packages:** none.
**Time:** about 2–3 hours.

---

## ⭐ Why this lesson matters

**This is the jump from "learning scripts" to "a real project."**
Scripts 01–20 each did one thing and were thrown away. Real software is **organized** so that:
- each file has **one job**
- pieces can be **reused** (the Streamlit UI in Phase 6 will import the same chain)
- you can **change one thing without breaking others** (swap Gemini for OpenAI by editing one file)

**Every earlier lesson has a home in this project:**
| Lesson | Where it lives now |
|---|---|
| 3.1 Chat models | `llm.py` |
| 3.2 Prompt templates + few-shot | `prompts.py` |
| 3.3 LCEL chains + `.assign()` | `chains.py` |
| 3.4 Structured output (Pydantic) | `schemas.py` |
| 3.5 Tools' lesson: *safety in code, errors as text* | `db.py` |
| Setup: `.env`, paths | `config.py` |

**Three reasons it's foundational:**
1. **Phases 4–6 plug into this structure.** RAG goes into a new `retriever.py`, LangGraph into `graph.py`, the UI into `ui/`, and none of them need a rewrite.
2. **It's what interviewers and teammates look at.** A clean structure shows you can build software, not just follow tutorials.
3. **It adds real safety layers:** a **read-only** DB connection plus a stricter SQL check. Defense in depth, for real this time.

---

## 1. The architecture

```
Gen_AI/
└── text2sql/
    ├── __init__.py          ← empty; tells Python "this folder is a package"
    ├── main.py              ← the command-line app (what you run)
    ├── data/chinook.db
    └── app/
        ├── __init__.py      ← empty
        ├── config.py        ← settings: paths, model name, row limit
        ├── llm.py           ← creates the chat model
        ├── db.py            ← connect (read-only), check SQL, run SQL
        ├── schemas.py       ← Pydantic shapes (SQLQuery)
        ├── prompts.py       ← all prompt text + few-shot examples
        └── chains.py        ← LCEL: v0.1 and v0.2 chains
```

### How the files depend on each other
```
            main.py
               │
           chains.py
      ┌────┬───┴────┬─────────┐
   llm.py db.py prompts.py schemas.py
      └──┬──┘
     config.py
```
**Arrows only point downward.** `config.py` knows nothing about the others, and `main.py` only talks to `chains.py`. That keeps changes local: editing a prompt only touches `prompts.py`.

### The data flow (v0.2)
```
{question}
   │ .assign(sql_query = SQL_PROMPT | llm.with_structured_output(SQLQuery))
   ▼
{question, sql_query}
   │ .assign(result = _execute)   ← check_sql → run on READ-ONLY db → text
   ▼
{question, sql_query, result}
   │ .assign(answer = ANSWER_PROMPT | llm | StrOutputParser())
   ▼
{question, sql_query, result, answer}
```

> 🏠 **Analogy:** a restaurant. `config.py` is the **manager's rulebook**, `llm.py` the **chef**, `prompts.py` the **recipes**, `schemas.py` the **plating standard**, `db.py` the **pantry with a lock on it**, `chains.py` the **kitchen workflow**, and `main.py` the **waiter** who talks to customers. Each person has one job, and you can replace the chef without rewriting the recipes.

---

## 2. Two Python ideas you need first

### 2.1 Packages and `__init__.py`
A folder with an `__init__.py` file (even an empty one) is a **package**. That's what lets you write:
```python
from text2sql.app.config import DB_PATH
```
Python reads the dots like folder paths: `text2sql` → `app` → `config.py`.

> 📦 **Analogy:** folders are rooms in a warehouse. `__init__.py` is the **signboard on the door**: *"This is a department; you can order items from here."*

**What `__init__.py` does:**
1. **Marks the folder as a regular package.** Empty is fine.
2. **Runs once**, the first time anything from the package is imported (sometimes used for setup).
3. **Can re-export shortcuts:** e.g. putting `from text2sql.app.chains import build_text2sql_chain` inside `app/__init__.py` lets others write `from text2sql.app import build_text2sql_chain`. Libraries like LangChain do this everywhere.

**Is it strictly required?** Since Python 3.3, folders without it can often still be imported ("namespace packages"). **Keep it anyway.** It's the standard, explicit marker, and tools (test runners, linters, packaging) expect it.

**Don't confuse it with `def __init__(self)` in a class.** The **file** initializes a *package*; the **method** sets up a new *object*. Both mean "initialize."

### 2.2 Running with `python -m`
From `F:\Project\Gen_AI` we run:
```powershell
python -m text2sql.main
```
`-m` means "run this **module** as part of its package." It makes imports like `text2sql.app...` work. Running `python text2sql\main.py` directly would fail with `ModuleNotFoundError: No module named 'text2sql'`.

---

## 3. The code (file by file)

First, create the two empty files:
```
text2sql\__init__.py
text2sql\app\__init__.py
```
(In VS Code: right-click the folder → New File → type the name → save it empty.)

### 3.1 `text2sql/app/config.py`: settings in one place

```python
"""Central settings for the Text2SQL app. Change things here, not all over the code."""
from pathlib import Path

from dotenv import load_dotenv

# This file is Gen_AI/text2sql/app/config.py, so parents[2] is the Gen_AI folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

DB_PATH = PROJECT_ROOT / "text2sql" / "data" / "chinook.db"
MODEL_NAME = "gemini-3.5-flash-lite"
TOP_K = 5  # default row limit for queries
```

**Why:**
- **No more `"sqlite:///text2sql/data/chinook.db"`** that only works from one folder. `Path(__file__)` means "where *this file* is," so the path works **no matter where you run from**.
- `parents[0]` = `app/`, `parents[1]` = `text2sql/`, `parents[2]` = `Gen_AI/`.
- Want to try another model? Change **one line** here.

---

### 3.2 `text2sql/app/llm.py`: the chat model

```python
"""Creates the chat model. The only file that knows which LLM provider we use."""
from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI

from text2sql.app.config import MODEL_NAME


@lru_cache  # create the model once, reuse it everywhere
def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=MODEL_NAME)
```

**Why:**
- **Only this file imports `langchain_google_genai`.** Switching to OpenAI or Claude later means editing this file and nothing else.
- `@lru_cache` = "remember the result." The first call creates the model, and every later call returns the **same object**. It's a simple way to share one instance.

> 🧠 **More on `@lru_cache`** (from `functools`, built into Python)
> - It gives a function a **memory**: same inputs → the saved result is returned without running the function again. **LRU = Least Recently Used**: when the cache is full (128 entries by default), the entry unused for the longest time is dropped.
> - **Analogy:** a student writes each solved problem's answer in a notebook and reads it back when asked again.
> - `get_llm()` and `get_db()` take **no inputs**, so there's exactly one cached result: **one shared model and one shared DB connection** for the whole app. Without it, every `run_sql()` call would open a new connection. Check it with `get_llm() is get_llm()` → `True`.
> - **Why not `llm = ChatGoogleGenerativeAI(...)` at the top of the file?** That runs the moment the file is imported, even for `--help`, and a missing API key would break the import. A cached function creates it **only when first needed** (*lazy creation*).
> - **Careful:** don't cache data that changes (you'd get stale results). The cache lasts until the program stops (restart after editing `.env`/`config.py`, or call `get_llm.cache_clear()`). Inputs must be hashable (no lists or dicts).

---

### 3.3 `text2sql/app/db.py`: the locked pantry 🔒

```python
"""Everything that touches the database: connecting, safety checks, running queries."""
import re
from functools import lru_cache

from langchain_community.utilities import SQLDatabase

from text2sql.app.config import DB_PATH

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|attach|detach|pragma|vacuum)\b",
    re.IGNORECASE,
)


@lru_cache
def get_db() -> SQLDatabase:
    """Open Chinook in READ-ONLY mode, so even a bad query can't change data."""
    uri = f"sqlite:///file:{DB_PATH.as_posix()}?mode=ro&uri=true"
    return SQLDatabase.from_uri(uri)


def check_sql(sql: str) -> str | None:
    """Return a reason if the SQL is not allowed, or None if it looks safe."""
    s = sql.strip().rstrip(";").strip()
    if not s:
        return "Empty query."
    if ";" in s:
        return "Only one statement is allowed."
    if "--" in s or "/*" in s:
        return "Comments are not allowed."
    if not re.match(r"^(select|with)\b", s, re.IGNORECASE):
        return "Only SELECT queries are allowed."
    if FORBIDDEN.search(s):
        return "Query contains a forbidden keyword."
    return None


def run_sql(sql: str) -> tuple[bool, str]:
    """Check and run a query. Returns (ok, rows_or_error). Never raises."""
    problem = check_sql(sql)
    if problem:
        return False, problem
    try:
        return True, get_db().run(sql.strip().rstrip(";")) or "(no rows)"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
```

**Why:**
- **Layer 1, read-only connection:** `mode=ro` tells SQLite to open the file read-only. **The database itself refuses writes**, whatever the SQL says. (`as_posix()` turns `F:\...` into `F:/...`, the forward-slash form URIs need.)
- **Layer 2, `check_sql`:** stricter than Lesson 3.4:
  - `";" in s` → blocks **stacked statements** like `SELECT 1; DROP TABLE Customer`
  - `"--"` / `"/*"` → blocks **comments**, a common trick for hiding code
  - `\b...\b` in the regex = **whole words only**, so the column `LastUpdate` wouldn't trigger `update`
- **`run_sql` never raises.** It returns `(False, "reason")`, following the Lesson 3.5 rule: **errors as information, not crashes**.
- **Why here and not in a Pydantic validator (like 3.4)?** A validator error would **crash the whole chain**. Here a bad query becomes a friendly message, and there's **one place** that guards the database.

---

### 3.4 `text2sql/app/schemas.py`: the shape of the LLM's answer

```python
"""The shapes of data the LLM must return (Pydantic models)."""
from pydantic import BaseModel, Field


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
```

**Why:**
- Same as Lesson 3.4, with one change: **`explanation` comes before `sql`**. The model writes fields in order, so it **explains its plan first, then writes the SQL**. It's a mini "think first" step (see the 3.4 deeper-knowledge table).

---

### 3.5 `text2sql/app/prompts.py`: all prompt text

```python
"""All prompt text lives here, so we can improve prompts without touching logic."""
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

SQL_SYSTEM = """You are an expert {dialect} developer for a music store database.
Given the schema below, write ONE {dialect} SELECT query that answers the user's question.

Rules:
- Only use tables and columns that exist in the schema.
- Only write SELECT queries.
- Unless the user asks for more, LIMIT results to {top_k} rows.
- If the schema does not contain the needed information, set is_answerable to false and leave sql empty.

Schema:
{schema}"""

# Example question -> SQL pairs that show the style we want.
EXAMPLES = [
    {
        "question": "How many artists are there?",
        "sql": "SELECT COUNT(*) FROM Artist;",
    },
    {
        "question": "List all albums by AC/DC.",
        "sql": "SELECT al.Title FROM Album al JOIN Artist ar ON al.ArtistId = ar.ArtistId "
               "WHERE ar.Name = 'AC/DC';",
    },
]

_few_shot = FewShotChatMessagePromptTemplate(
    example_prompt=ChatPromptTemplate.from_messages([("human", "{question}"), ("ai", "{sql}")]),
    examples=EXAMPLES,
)

SQL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SQL_SYSTEM),
    _few_shot,
    ("human", "{question}"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions about a music store database using ONLY the SQL result given. "
     "If the result starts with NOT_ANSWERABLE, politely say the database doesn't have that information. "
     "If it starts with ERROR, say the query failed and briefly why. Be concise."),
    ("human", "Question: {question}\nSQL result: {result}\n\nAnswer:"),
])
```

**Why:**
- **Prompts are separated from logic.** You'll edit this file often while improving accuracy, and nothing else needs to change.
- `_few_shot` starts with `_`, the Python convention for "internal, don't import this from outside."
- `ANSWER_PROMPT` knows about the **`NOT_ANSWERABLE:`** and **`ERROR:`** prefixes, so every outcome gets a sensible English reply.

---

### 3.6 `text2sql/app/chains.py`: the kitchen workflow

```python
"""Connects the pieces into LCEL chains. v0.1 = question -> SQL. v0.2 = + run + answer."""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough

from text2sql.app.config import TOP_K
from text2sql.app.db import get_db, run_sql
from text2sql.app.llm import get_llm
from text2sql.app.prompts import ANSWER_PROMPT, SQL_PROMPT
from text2sql.app.schemas import SQLQuery


def build_sql_chain() -> Runnable:
    """v0.1: {question} -> SQLQuery object."""
    db = get_db()
    prompt = SQL_PROMPT.partial(dialect=db.dialect, schema=db.get_table_info(), top_k=TOP_K)
    return prompt | get_llm().with_structured_output(SQLQuery)


def _execute(x: dict) -> str:
    """Run the generated SQL, turning every outcome into text the answer step can read."""
    query: SQLQuery = x["sql_query"]
    if not query.is_answerable:
        return f"NOT_ANSWERABLE: {query.explanation}"
    ok, rows = run_sql(query.sql)
    return rows if ok else f"ERROR: {rows}"


def build_text2sql_chain() -> Runnable:
    """v0.2: {question} -> {question, sql_query, result, answer}."""
    write_answer = ANSWER_PROMPT | get_llm() | StrOutputParser()
    return (
        RunnablePassthrough.assign(sql_query=build_sql_chain())
        .assign(result=_execute)
        .assign(answer=write_answer)
    )
```

**Why:**
- **"Builder" functions** (`build_...`) create chains on demand. `main.py` today and the Streamlit UI in Phase 6 will call the **same** functions.
- **v0.2 reuses v0.1**: `build_text2sql_chain` puts `build_sql_chain()` inside itself. Small pieces, combined.
- `_execute` covers **all three outcomes** (not answerable / error / success) and turns each into text, so the chain **never crashes** on a bad query.

---

### 3.7 `text2sql/main.py`: the command-line app

```python
"""Command-line app. Run from the Gen_AI folder:

    python -m text2sql.main                         (interactive)
    python -m text2sql.main "How many artists?"     (one question)
    python -m text2sql.main --sql-only "..."        (v0.1: show SQL only)
"""
import argparse

from text2sql.app.chains import build_sql_chain, build_text2sql_chain


def show(question: str, sql_only: bool, chain) -> None:
    try:
        if sql_only:
            q = chain.invoke({"question": question})
            print(f"  SQL:    {q.sql or '(none)'}")
            print(f"  Why:    {q.explanation}")
            return
        out = chain.invoke({"question": question})
        print(f"  SQL:    {out['sql_query'].sql or '(none)'}")
        print(f"  Result: {out['result']}")
        print(f"  Answer: {out['answer']}")
    except Exception as e:  # e.g. network error or rate limit
        print(f"  Something went wrong: {type(e).__name__}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions about the Chinook music store.")
    parser.add_argument("--sql-only", action="store_true", help="v0.1 mode: only generate SQL")
    parser.add_argument("question", nargs="*", help="a single question to answer, then exit")
    args = parser.parse_args()

    chain = build_sql_chain() if args.sql_only else build_text2sql_chain()

    if args.question:
        show(" ".join(args.question), args.sql_only, chain)
        return

    print("Text2SQL - ask about the Chinook music store. Type 'exit' to quit.")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"", "exit", "quit"}:
            break
        show(question, args.sql_only, chain)


if __name__ == "__main__":
    main()
```

**Why:**
- **`argparse`** (built into Python) reads command-line options: `--sql-only` switches to v0.1 mode, and anything else is the question.
- **The chain is built once** before the loop, not on every question, so the schema is read only once.
- **`try/except` in `show`**: a network error or rate limit prints a message instead of killing the app.
- `if __name__ == "__main__":` = "run `main()` only when this file is run directly, not when it's imported."

---

## 4. Run it! (from `F:\Project\Gen_AI`, with `(.venv)` active)

**v0.1: SQL only**
```powershell
python -m text2sql.main --sql-only "Which 3 countries have the most customers?"
```

**v0.2: full answer, one question**
```powershell
python -m text2sql.main "What is the phone number of the manager of Jane Peacock?"
```

**v0.2: interactive**
```powershell
python -m text2sql.main
```

**Help**
```powershell
python -m text2sql.main --help
```

### Results from our test run
```
=== v0.1 (--sql-only) ===
Q: Which 3 countries have the most customers?
  SQL:    SELECT Country, COUNT(*) AS CustomerCount FROM Customer GROUP BY Country
          ORDER BY CustomerCount DESC LIMIT 3
  Why:    This query groups customers by their country, counts them, and returns the top 3.

=== v0.2 ===
Q: What is the phone number of the manager of Jane Peacock?
  SQL:    SELECT m.Phone FROM Employee e JOIN Employee m ON e.ReportsTo = m.EmployeeId
          WHERE e.FirstName = 'Jane' AND e.LastName = 'Peacock' LIMIT 5
  Result: [('+1 (403) 262-3443',)]
  Answer: The phone number of the manager of Jane Peacock is +1 (403) 262-3443.

Q: What is the average age of our customers?
  SQL:    (none)
  Result: NOT_ANSWERABLE: The Customer table does not contain birth date information...
  Answer: The database doesn't have that information.

=== interactive ===
You: How many playlists are there?
  SQL:    SELECT COUNT(*) FROM Playlist;
  Result: [(18,)]
  Answer: There are 18 playlists.
```
The "manager" question needed a **self-join** (the Employee table joined to itself through `ReportsTo`). The LLM worked that out from the schema alone.

### Safety test results
```
'SELECT 1; DROP TABLE Customer'  → Only one statement is allowed.
'SELECT 1 -- hi'                 → Comments are not allowed.
'DELETE FROM Customer'           → Only SELECT queries are allowed.
'WITH t AS (SELECT 1) SELECT ...'→ allowed ✅

Bypassing check_sql and sending DELETE straight to the DB:
  → OperationalError: attempt to write a readonly database   ← Layer 1 held!
```

> 🧪 **Test the safety layers yourself** (a quick script or the Python REPL):
> ```python
> from text2sql.app.db import check_sql, run_sql, get_db
> print(check_sql("SELECT 1; DROP TABLE Customer"))
> print(run_sql("SELECT Age FROM Customer"))                  # (False, 'OperationalError: no such column...')
> get_db().run("DELETE FROM Customer WHERE CustomerId = 1")   # read-only error
> ```
> Run it with `python` from `F:\Project\Gen_AI` so the `text2sql` imports work.

---

## 5. Known limitations (honest list = the v0.3–v1.0 roadmap)

| Limitation we saw | Example | Fixed in |
|---|---|---|
| **Keyword check has false positives** | `WHERE Name LIKE '%update%'` gets **blocked** even though it's harmless | v0.4: proper SQL parsing instead of keyword matching |
| **Ambiguous questions** | *"Which album has the most tracks?"* returned **5** albums, because the `LIMIT {top_k}` rule beat the singular "album" | Better prompt + evaluation set (Phase 6) |
| **No retry** | A query with a wrong column → `ERROR:` → the answer just reports the failure | v0.4: LangGraph self-correction loop |
| **Whole schema in every prompt** | About 2,500 tokens each time; won't scale to 500 tables | v0.3: RAG over the schema (Phase 4) |
| **No memory** | *"Now only the ones from India"* doesn't know what "ones" means | v0.5: LangGraph memory |
| **No routing** | *"Hi!"* still goes through the SQL step | v0.5: the router from Lesson 3.4, as a graph node |
| **Terminal only** | No web UI | v1.0: Streamlit (Phase 6) |

> 💡 Keeping a list like this is a professional habit. **Every version has limits.** Knowing and documenting them is what separates engineers from tutorial-followers.

---

## 6. 🎯 Points to stress + deeper knowledge

### Points to stress (the "aha" moments)

**1. One job per file.**
When the SQL is wrong → look in `prompts.py`. When the DB errors → `db.py`. When you switch models → `llm.py`. **Structure = knowing where to look.**

**2. Defense in depth, for real.**
Four layers now protect the database: **prompt rules** (advice) → **`is_answerable`** (the model can decline) → **`check_sql`** (code) → **read-only connection** (the database itself). We proved the last layer holds even when the others are bypassed.

**3. Errors are data, everywhere.**
`run_sql` never raises, and `_execute` turns every outcome into text. The app **degrades gracefully** instead of crashing. That's the same principle as tools in 3.5.

**4. Reuse, don't copy.**
v0.2 **contains** v0.1. The Streamlit UI (Phase 6) will import `build_text2sql_chain()`. Nothing is copy-pasted.

**5. Paths relative to the file, not the terminal.**
`Path(__file__)` makes the app work from any folder. Hard-coded relative paths are one of the most common "works on my machine" bugs.

### Deeper knowledge (sounds great in interviews)

| Topic | What to know |
|---|---|
| **Separation of concerns** | The formal name for "one job per file." It's the most important software design principle behind this lesson. |
| **Configuration in one place** | Real apps read settings like `MODEL_NAME` from **environment variables** too, so dev and production can use different values without code changes (the "12-factor app" idea). |
| **Least privilege** | In production you'd connect with a **database user that only has SELECT permission** on specific tables. `mode=ro` is SQLite's version of this. |
| **Blocklists vs. allowlists** | Our `FORBIDDEN` list is a **blocklist** (block known-bad things). Blocklists are always incomplete. **Allowlists** (only permit known-good things, e.g. parse the SQL and allow only SELECT on approved tables) are stronger. That's the v0.4 upgrade. |
| **Why show the SQL** | Text2SQL answers can be **confidently wrong** (right-looking SQL, wrong JOIN). Showing the SQL and the raw result lets users verify. Production tools always do this. |
| **Caching** | `@lru_cache` gives a single shared model and DB connection. In web apps (Phase 6), shared resources like these are created once at startup. |

---

## 7. Common errors

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'text2sql'` | Ran `python text2sql\main.py`, or not from `Gen_AI` | Use `python -m text2sql.main` from `F:\Project\Gen_AI` |
| `ModuleNotFoundError: No module named 'text2sql.app'` | Missing `__init__.py` | Create the empty `text2sql\app\__init__.py` |
| `unable to open database file` | Wrong path, or chinook.db missing | Check that `text2sql\data\chinook.db` exists. `config.py` builds the path from the file location. |
| `API key required` | `.env` not found | `.env` must be in `F:\Project\Gen_AI` (config loads it from `PROJECT_ROOT`) |
| `attempt to write a readonly database` | Something tried to write | ✅ Working as intended! |
| `ImportError: cannot import name ...` | Typo in a function name between files | Compare the names in the `import` line and the file they come from |

---

## 8. 🏁 Phase 3 complete: what you now know

```
3.1 Chat models      → talk to any LLM through one interface
3.2 Prompt templates → reusable prompts, few-shot examples
3.3 LCEL chains      → connect steps with |, carry data with .assign()
3.4 Structured output→ typed objects, Literal routing, validation
3.5 Tools            → the LLM decides actions; the agent loop (ReAct)
3.6 Project          → Text2SQL v0.1 + v0.2, clean structure, 4 safety layers
```

**Text2SQL roadmap:**
| Version | Feature | Status |
|---|---|---|
| v0.1 | Question → SQL | ✅ Done |
| v0.2 | + execute + English answer | ✅ Done |
| v0.3 | Schema RAG (only relevant tables) | Phase 4 |
| v0.4 | LangGraph: validate → execute → self-correct loop | Phase 5 |
| v0.5 | Memory + router + human approval | Phase 5 |
| v1.0 | Streamlit UI, evaluation, deployment | Phase 6 |

---

## 9. Summary: what to tell your friend

1. We moved from scripts to a **package**: one job per file, dependencies pointing one way.
2. `config.py` holds settings, and `Path(__file__)` makes paths work from anywhere.
3. `llm.py` is the **only** provider-specific file, so switching models is a one-file change.
4. `db.py` = **read-only connection + SQL check + never-crash runner**.
5. `chains.py` builds v0.1 (SQL only) and v0.2 (SQL → run → answer), and v0.2 **reuses** v0.1.
6. Run with `python -m text2sql.main` from the project root.
7. We **documented the limitations**, and they're exactly what Phases 4–6 fix.

---

## 10. Quiz

1. Why do we run `python -m text2sql.main` instead of `python text2sql\main.py`?
2. What do `Path(__file__).resolve().parents[2]` and `mode=ro` each do?
3. Name the 4 safety layers protecting the database, from weakest to strongest.
4. Why is the SELECT check in `db.py` rather than in a Pydantic validator like Lesson 3.4?
5. Why is `explanation` placed *before* `sql` in `SQLQuery`?
6. You want to switch from Gemini to OpenAI. Which file(s) change?
7. `WHERE Name LIKE '%update%'` gets blocked. What kind of problem is this, and what's the better approach?

<details>
<summary>Answers</summary>

1. `-m` runs it as a module inside its package, so imports like `text2sql.app.config` work. Running the file directly doesn't set up the package, so you get `ModuleNotFoundError`.
2. `parents[2]` walks up from `app/config.py` to the `Gen_AI` folder, so paths work from anywhere. `mode=ro` opens SQLite read-only, so the database itself refuses writes.
3. Prompt rules → `is_answerable` (model can decline) → `check_sql` (code) → read-only connection (the database).
4. A validator error would crash the whole chain. In `db.py`, a bad query becomes a friendly `ERROR:` message, and there's a single place guarding the DB.
5. The model writes fields in order, so it states its plan first, then writes the SQL. It's a mini "think first" step that can improve accuracy.
6. Only `llm.py` (plus installing the new provider package and adding its key to `.env`). Maybe `MODEL_NAME` in `config.py`.
7. A **false positive** from a keyword **blocklist**. Better: parse the SQL properly and use an **allowlist** (only SELECT, only approved tables). That's the v0.4 upgrade.
</details>

---

**Next → Phase 4: RAG.** We learn embeddings and vector stores with a small "chat with a PDF" exercise, then build **v0.3**: instead of sending all 11 tables, retrieve **only the relevant ones** for each question.
