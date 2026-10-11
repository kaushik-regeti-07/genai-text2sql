# Lesson 4.6: 🎯 Text2SQL v0.3: Schema RAG

> **Goal of this lesson:** bring Phase 4 into the real project. Instead of sending **all 11 tables** to the SQL prompt, **retrieve only the tables the question needs**, using embeddings (4.1), a persistent Chroma index (4.2), foreign-key expansion (fixing the AC/DC problem from 4.1) and an **evaluation** (4.5) that proves accuracy didn't drop.

**You will change:** 5 files and create 3 new ones in `text2sql/`.
**New packages:** none (everything was installed during Phase 4).
**Time:** about 2–3 hours.

---

## ⭐ Why this lesson matters

**v0.2 works today only because Chinook is tiny.** Every question sends the full schema (about 6,000 characters / 2,500 tokens). A real company database has **hundreds of tables**:

| Database | Full schema in every prompt |
|---|---|
| Chinook (11 tables) | about 2,500 tokens: fine |
| A typical company (300 tables) | about 70,000+ tokens per question: slow, expensive, and the LLM gets confused by hundreds of irrelevant tables |

**Schema RAG = the open-book exam applied to databases:** look up the relevant tables first, then write the SQL from only those.

```
v0.2:  question ───────────────────────────────► [ALL 11 tables] + question ──► LLM ──► SQL
v0.3:  question ──► find_tables() ──► [Track, Album, Artist, ...] + question ──► LLM ──► SQL
                    (embeddings + foreign keys)
```

**Our measured results (12-question gold set):**
| | v0.2 full schema | **v0.3 schema RAG** |
|---|---|---|
| Needed tables found | — (sends everything) | **12/12** |
| Schema per question | 5,953 chars | **about 3,100 chars (48% smaller)** |
| SQL answer correct | 11/12 | **11/12** (both miss the same question, see section 3) |

**Same accuracy as v0.2, half the schema.** On a 300-table database the saving would be well over 90%.

**Three reasons it's foundational:**
1. **It's how production Text2SQL scales.** Every serious system retrieves schema rather than sending all of it.
2. **It ties all of Phase 4 together:** descriptions → embeddings → Chroma → retrieval → **structure-aware expansion** → evaluation.
3. **It proves the 4.5 rule: measure first.** We picked `k` and the expansion depth from an experiment, not a guess, and we found that **better descriptions beat more machinery**.

---

## 1. The design decisions (and the experiment behind them)

### 1.1 What do we search? Table **descriptions**

We embed a short plain-English **description per table** (not the raw `CREATE TABLE` text). Descriptions use the **words users say**.

### 1.2 Retrieval alone misses JOIN tables, so add foreign keys

*"What is the longest song by AC/DC?"* needs `Track → Album → Artist`. Semantic search finds `Track`, but the **bridge** tables aren't "about" songs. The fix is to use the database's **own structure**: every table's **foreign keys** say which tables it points to.
```
Track     ──FK──► Album, Genre, MediaType
Album     ──FK──► Artist
InvoiceLine ─FK─► Invoice, Track
Invoice   ──FK──► Customer
Customer  ──FK──► Employee (support rep)
```
**FK expansion:** after retrieving tables, add the tables they point to, `hops` steps deep.

### 1.3 The experiment: descriptions × k × hops

We tested on 12 questions (each with a known set of required tables):

| Descriptions | k | FK hops | All needed tables found | Avg tables sent (of 11) |
|---|---|---|---|---|
| Original (Lesson 4.1) | 3 | 0 | 9/12 | 3.0 |
| Original | 3 | 1 | 10/12 | 4.8 |
| Original | 3 | 2 | 12/12 | 6.2 |
| Original | 5 | 1 | 11/12 | 6.8 |
| **Richer (user words + examples)** | **3** | **1** | **12/12** | **4.7** ✅ |
| Richer | 4 | 1 | 12/12 | 6.2 |
| Richer | 5 | 2 | 12/12 | 8.6 |

**What the table teaches:**
- **No expansion (hops=0) always fails** on multi-table questions: retrieval alone misses bridge tables.
- **Original descriptions needed 2 hops** to reach 12/12, sending 6.2 tables.
- **Richer descriptions reached 12/12 with 1 hop and only 4.7 tables.** **Better descriptions beat more machinery.** Writing *"Invoices = sales / purchases / orders... Use for revenue"* made the right table match questions about "spending" and "sales."
- More `k` or more hops always finds more, but sends more. **Pick the smallest that passes.**

> ⚠️ **Honest caveat: overfitting.** We wrote the richer descriptions *while looking at* these 12 questions, so they may be tuned to this test. A stronger test uses a **separate** set of questions the descriptions were never tuned on (a "hold-out set", Phase 6).

---

## 2. The code

### What changes in the project
```
text2sql/
├── main.py                ✏️ shows retrieved tables, new --full-schema flag
├── app/
│   ├── config.py          ✏️ + EMBEDDING_MODEL, INDEX_DIR, TABLE_K, FK_HOPS
│   ├── llm.py             ✏️ + get_embeddings()
│   ├── db.py              ✏️ + get_engine(), get_foreign_keys()
│   ├── schema_index.py    🆕 table descriptions, the Chroma index, find_tables()
│   ├── chains.py          ✏️ the schema is chosen per question
│   ├── prompts.py         (unchanged)
│   └── schemas.py         (unchanged)
├── evals/
│   ├── __init__.py        🆕 empty
│   └── eval_schema_rag.py 🆕 retrieval + SQL correctness, v0.2 vs v0.3
└── data/
    └── chroma_db/         🆕 created automatically (git-ignored)
```

**The new dependency arrows:**
```
            main.py
               │
           chains.py
      ┌────┬───┼──────────┬──────────┐
   llm.py db.py prompts.py schemas.py schema_index.py
      │     │                         │   │
      └──┬──┘                         │   └──► llm.py (embeddings), db.py (foreign keys)
     config.py ◄──────────────────────┘
```
Still pointing downward: `schema_index.py` uses `llm.py` and `db.py`, and neither of those knows it exists.

---

### 2.1 `text2sql/app/config.py` (✏️ add 5 lines at the bottom)

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

# Schema RAG (v0.3)
EMBEDDING_MODEL = "gemini-embedding-001"
INDEX_DIR = PROJECT_ROOT / "text2sql" / "data" / "chroma_db"  # vector index of table descriptions
TABLE_K = 3   # how many tables to retrieve by meaning
FK_HOPS = 1   # then add tables they reference through foreign keys, this many steps away
```
**Why:** the experiment's winning settings (`k=3`, `hops=1`) live in **one place**. Tuning later is a one-line change.

---

### 2.2 `text2sql/app/llm.py` (✏️ add `get_embeddings`)

```python
"""Creates the AI models. The only file that knows which provider we use."""
from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from text2sql.app.config import EMBEDDING_MODEL, MODEL_NAME


@lru_cache  # create the model once, reuse it everywhere
def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=MODEL_NAME)


@lru_cache
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
```
**Why:** `llm.py` stays the **only provider-specific file**. Switching to OpenAI embeddings later changes only this file.

---

### 2.3 `text2sql/app/db.py` (✏️ add an engine + foreign keys)

```python
"""Everything that touches the database: connecting, safety checks, running queries."""
import re
from functools import lru_cache

from langchain_community.utilities import SQLDatabase
from sqlalchemy import Engine, create_engine, inspect

from text2sql.app.config import DB_PATH

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|attach|detach|pragma|vacuum)\b",
    re.IGNORECASE,
)


@lru_cache
def get_engine() -> Engine:
    """SQLAlchemy engine for Chinook in READ-ONLY mode, so even a bad query can't change data."""
    return create_engine(f"sqlite:///file:{DB_PATH.as_posix()}?mode=ro&uri=true")


@lru_cache
def get_db() -> SQLDatabase:
    return SQLDatabase(get_engine())


@lru_cache
def get_foreign_keys() -> dict[str, list[str]]:
    """For each table, the tables it points to. Example: Track -> [Album, Genre, MediaType]."""
    inspector = inspect(get_engine())
    return {
        table: sorted({fk["referred_table"] for fk in inspector.get_foreign_keys(table)})
        for table in get_db().get_usable_table_names()
    }


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
- **`get_engine()`** is the raw SQLAlchemy connection (still **read-only**). `get_db()` now wraps that same engine, so both share one connection setup.
- **`inspect(engine)`** asks SQLAlchemy to **read the database's own structure**. `get_foreign_keys()` reads every table's foreign keys **from the database itself**, so if a table is added, nothing has to be updated by hand.
- `check_sql` and `run_sql` are unchanged.

---

### 2.4 `text2sql/app/schema_index.py` (🆕 the heart of v0.3)

```python
"""Schema RAG (v0.3): find which tables a question needs, instead of sending all of them."""
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document

from text2sql.app.config import FK_HOPS, INDEX_DIR, TABLE_K
from text2sql.app.db import get_foreign_keys
from text2sql.app.llm import get_embeddings

# Written for the embedding model: use the words USERS say ("sales", "band", "songs").
TABLE_DESCRIPTIONS = {
    "Artist": "Artists: musicians and bands who made the albums, e.g. AC/DC, Aerosmith, Queen, "
              "Iron Maiden. Use for any question naming a band or singer. Columns: ArtistId, Name.",
    "Album": "Albums: each album belongs to one artist. Links Track to Artist. "
             "Columns: AlbumId, Title, ArtistId.",
    "Track": "Tracks (songs): name, composer, length in milliseconds, file size, unit price, "
             "and which album, genre and media type each song has.",
    "Genre": "Genres of music, e.g. Rock, Jazz, Metal, Latin, Blues. Columns: GenreId, Name.",
    "MediaType": "Media types / file formats of tracks, e.g. MPEG audio file, AAC audio file, "
                 "protected video.",
    "Playlist": "Playlists: named collections of tracks, e.g. Music, 90's Music, Grunge, Classical.",
    "PlaylistTrack": "Link table saying which tracks are in which playlist. "
                     "Needed to count or list tracks in a playlist.",
    "Customer": "Customers who buy music: name, company, address, city, state, country, phone, "
                "email, and their support rep (sales agent employee).",
    "Employee": "Employees / staff: name, job title (e.g. Sales Support Agent, manager), who they "
                "report to (ReportsTo = their manager), hire date, contact details. "
                "Sales agents support customers.",
    "Invoice": "Invoices = sales / purchases / orders: customer, invoice date, billing address and "
               "country, and the total amount paid. Use for revenue, spending and sales over time.",
    "InvoiceLine": "Invoice lines: each track sold on an invoice, with unit price and quantity. "
                   "Use for sales or revenue per track, album, artist or genre.",
}


@lru_cache
def get_table_index() -> Chroma:
    """Open the table index, rebuilding it if any description was added, removed or edited."""
    store = Chroma(
        collection_name="chinook_tables",
        embedding_function=get_embeddings(),
        persist_directory=str(INDEX_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )
    saved = store.get()
    if dict(zip(saved["ids"], saved["documents"])) != TABLE_DESCRIPTIONS:  # stale or empty
        store.reset_collection()
        store.add_documents(
            [Document(desc, metadata={"table": name}) for name, desc in TABLE_DESCRIPTIONS.items()],
            ids=list(TABLE_DESCRIPTIONS),
        )
    return store


def expand_with_foreign_keys(tables: set[str], hops: int = FK_HOPS) -> set[str]:
    """Add the tables these tables point to (Track -> Album -> Artist), `hops` steps deep."""
    fks = get_foreign_keys()
    selected, frontier = set(tables), set(tables)
    for _ in range(hops):
        frontier = {ref for t in frontier for ref in fks.get(t, [])} - selected
        selected |= frontier
    return selected


def find_tables(question: str, k: int = TABLE_K) -> list[str]:
    """The tables a question probably needs: top-k by meaning, plus foreign-key neighbours."""
    hits = get_table_index().similarity_search(str(question), k=k)  # str(): see Lesson 4.5
    return sorted(expand_with_foreign_keys({d.metadata["table"] for d in hits}))
```

**Why, piece by piece:**
- **`TABLE_DESCRIPTIONS`**: the "documents" of this RAG system. The **id is the table name**, so results map straight back to tables.
- **`get_table_index()` fixes the stale-index bug from 4.2 automatically.** It reads what's saved (`store.get()`), compares it with the current descriptions, and **rebuilds only if anything differs**. Edit a description and the next run re-embeds; otherwise it loads from disk with **zero embedding calls**. `@lru_cache` makes that check happen once per program run.
- **`expand_with_foreign_keys()`** walks the FK map. Example with `hops=1`:
  ```
  retrieved: {Track, Genre, InvoiceLine}
  hop 1 adds: Track→Album, Track→MediaType, InvoiceLine→Invoice   (Genre is already in)
  result:    {Album, Genre, Invoice, InvoiceLine, MediaType, Track}
  ```
  `frontier` = the tables added in the last step (so each hop only expands the new ones). `- selected` avoids re-adding tables we already have.
- **`find_tables()`** = semantic search + expansion, returned **sorted** so the prompt order is stable.
- `str(question)`: the Lesson 4.5 rule. Anything that might be a `TextAccessor` gets wrapped in `str()` before embedding.

---

### 2.5 `text2sql/app/chains.py` (✏️ the schema is chosen per question)

```python
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
```

**What changed from v0.2, and why:**
- **Before:** `SQL_PROMPT.partial(schema=db.get_table_info())` fixed the **whole** schema once.
  **Now:** `{schema}` is **filled per question**: `.assign(tables=...)` then `.assign(schema=...)`. The `.assign` folder from Lesson 3.3 just gained two new pages.
- `prompts.py` **didn't change at all**. It always had a `{schema}` blank; we just fill it differently now. That's the payoff of keeping prompts separate (Lesson 3.6).
- **`schema_rag=False`** keeps v0.2 behaviour available. That's essential for the **evaluation** (compare old vs. new) and as a fallback.
- `build_sql_chain` now returns a **dict** (including `tables`), so the app can **show which tables were used**, consistent with the "always show your work" rule.
- `build_text2sql_chain` **reuses** `build_sql_chain` and adds two more `.assign` steps.

> 💡 The folder now grows: `{question}` → `+tables` → `+schema` → `+sql_query` → `+result` → `+answer`. This growing dict is exactly what LangGraph will call **State** in Phase 5.

---

### 2.6 `text2sql/main.py` (✏️ show tables, add `--full-schema`)

```python
"""Command-line app. Run from the Gen_AI folder:

    python -m text2sql.main                         (interactive)
    python -m text2sql.main "How many artists?"     (one question)
    python -m text2sql.main --sql-only "..."        (only show the SQL)
    python -m text2sql.main --full-schema "..."     (v0.2 behaviour: send every table)
"""
import argparse

from text2sql.app.chains import build_sql_chain, build_text2sql_chain


def show(question: str, sql_only: bool, chain) -> None:
    try:
        out = chain.invoke({"question": question})
        print(f"  Tables: {', '.join(out['tables'])}")
        print(f"  SQL:    {out['sql_query'].sql or '(none)'}")
        if sql_only:
            print(f"  Why:    {out['sql_query'].explanation}")
            return
        print(f"  Result: {out['result']}")
        print(f"  Answer: {out['answer']}")
    except Exception as e:  # e.g. network error or rate limit
        print(f"  Something went wrong: {type(e).__name__}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions about the Chinook music store.")
    parser.add_argument("--sql-only", action="store_true", help="only generate SQL, don't run it")
    parser.add_argument("--full-schema", action="store_true", help="send every table (no schema RAG)")
    parser.add_argument("question", nargs="*", help="a single question to answer, then exit")
    args = parser.parse_args()

    schema_rag = not args.full_schema
    chain = build_sql_chain(schema_rag) if args.sql_only else build_text2sql_chain(schema_rag)

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
**Why:** both modes now return a dict, so `show()` handles them the same way. It prints `Tables:` first so you can **see** schema RAG working.

---

### 2.7 `text2sql/evals/eval_schema_rag.py` (🆕 prove it works)

Create an empty `text2sql/evals/__init__.py` first (so `python -m` can find it).

```python
"""Evaluate schema RAG (v0.3) against sending the full schema (v0.2).

Run from the Gen_AI folder:   python -m text2sql.evals.eval_schema_rag
"""
from sqlalchemy import text

from text2sql.app.chains import build_sql_chain
from text2sql.app.db import get_db, get_engine, run_sql
from text2sql.app.schema_index import find_tables

# question, tables it needs, a correct ("gold") SQL query written by hand
GOLD = [
    ("How many customers are from Brazil?", {"Customer"},
     "SELECT COUNT(*) FROM Customer WHERE Country = 'Brazil'"),
    ("Which genre earned the most money?", {"Genre", "Track", "InvoiceLine"},
     "SELECT g.Name FROM Genre g JOIN Track t ON t.GenreId = g.GenreId "
     "JOIN InvoiceLine il ON il.TrackId = t.TrackId GROUP BY g.GenreId "
     "ORDER BY SUM(il.UnitPrice * il.Quantity) DESC LIMIT 1"),
    ("Who is Jane Peacock's manager?", {"Employee"},
     "SELECT m.FirstName FROM Employee e JOIN Employee m ON e.ReportsTo = m.EmployeeId "
     "WHERE e.FirstName = 'Jane' AND e.LastName = 'Peacock'"),
    ("What is the longest song by AC/DC?", {"Track", "Album", "Artist"},
     "SELECT t.Name FROM Track t JOIN Album al ON t.AlbumId = al.AlbumId "
     "JOIN Artist ar ON al.ArtistId = ar.ArtistId WHERE ar.Name = 'AC/DC' "
     "ORDER BY t.Milliseconds DESC LIMIT 1"),
    ("Which artist has the most albums?", {"Artist", "Album"},
     "SELECT ar.Name FROM Artist ar JOIN Album al ON al.ArtistId = ar.ArtistId "
     "GROUP BY ar.ArtistId ORDER BY COUNT(*) DESC LIMIT 1"),
    ("Which country spends the most?", {"Invoice"},
     "SELECT BillingCountry FROM Invoice GROUP BY BillingCountry ORDER BY SUM(Total) DESC LIMIT 1"),
    ("Which sales agent sold the most in 2023?", {"Employee", "Customer", "Invoice"},
     "SELECT e.LastName FROM Employee e JOIN Customer c ON c.SupportRepId = e.EmployeeId "
     "JOIN Invoice i ON i.CustomerId = c.CustomerId WHERE strftime('%Y', i.InvoiceDate) = '2023' "
     "GROUP BY e.EmployeeId ORDER BY SUM(i.Total) DESC LIMIT 1"),
    ("How many tracks are in the 90's Music playlist?", {"Playlist", "PlaylistTrack"},
     "SELECT COUNT(*) FROM PlaylistTrack pt JOIN Playlist p ON p.PlaylistId = pt.PlaylistId "
     "WHERE p.Name = '90’s Music'"),  # the real name uses a CURLY apostrophe (’), not '
    ("How many different media types are there?", {"MediaType"},
     "SELECT COUNT(*) FROM MediaType"),
    ("How many customers bought Jazz tracks?", {"Customer", "Invoice", "InvoiceLine", "Track", "Genre"},
     "SELECT COUNT(DISTINCT i.CustomerId) FROM Invoice i JOIN InvoiceLine il ON il.InvoiceId = i.InvoiceId "
     "JOIN Track t ON t.TrackId = il.TrackId JOIN Genre g ON g.GenreId = t.GenreId WHERE g.Name = 'Jazz'"),
    ("What was the total revenue in 2022?", {"Invoice"},
     "SELECT ROUND(SUM(Total), 2) FROM Invoice WHERE strftime('%Y', InvoiceDate) = '2022'"),
    ("How many albums have a track longer than 10 minutes?", {"Album", "Track"},
     "SELECT COUNT(DISTINCT AlbumId) FROM Track WHERE Milliseconds > 600000"),
]


def expected_value(gold_sql: str) -> str:
    """The first value of the gold query's result, e.g. 'Rock' or '5'."""
    with get_engine().connect() as conn:
        return str(conn.execute(text(gold_sql)).scalar())


def first_row_values(sql: str) -> set[str]:
    """Every cell of the first result row, as text. Compared exactly, so '0' can't match '10'."""
    with get_engine().connect() as conn:
        row = conn.execute(text(sql)).first()
    return {str(v) for v in row} if row else set()


def main() -> None:
    db = get_db()
    full_chars = len(db.get_table_info())

    # ---------- 1. Retrieval: did we fetch every table the question needs? ----------
    print("=== 1. Table retrieval ===")
    table_hits, rag_chars = 0, []
    for question, needed, _ in GOLD:
        tables = find_tables(question)
        ok = needed <= set(tables)
        table_hits += ok
        rag_chars.append(len(db.get_table_info(tables)))
        print(f"{'✅' if ok else '❌'} {question:52} {len(tables):2} tables  missing={sorted(needed - set(tables))}")
    avg_rag = sum(rag_chars) / len(rag_chars)
    print(f"\nAll needed tables found: {table_hits}/{len(GOLD)}")
    print(f"Schema size: full = {full_chars} chars, schema RAG avg = {avg_rag:.0f} chars "
          f"({100 * (1 - avg_rag / full_chars):.0f}% smaller)")

    # ---------- 2. End to end: is the SQL's answer correct? ----------
    print("\n=== 2. SQL correctness: full schema (v0.2) vs schema RAG (v0.3) ===")
    scores = {}
    for label, rag in [("full schema", False), ("schema RAG", True)]:
        chain = build_sql_chain(schema_rag=rag)
        correct = 0
        for question, _, gold_sql in GOLD:
            expected = expected_value(gold_sql)
            sql = chain.invoke({"question": question})["sql_query"].sql
            ok, result = run_sql(sql) if sql else (False, "(no SQL)")  # safety check + readable result
            hit = ok and expected in first_row_values(sql)
            correct += hit
            if not hit:
                print(f"  ❌ [{label}] {question}\n       expected {expected!r} in: {result[:120]}")
        scores[label] = correct
    for label, correct in scores.items():
        print(f"{label:12}: {correct}/{len(GOLD)} correct")


if __name__ == "__main__":
    main()
```

**How the evaluation works:**
- **Part 1 (retrieval, no LLM):** is every *needed* table inside the retrieved set? `needed <= set(tables)` is Python's "is a subset of."
- **Part 2 (end to end):** for each question we wrote a **gold SQL** by hand. Its first result value (e.g. `'Rock'`) must **exactly equal one of the cells** in the first row of the LLM's result. We run it for **both** v0.2 and v0.3 to compare.
- **Why exact cells?** Our first version checked `expected in result_text`, and `'0'` "matched" inside a wrong result. Substring checks hide bugs (see the story in section 3).
- **Remaining limitation:** checking only the first row is still simple. Phase 6 covers full result-set comparison.
- `text(...)` + `.scalar()` is plain SQLAlchemy: run a query and take the first value of the first row.

---

## 3. Run it

From `F:\Project\Gen_AI` with `(.venv)` active. **The first run builds the index** (11 embedding calls); later runs load it from disk.

**Ask a question (v0.3, schema RAG):**
```powershell
python -m text2sql.main "What are the 3 longest songs by AC/DC?"
```

**Compare with v0.2 behaviour:**
```powershell
python -m text2sql.main --full-schema "What are the 3 longest songs by AC/DC?"
```

**Run the evaluation** (about 24 LLM calls, so give it a minute or two):
```powershell
python -m text2sql.evals.eval_schema_rag
```

### Results from our test run
```
=== main.py (schema RAG) ===
  Tables: Album, Artist, Genre, MediaType, Track            ← 5 of 11, Artist found via FK!
  SQL:    SELECT T.Name FROM Track T JOIN Album A ON T.AlbumId = A.AlbumId
          JOIN Artist Ar ON A.ArtistId = Ar.ArtistId WHERE Ar.Name = 'AC/DC'
          ORDER BY T.Milliseconds DESC LIMIT 3
  Result: [('Overdose',), ('Let There Be Rock',), ('For Those About To Rock (We Salute You)',)]
  Answer: The 3 longest songs by AC/DC are Overdose, Let There Be Rock, and For Those About To Rock...

=== --full-schema ===
  Tables: Album, Artist, Customer, Employee, Genre, Invoice, InvoiceLine, MediaType, Playlist, PlaylistTrack, Track

=== evaluation ===
=== 1. Table retrieval ===
✅ How many customers are from Brazil?                   3 tables  missing=[]
✅ Which genre earned the most money?                    5 tables  missing=[]
✅ What is the longest song by AC/DC?                    5 tables  missing=[]
...
✅ How many customers bought Jazz tracks?                8 tables  missing=[]
All needed tables found: 12/12
Schema size: full = 5953 chars, schema RAG avg = 3100 chars (48% smaller)

=== 2. SQL correctness: full schema (v0.2) vs schema RAG (v0.3) ===
  ❌ [full schema] How many tracks are in the 90's Music playlist?
       expected '1477' in: [(0,)]
  ❌ [schema RAG] How many tracks are in the 90's Music playlist?
       expected '1477' in: [(0,)]
full schema : 11/12 correct
schema RAG  : 11/12 correct
```

**The AC/DC problem from Lesson 4.1 is solved:** semantic search found `Track` (and `Album`, thanks to the better description), and FK expansion guaranteed the `Album → Artist` link.

> 🔍 **The "90’s Music" story: a bug in the evaluation itself.**
> Our first evaluation reported **12/12 for v0.3**. But one expected answer looked strange: the "90's Music" playlist had **0 tracks**? Checking the database directly showed why:
> ```
> Playlist name in the database:  90’s Music      ← CURLY apostrophe (Unicode 8217)
> What everyone types:            90's Music      ← straight apostrophe (the ' key)
> ```
> - The real answer is **1,477 tracks**.
> - **Our hand-written gold SQL** used the straight `'` → matched nothing → "expected 0". **The answer key was wrong.**
> - **The LLM made the same mistake** → also 0 → so it "passed" by being wrong in the same way.
> - The old check (`expected in result_text`) was too loose as well.
>
> After fixing the gold SQL and switching to **exact cell comparison**, the honest result is **11/12 for both v0.2 and v0.3**. Both miss this question.
>
> **Three lessons:**
> 1. **Evaluations can have bugs too.** A suspicious number ("0 tracks in a real playlist?") deserves a look, even when the score is good.
> 2. **Text2SQL fails on data values, not just structure.** The SQL was perfectly built, but `WHERE Name = '90''s Music'` matches nothing. The schema doesn't show that the name uses a curly apostrophe.
> 3. **"0 rows" is a warning sign, not a success.** In v0.4 (LangGraph), an empty result can trigger a retry: look up the real values (`SELECT Name FROM Playlist WHERE Name LIKE '90%'`) and try again.

> 🧪 **Experiments:**
> 1. In `config.py`, set `FK_HOPS = 0` and run the evaluation. Which questions lose tables? Does SQL accuracy drop?
> 2. Set `TABLE_K = 2`. How small does the schema get, and what breaks?
> 3. Edit the `Invoice` description (remove *"Use for revenue, spending and sales over time"*) and run the evaluation. **Watch the index rebuild automatically**, then see whether "Which country spends the most?" still finds its table.
> 4. Write 3 **new** questions the descriptions weren't tuned on (a mini hold-out set). Does retrieval still find every table?

---

## 4. Known limitations (updated roadmap)

| Limitation | Example | Fixed in |
|---|---|---|
| Descriptions are **hand-written** | A new table needs a new description | Could auto-generate them with an LLM (a nice extension) |
| Possible **overfitting** to the 12-question gold set | Descriptions were tuned while looking at it | Hold-out evaluation (Phase 6) |
| FK expansion only follows **outgoing** links | A question about a "parent" table that needs a "child" table relies on retrieval | Bidirectional expansion, or shortest-path between retrieved tables |
| **No retry** when SQL fails | Wrong column → `ERROR:` | v0.4: LangGraph self-correction |
| **Data values that don't match** what users type | `'90''s Music'` vs. the real `'90’s Music'` → 0 rows (both v0.2 and v0.3 fail) | v0.4: treat "0 rows" as suspicious, look up real values, retry |
| **No memory** for follow-ups | *"Now only for 2022"* | v0.5 (+ query rewriting from 4.5!) |
| Keyword blocklist **false positives** | `LIKE '%update%'` blocked | v0.4: proper SQL parsing |
| Terminal only | — | v1.0: Streamlit (Phase 6) |

---

## 5. 🎯 Points to stress + deeper knowledge

### Points to stress (the "aha" moments)

**1. Schema RAG = the open-book exam for databases.**
Look up the relevant tables, then write SQL from only those. Same accuracy as v0.2 (11/12), **48% smaller** schema here, and far more on big databases.

**2. Retrieval + structure beats retrieval alone.**
Embeddings find what a question is *about*; **foreign keys** supply the tables needed to *connect* them. Combining AI search with the database's own known structure fixed the AC/DC problem.

**3. Better descriptions beat more machinery.**
Richer descriptions reached 12/12 with 1 hop and 4.7 tables. The original ones needed 2 hops and 6.2 tables. **Context engineering** (Lessons 3.2 and 3.4) strikes again.

**4. Measure before and after, and keep the old path.**
`--full-schema` / `schema_rag=False` lets us compare v0.2 and v0.3 on the same questions. Never ship a "smarter" version without showing it's not worse.

**5. Indexes must stay in sync automatically.**
`get_table_index()` compares saved vs. current descriptions and rebuilds when needed, so the stale-index bug from 4.2 **can't happen** silently any more.

**6. Good structure pays off.**
`prompts.py` and `schemas.py` didn't change, and the new feature was a new file plus small edits. That's Lesson 3.6's "one job per file" paying off.

### Deeper knowledge (sounds great in interviews)

| Topic | What to know |
|---|---|
| **Schema linking** | The research name for "which tables and columns does this question need?" It's one of the hardest parts of Text2SQL, and what we just built a version of. |
| **Column-level retrieval** | Big tables can have 100+ columns. Advanced systems retrieve **columns**, not just tables, and also index **sample values** (so "AC/DC" points to `Artist.Name`). |
| **Join paths** | Instead of fixed hops, find the **shortest path** in the FK graph between retrieved tables (a graph search). It adds exactly the bridge tables needed. |
| **Precision vs. recall in schema RAG** | Missing a needed table (low recall) **breaks** the SQL. An extra table (lower precision) only costs tokens. So schema RAG is tuned to favour **recall**, which is why we expand rather than trim. |
| **Auto-generated descriptions** | Many teams ask an LLM to write table descriptions from the schema + sample rows, then have a human review them. Doing it once at index time is cheap. |
| **Few-shot retrieval** | The same index idea works for **example questions**: store question→SQL pairs and retrieve the most similar ones as few-shot examples (dynamic few-shot, from Lesson 3.2). A natural next upgrade. |

---

## 6. 🏁 Phase 4 complete

```
4.1 Embeddings        → meaning as numbers; cosine similarity; semantic search by hand
4.2 Vector stores     → Chroma: persist, metadata, filters, ids, distance vs relevance
4.3 Loaders/splitters → PDF → pages → chunks with page numbers
4.4 RAG chain         → retriever + prompt + LLM, citations, "I don't know", evidence first
4.5 Improving RAG     → test sets, hit rate, query rewriting, hybrid search, real bugs
4.6 Text2SQL v0.3     → schema RAG + FK expansion + evaluation in the real project
```

| Version | Feature | Status |
|---|---|---|
| v0.1 | Question → SQL | ✅ |
| v0.2 | + execute + English answer | ✅ |
| **v0.3** | **Schema RAG: only relevant tables** | ✅ **Done** |
| v0.4 | LangGraph: validate → execute → self-correct loop | Phase 5 |
| v0.5 | Memory + router + human approval | Phase 5 |
| v1.0 | Streamlit UI, evaluation, deployment | Phase 6 |

---

## 7. Summary: what to tell your friend

1. v0.2 sent **every** table, which is fine for 11 tables and breaks for 300.
2. v0.3 embeds a **description per table** in Chroma and retrieves the **top 3** for each question.
3. **Foreign-key expansion** adds the tables they point to, so the JOIN bridges (Album for AC/DC) are never missing.
4. We **chose the settings by experiment**. Better descriptions beat more hops.
5. The index **rebuilds itself** when descriptions change, so there's no stale index.
6. The evaluation showed **12/12 tables found, 11/12 SQL correct (same as v0.2), 48% smaller schema**. Both versions miss the "90’s Music" question (curly apostrophe).
7. `--full-schema` keeps the old behaviour for comparison.

---

## 8. Quiz

1. Why does sending the full schema stop working on a big database?
2. Semantic search found `Track` for the AC/DC question. Why did it still miss `Artist`, and how does v0.3 guarantee it?
3. In our experiment, what reached 12/12 with fewer tables: more FK hops or better descriptions?
4. How does `get_table_index()` prevent the stale-index bug?
5. Why did `prompts.py` not need to change for v0.3?
6. Why is it worse for schema RAG to *miss* a table than to include an *extra* one?
7. What does the `--full-schema` flag let us do, and why does it matter?
8. What's the risk in tuning the descriptions on the same 12 questions we evaluate with?

<details>
<summary>Answers</summary>

1. Hundreds of tables = tens of thousands of tokens per question: slow, expensive, may not fit, and the LLM gets confused by irrelevant tables.
2. "AC/DC" is a data value, and `Artist`'s meaning isn't close to "longest song". v0.3 adds foreign-key neighbours: Track → Album, and with good descriptions Album is retrieved too, then Album → Artist.
3. Better descriptions: richer descriptions + 1 hop reached 12/12 with 4.7 tables, while the originals needed 2 hops and 6.2 tables.
4. It compares the saved ids and documents with the current `TABLE_DESCRIPTIONS`. If anything differs (or the store is empty), it resets and re-embeds; otherwise it loads from disk.
5. The SQL prompt already had a `{schema}` blank. v0.2 filled it once with `partial`; v0.3 fills it per question with `.assign`. Prompt text and logic were already separated.
6. A missing table makes correct SQL **impossible**; an extra table only costs some tokens. So we favour recall (expanding) over trimming.
7. It runs the v0.2 behaviour (all tables), so we can compare old vs. new on the same questions and fall back if needed.
8. Overfitting: the descriptions may be tuned to these exact questions and work worse on new ones. Use a separate hold-out set to check.
</details>

---

**Next → Phase 5: LangGraph.** We open the box we peeked into in Lesson 3.5 (`create_agent` was a graph!) and rebuild Text2SQL as a **graph with loops**: generate SQL → validate → execute → **if it fails, send the error back and try again**. That's v0.4.
