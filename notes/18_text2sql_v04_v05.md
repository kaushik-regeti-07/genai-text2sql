# Lesson 5.7: 🎯 Text2SQL v0.4 + v0.5: the LangGraph App

> **Goal:** rebuild Text2SQL as a **graph** that **checks its own SQL and fixes it** (v0.4), then wrap it in a **conversation** with routing, memory and human approval (v0.5).
> **Status:** ✅ built and tested in the real project (`text2sql/app/graph.py`).

**New package:** `sqlglot` (proper SQL parsing), added to `requirements.txt`.
**Files changed:** `db.py`, `prompts.py`, `schemas.py`, `config.py`, `main.py`. **New:** `graph.py`, `evals/eval_graph.py`.
**Kept as-is:** `chains.py` (v0.3) for comparison, `schema_index.py`, `llm.py`.

---

## ⭐ Why this lesson matters

**Remember the "90’s Music" mystery from v0.3?** The SQL was perfectly built, but used `'90''s Music'` with a straight apostrophe while the database stores `'90’s Music'` (curly). Result: **0 rows**. v0.3 **ran it once and trusted the answer**.

**v0.4 doesn't trust its first attempt.** It **checks** the SQL before running it, and when something's wrong it **sends the problem back to the LLM and tries again**:

```
Attempt 1:  WHERE p.Name = '90''s Music'
            🔧 validate: Playlist.Name has no value "90's Music". Did you mean: '90’s Music'?
Attempt 2:  WHERE p.Name = '90’s Music'      →  [(1477,)]  ✅
Answer:     "There are 1,477 tracks in the 90's Music playlist."
```
That's the loop we've been promising since Lesson 3.3. **A chain can't do this. A graph can.**

**v0.5 makes it a real assistant:** it greets you, refuses off-topic questions, **remembers the conversation** ("What is *her* phone number?") and can **ask before running** any query.

---

## 1. v0.4: the self-correcting SQL graph

### 1.1 The graph
```
            START
              │
         pick_tables          (schema RAG from v0.3)
              │
     ┌──► write_sql  ◄───────────────────┐
     │        │  ▲                       │  problem + attempts < 3 → try again
     │        │  └── widen_schema ◄── "not answerable" with only SOME tables (once)
     │        ├── not answerable (with all tables) ──► explain
     │        ▼                          │
     │    validate ──── problem ─────────┤  (safety check + value check)
     │        │ ok                       │
     │        ▼                          │
     │    execute ───── DB error ────────┘
     │        │ ok
     │        ▼
     └──  explain ──► END         (3 attempts used up → explain the error)
```
**Mermaid (from `build_sql_graph().get_graph().draw_mermaid()`):**
```
__start__ --> pick_tables;
pick_tables --> write_sql;
write_sql -.-> validate;      write_sql -.-> widen_schema;  write_sql -.-> explain;
widen_schema --> write_sql;
validate -.-> execute;        validate -.-> write_sql;      validate -.-> explain;
execute -.-> explain;         execute -.-> write_sql;
explain --> __end__;
```
The **dotted arrows back to `write_sql`** are the self-correction loop.

### 1.2 The State: the logbook every node shares
```python
class State(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]   # conversation (v0.5)
    route: str                        # v0.5: database_question / greeting / off_topic
    question: str                     # the standalone question
    tables: list[str]                 # from schema RAG
    schema: str
    sql_query: Optional[SQLQuery]     # the LLM's latest attempt
    attempts: int                     # how many times SQL was written
    error: Optional[str]              # why the latest attempt failed (None = fine)
    fix_log: list[str]                # every problem that got fixed (for display)
    result: Optional[str]
    answer: Optional[str]
```
Compare with v0.3's `.assign()` folder: `{question, tables, schema, sql_query, result, answer}`. It's the **same idea**, plus `attempts`, `error` and `fix_log` for the loop.

### 1.3 The nodes (one job each)
| Node | Job | Uses |
|---|---|---|
| `pick_tables` | Choose relevant tables + their schema | `find_tables()` (v0.3) |
| `write_sql` | Ask the LLM for SQL. **On a retry, show it its previous query and what went wrong.** | `SQL_PROMPT`, `RETRY_MESSAGE` |
| `widen_schema` | If the LLM says "not answerable" while it only saw **some** tables, give it **all** tables and try once more | `get_usable_table_names()` |
| `validate` | **Before running:** is it safe? Do its text values exist? | `check_sql()`, `check_values()` |
| `execute` | Run it (read-only) | `run_sql()` |
| `explain` | Turn the outcome into English | `ANSWER_PROMPT` |

**How a retry looks to the LLM** (`write_sql`):
```
[system]  rules + schema
[human]   example question     [ai] example SQL        (few-shot)
[human]   How many tracks are in the 90's Music playlist?
[ai]      SELECT COUNT(pt.TrackId) ... WHERE p.Name = '90''s Music'      ← its own previous answer
[human]   Your previous query failed.
          Problem: Playlist.Name has no value "90's Music". Did you mean: '90’s Music', 'Music'?
          Write a corrected query. Only use tables, columns and values that exist.
```
The LLM sees **its mistake + a specific hint**, which is the same "errors as information" idea as tools in Lesson 3.5.

### 1.4 The routing functions (the decisions)
```python
def after_write(state):      # answerable? → validate, otherwise → explain
def retry_or_give_up(state): return "write_sql" if state["attempts"] < MAX_ATTEMPTS else "explain"
def after_validate_auto(state): return retry_or_give_up(state) if state.get("error") else "execute"
def after_execute(state):       return retry_or_give_up(state) if state.get("error") else "explain"
```
`MAX_ATTEMPTS = 3` in `config.py`: 1 try + up to 2 fixes. The `recursion_limit` (Lesson 5.3) is the backup safety net.

---

## 2. Two upgrades in `db.py`

### 2.1 `check_sql`: from keyword blocklist → parsed **allowlist**
v0.2–v0.3 searched the text for words like `update`. That **blocked harmless queries** (`LIKE '%update%'`, a known limitation since v0.2) and could miss tricks.

v0.4 **parses** the SQL with **`sqlglot`** into a tree, then checks:
1. It parses at all (otherwise: "SQL syntax error", sent back to the LLM to fix)
2. Exactly **one** statement
3. That statement is a **SELECT** (or `UNION` of SELECTs). That's an **allowlist**: anything else is refused
4. No write operation **anywhere** in the tree

**Tested:**
```
DELETE FROM Customer                      → Only SELECT queries are allowed (got DELETE).
SELECT 1; DROP TABLE Customer             → Only one statement is allowed.
PRAGMA table_info(Track)                  → Only SELECT queries are allowed (got PRAGMA).
ATTACH 'x' AS y                           → Only SELECT queries are allowed (got ATTACH).
SELECT ... WHERE Name LIKE '%update%'     → allowed ✅  (the old false positive is FIXED)
WITH t AS (...) SELECT ...                → allowed ✅
```
> 💡 **Blocklist vs. allowlist** (Lesson 3.6, deeper knowledge): a blocklist lists *bad* things and always misses some. An allowlist lists the *only good* thing (a single SELECT). PRAGMA and ATTACH are refused **without anyone having to think of them**.

The **read-only connection** (`mode=ro`) is still there as the final layer.

### 2.2 `check_values`: the fix for "90’s Music"
For every `column = 'text'` or `column IN ('a', 'b')` in the query:
1. Find the real table (resolving aliases like `p` → `Playlist`)
2. Does that exact value exist in that column? If yes → fine.
3. If not → look for **similar** values (Python's `difflib`). If any are close → report them.
4. If nothing is similar (e.g. `'Atlantis'`) → **no complaint**: "no rows" is the honest answer.

**Tested:**
```
P.Name = '90''s Music'      → Playlist.Name has no value "90's Music". Did you mean: '90’s Music', 'Music'?
Country = 'brazil'          → Customer.Country has no value 'brazil'. Did you mean: 'Brazil'?
ar.Name = 'ACDC'            → Artist.Name has no value 'ACDC'. Did you mean: 'AC/DC', 'Accept'?
Country IN ('Brasil','USA') → Customer.Country has no value 'Brasil'. Did you mean: 'Brazil'?
Country = 'Atlantis'        → (no complaint, honest "no rows")
Country = 'Brazil'          → (exists, no complaint)
```
> 🔑 This combines **AI** (the LLM writes and fixes SQL) with **plain code** (exact value lookups and fuzzy matching). The code gives the LLM **precise, checkable facts** instead of hoping it guesses right.

---

## 3. v0.5: the conversation graph

### 3.1 The graph
```
START → router ─┬─ greeting  → greet ─────────────────────────► END
                ├─ off_topic → off_topic ─────────────────────► END
                └─ database  → pick_tables → write_sql → validate ─┬→ execute → explain → END
                                    (the v0.4 loop, unchanged)     └→ approve (if --approve)
```
`build_chat_graph()` reuses **the same v0.4 nodes** (`_add_sql_nodes`) and adds three things:

| Feature | How | Lesson |
|---|---|---|
| **Router** | One LLM call (structured `Route`) classifies **and** rewrites follow-ups into standalone questions | 5.2 + 4.5 (query rewriting) |
| **Memory** | `messages` with `add_messages` + a **checkpointer** + one **thread id** per conversation | 5.4 |
| **Human approval** | An `approve` node calls `interrupt()` with the SQL; `main.py` asks y/n and resumes with `Command(resume=...)` | 5.5 |

### 3.2 A gotcha we had to handle: memory remembers *everything*
(`widened` is reset too. See 3.3.)
With a checkpointer, the state is **saved between turns**. So on turn 2, `error`, `attempts`, `fix_log` and `sql_query` from turn 1 are **still there**, and a leftover `attempts = 2` would wrongly allow only one try. So the **router resets the per-question fields** at the start of every turn:
```python
return {"route": ..., "question": ...,
        "tables": [], "schema": "", "sql_query": None, "attempts": 0,
        "error": None, "fix_log": [], "result": None, "answer": None}
```
Only `messages` (the conversation) is meant to carry over.

### 3.3 A bug the live test found: "tracks by Nirvana" (and the `widen_schema` fix)
Testing the real app with a follow-up:
```
You: How many tracks are in the 90's Music playlist?     → 1,477 ✅
You: Which of those are by Nirvana?
     → rewritten: "Which of the tracks in the 90's Music playlist are by Nirvana?"   ✅
     → tables picked: Album, Genre, MediaType, Playlist, PlaylistTrack, Track      ← no Artist!
     → "The database doesn't have that information."                              ❌
```
The rewrite was perfect. **Schema RAG missed the `Artist` table** (it's 2 foreign-key hops from Track, and we only follow 1), so the LLM honestly said "not answerable". **The LLM was right given what it saw; what it saw was incomplete.**

**The fix is a graph-shaped one: a fallback path.** If the answer is "not answerable" **and** the LLM only saw *some* tables, go to `widen_schema` (all 11 tables) and try **once** more:
```python
def after_write(state):
    if query.is_answerable and query.sql.strip():
        return "validate"
    if not state.get("widened") and len(state["tables"]) < len(all_tables) and state["attempts"] < MAX_ATTEMPTS:
        return "widen_schema"          # maybe schema RAG missed a table: try with everything
    return "explain"                   # truly not answerable
```
**After the fix, the first try still failed, and taught us one more thing:**
```
Attempt 1: not answerable (6 tables)       → 🔧 widen: retrying with all 11 tables
Attempt 2: WHERE P.Name = '90''s Music'    → 🔧 "Did you mean: '90’s Music'?"
Attempt 3: is_answerable = False, "The playlist '90's Music' does not exist."   ❌
```
The LLM **misread the hint**: it decided the playlist doesn't exist instead of using the suggestion. So we made the retry message **explicit**:
```
"If the problem says 'Did you mean', the user meant that value: copy it EXACTLY
 (including any special characters such as ’) into the query, and do not mark the
 question as not answerable because of the spelling difference."
```
**Final result:** two different fixes chained within the 3-attempt limit:
```
Q: How many tracks by Nirvana are in the 90's Music playlist?   (attempts = 3)
   🔧 Not answerable with tables Album, Genre, MediaType, Playlist, PlaylistTrack, Track; retrying with all 11 tables.
   🔧 Playlist.Name has no value "90's Music". Did you mean: '90’s Music', 'Music'?
   SQL:    ... JOIN Artist ar ... WHERE ar.Name = 'Nirvana' AND p.Name = '90’s Music'
   Answer: There are 29 tracks by Nirvana in the 90's Music playlist.      ✅ (checked against the DB: 29)
```
**Feedback must be unambiguous.** "Did you mean X?" seemed clear to us, but the model needed to be told exactly what to *do* with it.

> 🔑 **Lesson:** retrieval (schema RAG) can fail **silently**. A cheap **fallback** (use everything once) turns a wrong "I can't" into a right answer. The cost: one extra LLM call for questions that are truly unanswerable (like "average age").

### 3.4 A safety warning we fixed: allow-listing our own types in memory
The first run printed: *"Deserializing unregistered type text2sql.app.schemas.SQLQuery from checkpoint. This will be blocked in a future version."* The checkpointer saves our `SQLQuery` objects and rebuilds them when loading. LangGraph wants us to **explicitly allow** that (rebuilding arbitrary classes from storage can be a security risk). Fixed with:
```python
def make_checkpointer() -> InMemorySaver:
    serde = JsonPlusSerializer(allowed_msgpack_modules=[("text2sql.app.schemas", "SQLQuery")])
    return InMemorySaver(serde=serde)
```

---

## 4. Run it

```powershell
python -m text2sql.main
```
```powershell
python -m text2sql.main --approve
```
```powershell
python -m text2sql.main "How many tracks are in the 90's Music playlist?"
```
```powershell
python -m text2sql.evals.eval_graph
```

### Results from our test runs

**v0.4: self-correction**
```
Q: How many tracks are in the 90's Music playlist?   (attempts = 2)
   🔧 fixed: Playlist.Name has no value "90's Music". Did you mean: '90’s Music', 'Music'?
   SQL:    SELECT COUNT(pt.TrackId) ... WHERE p.Name = '90’s Music'
   Answer: There are 1,477 tracks in the 90's Music playlist.          ✅ (v0.3 said 0)

Q: What is the average age of customers?   (attempts = 1)
   Answer: The database doesn't have that information.                 ✅ (no pointless retries)
```

**Watching the graph step by step** (`stream_mode="updates"`):
```
node: pick_tables | keys: ['schema', 'tables']
node: write_sql   | keys: ['attempts', 'error', 'fix_log', 'sql_query']
node: validate    | keys: ['error']
node: execute     | keys: ['result']
node: explain     | keys: ['answer', 'messages', 'result']
```

**v0.5: a conversation with memory**
```
You: Hi there!
Bot: Hi! I can answer questions about the Chinook music store: ...                (greeting, no SQL)

You: Who is Jane Peacock's manager?
Bot: Jane Peacock's manager is Nancy Edwards.

You: What is her phone number?
     → rewritten: "What is Nancy Edwards's phone number?"
Bot: Nancy Edwards's phone number is +1 (403) 262-3443.

You: And how many customers does Jane support?
     → rewritten: "How many customers does Jane Peacock support?"
Bot: Jane Peacock supports 21 customers.

You: What's the weather in Paris?
Bot: Sorry, I can only answer questions about the Chinook music store database ...  (off-topic)
```
- **"her" was resolved to Nancy Edwards**, the person named in the *previous answer*. A user might have meant Jane, so **pronouns are genuinely ambiguous**, and showing the rewritten question lets the user see how it was understood.
- A **new thread** asking *"What is her phone number?"* got *"The database doesn't have that information."* Different thread = no memory of Jane or Nancy. ✅

**v0.5: human approval**
```
PAUSED at: ('approve',)
  SQL: SELECT COUNT(*) FROM Invoice;
  Run this query? (y/n): y   →  "There are 412 invoices in the database."
  Run this query? (y/n): n   →  "The query was not run because the user declined it."
```

**Evaluation: v0.3 chain vs. v0.4 graph** (12 gold questions + 3 "messy value" questions):
```
✅ ✅  How many customers are from Brazil?                  attempts=1
✅ ✅  Which genre earned the most money?                   attempts=1
 ...   (10 more where both are correct on the first try)
❌ ✅  How many tracks are in the 90's Music playlist?      attempts=2
        🔧 fixed: Playlist.Name has no value "90's Music". Did you mean: '90’s Music', 'Music'?
✅ ✅  How many customers are from Brasil?                  attempts=1
✅ ✅  How many albums does ACDC have?                      attempts=1
❌ ✅  How many tracks are in the 90s Music playlist?       attempts=2
        🔧 fixed: Playlist.Name has no value '90s Music'. Did you mean: '90’s Music', 'Music'?

v0.3 chain: 13/15 correct
v0.4 graph: 15/15 correct
v0.4 problems caught and fixed by the loop: 2
```
**What the numbers say:**
- **v0.4 fixed exactly the questions v0.3 got wrong**, and broke none of the others.
- **13 of 15 needed only 1 attempt**, so the loop costs nothing extra when the first try is right. It only kicks in when there's a real problem.
- "Brasil" and "ACDC" passed on the first try in **both** versions: Gemini corrected those spellings itself. The value check is the safety net for the cases it *doesn't* catch, like the curly apostrophe nobody can see.

---

## 5. Known limitations (what's left for Phase 6)

| Limitation | Example | Idea |
|---|---|---|
| Memory lives in RAM | Restart the app → the conversation is forgotten | `SqliteSaver`/`PostgresSaver` (Phase 6) |
| Pronouns are ambiguous | "her" → Nancy, maybe the user meant Jane | Show the rewritten question (we do) and let the user correct it |
| The value check covers `=` and `IN` only | `LIKE 'acdc%'` isn't checked | Extend to LIKE, or retrieve sample values into the prompt |
| Approval for every query can be tiring | — | Ask only for expensive or unusual queries |
| Terminal only | — | **Streamlit UI** (Phase 6) |
| Evaluation is small (15 questions) | — | A bigger hold-out set + LangSmith tracing (Phase 6) |

---

## 6. 🎯 Points to stress + deeper knowledge

### Points to stress
1. **v0.4 = the first version that doesn't trust its first try.** Validate → send the problem back → retry, at most 3 times.
2. **Good feedback makes good fixes.** "Did you mean '90’s Music'?" is far more useful to the LLM than "0 rows".
3. **AI + plain code beats AI alone.** Code checks facts exactly (values, safety); the LLM handles language and intent.
4. **Allowlist > blocklist.** Parsing the SQL fixed a false positive **and** blocked PRAGMA/ATTACH automatically.
5. **v0.5 reuses v0.4's nodes.** Router, memory and approval were **added around** the core, not rewritten.
6. **Memory means resetting per-question fields.** Otherwise old errors leak into new questions.

### Deeper knowledge (interview level)
| Topic | What to know |
|---|---|
| **Self-correction / reflection** | Feeding execution errors back to the model is a standard Text2SQL technique and a common agent pattern ("reflection"). Concrete error messages matter more than the number of retries. |
| **Validation before execution** | Checking *before* running is cheaper and safer than running and inspecting. Production systems also check **cost** (e.g. `EXPLAIN`, row limits) before execution. |
| **Deterministic guardrails** | Rules in code (allowlists, value checks, read-only connections) are **testable and predictable**. Prompt instructions are not. Use both, and trust code for safety. |
| **Graph reuse** | One builder function adding a group of nodes (`_add_sql_nodes`) lets several graphs share logic. LangGraph also supports full **subgraphs** (a compiled graph as a node, as in Script 9). |
| **Interrupts in web apps** | In a web UI, the approval click comes in a *different request*. That works because the paused state lives in the checkpointer, keyed by thread id. |

## 7. Quiz

1. Why did v0.3 answer "0" for the 90's Music question, and how does v0.4 fix it?
2. What does the LLM see on a retry that it didn't see on the first try?
3. Why is parsing the SQL (allowlist) better than searching for keywords (blocklist)? Give two examples from our tests.
4. `check_values` doesn't complain about `'Atlantis'`. Why is that the right behaviour?
5. With memory on, why must the router reset `attempts`, `error` and `fix_log` every turn?
6. How does `main.py` know the graph is waiting for approval, and how does it continue?
7. What did the v0.5 tests show about two different thread ids?

<details>
<summary>Answers</summary>

1. The database stores `90’s Music` (curly apostrophe), but the SQL used a straight one → no match → `COUNT = 0`. v0.4's `check_values` finds that the value doesn't exist, suggests `'90’s Music'`, and the retry uses it → 1,477.
2. Its own previous SQL (as an AI message) plus a human message with the specific problem ("Did you mean: '90’s Music'?") and a request to correct it.
3. Keyword search blocked the harmless `LIKE '%update%'` (false positive) and needed every dangerous word listed. The parsed allowlist only accepts a single SELECT, so `LIKE '%update%'` passes and `PRAGMA`/`ATTACH` are refused without being listed.
4. Nothing similar exists, so the user isn't mistyping a real value. The honest answer is "no customers from Atlantis", and retrying would waste calls.
5. The checkpointer saves the whole state between turns. Old values would leak into the next question; e.g. a leftover `attempts=2` would allow only one try.
6. `graph.invoke` returns a dict containing `"__interrupt__"` with the SQL. `main.py` asks the user, then calls `graph.invoke(Command(resume=answer), config)` with the same thread id.
7. They're separate memories: a new thread asking "What is her phone number?" didn't know who "her" was.
</details>
