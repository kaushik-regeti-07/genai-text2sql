# Phase 6: Production (Text2SQL v1.0)

> **Goal of this phase:** turn a working terminal program into something **other people can use**: a web app, memory that survives restarts, a web API, tracing, an honest evaluation and a path to a public link.

**New packages:** `streamlit` (web app), `langgraph-checkpoint-sqlite` (saved memory), `fastapi` + `uvicorn` (web API). All in `requirements.txt`.
**New files:** `text2sql/ui/app.py`, `text2sql/api.py`, `text2sql/evals/eval_holdout.py`, `.streamlit/config.toml`, `.streamlit/secrets.toml.example`.

---

## ⭐ Why this phase matters

Phases 3–5 made the app **smart**. Phase 6 makes it **usable, trustworthy and shareable**:

| Before Phase 6 | After Phase 6 |
|---|---|
| Terminal only | **Web app** with tables, charts and saved chats |
| Memory forgotten when the program stops | **Saved to SQLite**, survives restarts |
| Only usable from Python | **Web API** any app can call |
| "Why did it answer that?" → guesswork | **Trace** of every step (in-app + LangSmith) |
| Scores from questions we tuned on | **Hold-out** evaluation on 24 new questions |
| Runs only on your laptop | **Deployment-ready** (Streamlit Community Cloud) |

> 🏪 **Analogy:** Phases 3–5 trained a brilliant expert and built them an efficient office. Phase 6 **opens the shop to the public**: a shop front (web app), a filing system that survives the night (saved memory), a phone line for other businesses (API), CCTV (tracing), a mystery shopper (hold-out evaluation) and a real address (deployment).

**The key design idea: one brain, many faces.**
```
            ┌──── text2sql/main.py   (terminal)
graph.py ───┼──── text2sql/ui/app.py (web app)
(the brain) └──── text2sql/api.py    (web API)
                       │
               memory.sqlite (shared saved conversations)
```
The LangGraph graph from Phase 5 **didn't change its logic**. Each "face" is a thin layer that sends messages in and draws the results.

---

## 6.1 The web app (Streamlit)

**Run:**
```powershell
streamlit run text2sql/ui/app.py
```

**What Streamlit is:** a Python library that turns a script into a web page. You write `st.title(...)`, `st.chat_input(...)`, `st.dataframe(...)`, and Streamlit draws the page. **Every click or message re-runs the script from top to bottom**, which is the one idea that makes Streamlit click.

| Feature | How |
|---|---|
| Chat | `st.chat_input` + `st.chat_message` |
| **Results table** | `st.dataframe` from the new `columns` / `rows` in the graph state |
| **Automatic chart** | Label + number columns with 2–25 rows → `st.bar_chart(..., sort=False)` (keeps the SQL's ranking) |
| Single number | `st.metric` ("Result: 1,477") |
| 🔍 "How I got this answer" | An expander with the rewritten question, tables, 🔧 fixes, SQL and raw result |
| Saved conversations | Sidebar list from `list_threads()`; ➕ New / 🗑️ Delete |
| Approval | A toggle; a pending query shows **✅ Run it / ❌ Don't run** buttons that resume the graph with `Command(resume=...)` |
| Expensive objects created once | `@st.cache_resource` for the graph and checkpointer (Streamlit's `lru_cache`) |
| Optional password | `APP_PASSWORD` in `.env` or Streamlit secrets |

**New in the graph (`graph.py`) to support a UI:**
- `execute` now stores a **table** (`columns`, `rows`) via `run_table()` in `db.py`, not just text.
- A new state field **`turns: Annotated[list[dict], operator.add]`** appends one record per question (question, SQL, fixes, table, answer). The UI redraws any conversation from it, even after a restart. (`operator.add` = the "append" reducer from Lesson 5.1.)

**Results from testing (browser + Streamlit's own headless tester, 0 errors):**
```
"Top 5 countries by total sales"         → table + bar chart (USA 523.06, Canada 303.96, ...)
"How many tracks are in the 90's Music…" → "Result: 1,477"   🔧 Fix 1 shown in the expander
"How many of those are by Nirvana?"      → 29, "Understood as: How many tracks in the 90's Music playlist are by Nirvana?"
Approval toggle on                       → "I'd like to run this query. Approve?" → ✅ Run it → "There are 8 employees"
Page refresh                             → old conversations still in the sidebar, fully redrawn
```

**Bugs and polish found while testing:**
| Found | Fix |
|---|---|
| Bars sorted alphabetically (Brazil first), not by the SQL's ranking | `sort=False` |
| Metric labelled `COUNT(*)` and showing `1477` | Label → "Result" for non-name columns; format numbers with commas |
| `use_container_width` deprecated in Streamlit 1.67 | `width="stretch"` |
| `streamlit run` couldn't `import text2sql` | Add the project folder to `sys.path` (same as Lesson 5.6) |
| Streamlit sends anonymous usage statistics by default | Turned off in `.streamlit/config.toml` |

---

## 6.2 Persistent memory (SQLite checkpointer)

Phase 5 used `InMemorySaver`, so memory was lost when the program stopped. Now:
```python
def make_checkpointer(path=None):
    serde = JsonPlusSerializer(allowed_msgpack_modules=[("text2sql.app.schemas", "SQLQuery")])
    if path is None:
        return InMemorySaver(serde=serde)                      # Phase 5
    conn = sqlite3.connect(str(path), check_same_thread=False)  # web servers use several threads
    return SqliteSaver(conn, serde=serde)                       # v1.0: saved to text2sql/data/memory.sqlite
```

**Restart test (two separate runs of the program):**
```
Run 1:  python -m text2sql.main --thread restart-test "Which artist has the most albums?"
        → Iron Maiden                                  (program ends; memory saved to the file)
Run 2:  python -m text2sql.main --thread restart-test "How many tracks do they have in total?"
        → understood as "How many tracks do Iron Maiden have in total?" → 213 ✅
```

**New CLI options:** `--thread NAME` (continue a named conversation), `--list` (show saved ones), `--no-save` (RAM only, the Phase 5 behaviour).

> 🔐 `memory.sqlite` holds people's questions, so it's **git-ignored** (`*.sqlite`) and never pushed.

---

## 6.3 Tracing

**Built in:** the 🔍 expander in the web app and `fix_log` in the CLI show *what happened* for each question.

**LangSmith** (LangChain's tracing service) shows **every node, every LLM call, the exact prompts, token counts and timings**. It needs **no code changes**: LangChain sends traces automatically when these are set (in `.env`, see `.env.example`):
```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key_here     ← free account at smith.langchain.com
LANGSMITH_PROJECT=text2sql
```
Every run is labelled (`run_name="text2sql-chat"`, tags `cli` / `web` / `api`, plus the `thread_id`), so you can filter traces by where they came from or by conversation. The web app's sidebar shows "📡 LangSmith tracing on" when it's active.

> ⚠️ Tracing sends prompts and answers to LangSmith's servers. That's fine for this sample database, but think twice for real company data.

---

## 6.4 Hold-out evaluation (the honest test)

**Why:** the earlier test sets were partly used while choosing table descriptions and the retry message, so their scores could be **optimistic** (overfitting, flagged in Lesson 4.6). A **hold-out set** = new questions the system was **never tuned on**.

**The set (`evals/eval_holdout.py`):** 24 questions in 6 categories: simple, join, aggregation, ranking, date, messy value (lowercase names like "led zeppelin") and **unanswerable** (should be declined). Every gold answer was checked against the database, and a 3-way tie ("city with the most customers") was **dropped** as unfair.

**Fairer grading than before:**
- Numbers match **within rounding** (5.65 vs 5.6533)
- Text matches if **contained**, ignoring case ("Jane" in "Jane Peacock")
- Unanswerable questions are correct only if the system **declines**

**Results** (run 2026-11-12):
```
category        v0.3 chain   v0.4+ graph
simple             3/3          3/3
join               7/7          7/7
aggregation        3/3          2/3     ← graph miss (see below)
ranking            5/5          5/5
date               2/2          2/2
messy value        1/2          2/2     ← chain miss: "led zeppelin" (lowercase)
unanswerable       2/2          2/2     (graph tried the full schema first, then declined ✅)
-------------------------------------------
total            23/24 (96%)  23/24 (96%)
```

**Reading the result honestly:**
- **Both are equally accurate on new questions (96%)**, so our earlier 15/15 vs 13/15 *was* a bit flattered by tuning. That's exactly why hold-out sets exist.
- **They missed different questions.** The chain failed a **messy value** ("led zeppelin" in lowercase), which is what the graph's safety nets are built for. The graph failed an ordinary question by random LLM variation:
  ```
  Q: How many albums have more than 20 tracks?      (correct answer: 17)
  graph SQL: SELECT AlbumId FROM Track GROUP BY AlbumId HAVING COUNT(*) > 20 LIMIT 5
  result:    [(23,), (24,), (39,), (51,), (73,)]    ← LISTED album ids instead of COUNTING them
  ```
  The SQL was safe and valid, and every value existed, so **no check could catch it**. It's **correct-looking SQL that answers a slightly different question**. The same model got it right in the chain run, which shows LLM output varies from run to run.
- **Unanswerable questions cost 2 attempts in the graph** (it widens to all tables before declining): safer, but slower.

**What would catch the "listed instead of counted" mistake?** A **result sanity check** ("a *how many* question should return one number") or an **LLM self-review node** ("does this SQL answer the question?") placed before `execute`. That's a natural next step, and one more node in the graph.

> 💡 **The lesson:** checks catch *invalid* SQL; only evaluation catches *valid-but-wrong* SQL. Keep a hold-out set, run it after every change, and read the failures, not just the score.

Results are also saved as JSON in `text2sql/evals/results/` so future versions can be compared.

---

## 6.5 The web API (FastAPI)

**Run:**
```powershell
uvicorn text2sql.api:app --reload
```
Then open **http://127.0.0.1:8000/docs**, an interactive page generated automatically from the code, where you can try every endpoint.

| Endpoint | Does |
|---|---|
| `POST /chat` | `{message, thread_id, approval}` → answer, SQL, tables, fixes, columns, rows (or `needs_approval`) |
| `POST /approve` | `{thread_id, decision: "y"/"n"}` → resumes a paused query |
| `GET /threads` | Saved conversation ids |
| `GET /threads/{id}` | All turns in a conversation |
| `DELETE /threads/{id}` | Forget a conversation |
| `GET /health` | `{"status": "ok"}`, for monitoring |

**Pydantic again (Lesson 3.4):** request and response shapes are Pydantic models, so FastAPI **validates input** and **documents** the API automatically.

**Tested (FastAPI's test client):**
```
/health → ok · unknown thread → 404 · approve with nothing pending → 409
/chat (approval on) → needs_approval, SQL: SELECT COUNT(*) FROM Employee;
/approve y          → "There are 8 employees in the database."  columns ['COUNT(*)'], rows [[8]]
/chat follow-up     → "How many employees are Sales Support Agents?" → 3
```

---

## 6.6 Deployment (Streamlit Community Cloud)

Everything is ready. The last steps need **your** accounts, so they're yours to click. It takes about 10 minutes:

1. Go to **share.streamlit.io** and sign in with **GitHub** (account `kaushik-regeti-07`).
2. **Create app** → repository `kaushik-regeti-07/genai-text2sql`, branch `main`, main file **`text2sql/ui/app.py`**.
3. **Advanced settings** → Python **3.12** → **Secrets**: paste (see `.streamlit/secrets.toml.example`):
   ```
   GOOGLE_API_KEY = "your real key"
   APP_PASSWORD = "a password you choose"
   ```
4. **Deploy.** The first question takes longer while the table index builds (11 embedding calls).

**Things to know (honest list):**
| Point | Why |
|---|---|
| **Set `APP_PASSWORD`** | A public URL lets anyone use **your** Gemini free-tier quota (100 embedding requests/minute, daily limits) |
| Saved chats reset on the cloud | The cloud disk is temporary. Chats survive page refreshes but not app restarts or sleep. Real persistence = a hosted database (e.g. Postgres with `PostgresSaver`). |
| The app sleeps when unused | The free tier wakes it on the next visit (takes a minute) |
| Not deployed by me, so not yet tested in the cloud | If the cloud logs show a SQLite version error from Chroma, the usual fix is adding `pysqlite3-binary` and swapping it in before importing Chroma. Ask me and I'll add it. |

---

## 🎯 Points to stress + deeper knowledge

### Points to stress
1. **One brain, many faces.** CLI, web app and API all use the **same graph**. Production work is mostly wrapping, not rewriting.
2. **State designed for display.** `turns` and the result table exist so any face can redraw a conversation.
3. **Persistence is a one-line swap**: `InMemorySaver` → `SqliteSaver`, because the checkpointer is a pluggable piece.
4. **Honest evaluation needs unseen questions.** Tuning on your test set inflates scores.
5. **Security doesn't stop at SQL:** a password for public demos, secrets never committed, private data (`memory.sqlite`) git-ignored, telemetry off.

### Deeper knowledge (interview level)
| Topic | What to know |
|---|---|
| **Streamlit's execution model** | The script re-runs on every interaction; `st.session_state` keeps per-user values; `st.cache_resource` keeps shared objects (models, DB connections) across re-runs and users. |
| **Stateless API + stateful checkpointer** | The API process keeps nothing between requests. All conversation state lives in the checkpointer, keyed by `thread_id`, so you can run several API servers behind a load balancer (with a shared database such as Postgres). |
| **Interrupts over HTTP** | `/chat` returns `needs_approval`; minutes later `/approve` resumes. That works only because the paused graph is saved in the checkpointer. |
| **Observability** | Production LLM apps log traces (inputs, outputs, latency, tokens, errors) to debug and monitor cost. LangSmith, Langfuse and OpenTelemetry-based tools are common. |
| **Evaluation hygiene** | Keep separate **dev** (for tuning) and **hold-out** (for reporting) sets; save results per version to catch regressions. |
| **Cost and abuse control** | Authentication, rate limits and usage caps matter as much as model quality once an app is public. |

---

## Quiz

1. What does "one brain, many faces" mean in this project?
2. Why did the graph need new `turns`, `columns` and `rows` fields for the web app?
3. What changed to make memory survive restarts, and why `check_same_thread=False`?
4. Why is a hold-out evaluation more honest than our earlier test sets?
5. How does an API client approve a query, and why does that work across two separate requests?
6. Why set `APP_PASSWORD` before deploying publicly?

<details>
<summary>Answers</summary>

1. The CLI, web app and API all use the same LangGraph graph; each is a thin layer that sends messages in and displays results.
2. To display results as a real table/chart and to redraw any old conversation (question, SQL, fixes, table, answer) from saved state.
3. The checkpointer switched from `InMemorySaver` to `SqliteSaver` writing to `memory.sqlite`. `check_same_thread=False` because web servers handle requests on different threads.
4. The earlier questions influenced design choices (descriptions, retry wording), which can inflate scores. New, unseen questions measure how it really generalizes.
5. `/chat` with `approval: true` returns `needs_approval` and the SQL; the client then calls `/approve` with the same `thread_id`. The paused graph is saved in the checkpointer, so the second request can resume it.
6. Anyone with the URL could use your Gemini API quota (and your key's limits); a password keeps the demo to people you share it with.
</details>
