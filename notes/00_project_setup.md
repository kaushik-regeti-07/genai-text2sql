# Session 0: Project Setup

> **Goal of this session:** get a clean, working project folder so we can start learning LangChain.
> By the end, you should be able to explain **what** we installed, **why** we need it, and **where** it lives.

---

## 1. The big picture: what are we building?

We are learning Generative AI by building one project step by step: **Text2SQL**.

```
You type:   "Which 5 customers spent the most money?"
     ↓
The LLM writes SQL:   SELECT ... FROM Customer JOIN Invoice ... LIMIT 5;
     ↓
Our code runs the SQL on a real database
     ↓
The LLM explains:   "Your top customer is Helena Holý with $49.62..."
```

Along the way we learn **LangChain** (Phase 3), **RAG** (Phase 4), **LangGraph** (Phase 5) and **deployment** (Phase 6).

---

## 2. The tools on our computer

### 2.1 Python
The programming language everything is written in. We use **Python 3.12**.

Check it:
```powershell
python --version
```

### 2.2 uv (the package manager)
**What:** a tool that installs Python packages and creates virtual environments.
**Why uv and not plain pip?** It does the same job but is **10–100x faster**, and it's what many modern Python projects use.

Install it once for the whole computer:
```powershell
winget install --id=astral-sh.uv -e
```

> 💡 **Common doubt:** "I installed uv inside my project folder. Is it only for this project?"
> **No.** uv is a program installed on the computer, like Chrome. The folder you were in doesn't matter. It works everywhere.

### 2.3 Virtual environment (`.venv`)
**What:** a private box of Python packages that belongs to **one project only**.
**Why:** Project A might need LangChain version 1, and Project B version 2. Without separate boxes they'd clash.

**Analogy:**
| Thing | Analogy | Scope |
|---|---|---|
| uv | The kitchen | Whole computer (install once) |
| `.venv` | A lunchbox for one project | This project folder only |
| Packages (langchain...) | Food inside the lunchbox | Only this project sees them |

Create it (run **inside** the project folder, because it is created where you stand):
```powershell
cd F:\Project\Gen_AI
uv venv
```

Activate it (do this **every time** you open a new terminal):
```powershell
.venv\Scripts\activate
```

✅ It worked if your prompt starts with `(.venv)`:
```
(.venv) PS F:\Project\Gen_AI>
```

> ⚠️ If you see *"running scripts is disabled on this system"*, run this once, then activate again:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```

To leave the environment: `deactivate`

---

## 3. The packages we install (and why)

All of them are listed in `requirements.txt`. Install everything in one go (with `(.venv)` active):
```powershell
uv pip install -r requirements.txt
```

| Package | What it is | Why we need it | First used in |
|---|---|---|---|
| **langchain** | The core framework for building LLM apps | Gives us prompt templates, chains (`prompt \| model \| parser`) and output parsers, so we don't write everything from scratch | Phase 3 |
| **langchain-google-genai** | The "adapter" between LangChain and Google Gemini | LangChain itself doesn't talk to any AI company. Each provider has its own adapter package. This one is for Gemini. | Phase 3 |
| **langchain-community** | Extra integrations built by the community | Has the `SQLDatabase` helper (reads our database schema), document loaders and vector stores | Phase 3–4 |
| **langgraph** | A framework for building agents as graphs | Lets our app **loop and make decisions**, for example: SQL failed → fix it → try again | Phase 5 |
| **python-dotenv** | Reads the `.env` file | Loads our secret API key into the program without writing it in the code | Phase 3 |
| **sqlalchemy** | A universal Python library for talking to databases | LangChain's SQL tools use it underneath to connect to SQLite (and later Postgres or MySQL) | Phase 3 |

> 💡 **Why is LangChain split into many packages?**
> So you only install what you use. Using Gemini? Install `langchain-google-genai`. Switching to OpenAI later? Install `langchain-openai` and change **one line** of code. The rest stays the same.

**Coming later (not installed yet):**
| Package | Why | When |
|---|---|---|
| `langchain-chroma` | Vector database for RAG | Phase 4 |
| `streamlit` | Web UI for the app | Phase 6 |
| `fastapi`, `uvicorn` | Backend API | Phase 6 |

---

## 4. The LLM provider: why Gemini?

| | Groq | **Gemini** ✅ |
|---|---|---|
| Cost to start | Free | Free |
| Speed | Very fast | Fast |
| Chat models | Yes (open-source Llama, Qwen...) | Yes (Google's models) |
| **Embedding models** | ❌ No | ✅ Yes |

**Deciding reason:** in Phase 4 (RAG) we need **embeddings**, which turn text into vectors. Gemini gives us chat **and** embeddings with **one free key**.

**Get a key:** go to **aistudio.google.com** → sign in → **Get API key**.

> ⚠️ Free-tier data may be used by Google to improve their products. Don't send private or sensitive data.

---

## 5. Files in the project and why each exists

```
Gen_AI/
├── .venv/              ← the lunchbox (packages live here). Never edit by hand.
├── .env                ← YOUR SECRET API KEY. Never share or upload it.
├── .env.example        ← a safe template showing which keys are needed (no real values)
├── .gitignore          ← tells Git which files to NEVER upload (.env, .venv)
├── requirements.txt    ← list of packages, so anyone can recreate the setup
├── notes/              ← these teaching notes
├── learning/
│   ├── phase3/         ← small practice scripts, one per lesson
│   ├── phase4/
│   └── phase5/
└── text2sql/           ← the real project
    ├── README.md       ← project roadmap (v0.1 → v1.0)
    ├── app/            ← core logic: chains, graph, tools
    ├── ui/             ← Streamlit front end (Phase 6)
    ├── data/           ← the Chinook database file
    └── evals/          ← test questions to check accuracy (Phase 6)
```

### Why `.env` + `.env.example`?
- `.env` holds the **real** key → it's in `.gitignore`, so it never reaches GitHub.
- `.env.example` holds a **fake** placeholder → it **is** uploaded, so your friend knows what to fill in.

Create your `.env` from the template:
```powershell
copy .env.example .env
```
Then open `.env` and replace `your_key_here` with your real key:
```
GOOGLE_API_KEY=AIza...your real key...
```

> 🔐 **Golden rule:** API keys go in `.env`, **never** inside `.py` files. If a key leaks, anyone can use it on your account.

---

## 6. The database: Chinook

**What:** a free sample database of a **digital music store**, stored in one SQLite file (about 1 MB).
**Why SQLite:** no server to install. The whole database is a single file.

Download it into `text2sql\data\`:
```powershell
curl.exe -L -o text2sql\data\chinook.db https://github.com/lerocha/chinook-database/raw/master/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite
```
(If the link fails: download `Chinook_Sqlite.sqlite` from the **Releases** page of `github.com/lerocha/chinook-database`, rename it to `chinook.db` and put it in `text2sql\data\`.)

**Reading the curl output:** `100 984.0k 100 984.0k` means 100% of 984 KB was downloaded. ✅
(`curl.exe` = download tool built into Windows, `-L` = follow redirects, `-o` = save to this file.)

**The 11 tables (with row counts):**
| Group | Tables | Holds |
|---|---|---|
| People | `Customer` (59), `Employee` (8) | Who buys, who sells |
| Money | `Invoice` (412), `InvoiceLine` (2,240) | Every purchase and its line items |
| Music catalog | `Artist` (275) → `Album` (347) → `Track` (3,503) | What's for sale |
| Details | `Genre` (25), `MediaType` (5), `Playlist` (18), `PlaylistTrack` (8,715) | Categories and playlists |

### Why Chinook, and why only Chinook?
**Where it came from:** Microsoft made a sample database called *Northwind* years ago. Chinook was created as a free, modern replacement for it, based on real iTunes-style data. It's named after the Chinook winds, as a nod to Northwind.

**Why it's perfect for learning Text2SQL:**
1. **It looks like real business data:** customers, sales, employees, products. These are the questions real companies ask.
2. **Tables are linked:** `Artist → Album → Track → InvoiceLine → Invoice → Customer`. The LLM has to figure out **JOINs**, which is the hard part of Text2SQL.
3. **Small enough for the prompt:** 11 tables fit easily in one LLM prompt, which is what v0.1 needs.
4. **One file, no setup:** SQLite means no database server to install.
5. **It's the standard:** LangChain's official SQL tutorials use Chinook, so answers you find online will match.

**Only Chinook?** Only for **learning**. The code we write works with **any** database, because SQLAlchemy just needs a different connection string:
```
sqlite:///text2sql/data/chinook.db      ← now
postgresql://user:pass@host/dbname      ← later, a real company database
```
Once the app works on Chinook, you can point it at a bigger or different database. That's a good v1.0 stretch goal.

**Example questions we'll ask it:**
- "How many customers are from Brazil?" *(easy, one table)*
- "Which artist has the most albums?" *(medium, one JOIN)*
- "What's the best-selling genre in each country?" *(hard, several JOINs plus grouping)*

---

## 7. Setup checklist ✅

- [ ] `python --version` shows 3.10 or newer
- [ ] `uv --version` works (in a **new** terminal)
- [ ] `.venv` created inside `F:\Project\Gen_AI`
- [ ] Terminal shows `(.venv)` after activating
- [ ] `uv pip install -r requirements.txt` finished without errors
- [ ] `.env` exists and contains your real `GOOGLE_API_KEY`
- [ ] `text2sql\data\chinook.db` exists (about 1 MB)

---

## 8. Quick quiz (to check understanding)

1. If you install uv while standing in `F:\Project\Gen_AI`, can you use it from `C:\`? *Why?*
2. What goes wrong if two projects share one environment?
3. Why does LangChain have a separate `langchain-google-genai` package?
4. Why do we have both `.env` **and** `.env.example`?
5. Which package lets our app retry a failed SQL query in a loop?

<details>
<summary>Answers</summary>

1. Yes. uv is installed for the whole computer. Only `.venv` is tied to a folder.
2. Package versions clash. One project's upgrade can break the other.
3. LangChain is provider-neutral. Each AI provider has its own adapter, so you only install what you use and can switch with one line.
4. `.env` holds the real secret and is never uploaded. `.env.example` is a safe template that shows others what to fill in.
5. `langgraph`.
</details>

---

**Next session → Lesson 3.1: What is LangChain, and your first chat model call.**
