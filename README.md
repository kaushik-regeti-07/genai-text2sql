# Gen_AI: Learning Generative AI by Building Text2SQL

A hands-on journey through **LangChain, RAG and LangGraph**, built around one growing project:
a **Text2SQL assistant** that answers plain-English questions about a music store database
(the [Chinook](https://github.com/lerocha/chinook-database) sample database).

```
You:  How many tracks are in the 90's Music playlist?
      🔧 Fix 1: Playlist.Name has no value "90's Music". Did you mean: '90’s Music'?
      SQL:    SELECT COUNT(*) FROM PlaylistTrack pt JOIN Playlist p ... WHERE p.Name = '90’s Music'
Bot:  There are 1,477 tracks in the 90's Music playlist.

You:  How many of those are by Nirvana?
      (rewritten: "How many tracks in the 90's Music playlist are by Nirvana?")
      🔧 Fix 1: Not answerable with 6 tables; retrying with all 11 tables.
      🔧 Fix 2: Playlist.Name has no value "90's Music". Did you mean: '90’s Music'?
Bot:  There are 29 tracks in the 90's Music playlist by Nirvana.
```

## What's inside
| Folder | Contents |
|---|---|
| [`text2sql/`](text2sql/) | **The project** (currently v0.5). See its [README](text2sql/README.md). |
| [`learning/`](learning/) | Small practice scripts, one per lesson (Phase 3: 20, Phase 4: 18, Phase 5: 9) |
| [`notes/`](notes/) | Lesson notes written for teaching: concepts, analogies, tested code, real results, quizzes |

## The roadmap
| Phase | Topic | Project version | Status |
|---|---|---|---|
| 3 | **LangChain**: chat models, prompts, LCEL chains, structured output, tools | v0.1–v0.2: question → SQL → answer | ✅ |
| 4 | **RAG**: embeddings, Chroma, loaders/splitters, retrieval, evaluation | v0.3: schema RAG (only relevant tables) | ✅ |
| 5 | **LangGraph**: state, routing, loops, memory, human-in-the-loop, multi-agent | v0.4–v0.5: self-correcting graph + chat | ✅ |
| 6 | **Production**: web UI, persistent memory, tracing, evaluation, deployment | v1.0 | ⬜ |

**Start reading at:** [notes/00_project_setup.md](notes/00_project_setup.md). Each phase has a story-style
recap: [Phase 3](notes/07_phase3_recap.md), [Phase 4](notes/14_phase4_recap.md), [Phase 5](notes/19_phase5_recap.md).

## Highlights
- **Self-correcting SQL** (LangGraph): validate → retry with specific feedback, up to 3 attempts
- **Layered safety**: parsed SQL allowlist (sqlglot), read-only database, optional human approval
- **Schema RAG**: retrieve relevant tables + foreign-key expansion (48% smaller prompts, same accuracy)
- **Measured, not guessed**: evaluation scripts for every major change (v0.4: 15/15 vs v0.3: 13/15)

## Setup
Requires Python 3.12 and a free [Google AI Studio](https://aistudio.google.com) API key (Gemini).
```powershell
uv venv
.venv\Scripts\activate
uv pip install -r requirements.txt
copy .env.example .env        # then put your GOOGLE_API_KEY in .env
python -m text2sql.main
```

## Tech
Python · LangChain · LangGraph · Google Gemini (`gemini-3.5-flash-lite`, `gemini-embedding-001`) ·
Chroma · SQLite · SQLAlchemy · sqlglot · Pydantic
