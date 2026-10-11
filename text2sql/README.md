# Text2SQL

Chat with a database in plain English: the app writes SQL, checks it, fixes its own mistakes,
runs it read-only and explains the result. Follow-up questions work.

## Run it
From the `Gen_AI` folder with the virtual environment active:
```powershell
python -m text2sql.main                                  # chat (memory + follow-ups)
python -m text2sql.main --approve                        # ask before running each query
python -m text2sql.main "Which genre sells the most?"    # one question
python -m text2sql.evals.eval_graph                      # v0.3 chain vs v0.4 graph
python -m text2sql.evals.eval_schema_rag                 # v0.2 full schema vs v0.3 schema RAG
```

## Roadmap
| Version | Feature | Phase | Status |
|---|---|---|---|
| v0.1 | Full schema in prompt -> SQL (printed only) | 3 | ✅ |
| v0.2 | Execute SQL, explain result in plain English | 3 | ✅ |
| v0.3 | Schema RAG: retrieve only relevant tables + foreign-key expansion | 4 | ✅ |
| v0.4 | LangGraph: validate -> execute -> self-correct loop, value check, schema fallback | 5 | ✅ |
| v0.5 | Router, conversation memory + follow-up rewriting, human approval | 5 | ✅ |
| v1.0 | Streamlit UI, persistent memory, tracing, bigger evals, deployment | 6 | |

**Evaluations**
- v0.3 vs v0.2 (12 gold questions): needed tables found 12/12, schema 48% smaller, SQL correct 11/12 for both.
- v0.4 vs v0.3 (12 gold + 3 "messy value" questions): **v0.4 15/15**, v0.3 13/15. Both misses were the
  "90’s Music" playlist (curly apostrophe), which v0.4's value check + retry fixes.

## How v0.4/v0.5 works (`app/graph.py`)
```
START → router ─┬─ greeting  → greet → END
                ├─ off_topic → off_topic → END
                └─ database  → pick_tables → write_sql ⇄ (widen_schema) → validate → [approve] → execute → explain → END
                                                ▲                            │                      │
                                                └──── problem, attempts < 3 ─┴──────────────────────┘
```
- `validate`: parsed allowlist (`check_sql`, sqlglot) + value check (`check_values`: "Did you mean '90’s Music'?")
- `widen_schema`: if the LLM says "not answerable" with only some tables, retry once with all tables
- `approve`: `interrupt()` shows the SQL; `main.py` resumes with `Command(resume="y"/"n")`
- Memory: checkpointer + one thread id per conversation (`make_checkpointer()` allow-lists `SQLQuery`)

## Layout
- `main.py`                  - chat CLI (v0.5)
- `app/config.py`            - settings (paths, models, row limit, TABLE_K, FK_HOPS, MAX_ATTEMPTS)
- `app/llm.py`               - chat + embedding models (the only provider-specific file)
- `app/db.py`                - read-only engine, foreign keys, columns, SQL allowlist, value check, safe runner
- `app/schema_index.py`      - table descriptions, Chroma index (auto-rebuilds), find_tables()
- `app/schemas.py`           - Pydantic output shapes (SQLQuery, Route)
- `app/prompts.py`           - prompt text, few-shot examples, retry message, router prompt
- `app/chains.py`            - v0.3 LCEL chain (kept for comparison)
- `app/graph.py`             - v0.4 SQL graph + v0.5 chat graph
- `evals/eval_schema_rag.py` - schema RAG evaluation
- `evals/eval_graph.py`      - self-correcting graph evaluation
- `data/`                    - Chinook SQLite database + `chroma_db/` index (generated, git-ignored)

## Safety layers
1. Prompt rules (SELECT only)
2. `is_answerable` lets the model decline
3. `check_sql`: the query is parsed; exactly one SELECT, no write operation anywhere (allowlist)
4. Optional human approval before execution (`--approve`)
5. Database opened read-only (`mode=ro`)

## Known limitations
- Memory is in RAM: restarting the app forgets the conversation
- Pronouns can be ambiguous ("her" → the last person mentioned)
- The value check covers `=` and `IN`, not `LIKE`
- Table descriptions are hand-written; evaluation sets are small (overfitting risk)
- Terminal only
