# Phase 6 Recap: Production, as a Story

> The final chapter of the story from Phases 3 (notes/07), 4 (notes/14) and 5 (notes/19). Details: notes/20.

---

## 🏢 The story: the shop opens to the public

**Previously:**
- **Phase 3:** a brilliant **expert with amnesia** in a **sealed room**, given forms, an assembly line, answer boxes and a phone.
- **Phase 4:** a **librarian** who finds the right pages by meaning.
- **Phase 5:** an **office workflow**: junctions, ring roads that fix mistakes, a notebook per customer, a checkpoint booth.

**The new problem:** the office works brilliantly, but **only for the people inside it**. Customers can't walk in (terminal only). The notebooks are thrown away every night (memory in RAM). Other businesses can't call (no API). Nobody can see what happened when something goes wrong. And all the praise came from **practice questions the staff had already seen**.

**Phase 6 opens the shop to the public.**

| Lesson | In the story | In the code |
|---|---|---|
| **6.1 Web app** | A **shop front** with a counter where anyone can ask, and answers come back with **tables and charts**. A "🔍 how I got this" drawer shows the work. | Streamlit `text2sql/ui/app.py` |
| **6.2 Persistent memory** | The notebooks go into a **fireproof filing cabinet**, so tomorrow the expert still knows what you asked today. | `SqliteSaver` → `memory.sqlite` |
| **6.3 Tracing** | **CCTV** in every room: replay exactly what each room did, which notes were passed and how long it took. | LangSmith (env vars) + the in-app trace |
| **6.4 Hold-out evaluation** | A **mystery shopper** asks questions the staff have **never practised**. Score: 96%, and one surprise: the expert once **listed** albums when asked **how many**. | `evals/eval_holdout.py` |
| **6.5 Web API** | A **phone line for other businesses**: they call, ask, and get structured answers (and can say "yes, run it" on a later call). | FastAPI `text2sql/api.py` |
| **6.6 Deployment** | A **real street address** anyone can visit, with a **lock on the door** (password) so strangers don't use up the shop's supplies (API quota). | Streamlit Community Cloud + `APP_PASSWORD` |

**The key idea: one brain, many faces.** The expert and the office (the LangGraph graph) **didn't change**. Phase 6 added doors: a shop front, a phone line and a terminal, all leading to the same office and the same filing cabinet.

---

## All four chapters

| | Phase 3 | Phase 4 | Phase 5 | Phase 6 |
|---|---|---|---|---|
| **Problem** | Doesn't know **how** to work | Doesn't know **your facts** | Can't **go back, decide, remember, wait** | Nobody **outside** can use it |
| **We added** | Forms, assembly line, boxes, phone | Librarian, cabinet, flashcards | Office workflow | Shop front, fireproof cabinet, CCTV, mystery shopper, phone line, address |
| **Controls** | **How** it works | **What** it knows | **The flow** | **Who** can use it, and **how we know** it works |
| **Version** | v0.1–v0.2 | v0.3 | v0.4–v0.5 | **v1.0** |

**The one sentence:**
> *"Phase 6 opens the shop: a web page and an API in front of the same brain, memory that survives the night, cameras to see what happened, a mystery shopper to keep us honest, and a real address with a lock on the door."*

---

## 🐛 Real things we hit in Phase 6

| What happened | Lesson |
|---|---|
| Bars sorted alphabetically instead of by the SQL's ranking | Charts must respect the query's `ORDER BY` (`sort=False`) |
| Metric showed `COUNT(*)` and `1477` | Small polish (labels, number formatting) makes a demo look professional |
| `use_container_width` deprecated | Libraries move fast; check warnings |
| Streamlit telemetry on by default | Turn off data sharing you don't need |
| Hold-out score 96% for **both** chain and graph | Earlier scores were flattered by tuning; read the failures, not just the score |
| Graph **listed** album ids instead of **counting** | Valid-but-wrong SQL passes every check; only evaluation (or a self-review step) catches it |
| A 3-way tie in a candidate test question | Check your answer key before trusting the grader |

---

## Self-check

1. What does "one brain, many faces" mean?
2. What single change made memory survive restarts?
3. Why is a hold-out set more honest than the earlier tests?
4. What kind of mistake can't any of our checks catch?
5. Why lock a public demo with a password?

<details>
<summary>Answers</summary>

1. CLI, web app and API all use the same LangGraph graph; each is a thin layer around it.
2. Swapping `InMemorySaver` for `SqliteSaver` (saving to `memory.sqlite`).
3. Its questions were never used to tune descriptions or prompts, so it measures real generalization.
4. Valid, safe SQL that answers a slightly different question (listing instead of counting).
5. Anyone with the URL would use your Gemini quota and key limits.
</details>
