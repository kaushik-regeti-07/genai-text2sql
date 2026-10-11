# Phase 5 Recap: LangGraph, as a Story + All 9 Scripts

> Continues the story from Phase 3 (notes/07) and Phase 4 (notes/14).

---

## 🏢 The story: the office gets a workflow

**Previously:**
- **Phase 3:** the Chinook store hired a brilliant **expert with amnesia** in a **sealed room**, who communicates by **notes under the door**. We gave them forms, an assembly line, answer boxes and a phone to the archive.
- **Phase 4:** we hired a **librarian** who finds the right pages by meaning and slides them under the door.

**The new problem:** the **assembly line** (the chain) only moves **forward**. Look what happened with the 90's Music playlist:
```
Expert writes SQL with '90''s Music' → archive returns 0 → line moves on → "There are 0 tracks." ❌
```
Nobody on the line could say **"wait, that looks wrong, go back."** The line also couldn't:
- tell a greeting from a real question,
- remember what the customer said a minute ago,
- stop and ask the manager before doing something important.

**Phase 5 replaces the assembly line with an office workflow chart**, a map of rooms and corridors where work can **turn, loop back, wait and remember**. That chart is **LangGraph**.

| Lesson | In the story | In the code |
|---|---|---|
| **5.1 State, nodes, edges** | A **logbook** travels from room to room. Each **room** does one job and writes **only its own lines** in the logbook. **Corridors** connect the rooms. Some pages get **overwritten** (current step), others **keep adding lines** (the conversation). | `State` (TypedDict), nodes return `dict` updates, `add_edge`, reducers (`add_messages`, `operator.add`) |
| **5.2 Routing** | At a **junction**, an officer reads the logbook and points: refunds ≤ $20 → agent's desk, bigger → manager's office. A **receptionist** (LLM) decides: greeting, real question or off-topic. | `add_conditional_edges` + a routing function; the LLM router with `Literal` |
| **5.3 Loops** | A **ring road** back to an earlier room: "not good enough, do it again". Always with a rule: **maximum 3 laps**, plus a building-wide limit as a backup. The expert's phone calls from Phase 3 become rooms: think ⇄ tools. | Conditional edge pointing back, `attempts` in state, `recursion_limit`, `ToolNode`, `tools_condition` |
| **5.4 Memory** | **One notebook per customer**, saved after every step. When Ravi returns, open *his* notebook; Priya's never mixes in. | Checkpointer (`InMemorySaver`) + `thread_id` |
| **5.5 Human approval** | A **checkpoint booth**: the work **stops**, the manager reads the papers and says yes or no, and work continues **from exactly that spot**. | `interrupt()`, `Command(resume=...)` |
| **5.6 Multi-agent** | A **reception desk** that sends each visitor to the right **department**: HR (handbook RAG) for policies, the data team (Text2SQL) for numbers. | A supervisor node + specialist nodes; a graph inside a graph |
| **5.7 Text2SQL v0.4 + v0.5** | The expert's SQL now goes through a **quality inspector** who checks the archive's real labels: *"There's no '90's Music' shelf. Did you mean '90’s Music'?"*, and sends it **back** with that note. Then the whole office gets a reception desk, customer notebooks and a manager's approval booth. | `graph.py`: `validate` → `write_sql` loop, `check_values`, router, memory, `approve` |

**Phase 3 vs. 4 vs. 5:**
| | Phase 3 | Phase 4 | Phase 5 |
|---|---|---|---|
| The expert's problem | Doesn't know **how** to work | Doesn't know **your facts** | Can't **go back, decide, remember or wait** |
| What we gave them | Forms, assembly line, boxes, phone | Librarian, filing cabinet, flashcards | An office workflow chart: junctions, ring roads, notebooks, checkpoints |
| Main idea | Control **how** the LLM works | Control **what** it knows | Control **the flow** of the work |

**The one sentence:**
> *"A chain is an assembly line that only moves forward. LangGraph turns it into an office workflow: a shared logbook, junctions that decide, ring roads that retry, a notebook per customer and a checkpoint where a human can approve, so our Text2SQL app can catch its own mistakes and fix them."*

---

## 🗺️ The 9 scripts

```
5.1  Graph basics      01 → 02          (no LLM)
5.2  Routing           03 (no LLM) → 04
5.3  Loops             05 (no LLM) → 06
5.4  Memory            07
5.5  Human approval    08 (no LLM)
5.6  Multi-agent       09
5.7  The project       text2sql v0.4 + v0.5
```

| # | Script | The one idea |
|---|---|---|
| 01 | `first_graph` | State + nodes + edges; `stream` shows each node's update; `draw_mermaid` draws it |
| 02 | `reducers` | Replace vs. append: `add_messages` keeps the whole conversation |
| 03 | `conditional_edges` | Refunds ≤ $20 → agent, more → manager (the handbook rule) |
| 04 | `llm_router` | The 3.4 classifier as the first node: one LLM call, cheap paths for greetings |
| 05 | `loop_with_limit` | Fail, fail, OK. Your limit + `recursion_limit` |
| 06 | `agent_graph` | Lesson 3.5's Script 19 as a graph: `think ⇄ tools` → ₹1,045.44 |
| 07 | `memory` | Ravi and Priya: two threads, two separate memories |
| 08 | `human_approval` | `interrupt()` pauses the $49.50 refund, `Command(resume=...)` continues |
| 09 | `supervisor` | Policies → handbook RAG, numbers → Text2SQL, pizza → nobody |

**Text2SQL (notes/18):** v0.4 = validate + self-correct loop + `widen_schema` fallback + parsed allowlist + value check (**90’s Music fixed: 0 → 1,477; evaluation 15/15 vs. v0.3's 13/15**). v0.5 = router + memory with follow-up rewriting + optional human approval.

---

## 🐛 Real things we hit in Phase 5

| What happened | Lesson |
|---|---|
| Script 09 couldn't `import text2sql` | Python only looks for packages next to the script; add the project folder to `sys.path` (or run with `-m`) |
| Memory warning: *"Deserializing unregistered type SQLQuery"* | Checkpointers rebuild saved objects; **allow-list** your own types (`make_checkpointer()`) |
| With memory, old `attempts`/`error` would leak into the next question | Reset per-question fields at the start of each turn (the router does this) |
| "What is *her* phone number?" → Nancy (the manager), not Jane | Pronouns are ambiguous; show the rewritten question so users can spot it |
| Script 06 printed a raw content list with a long signature | Use `.text`, not `.content` (Lesson 3.1 again!) |
| "Which of those are by Nirvana?": schema RAG missed `Artist`, so the LLM said "not answerable" | Add a **fallback path** (`widen_schema`): retry once with every table |
| The LLM then misread "Did you mean '90’s Music'?" and gave up | Make feedback **explicit about what to do**: "copy that value exactly" |

---

## Self-check

1. What three things make up every LangGraph app?
2. What does a reducer decide? Which one keeps chat history?
3. How do you make a loop, and what two things stop it running forever?
4. What two pieces give a graph memory?
5. What does `interrupt()` do, and how do you continue?
6. What does a supervisor do in a multi-agent graph?
7. How does Text2SQL v0.4 fix the 90's Music question?
8. Why does the v0.5 router reset some fields every turn?

<details>
<summary>Answers</summary>

1. State, nodes, edges.
2. How an update combines with the old value (replace or append). `add_messages` keeps the chat history.
3. A conditional edge pointing back to an earlier node. Your own attempt limit (graceful) and `recursion_limit` (hard stop).
4. A checkpointer (saves state after every step) and a thread id (which conversation).
5. It pauses the graph and hands a value (a question, the SQL) to the outside. Continue with `graph.invoke(Command(resume=answer), config)`.
6. It routes each question to the right specialist. It doesn't answer itself.
7. `validate` → `check_values` finds that `'90''s Music'` doesn't exist and suggests `'90’s Music'` → back to `write_sql` with that feedback → the corrected query returns 1,477.
8. The checkpointer saves the whole state between turns, so old `attempts`, `error` and `fix_log` would leak into the next question.
</details>
