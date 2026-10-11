# Lesson 5.6: Multi-Agent: A Supervisor with Specialists

> **Goal:** combine everything you've built (the **handbook RAG assistant** from Phase 4 and **Text2SQL** from the project) into one assistant, where a **supervisor** decides which specialist answers each question.

**Script:** `learning/phase5/09_supervisor.py`
**Needs:** the handbook index from Lesson 4.3 (Script 11) in `learning/phase4/chroma_db/`.

---

## ⭐ Why this lesson matters

Real companies have **many kinds of knowledge**: policies live in documents, numbers live in databases. One giant prompt that tries to do everything gets confused. Instead:

> 🏢 **Analogy: a company reception desk.** The **receptionist (supervisor)** listens to your question and sends you to the right **department (specialist)**: HR for policies, the data team for numbers. Each department is good at **one** thing.

```
                     ┌──► handbook_agent  (Phase 4 RAG: policies, with page citations)
question ─► supervisor ─┼──► database_agent  (Text2SQL v0.4 graph: facts and numbers)
                     └──► no_agent        (polite "can't help")
```

**Why this design is powerful:**
1. **Each specialist stays simple and testable.** You already evaluated both separately.
2. **Easy to extend:** add a "web search agent" or "email agent" as another node + one more `Literal` option.
3. **Graphs inside graphs:** `database_agent` calls `build_sql_graph()`, a whole **compiled graph used inside a node**. A compiled graph is a Runnable, so it plugs in anywhere.

---

## 1. Key concepts

| Concept | In the script |
|---|---|
| **Supervisor** | An LLM node with structured output: `specialist: Literal["handbook", "database", "none"]` |
| **Specialist** | A node that does one job (RAG chain, or the Text2SQL graph) |
| **Routing** | A conditional edge from the supervisor to the chosen specialist (Lesson 5.2) |
| **Descriptions decide the routing** | The `Field(description=...)` says *handbook = POLICIES and rules, database = FACTS and numbers*. This is the most important line in the script. |

> 💡 It's the **same pattern as v0.5's router** (greeting / off-topic / database), just with bigger destinations. Routing is the backbone of multi-agent systems.

---

## 2. Results from our test run

```
Q: Who has to approve a $50 refund?
   → handbook: A refund above $20 must be approved by the Sales Manager (page 3).
Q: How many customers do we have in Canada?
   → database: We have 8 customers in Canada.
Q: How many vacation days do full-time employees get?
   → handbook: Full-time employees receive 20 days of paid vacation per year (page 5).
Q: Which sales agent sold the most in 2023?
   → database: Jane Peacock sold the most in 2023, with a total of $184.34.
Q: What's a good pizza place nearby?
   → none: Neither specialist can help with that.
```
**5 of 5 routed correctly.** Each answer matches what the specialist gave on its own in Phases 4 and 4.6.

**What to notice:**
- **Policy questions** ("who approves a $50 refund?", "vacation days") went to the **handbook** and came back **with page citations**, from the Phase 4 RAG system.
- **Data questions** ("customers in Canada", "best sales agent in 2023") went to **Text2SQL** and came back with real numbers from the database.
- **Off-topic** ("pizza") went to neither.
- The supervisor never sees the data. It only decides **who** should answer.

---

## 3. 🎯 Points to stress + deeper knowledge

### Points to stress
1. **Multi-agent = routing + specialists.** The supervisor is a router, and each specialist is a chain or graph.
2. **Specialists stay small and testable.** Build and evaluate them separately, then connect them.
3. **A compiled graph can be a node** inside another graph (graphs are Runnables).
4. **The supervisor's descriptions are the most important prompt.** Vague descriptions mean questions go to the wrong department.

### Deeper knowledge (interview level)
| Topic | What to know |
|---|---|
| **Supervisor vs. swarm** | **Supervisor:** one central router decides every hand-off (simple, predictable). **Swarm / network:** agents hand off to each other directly (flexible, harder to debug). Start with a supervisor. |
| **Tool-calling supervisor** | Another style: give the supervisor each specialist **as a tool** (Lesson 3.5) and let it call several in one turn, e.g. "compare our refund policy with how many refunds were issued". |
| **Don't over-agent** | Every extra agent adds LLM calls, latency and failure points. If one well-prompted chain works, **use the chain**. Multi-agent is for genuinely different jobs. |
| **Evaluation still applies** | Test the **routing** itself (did each question go to the right specialist?) with a small labelled set, just like the hit rate in 4.5. |

## 4. Quiz

1. What does the supervisor decide, and what does it *not* do?
2. How does `database_agent` reuse the whole Text2SQL v0.4 graph?
3. A question about "refund policy" keeps going to the database specialist. What do you fix first?
4. When is a multi-agent design a bad idea?

<details>
<summary>Answers</summary>

1. It decides **which specialist** answers. It doesn't look up data or write the answer itself.
2. It calls `build_sql_graph().invoke({"question": ...})`. A compiled graph is a Runnable, so it can run inside another graph's node.
3. The supervisor's `Field(description=...)` for the categories. Make "POLICIES and rules" vs. "FACTS and numbers" clearer, then re-test the routing.
4. When one chain or graph can handle the job well. Extra agents add cost, latency and more things that can break.
</details>

**Next → Lesson 5.7: 🎯 Text2SQL v0.4 + v0.5** (notes/18).
