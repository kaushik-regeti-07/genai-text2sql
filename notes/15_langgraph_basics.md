# Phase 5: LangGraph

## 🗺️ Phase 5 overview

**The problem LangGraph solves:** LCEL chains (Phase 3) run in a **straight line**: A → B → C, always in that order, once. Real apps need to:
- **Decide** what to do next ("is this a greeting or a database question?")
- **Loop** ("the SQL failed, so fix it and try again")
- **Remember** ("what about *her* phone number?")
- **Pause** ("a human must approve this before it runs")

A chain can't do any of that. **A graph can.**

> 🗺️ **Analogy: a chain is a train track; a graph is a road map.** On a track you can only go forward to the next station. On a road map you can turn at junctions, take a ring road back to where you were, stop at a checkpoint, and the car carries a **logbook** of the whole trip.

**You've already met it:** in Lesson 3.5, `create_agent(...)` returned a **`CompiledStateGraph`**. LangChain's agent *is* a LangGraph graph. Phase 5 opens that box.

| Lesson | Topic | Scripts |
|---|---|---|
| **5.1** | **State, nodes, edges, reducers** | 01, 02 (no LLM) |
| **5.2** | **Conditional edges & routing** | 03 (no LLM), 04 |
| 5.3 | Loops: retry loops and the agent loop as a graph | 05 (no LLM), 06 |
| 5.4 | Memory: checkpointers and threads | 07 |
| 5.5 | Human-in-the-loop: `interrupt()` | 08 (no LLM) |
| 5.6 | Multi-agent: a supervisor with specialists | 09 |
| 5.7 | 🎯 **Text2SQL v0.4 + v0.5** | the project |

**New packages:** none for LangGraph itself (`langgraph` was installed in setup). The project adds **`sqlglot`** for proper SQL parsing (Lesson 5.7).

> 💡 Several Phase 5 scripts need **no API key and no LLM at all**. LangGraph's ideas are about **control flow**, not AI, so they're free and instant to experiment with.

---

# Lesson 5.1: State, Nodes, Edges

> **Goal:** build and run your first graph, and understand how data flows through it.

## ⭐ Why this lesson matters

Every LangGraph app, from a 3-node toy to a production agent, is made of **just three things**:

| Piece | What it is | Story version |
|---|---|---|
| **State** | A dictionary that travels through the graph | The **logbook** in the car |
| **Nodes** | Python functions that do one job each | The **stops** on the trip |
| **Edges** | Arrows saying which node runs next | The **roads** between stops |

**You already know the State idea.** In Lesson 3.3 the `.assign()` "folder" grew page by page: `{question}` → `+query` → `+result` → `+answer`. **LangGraph's State is that folder, made official**, with a declared shape (a `TypedDict`), so every node knows what's in it.

---

## 1. Key concepts

### 1.1 State: the declared shape of the folder
```python
from typing import TypedDict

class State(TypedDict):
    question: str
    cleaned: str
    word_count: int
```

### 1.2 Nodes: functions that return only what they CHANGE
```python
def clean(state: State) -> dict:
    return {"cleaned": state["question"].strip().lower()}   # ← only the key it changed
```
> 🔑 **The #1 rule of nodes:** return a **dict of updates**, not the whole state. LangGraph **merges** the updates into the state for you.

### 1.3 Edges and START / END
```python
builder.add_edge(START, "clean")         # where the graph begins
builder.add_edge("clean", "count_words") # then this
builder.add_edge("make_report", END)     # where it finishes
```

### 1.4 The build → compile → run pattern
```python
builder = StateGraph(State)     # 1. a blueprint
builder.add_node(...)           # 2. stops
builder.add_edge(...)           # 3. roads
graph = builder.compile()       # 4. a runnable graph (a Runnable! invoke/stream/batch)
graph.invoke({...})             # 5. run it
```

### 1.5 Reducers: replace or append?

When a node returns `{"key": value}`, what happens to the old value?

| Declared as | Behaviour | Use for |
|---|---|---|
| `current_step: str` | **Replaced** by the new value | Most fields |
| `visited: Annotated[list, operator.add]` | New items **appended** | Logs, collected results |
| `messages: Annotated[list, add_messages]` | Messages **appended** (and matched by id) | Chat history |

A **reducer** is a small function that says *how to combine* the old value with the update. `add_messages` is the one you'll use most. It's why the conversation keeps growing instead of being overwritten.

> 💡 **`MessagesState`** is a ready-made State with just `messages: Annotated[list, add_messages]`, the standard choice for chatbots and agents.

---

## 2. Hands-on

### Script 1: `learning/phase5/01_first_graph.py`: build, run, watch, draw (no LLM)

A 3-node graph: clean the question → count its words → make a report.

**Results from our test run:**
```
Final state: {'question': '   Which GENRE sells the most?  ', 'cleaned': 'which genre sells the most?',
              'word_count': 5, 'report': "'which genre sells the most?' has 5 words"}

Step by step:
   {'clean': {'cleaned': 'how many customers are from brazil?'}}
   {'count_words': {'word_count': 6}}
   {'make_report': {'report': "'how many customers are from brazil?' has 6 words"}}
```
**What to notice:**
- **`invoke`** returns the **final state**, with every key filled in by a different node.
- **`stream(..., stream_mode="updates")`** shows **what each node changed**, one step at a time. This is your #1 debugging tool for graphs.
- Each node returned **only its own key**, and LangGraph merged them.
- **`draw_mermaid()`** prints the graph as text. Paste it into **mermaid.live** to see a picture.

### Script 2: `learning/phase5/02_reducers.py`: replace vs. append (no LLM)

**Results from our test run:**
```
current_step: B                        ← replaced (A, then B)
visited:      ['start', 'A', 'B']      ← appended by operator.add
messages:
   HumanMessage Hi graph
   AIMessage    Hello from A           ← appended by add_messages
   AIMessage    Hello from B
```
**What to notice:** the **same kind of update** (`{"key": [...]}`) gives **different results** depending on the reducer. Forgetting the reducer on `messages` is a classic bug: the chat history gets **replaced** by the last message, and the bot "forgets" everything.

---

# Lesson 5.2: Conditional Edges & Routing

> **Goal:** let the graph **choose** the next node based on the state, first with a plain rule and then with an LLM.

## ⭐ Why this lesson matters

**Decisions are what make an app smart.** A chain does the same steps for every input. A graph can send a greeting to a quick reply, a database question to Text2SQL and nonsense to a polite refusal. **Routing** is the first node of almost every real LLM app, including our v0.5.

## 1. Key concept: a routing function

```python
def who_approves(state) -> str:                      # reads the state…
    return "agent_approves" if state["amount"] <= 20 else "manager_approves"   # …returns a NODE NAME

builder.add_conditional_edges("receive", who_approves, ["agent_approves", "manager_approves"])
```
- The **routing function** returns the **name** of the next node.
- The list at the end declares **all possible destinations** (used for drawing, and to catch typos).
- In the diagram, conditional edges are **dotted arrows** (`-.->`).

> 🛣️ **Analogy:** a **junction with a traffic officer**. The officer (routing function) looks at your logbook (state) and points: "left to the agent" or "right to the manager."

## 2. Hands-on

### Script 3: `learning/phase5/03_conditional_edges.py`: refund routing (no LLM)

Uses the **Chinook handbook's real rule** (page 3): agents may approve refunds up to $20; above that, the Sales Manager decides.

**Results from our test run:**
```
Refund $9.99:   → approved by the Sales Support Agent
Refund $20:     → approved by the Sales Support Agent      (≤ 20, the edge case works)
Refund $49.5:   → sent to the Sales Manager
```
Mermaid shows `receive -.-> agent_approves` and `receive -.-> manager_approves`, the dotted conditional edges.

### Script 4: `learning/phase5/04_llm_router.py`: the LLM decides the route

The **Lesson 3.4 classifier** (`Literal["database_question", "greeting", "off_topic"]`) becomes the first node. Only that node uses the LLM; the three destinations are plain Python.

**Results from our test run:**
```
Hello!                                 → greeting           → Hi! Ask me anything about the Chinook music store.
Which artist has the most albums?      → database_question  → (would run Text2SQL for: ...)
What's the weather in Paris?           → off_topic          → Sorry, I only answer questions about the music store.
```
**What to notice:**
- **Structured output + conditional edge = routing.** `Literal` guarantees the category is one of 3 known strings, so the routing function can safely map it to a node name.
- **Only one LLM call per question.** The cheap paths (greeting, refusal) cost nothing extra. Good graphs use the LLM **only where judgement is needed**.

---

## 3. 🎯 Points to stress + deeper knowledge (5.1–5.2)

### Points to stress
1. **A graph = State + Nodes + Edges.** That's the whole vocabulary.
2. **State is the Lesson 3.3 `.assign()` folder, made official**, with a declared shape.
3. **Nodes return only what they change.** LangGraph merges the updates.
4. **Reducers decide replace vs. append.** Chat history needs `add_messages`.
5. **Conditional edges = decisions.** A routing function returns the next node's name.
6. **Use the LLM only where judgement is needed.** Most nodes can be plain Python.

### Deeper knowledge (interview level)
| Topic | What to know |
|---|---|
| **Why "graph"?** | Nodes and edges form a directed graph that can contain **cycles** (loops). Chains are graphs **without** cycles. That one difference unlocks retries and agents. |
| **Pregel / super-steps** | Under the hood LangGraph runs in **steps**: all nodes scheduled for a step run, their updates are merged with reducers, then the next nodes are chosen. Parallel branches merge cleanly thanks to reducers. |
| **A compiled graph is a Runnable** | It has `invoke`, `stream`, `batch` and async versions, like everything since Lesson 3.1. A graph can even be **used inside another graph's node** (Script 9). |
| **stream modes** | `"updates"` (what each node changed), `"values"` (full state after each step), `"messages"` (LLM tokens as they're generated, for chat UIs). |

---

## 4. Quiz (5.1–5.2)

1. Name the three building blocks of every LangGraph app.
2. What should a node function return?
3. `messages` is declared without `add_messages`. What goes wrong?
4. What does a routing function return?
5. In Script 4, why are the greeting and refusal nodes plain Python instead of LLM calls?

<details>
<summary>Answers</summary>

1. State, nodes, edges.
2. A dict with **only the keys it changed**. LangGraph merges them into the state.
3. Each update **replaces** the list, so the history shrinks to the latest message and the bot forgets the conversation.
4. The **name** of the next node (one of the destinations listed in `add_conditional_edges`).
5. They don't need judgement, just a fixed reply. Skipping the LLM makes them instant and free. Use the LLM only where it adds value.
</details>

**Next → Lesson 5.3: Loops.** Edges that point **backwards**: retry until success (with a limit), and the Lesson 3.5 agent loop rebuilt as a graph.
