# Lessons 5.3–5.5: Loops, Memory, Human-in-the-Loop

> These three lessons add the abilities that chains could never have: **going back**, **remembering**, and **pausing for a person**. Together with routing (5.2), they're everything our Text2SQL v0.4/v0.5 is made of.

---

# Lesson 5.3: Loops

> **Goal:** make an edge point **backwards** so the graph can retry, and always put a **limit** on it. Then rebuild the Lesson 3.5 agent loop as a graph.

## ⭐ Why this lesson matters

**Loops are the reason LangGraph exists.** In Lesson 3.3 we wrote: *"LCEL can't loop or retry based on the result. That's why LangGraph exists."* Here's the payoff:
- **Retry loops:** the SQL failed → send the error back → try again (v0.4)
- **Agent loops:** think → use a tool → look at the result → think again (Lesson 3.5, Script 19)

> 🔁 **Analogy:** a **ring road**. When a stop says "not good enough", the road leads back to an earlier stop. And every ring road needs a rule like **"maximum 3 laps"**, or a confused driver circles forever (burning tokens).

## 1. Key concepts

**A loop is just a conditional edge whose destination is an earlier node:**
```python
def retry_or_stop(state) -> str:
    if state["succeeded"]:
        return "done"
    if state["attempts"] < MAX_ATTEMPTS:
        return "call_flaky_service"          # ← BACK to the same node = a loop
    return "give_up"
```

**Two safety nets against infinite loops:**
| Safety net | Who sets it | What happens |
|---|---|---|
| **Your own limit** (`attempts < MAX_ATTEMPTS`) | You, in the routing function | The graph **gives up gracefully** (e.g. "gave up after 3 attempts") |
| **`recursion_limit`** (default 25 steps) | LangGraph, per run | Raises `GraphRecursionError`: a **hard stop** if your own logic has a bug |

**Prebuilt agent pieces:**
| Piece | What it does |
|---|---|
| `ToolNode(tools)` | A node that runs **every** tool call in the last AIMessage and returns `ToolMessage`s |
| `tools_condition` | A ready-made routing function: tool calls? → `"tools"`, none? → `END` |

## 2. Hands-on

### Script 5: `learning/phase5/05_loop_with_limit.py`: retry until success (no LLM)

A pretend service fails twice and works on the third call.

**Results from our test run:**
```
   attempt 1: failed
   attempt 2: failed
   attempt 3: OK
   finished successfully

Recursion limit hit: Recursion limit of 2 reached without hitting a stop condition. ...
```
**What to notice:**
- `call_flaky_service -.-> call_flaky_service`: a node with an edge **to itself**.
- The count lives **in the state** (`attempts`), so each lap knows how many laps came before.
- The second run uses `{"recursion_limit": 2}` to **prove** the hard safety net works.

### Script 6: `learning/phase5/06_agent_graph.py`: Script 19 from Lesson 3.5, as a graph

```python
builder = StateGraph(MessagesState)
builder.add_node("think", think)                         # LLM decides
builder.add_node("tools", ToolNode(tools))               # runs requested tools
builder.add_edge(START, "think")
builder.add_conditional_edges("think", tools_condition)  # tool calls? → tools : END
builder.add_edge("tools", "think")                       # results go back → THE LOOP
```

**Results from our test run:**
```
HumanMessage A customer buys 12 tracks at 0.99 dollars each. What's the total in rupees?
AIMessage    [{'name': 'multiply', 'args': {'b': 0.99, 'a': 12}, ...}]           ← think
ToolMessage  11.879999999999999                                                   ← tools
AIMessage    [{'name': 'get_track_price_in_inr', 'args': {'price_usd': 11.88}, ...}]  ← think
ToolMessage  ₹1045.44                                                             ← tools
AIMessage    The total for 12 tracks at $0.99 each is $11.88, which is ₹1,045.44.  ← think → END

Mermaid:  __start__ --> think;  think -.-> tools;  think -.-> __end__;  tools --> think;
```
The diagram **is** the ReAct loop from Lesson 3.5: `think ⇄ tools`, exit when there are no tool calls.

**Compare with Lesson 3.5, Script 19:**
| Script 19 (hand-written loop) | Script 6 (graph) |
|---|---|
| `for step in range(5):` | `recursion_limit` |
| `ai_msg = llm_with_tools.invoke(messages)` | the `think` node |
| `if not ai_msg.tool_calls: break` | `tools_condition` → `END` |
| run each tool, append `ToolMessage` | `ToolNode(tools)` |
| `messages.append(...)` | `MessagesState` + `add_messages` |

**Same loop, but now it's a graph**, so it gets memory, streaming, pausing and drawing for free. This is essentially what `create_agent` builds (Lesson 3.5 bonus).

---

# Lesson 5.4: Memory (Checkpointers & Threads)

> **Goal:** let a graph remember conversations, separately for each user.

## ⭐ Why this lesson matters

**LLMs remember nothing** (Lesson 3.1). Until now *we* kept the history list. A **checkpointer** makes LangGraph do it: after **every step** it saves the whole state, filed under a **thread id**. Next time you invoke with the same thread id, the state is **loaded back** and the new message is **added** to it.

> 📒 **Analogy:** a **separate notebook for every customer**. The thread id is the name on the cover. When Ravi comes back, you open *Ravi's* notebook. Priya's notes never mix with his.

## 1. Key concepts
```python
graph = builder.compile(checkpointer=InMemorySaver())           # 1. switch memory on
config = {"configurable": {"thread_id": "ravi"}}                # 2. pick the notebook
graph.invoke({"messages": [HumanMessage("Hi, I'm Ravi")]}, config)   # 3. send ONLY the new message
graph.get_state(config).values                                   # 4. look inside the notebook
```
| Checkpointer | Saves to | Use for |
|---|---|---|
| `InMemorySaver` | RAM (gone when the program stops) | Learning, tests, one CLI session |
| `SqliteSaver` / `PostgresSaver` (separate packages) | A database file / server | Real apps where memory must survive restarts |

## 2. Hands-on

### Script 7: `learning/phase5/07_memory.py`: two users, two memories

**Results from our test run:**
```
[ravi]  You: Hi, my name is Ravi and I love AC/DC.
        Bot: Nice to meet you, Ravi, rock on with AC/DC!
[priya] You: Hi, I'm Priya and I prefer Jazz.
        Bot: Nice to meet you, Priya, I'll be sure to keep the jazz vibes playing for you!
[ravi]  You: What's my name and favourite music?
        Bot: Your name is Ravi and your favorite music is AC/DC!
[priya] You: What's my name and favourite music?
        Bot: Your name is Priya and your favorite music is jazz!

Ravi's thread holds 4 messages:
   HumanMessage Hi, my name is Ravi and I love AC/DC.
   AIMessage    Nice to meet you, Ravi, rock on with AC/DC!
   HumanMessage What's my name and favourite music?
   AIMessage    Your name is Ravi and your favorite music is AC/DC!
```

**What to notice:**
- We sent **only the new message** each time. The checkpointer added it to the saved history.
- **Two thread ids = two separate memories.** Ravi's AC/DC never leaked into Priya's conversation.
- `get_state(config)` lets you **inspect** what's remembered, which is great for debugging.

> ⚠️ **Memory has a cost.** Every turn sends the **whole** history to the LLM, so long conversations get slower and more expensive. Real apps **trim** or **summarize** old messages.

---

# Lesson 5.5: Human-in-the-Loop

> **Goal:** pause a graph mid-run, ask a person, and continue with their answer.

## ⭐ Why this lesson matters

Some actions are too important to let an AI do alone: refunds, sending emails, running database queries, deleting anything. **Human-in-the-loop** = the graph **stops** at a checkpoint, shows a person what it's about to do, and **continues only with their decision**.

> 🛂 **Analogy:** a **border checkpoint**. The car (graph) stops, the officer (human) looks at the papers (state), and only after they say "go" does the trip continue. The car **waits** at the checkpoint, because the checkpointer saved exactly where it stopped.

## 1. Key concepts
```python
from langgraph.types import interrupt, Command

def ask_manager(state):
    answer = interrupt("Approve a $49.50 refund? (yes/no)")   # ← the graph STOPS here
    return {"status": "approved" if answer == "yes" else "rejected"}   # ← continues HERE later

out = graph.invoke(inputs, config)            # returns early, with out["__interrupt__"]
graph.invoke(Command(resume="yes"), config)   # continue from the same spot with the answer
```
| Piece | Meaning |
|---|---|
| `interrupt(value)` | Pause and hand `value` (a question, the SQL...) to the outside world |
| `out["__interrupt__"]` | How you know the graph paused, and what it's asking |
| `graph.get_state(config).next` | Which node it's waiting at |
| `Command(resume=answer)` | Continue. `answer` becomes the **return value of `interrupt()`** |

> 🔑 **Pausing requires a checkpointer.** The graph must save where it stopped so it can resume later, even minutes later, or from a different web request.

## 2. Hands-on

### Script 8: `learning/phase5/08_human_approval.py`: refund approval (no LLM)

Handbook rule again: refunds **≤ $20** are approved automatically; bigger ones need the **manager**.

**Results from our test run** (typing `no`):
```
approved automatically (≤ $20)

PAUSED. Question for the human: Approve a $49.5 refund for Priya? (yes/no)
Waiting at node: ('ask_manager',)
Your decision (yes/no): no
Final status: rejected by manager
```
**What to notice:**
- The $9.99 refund **never paused**. Conditional edges decide **when** a human is needed.
- The $49.50 refund **stopped** at `ask_manager`, and `get_state(...).next` shows exactly where.
- Resuming **didn't re-run** `check_amount`. It continued **inside** `ask_manager`, right after `interrupt()`.

---

## 3. 🎯 Points to stress + deeper knowledge (5.3–5.5)

### Points to stress
1. **A loop = a conditional edge pointing backwards.** The count of attempts lives in the state.
2. **Always two safety nets:** your own limit (graceful) + `recursion_limit` (hard stop).
3. **The agent loop is a graph:** think ⇄ tools, with `tools_condition` deciding when to stop.
4. **Memory = checkpointer + thread id.** Send only the new message; LangGraph loads the rest.
5. **Threads keep users separate.** One notebook per conversation.
6. **`interrupt()` pauses, `Command(resume=...)` continues**, and it needs a checkpointer.

### Deeper knowledge (interview level)
| Topic | What to know |
|---|---|
| **Persistence = superpowers** | Because every step is saved, LangGraph can also **resume after a crash**, **time-travel** (re-run from an earlier checkpoint) and support pausing for humans. |
| **Short-term vs. long-term memory** | A thread's messages = **short-term** memory (one conversation). Facts that should survive across conversations ("Ravi likes AC/DC") go in a **store** = **long-term** memory. |
| **Trimming history** | Production chat apps keep the last N messages or **summarize** older ones to control cost and the context window. |
| **When to ask a human** | Before **irreversible** or **expensive** actions, or when the model is **unsure**. Asking for everything makes people click "yes" without reading ("approval fatigue"). |
| **Checkpointer security** | Saved state is loaded back into Python objects. LangGraph only rebuilds **allow-listed** custom types (see `make_checkpointer()` in v0.5). |

## 4. Quiz (5.3–5.5)

1. What makes an edge a "loop"? Where does the attempt count live?
2. What's the difference between your own retry limit and `recursion_limit`?
3. In Script 6, what does `tools_condition` decide?
4. Why do we send only the **new** message when memory is on?
5. Two users chat at the same time. How does LangGraph keep their memories apart?
6. Why does `interrupt()` need a checkpointer?
7. After `Command(resume="no")`, does the graph start again from the beginning?

<details>
<summary>Answers</summary>

1. A conditional edge whose destination is an earlier (or the same) node. The count lives in the **state** (e.g. `attempts`).
2. Your limit ends the loop **gracefully** (e.g. "give up" node). `recursion_limit` is LangGraph's **hard stop** (raises `GraphRecursionError`) in case your logic is buggy.
3. Whether the LLM's last message requested tools (→ `"tools"` node) or not (→ `END`, the answer is final).
4. The checkpointer already saved the history for that thread and **appends** the new message (via `add_messages`).
5. Different **thread ids**. Each thread has its own saved state.
6. The graph must **save exactly where it stopped** so it can continue later. Without saved state there's nothing to resume.
7. No. It continues **inside the paused node**, right after `interrupt()`, which returns `"no"`.
</details>

**Next → Lesson 5.6: Multi-agent.** A supervisor that sends each question to the right specialist: the handbook RAG assistant (Phase 4) or Text2SQL.
