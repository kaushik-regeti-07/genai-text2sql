# Lesson 3.5: Tools / Function Calling

> **Goal of this lesson:** let the LLM **decide for itself** which of *your* Python functions to call, with what arguments, and in what order.
> We finish with a **Text2SQL agent** that explores the database on its own: list tables → read schema → write and run SQL → answer. 🎯

**You will create:** 4 scripts in `learning/phase3/` (17 to 20).
**New packages:** none. `@tool` is in `langchain-core`, and `create_agent` is in `langchain`.
**Time:** about 2–2.5 hours.

---

## ⭐ Why this lesson matters

**This is the moment an LLM becomes an *agent*.** Until now, **we** decided every step:
```
Lessons 3.3–3.4 (a chain):  we hard-wired  →  write SQL → run it → answer
Lesson 3.5 (an agent):      the LLM decides →  "First I'll list the tables... now I need the
                                               schema of Genre and Track... now I'll run this query."
```

**An LLM on its own can only produce text.** It can't read your database, check today's date, call an API or do exact maths. **Tools give it hands.** It asks for an action, *your* code performs it, and it reads the result.

**In our Text2SQL project, this changes the design:**
- Before: we stuffed the **whole schema** into every prompt (about 2,500 tokens).
- Now: the agent **asks** for only the tables it needs (`get_schema(['Genre', 'Track'])`). It's the same idea as RAG (Phase 4), only fetch what's relevant, but driven by the LLM itself.
- Errors become **information**: if a query fails, the error goes back to the LLM so it can fix the query.

**Three reasons it's foundational:**
1. **It's the core of every AI agent:** Claude Code, ChatGPT's browsing, Copilot. All of them are "LLM + tools + a loop."
2. **It's exactly what LangGraph formalizes.** The loop you write by hand in Script 19 **is** the graph you'll build in Phase 5. We'll even see that LangChain's built-in agent **is a LangGraph graph**.
3. **It's the same mechanism as structured output (3.4).** A tool call is just structured output: `{name, args}`. You already understand the hard part.

---

## 1. The key idea: the LLM never runs anything

This is the **#1 misunderstanding** about tools, so stress it:

> 🚨 **The LLM does NOT execute your function.** It only **asks** you to run it, by returning a structured request. **Your code** runs it and sends the result back.

```
 You: "What's 12 × 0.99 in rupees?"  + a list of available tools
        │
        ▼
 LLM:  AIMessage(tool_calls=[{name: "multiply", args: {a: 12, b: 0.99}, id: "call_1"}])
        │      ↑ just a REQUEST, nothing has run yet
        ▼
 YOUR CODE runs multiply(12, 0.99) → 11.88
        │
        ▼
 You send back: ToolMessage(content="11.88", tool_call_id="call_1")
        │
        ▼
 LLM:  another tool call (convert to rupees)... or a final text answer
```

> 🍽️ **Analogy:** the LLM is a **head chef who can't leave the office**. You're the kitchen staff. The chef reads the menu of what the kitchen can do (the **tool list**), then shouts orders: *"Chop 2 onions!"* (a tool call). **You** chop the onions and report back: *"Done, 2 onions chopped"* (a ToolMessage). The chef decides the next order, until the dish is ready (the final answer).

### The new message type

| Message | Who creates it | Contains |
|---|---|---|
| `AIMessage` with `.tool_calls` | The LLM | A list of `{name, args, id}` requests. `.text` is usually empty. |
| `ToolMessage` | **Your code** | The tool's result + `tool_call_id` (which request it answers) |

> 🔑 The `id` links each result to its request. When the LLM asks for **two tools at once**, the ids tell it which result belongs to which call.

---

## 2. The building blocks

| Piece | What it does |
|---|---|
| `@tool` | Decorator that turns a Python function into a LangChain tool |
| Function **name** | Becomes the tool's name that the LLM sees |
| **Docstring** | Becomes the tool's **description**. This is how the LLM decides *when* to use it! |
| **Type hints** (`a: float`) | Become the **argument schema** (like Pydantic in 3.4) |
| `tool.invoke({...})` | Run a tool yourself (tools are Runnables too!) |
| `llm.bind_tools([...])` | Gives the LLM the "menu" of tools. It returns a new Runnable. |
| `ai_msg.tool_calls` | The LLM's requests: a list of `{name, args, id}` |
| `ToolMessage(content, tool_call_id)` | How you send a result back |
| `create_agent(...)` | LangChain's ready-made agent that runs the loop for you |

---

## 3. Hands-on

> Run each script from `F:\Project\Gen_AI` with `(.venv)` active.

### Script 17: `learning/phase3/17_first_tool.py`: what a tool looks like

```python
from langchain_core.tools import tool

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

@tool
def get_track_price_in_inr(price_usd: float) -> str:
    """Convert a track price from US dollars to Indian rupees."""
    return f"₹{price_usd * 88:.2f}"      # rough exchange rate, fine for learning

# 1. The decorator turned the function into a tool object.
print("Type:       ", type(multiply).__name__)      # StructuredTool
print("Name:       ", multiply.name)                # multiply
print("Description:", multiply.description)         # from the docstring
print("Args:       ", multiply.args)                # from the type hints

# 2. You can run a tool yourself. Tools are Runnables, so they have .invoke().
print("Result:     ", multiply.invoke({"a": 6, "b": 7}))                  # 42.0
print("In rupees:  ", get_track_price_in_inr.invoke({"price_usd": 0.99}))
```

**What to notice:**
- **Name + docstring + type hints = everything the LLM knows about your tool.** Nothing else is sent.
- A tool without a good docstring is like a button with no label. The LLM won't know when to press it.

---

### Script 18: `learning/phase3/18_bind_tools.py`: the LLM chooses

```python
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

@tool
def get_track_price_in_inr(price_usd: float) -> str:
    """Convert a track price from US dollars to Indian rupees."""
    return f"₹{price_usd * 88:.2f}"

# Give the LLM the "menu" of tools.
llm_with_tools = llm.bind_tools([multiply, get_track_price_in_inr])

questions = [
    "What is 23.5 times 4?",                                   # → multiply
    "How much is a 0.99 dollar track in rupees?",              # → currency tool
    "What is SQL? One sentence.",                              # → no tool needed
    "What is 3 times 4, and what is 1.99 dollars in rupees?",  # → BOTH tools at once
]
for q in questions:
    msg = llm_with_tools.invoke(q)
    print(f"\nQ: {q}")
    print("   tool_calls:", msg.tool_calls)
    print("   text:      ", repr(msg.text[:80]))
```

**Results from our test run:**
```
Q: What is 23.5 times 4?
   tool_calls: [{'name': 'multiply', 'args': {'a': 23.5, 'b': 4}, 'id': 'call_629485', ...}]
   text:       ''                                      ← no answer yet, just a request

Q: What is SQL? One sentence.
   tool_calls: []                                      ← decided no tool was needed
   text:       'SQL stands for Structured Query Language...'

Q: What is 3 times 4, and what is 1.99 dollars in rupees?
   tool_calls: [{'name': 'multiply', ...}, {'name': 'get_track_price_in_inr', ...}]   ← 2 at once!
```

**What to notice:**
- The LLM **picked the right tool** and **filled in the arguments** (`23.5`, `4`) from plain English.
- For "What is SQL?", **no tool call**. Tools are *offered*, not forced.
- **Parallel tool calls:** it asked for two tools in one reply.
- **Nothing ran yet!** `text` is empty. The LLM is waiting for us to run the tools (next script).

---

### Script 19: `learning/phase3/19_tool_loop.py`: the agent loop, by hand

This is **the most important script in Phase 3**. It's the heart of every AI agent.

```python
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

@tool
def get_track_price_in_inr(price_usd: float) -> str:
    """Convert a track price from US dollars to Indian rupees."""
    return f"₹{price_usd * 88:.2f}"

tools = [multiply, get_track_price_in_inr]
tools_by_name = {t.name: t for t in tools}     # lookup: "multiply" → the tool
llm_with_tools = llm.bind_tools(tools)

# The conversation. It grows as we go.
messages = [HumanMessage("A customer buys 12 tracks at 0.99 dollars each. "
                         "What's the total in rupees?")]

# ---------------- THE AGENT LOOP ----------------
for step in range(5):                              # safety limit: max 5 rounds
    ai_msg = llm_with_tools.invoke(messages)       # 1. THINK: LLM decides
    messages.append(ai_msg)

    if not ai_msg.tool_calls:                      # 2. DONE? No tool calls = final answer
        print("\nFINAL ANSWER:", ai_msg.text)
        break

    for call in ai_msg.tool_calls:                 # 3. ACT: run each requested tool
        tool_fn = tools_by_name[call["name"]]
        result = tool_fn.invoke(call["args"])
        print(f"Step {step}: {call['name']}({call['args']}) → {result}")

        messages.append(ToolMessage(               # 4. OBSERVE: send the result back
            content=str(result),
            tool_call_id=call["id"],
        ))
# ------------------------------------------------

print("\nFull conversation:", [type(m).__name__ for m in messages])
```

**Results from our test run:**
```
Step 0: multiply({'a': 12, 'b': 0.99}) → 11.879999999999999
Step 1: get_track_price_in_inr({'price_usd': 11.88}) → ₹1045.44

FINAL ANSWER: The total cost for 12 tracks at $0.99 each is ₹1,045.44.

Full conversation: ['HumanMessage', 'AIMessage', 'ToolMessage', 'AIMessage', 'ToolMessage', 'AIMessage']
```

**What to notice:**
- The LLM **planned two steps on its own**: multiply first, then convert the *result*. We never told it the order.
- **The loop = Think → Act → Observe → repeat.** This pattern is called **ReAct** (Reasoning + Acting).
- The conversation grows: `Human → AI(tool call) → Tool(result) → AI(tool call) → Tool(result) → AI(answer)`. The LLM sees the whole history every time. (Remember Lesson 3.1: LLMs have no memory!)
- `range(5)` is a **safety limit**. Without it, a confused agent could loop forever and burn tokens.
- `11.879999999999999`: that's normal computer floating-point maths, not an AI mistake. For money, real apps use `Decimal`.

> 💡 **Shortcut:** `tool_fn.invoke(call)` (passing the whole `call`, not `call["args"]`) returns a ready-made `ToolMessage` with the right `tool_call_id`.

---

### Script 20: `learning/phase3/20_text2sql_agent.py`: 🎯 a Text2SQL agent

Now we give the LLM **database tools** and let it explore Chinook by itself.

```python
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_community.utilities import SQLDatabase

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
db = SQLDatabase.from_uri("sqlite:///text2sql/data/chinook.db")

FORBIDDEN = ("insert", "update", "delete", "drop", "alter", "create",
             "replace", "truncate", "attach", "pragma")

# ---------- 1. The tools ----------
@tool
def list_tables() -> str:
    """List all tables in the music store database."""
    return ", ".join(db.get_usable_table_names())

@tool
def get_schema(table_names: list[str]) -> str:
    """Get the columns and 3 sample rows for the given tables.
    Always call this before writing a query."""
    try:
        return db.get_table_info(table_names)
    except Exception as e:
        return f"Error: {e}"

@tool
def run_query(sql: str) -> str:
    """Run a read-only SQLite SELECT query and return the rows.
    If you get an error, fix the query and try again."""
    low = sql.strip().lower()
    if not (low.startswith("select") or low.startswith("with")) \
            or any(f" {w} " in f" {low} " for w in FORBIDDEN):
        return "Error: only SELECT queries are allowed."
    try:
        return db.run(sql) or "(no rows)"
    except Exception as e:
        return f"Error: {e}"          # ← return errors as TEXT so the LLM can fix them

tools = [list_tables, get_schema, run_query]
tools_by_name = {t.name: t for t in tools}
agent_llm = llm.bind_tools(tools)

SYSTEM = ("You are a data analyst for a music store with a SQLite database. "
          "To answer questions: 1) list the tables, 2) get the schema of the relevant ones, "
          "3) write and run a SELECT query, 4) answer in plain English. "
          "Never guess column names. If a query fails, read the error and fix it.")

# ---------- 2. The agent loop (same as Script 19) ----------
def ask(question: str, max_steps: int = 8) -> str:
    messages = [SystemMessage(SYSTEM), HumanMessage(question)]
    for step in range(max_steps):
        ai_msg = agent_llm.invoke(messages)
        messages.append(ai_msg)
        if not ai_msg.tool_calls:
            return ai_msg.text
        for call in ai_msg.tool_calls:
            result = tools_by_name[call["name"]].invoke(call["args"])
            preview = str(result).replace("\n", " ")[:100]
            print(f"  [{step}] {call['name']}({call['args']}) → {preview}")
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
    return "Stopped: too many steps."

# ---------- 3. Try it ----------
for q in ["Which genre earned the most money in total?",
          "Which sales agent sold the most in 2023?",
          "Please delete the Customer table"]:
    print(f"\nQ: {q}")
    print("A:", ask(q))
```

**Results from our test run:**
```
Q: Which genre earned the most money in total?
  [0] list_tables({}) → Album, Artist, Customer, Employee, Genre, Invoice, InvoiceLine, ...
  [1] get_schema({'table_names': ['Genre', 'Track', 'InvoiceLine']}) → CREATE TABLE "Genre" ...
  [2] run_query({'sql': 'SELECT g.Name, SUM(il.UnitPrice * il.Quantity) AS TotalEarnings
                         FROM Genre g JOIN Track t ... JOIN InvoiceLine il ... LIMIT 1;'}) → [('Rock', 826.65)]
A: The genre that earned the most money in total is Rock, with total earnings of $826.65.

Q: Which sales agent sold the most in 2023?
  [0] list_tables({}) → ...
  [1] get_schema({'table_names': ['Employee', 'Customer', 'Invoice']}) → ...
  [2] run_query({'sql': "SELECT e.FirstName, e.LastName, SUM(i.Total) ...
                         WHERE STRFTIME('%Y', i.InvoiceDate) = '2023' ..."}) → [('Jane', 'Peacock', 184.34)]
A: The sales agent who sold the most in 2023 was Jane Peacock, with total sales of $184.34.

Q: Please delete the Customer table
A: I cannot delete the Customer table, as I only have permission to run read-only SELECT queries.
```

**What to notice:**
- **The agent explored on its own:** it listed the tables, then asked for the schema of **only the 3 relevant tables**, then wrote the query. We didn't hard-wire any of it.
- It fetched **3 tables' schema instead of all 11**, so fewer tokens and a more focused prompt.
- It worked out the `Employee → Customer (SupportRepId) → Invoice` chain for "sales agent" **by reading the schema**.
- **Tools return errors as text, never crash.** If a query fails, the LLM reads `Error: no such column...` and tries again. That's **self-correction**, which LangGraph makes more robust in v0.4.
- The **safety check lives inside the tool**, in code. Even if the LLM is tricked, `run_query` refuses non-SELECT queries.

> 🧪 **Experiments:**
> 1. Remove the docstring from `get_schema`. Does the agent still use it? (Probably not properly.)
> 2. Ask a question that needs a correction, e.g. *"What's the average track length in minutes per album for AC/DC?"* Watch for an `Error:` followed by a fixed query.
> 3. Set `max_steps=2` and ask a hard question. What happens?
> 4. Ask *"Show 3 customers; DROP TABLE Customer"*. Which layer stops it: the LLM or `run_query`?

---

### Bonus: LangChain's ready-made agent (add at the bottom of Script 20)

Writing the loop by hand taught you how agents work. In practice, LangChain provides it:

```python
from langchain.agents import create_agent

agent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM)
print("Agent type:", type(agent).__name__)       # CompiledStateGraph ← a LangGraph graph!

result = agent.invoke({"messages": [{"role": "user", "content": "How many customers are from Canada?"}]})
print([type(m).__name__ for m in result["messages"]])
print(result["messages"][-1].text)
```

**Results from our test run:**
```
Agent type: CompiledStateGraph
['HumanMessage', 'AIMessage', 'ToolMessage', 'AIMessage', 'ToolMessage', 'AIMessage', 'ToolMessage', 'AIMessage']
There are 8 customers from Canada.
```

**What to notice:**
- The type is **`CompiledStateGraph`**. **LangChain's agent IS a LangGraph graph.** Under the hood it's the exact loop you wrote in Script 19: a "model" node and a "tools" node, joined by a loop.
- The message list has the same pattern you built by hand: `AI → Tool → AI → Tool → ... → AI`.
- **Phase 5 = opening this box** and building our own graph, with custom steps (validate SQL, retry with a limit, ask a human before running) that a ready-made agent doesn't give you.

---

## 4. 🎯 Points to stress + deeper knowledge

### Points to stress (the "aha" moments)

**1. The LLM never executes anything. It only asks.**
It returns `{name, args, id}`. Your code decides whether and how to run it. That's why **safety checks belong inside tools**: that's the one place the LLM can't bypass.

**2. The docstring is the tool's user manual for the LLM.**
Name + docstring + type hints are all it sees. Vague docstring = wrong tool choices. `"Always call this before writing a query"` in `get_schema` is **why** the agent checks the schema before guessing column names.

**3. Agent = LLM + tools + a loop.**
Think → Act → Observe → repeat, until there are no tool calls left. That's the **ReAct** pattern, and it's in every AI agent you've heard of.

**4. Return errors as text; don't crash.**
An error message *is* information. `"Error: no such column CustomerAge"` lets the LLM fix itself. An exception just kills the program.

**5. Always put a step limit on the loop.**
Agents can get stuck repeating the same failing call. `max_steps` (or LangGraph's recursion limit) protects your time and your token bill.

**6. Chain vs. agent is a real design choice.**
| | Chain (3.3–3.4) | Agent (3.5) |
|---|---|---|
| Who decides the steps? | You (fixed) | The LLM (dynamic) |
| Predictable? | ✅ Very | ⚠️ Less |
| Cost / speed | 2 LLM calls | 3–8 LLM calls |
| Handles surprises? | ❌ No | ✅ Yes (explores, retries) |

Use a **chain** when the steps are always the same. Use an **agent** when the path depends on the question. **LangGraph (Phase 5)** lets you mix both: fixed steps where you want control, and LLM decisions where you need flexibility.

### Deeper knowledge (sounds great in interviews)

| Topic | What to know |
|---|---|
| **ReAct** | "Reasoning + Acting", a 2022 research paper. The model alternates between thinking and calling tools. Nearly every modern agent is built on this idea. |
| **Parallel tool calls** | Models can request several independent tools in one turn (Script 18). Your loop should run **all** of them before calling the LLM again. |
| **`tool_choice`** | `llm.bind_tools(tools, tool_choice="any")` **forces** a tool call. Use a tool name to force a specific one. Useful when you *always* want structured action. |
| **Tool count matters** | Every tool's description is sent on **every** call. 50 tools = a huge prompt and more confusion. Keep tools few, focused and clearly named. |
| **Least privilege** | Give agents the **smallest power** they need: read-only DB users, allow-listed tables, no shell access. Assume the LLM *will* eventually be tricked (prompt injection) and limit the damage it could do. |
| **Ready-made SQL tools** | `langchain-community` has a `SQLDatabaseToolkit` with similar tools. Writing your own (as we did) gives you full control over safety and error messages. |
| **MCP (Model Context Protocol)** | An open standard for sharing tools between AI apps. Instead of writing `@tool` for each app, a tool server can plug into any MCP-compatible agent. It's the "USB-C for AI tools," and a good topic after this course. |

---

## 5. Common errors

| Error | Cause | Fix |
|---|---|---|
| `ValueError: Function must have a docstring` | `@tool` needs a docstring | Add `"""What this tool does."""` |
| LLM never calls your tool | Unclear name or docstring | Describe **when** to use it, not just what it does |
| `KeyError` in `tools_by_name[...]` | The LLM called a tool name that doesn't exist | Rare. Check that the names match. |
| Error about missing tool responses | You didn't send a `ToolMessage` for **every** tool call | Loop over **all** `ai_msg.tool_calls` |
| Loop never ends | The agent keeps retrying | Keep `max_steps`, and improve the error messages from tools |
| `429 RESOURCE_EXHAUSTED` | Agents make many calls quickly | Wait a minute. Agents use more quota than chains. |

---

## 6. Summary: what to tell your friend

1. A **tool** = a Python function with `@tool`. Its **name, docstring and type hints** are what the LLM sees.
2. `llm.bind_tools([...])` gives the LLM a "menu." It replies with **`tool_calls`** (requests), not results.
3. **The LLM never runs anything.** Your code runs the tool and sends back a **`ToolMessage`**.
4. **Agent = LLM + tools + loop** (Think → Act → Observe), the **ReAct** pattern.
5. Tools should **return errors as text** (for self-correction) and **enforce safety in code**.
6. Always **limit the steps**.
7. `create_agent` is LangChain's ready-made agent, and **it's a LangGraph graph**, which is exactly what we open up in Phase 5.

---

## 7. Quiz

1. When the LLM returns a tool call, has the function already run? Who runs it?
2. What three things about a tool does the LLM see?
3. What is the `tool_call_id` in a `ToolMessage` for?
4. Describe the agent loop in 4 words. When does it stop?
5. Why does `run_query` return `"Error: ..."` as text instead of raising an exception?
6. Why is the SELECT-only check **inside the tool** and not just in the system prompt?
7. When would you choose a chain over an agent?
8. What type of object does `create_agent` return, and why does that matter for Phase 5?

<details>
<summary>Answers</summary>

1. No. The LLM only **requests** it. Your code runs it.
2. Its name, its docstring (description) and its argument types (from the type hints).
3. It links the result to the request it answers, which matters when there are several parallel tool calls.
4. Think → Act → Observe → Repeat. It stops when the LLM replies with no tool calls (the final answer), or when the step limit is hit.
5. So the LLM can read the error and fix its query (self-correction). An exception would crash the loop.
6. Prompts are advice and can be ignored or bypassed through prompt injection. Code inside the tool always runs; the LLM can't skip it.
7. When the steps are always the same: it's more predictable, cheaper and faster. Use an agent when the path depends on the question.
8. A `CompiledStateGraph`, which is a LangGraph graph. LangChain's agent is built on LangGraph, and in Phase 5 we build our own graph with custom steps.
</details>

---

**Next → Lesson 3.6: Putting it together.** We move from `learning/` into the real project folder `text2sql/app/` and build **Text2SQL v0.1 + v0.2** as clean, reusable modules: `config.py`, `db.py`, `prompts.py`, `chains.py` and a small command-line app.
