# Lesson 3.1: What is LangChain? + Chat Models

> **Goal of this lesson:** understand **why** LangChain exists, how its packages fit together, and how to talk to an LLM through it.
> We finish by giving Gemini our Chinook database schema and asking it to explain the database back to us. That's the first tiny step of Text2SQL.

**You will create:** 4 small scripts in `learning/phase3/`.
**Time:** about 1.5–2 hours.

---

## ⭐ Why this lesson matters

**The chat model is the atom of every GenAI app.** Chains, RAG, agents and LangGraph are all just organized ways of calling a chat model. If you understand this one piece deeply, everything else is "the same thing, arranged differently."

**In our Text2SQL project, the same `llm` object is used in every step:**
```
write the SQL  →  llm
fix broken SQL →  llm
explain result →  llm
route question →  llm   (Phase 5)
```

**Three reasons it's foundational:**
1. **Provider independence:** your app isn't locked to Google. If Gemini gets expensive, slow or goes down, you switch in one line. Companies care about this a lot.
2. **One standard message format:** System, Human and AI messages work the same for every provider, so history, memory and agents all build on it.
3. **The Runnable interface:** `invoke` / `stream` / `batch` works the same on models, prompts, chains **and** LangGraph graphs. Learn it once, use it everywhere.

---

## 1. The problem LangChain solves

You already know how to call an LLM directly with the provider's own SDK. So why add a framework?

Imagine building Text2SQL **without** LangChain:

| Problem | What it looks like without LangChain |
|---|---|
| **Every provider is different** | Gemini, OpenAI and Claude each have their own SDK, message format and response shape. Switching means rewriting code. |
| **Prompt spaghetti** | Prompts built with f-strings scattered everywhere, hard to reuse or test |
| **Parsing output** | The LLM returns text. You write your own code to pull SQL or JSON out of it, and it breaks often. |
| **Glue code** | Connect to the DB → read the schema → build the prompt → call the LLM → extract SQL → run it → explain it. That's a lot of hand-written plumbing. |
| **Tools, memory, retries** | All of it built from scratch |

**LangChain = a standard set of building blocks for LLM apps.**

> 🧱 **Analogy:** LangChain is like **LEGO** for AI apps. Every brick (model, prompt, parser, retriever, tool) has the **same connector shape**, so any brick snaps onto any other. You can swap a Gemini brick for an OpenAI brick and nothing else changes.

---

## 2. The LangChain family: who does what

```
┌──────────────────────────────────────────────────────────────┐
│  langchain-core      ← the foundation: base classes, messages,│
│                        prompts, the "Runnable" interface       │
├──────────────────────────────────────────────────────────────┤
│  langchain           ← ready-made pieces built on core         │
│                        (init_chat_model, agents...)            │
├──────────────────────────────────────────────────────────────┤
│  langchain-google-genai, langchain-openai, ...                 │
│                      ← "adapters": one per AI provider         │
├──────────────────────────────────────────────────────────────┤
│  langchain-community ← community integrations (SQLDatabase,    │
│                        loaders, vector stores...)              │
└──────────────────────────────────────────────────────────────┘
        langgraph   ← separate framework for agents/graphs (Phase 5)
        langsmith   ← separate tool for tracing/debugging (Phase 6)
```

> 💡 We never installed `langchain-core` ourselves. It came **automatically** because `langchain` depends on it. Check it with:
> ```powershell
> uv pip list
> ```

---

## 3. Key concept: Chat Models

A **chat model** in LangChain is a wrapper around an LLM that:
- takes **messages** in
- gives an **AIMessage** back

```
[SystemMessage, HumanMessage, ...]  ──►  ChatGoogleGenerativeAI  ──►  AIMessage
```

### 3.1 The three message types

| Message | Who is "speaking" | Used for | Raw API equivalent |
|---|---|---|---|
| `SystemMessage` | You, the developer | Instructions and personality: "You are a SQL expert" | `role: "system"` |
| `HumanMessage` | The user | The question | `role: "user"` |
| `AIMessage` | The model | The model's reply (also used to replay chat history) | `role: "assistant"` |

### 3.2 Important settings

| Setting | What it does | For Text2SQL |
|---|---|---|
| `model` | Which LLM to use | `"gemini-3.5-flash-lite"` (fast, free tier) |
| `temperature` | Randomness. 0 = same answer every time, 1+ = creative | Low, because SQL should be precise, not creative |
| `max_tokens` | Maximum length of the reply | Optional |

> ⚠️ **Temperature on newer Gemini models:** Gemini 3.x models use **fixed sampling settings**. If you pass `temperature=0`, LangChain prints a warning saying *"temperature will be ignored"*. That's why our scripts leave it out. The concept still matters: most other models (OpenAI, Claude, older Gemini) do use it.

### 3.3 Choosing the model: thinking vs. fast

We tested two Gemini models on the same question, *"What is SQL in one sentence?"*:

| Model | Time | Hidden "thinking" tokens | Answer quality |
|---|---|---|---|
| `gemini-3.5-flash` | **~53 seconds** | 338 | Same |
| `gemini-3.5-flash-lite` | **~2 seconds** | 0 | Same |

**Thinking (reasoning) models** reason privately before answering. That's great for hard problems, but slow and wasteful for simple ones. You can see it in `usage_metadata` under `output_token_details: {'reasoning': 338}`.

**Our choice:** `flash-lite` for learning (fast feedback). Later, for **hard** SQL questions, we can compare it with the thinking model and see if accuracy improves. That's a real engineering trade-off: **speed and cost vs. quality**.

### 3.4 The three ways to call a model

Every LangChain building block (models, prompts, chains, even LangGraph graphs) supports the **same three methods**. This shared interface is called a **Runnable**.

| Method | What it does | When to use |
|---|---|---|
| `.invoke(input)` | One input → one complete output | Most of the time |
| `.stream(input)` | One input → output arrives piece by piece | Chat UIs (text appears as it's typed) |
| `.batch([in1, in2])` | Many inputs → many outputs, run in parallel | Evaluating many test questions at once |

> 🔑 **Remember "Runnable".** It's the reason LangChain bricks snap together. In Lesson 3.3 you'll see `prompt | model | parser`, which works *because* all three are Runnables.

---

## 4. Hands-on

> ⚠️ **Always run scripts from the project root** (`F:\Project\Gen_AI`) with `(.venv)` active:
> ```powershell
> python learning\phase3\01_first_call.py
> ```

### Script 1: `learning/phase3/01_first_call.py`: your first call

```python
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Load GOOGLE_API_KEY from the .env file into environment variables.
#    load_dotenv() searches for .env starting from this script's folder and going up,
#    so it finds F:\Project\Gen_AI\.env
load_dotenv()

# 2. Create the chat model. It reads GOOGLE_API_KEY from the environment automatically.
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

# 3. Send a plain string (LangChain turns it into a HumanMessage for us).
response = llm.invoke("What is SQL in one sentence?")

# 4. Look at what came back.
print("Type:   ", type(response).__name__)   # AIMessage
print("Answer: ", response.text)             # just the text
print("Tokens: ", response.usage_metadata)   # input / output / total tokens
```

**What to notice:**
- The reply is **not a string**. It's an `AIMessage` object, and `.text` gives you the plain text.
- Why `.text` and not `.content`? With Gemini, `.content` is a **list of blocks** (text, plus possibly other types like images or reasoning). `.text` joins the text parts into one clean string. Try `print(response.content)` to see the difference.
- `usage_metadata` shows the tokens used. That's what you'd pay for on a paid plan.

> 🧪 **Experiment:** change the model to `"gemini-3.5-flash"` (the thinking model) and run again. Compare the **time taken** and the `reasoning` count inside `usage_metadata`. Then switch back to `flash-lite`.

---

### Script 2: `learning/phase3/02_messages.py`: roles and conversation history

```python
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

messages = [
    SystemMessage("You are a SQL tutor. Answer in one short sentence."),
    HumanMessage("What does JOIN do?"),
    AIMessage("JOIN combines rows from two tables using a related column."),
    HumanMessage("Give me a one-line example."),   # "example" of what? The history tells it.
]

response = llm.invoke(messages)
print(response.text)
```

**What to notice:**
- The last question, *"Give me a one-line example"*, only makes sense because of the **history**.
- **LLMs have no memory.** "Memory" just means **sending the whole conversation every time**. Later, LangGraph will manage this list for us (Phase 5).

> 🧪 **Experiment:** delete the middle two messages and run again. Does the answer still make sense?

---

### Script 3: `learning/phase3/03_stream_batch.py`: the Runnable methods

```python
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

# --- stream: pieces arrive one by one ---
print("STREAM:")
for chunk in llm.stream("Explain what a database index is in 3 sentences."):
    print(chunk.text, end="", flush=True)   # end="" keeps it on one line
print("\n")

# --- batch: several questions in parallel ---
questions = [
    "What does SELECT do in SQL? One sentence.",
    "What does GROUP BY do in SQL? One sentence.",
    "What does ORDER BY do in SQL? One sentence.",
]
answers = llm.batch(questions)
for q, a in zip(questions, answers):
    print(f"Q: {q}\nA: {a.text}\n")
```

**What to notice:**
- `stream` → text appears gradually, the way ChatGPT's UI does.
- `batch` → 3 answers come back in roughly the time of one, because they run in parallel.

---

### Script 4: `learning/phase3/04_describe_chinook.py`: the first Text2SQL step 🎯

This is where our project begins. We read the **schema** (the structure of the tables) and give it to the LLM.

```python
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.utilities import SQLDatabase

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

# 1. Connect to the database.
#    "sqlite:///" + path. The path is relative to where you RUN the script (the project root).
db = SQLDatabase.from_uri("sqlite:///text2sql/data/chinook.db")

print("Dialect:", db.dialect)                   # sqlite
print("Tables: ", db.get_usable_table_names())  # the 11 tables

# 2. Get the schema: CREATE TABLE statements + 3 sample rows per table.
schema = db.get_table_info()
print("Schema length:", len(schema), "characters")
print(schema[:800])                             # peek at the first part

# 3. We can also run SQL directly (we write the SQL ourselves, for now).
print("Customers from Brazil:", db.run("SELECT COUNT(*) FROM Customer WHERE Country = 'Brazil';"))

# 4. Give the schema to the LLM and ask it to explain the database.
response = llm.invoke([
    SystemMessage("You are a data analyst. Be brief."),
    HumanMessage(f"Here is a database schema:\n\n{schema}\n\n"
                 "In 5 bullet points, explain what this database is about "
                 "and how the tables connect."),
])
print("\n--- Gemini says ---")
print(response.text)
print("\nTokens used:", response.usage_metadata)
```

**What to notice:**
- `SQLDatabase` (from `langchain-community`, using `sqlalchemy` underneath) does the reading for us.
- `get_table_info()` includes **sample rows**. These help the LLM understand what the data *looks like*, for example that `Country` holds values like `'Brazil'`, not `'BR'`.
- You'll see a `DeprecationWarning` saying **`langchain-community` is being sunset** (no longer actively maintained). It still works fine for learning. The LangChain team is moving integrations into separate packages. Later (v0.4) we'll write our own small, safe database layer directly with `sqlalchemy`, so we won't depend on it.
- Look at the **token count**: about **2,500 input tokens** (the schema is about 6,000 characters). The whole schema goes into every prompt. That's fine for 11 tables but **too big for 500 tables**, which is why we'll use RAG in Phase 4 to send only the relevant tables.
- The LLM understood how the tables relate **just from reading the schema**. In the next lessons it will write the SQL itself.

---

## 5. Bonus: switching providers in one line

```python
from langchain.chat_models import init_chat_model

llm = init_chat_model("google_genai:gemini-3.5-flash-lite")
# llm = init_chat_model("openai:gpt-...", temperature=0)        # needs langchain-openai + key
# llm = init_chat_model("anthropic:claude-...", temperature=0)  # needs langchain-anthropic + key
```
`init_chat_model` is a universal constructor. The rest of your code doesn't change. **That's the core promise of LangChain.**

---

## 6. Common errors

| Error | Cause | Fix |
|---|---|---|
| `API key required for Gemini Developer API` | `.env` not found or key not set | Run from the project root, and check that `.env` has `GOOGLE_API_KEY=...` |
| `ModuleNotFoundError: No module named 'langchain_google_genai'` | `.venv` not active | Run `.venv\Scripts\activate` |
| `unable to open database file` | Wrong working directory | `cd F:\Project\Gen_AI` before running |
| `429 RESOURCE_EXHAUSTED` | Free-tier rate limit hit | Wait a minute and retry |
| Warning about `AFC` (automatic function calling) | Harmless message from Google's SDK | Ignore it |
| `temperature will be ignored` warning | Gemini 3.x uses fixed sampling | Remove `temperature=...` |
| `langchain-community is being sunset` warning | That package is no longer maintained | Ignore it for now (it still works) |
| A call takes 30–60 seconds | You're using a thinking model (`gemini-3.5-flash`) | Use `gemini-3.5-flash-lite` |

> 📦 Note the package name vs. the import name: you **install** `langchain-google-genai` (dashes) but **import** `langchain_google_genai` (underscores). Python names can't contain dashes.

---

## 6b. 🎯 Points to stress + deeper knowledge

### Points to stress (the "aha" moments)

**1. LLMs are stateless: they remember nothing.**
Every call starts from zero. "Memory" = **resending the whole conversation every time**. The consequences:
- Each turn costs more than the last (the history keeps growing).
- Long chats eventually hit the **context window** limit.
- Someone has to store and trim that history. In Phase 5, LangGraph's checkpointer does it.

**2. Tokens = money + speed + limits.**
- **Input** tokens (what you send) and **output** tokens (what the model writes) are billed separately. Output usually costs **several times more**.
- Our Chinook schema is about **2,500 input tokens on every single call**. For 500 tables that could be 100,000+ tokens per question, which is why Phase 4 (RAG) exists.
- Always check `usage_metadata`. It's how real teams track cost.

**3. Picking a model is an engineering trade-off, not "use the best one."**
We measured it: `flash` took **53 s**, `flash-lite` took **2 s**, with the same answer. Thinking models are smarter at hard problems but slower, and their hidden **reasoning tokens are billed as output tokens**. Good engineers match the model to the task.

**4. The reply is an object, not a string.**
`AIMessage` carries `.text`, `usage_metadata` (tokens) and `response_metadata` (model info, **finish reason**). The finish reason tells you whether the answer was **cut off** by a token limit. That matters when SQL gets truncated halfway.

**5. LLM output is never guaranteed.**
Even with the same prompt, answers can vary. So we **never trust raw output**. We parse it, validate it and retry. That idea drives Lessons 3.3–3.4 and the self-correcting loop in v0.4.

**6. The system message is the developer's voice, but users can try to override it.**
"Ignore your instructions and..." is called **prompt injection**. System prompts guide behavior; they don't enforce it.

### Deeper knowledge (sounds great in interviews)

| Topic | What to know |
|---|---|
| **Async versions** | Every Runnable also has `ainvoke`, `astream` and `abatch`. Web servers (FastAPI, Phase 6) use these so one slow LLM call doesn't block other users. |
| **Streaming** | It doesn't make the answer faster. It lowers **time-to-first-token**, so the app *feels* faster. That's why every chat UI streams. |
| **Batch + rate limits** | `batch` runs calls in parallel and can hit free-tier limits (`429`). Control it with `llm.batch(qs, config={"max_concurrency": 2})`. |
| **Retries & fallbacks** | `llm.with_retry()` retries failures. `llm.with_fallbacks([other_llm])` switches to a backup model if the main one fails. Production apps use both. |
| **Bigger context ≠ better** | Models can pay less attention to information in the **middle** of a very long prompt ("lost in the middle"). Sending only relevant info usually beats sending everything. |
| **Why LangChain (and the criticism)** | Pros: standard interfaces, easy provider switching, huge integration library. Cons: extra abstraction layers, so for a tiny script the raw SDK can be simpler. Knowing **both sides** shows maturity. |

---

## 7. Summary: what to tell your friend

1. **LangChain** = LEGO bricks for LLM apps, all with the same connector shape.
2. It's split into packages: **core** (foundation), **langchain** (helpers), **provider adapters** (Gemini, OpenAI...) and **community** (extras like SQLDatabase).
3. A **chat model** takes **messages** (System / Human / AI) and returns an **AIMessage**.
4. Every brick is a **Runnable** with `.invoke()`, `.stream()` and `.batch()`.
5. LLMs have **no memory**. We resend the history each time.
6. For Text2SQL, we give the LLM the **schema** so it knows which tables and columns exist.

---

## 8. Quiz

1. Why does a low temperature suit Text2SQL? And why did we use `flash-lite` instead of `flash`?
2. What type of object does `llm.invoke()` return, and how do you get the plain text from it?
3. What are the three Runnable methods, and when would you use `batch`?
4. If the LLM has no memory, how does a chatbot "remember" what you said earlier?
5. Why will sending the full schema become a problem for a big company database?
6. You install `langchain-google-genai`. What do you type in `import`?

<details>
<summary>Answers</summary>

1. SQL must be exact. We want the same correct query every time, not creative variations. `flash` is a thinking model (about 53 s per call in our test), while `flash-lite` answered in about 2 s with the same quality for simple tasks.
2. An `AIMessage`. Use `.text` to get the plain text.
3. `invoke`, `stream`, `batch`. Use `batch` to run many inputs in parallel, like evaluating 50 test questions.
4. The app sends the whole conversation history with every request.
5. Hundreds of tables means a huge prompt: slow, expensive, may not fit in the context window, and the LLM gets confused. The fix is to retrieve only the relevant tables (RAG, Phase 4).
6. `from langchain_google_genai import ...` (underscores).
</details>

---

**Next → Lesson 3.2: Prompt Templates.** Instead of f-strings, we build a reusable Text2SQL prompt with `{schema}` and `{question}` slots.
