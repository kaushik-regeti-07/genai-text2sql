# Lesson 3.3: LCEL Chains (`prompt | llm | parser`)

> **Goal of this lesson:** connect LangChain pieces into **pipelines** using the `|` (pipe) operator.
> We finish with a **complete Text2SQL chain**: question → SQL → run it → plain-English answer, all in one `.invoke()`. 🎯

**You will create:** 4 scripts in `learning/phase3/` (09 to 12).
**New packages:** none. LCEL lives in `langchain-core`.
**Time:** about 2 hours.

---

## ⭐ Why this lesson matters

**LCEL is the "glue" of LangChain.** Lessons 3.1 and 3.2 gave you bricks (models, prompts). LCEL is how you **snap bricks together** into a working machine.

Look at how Script 8 (Lesson 3.2) worked, step by step, by hand:
```python
filled = text2sql_prompt.invoke({"question": q})   # step 1
sql    = llm.invoke(filled).text.strip()          # step 2
result = db.run(sql)                              # step 3
```
With LCEL, the same flow becomes one object you can call, stream, batch, reuse and debug:
```python
chain = text2sql_prompt | llm | parser
chain.invoke({"question": q})
```

**In our Text2SQL project, this lesson produces the first real pipeline:**
```
question ──► write SQL ──► clean SQL ──► run on DB ──► explain in English
```

**Three reasons it's foundational:**
1. **Composition:** small tested pieces combine into big systems. You can swap one piece (a different model, a different parser) without touching the rest.
2. **Free superpowers:** any chain automatically gets `invoke`, `stream`, `batch` and async, because a chain is itself a Runnable.
3. **The mental model for LangGraph:** a LangGraph node is usually a small LCEL chain. Phase 5 is "chains + decisions + loops."

---

## 1. The core idea: the pipe `|`

```
chain = prompt | llm | parser
```
means: **the output of the left side becomes the input of the right side.**

```
{"question": "..."}                 ← you pass a dict
      │
      ▼  prompt      fills the template
ChatPromptValue (messages)
      │
      ▼  llm         calls Gemini
AIMessage
      │
      ▼  parser      pulls out the text
"A primary key is..."               ← you get a plain string
```

> 🔧 **Analogy:** a **factory assembly line**. Each station does one job and passes its product to the next. The `|` is the conveyor belt. You feed raw material in one end and get a finished product out the other.

> 🔑 **The one rule of LCEL:** the **output type** of each step must match the **input type** of the next. Most LCEL bugs are a mismatch here, for example passing a string where a dict is expected.

---

## 2. The new building blocks

| Block | What it does | Example |
|---|---|---|
| `StrOutputParser()` | `AIMessage` → plain `str` | End of almost every chain |
| `RunnableLambda(func)` | Turns **any Python function** into a chain step | Cleaning SQL, running SQL |
| A plain function in a chain | LangChain wraps it in `RunnableLambda` **automatically** | `llm \| parser \| clean_sql` |
| `RunnableParallel({...})` | Runs several steps on the **same input**, returns a dict | Get SQL **and** keep the question |
| A plain dict in a chain | Automatically becomes a `RunnableParallel` | `{"a": step1, "b": step2}` |
| `RunnablePassthrough()` | Passes the input through **unchanged** | Keep the original question |
| `RunnablePassthrough.assign(key=step)` | Keeps the input dict **and adds** a new key | Add `query`, then `result`, then `answer` |

> 💡 `.assign()` is the most useful one for Text2SQL. It **builds up a dict step by step**, like filling in a form, so later steps can see everything earlier steps produced.

---

## 3. Hands-on

> Run each script from `F:\Project\Gen_AI` with `(.venv)` active.

### Script 9: `learning/phase3/09_first_chain.py`: your first chain

```python
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a SQL tutor. Answer in one sentence."),
    ("human", "{question}"),
])

# 1. Build the chain: prompt → model → parser
chain = prompt | llm | StrOutputParser()
print("Chain type:", type(chain).__name__)     # RunnableSequence

# 2. invoke: dict in, string out (no more .text!)
answer = chain.invoke({"question": "What is a primary key?"})
print("Type:", type(answer).__name__)           # TextAccessor (see note below)
print("Is it a str?", isinstance(answer, str))  # True
print(answer)

# 3. stream: works automatically on the whole chain
print("\nSTREAM:")
for piece in chain.stream({"question": "What is a database index?"}):
    print(piece, end="", flush=True)
print()

# 4. batch: also automatic
answers = chain.batch([
    {"question": "What is NULL in SQL?"},
    {"question": "What is a SQL view?"},
])
for a in answers:
    print("-", a)
```

**What to notice:**
- The chain is a `RunnableSequence`, which is **itself a Runnable**, so it has invoke/stream/batch for free.
- Thanks to `StrOutputParser`, the result is a string. No more `.text`.
- The type prints as **`TextAccessor`**. That's LangChain's own **subclass of `str`**, so it behaves exactly like a normal string (`isinstance(answer, str)` is `True`). Treat it as a string.
- Streaming still works **through** the parser. The pieces flow down the assembly line as they arrive.

---

### Script 10: `learning/phase3/10_custom_steps.py`: your own functions in a chain

LLMs sometimes wrap SQL in markdown (```` ```sql ... ``` ````), which breaks `db.run()`. Let's clean it **automatically** as a chain step.

```python
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

# 1. A normal Python function: str in, str out.
def clean_sql(text: str) -> str:
    """Remove ```sql ... ``` markdown fences around a query."""
    text = text.strip()
    text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)  # opening fence
    text = re.sub(r"\s*```$", "", text)                                # closing fence
    return text.strip()

# Test the function alone first (always test pieces separately!)
print(repr(clean_sql("```sql\nSELECT 1;\n```")))    # 'SELECT 1;'

# 2. Explicitly wrapped with RunnableLambda...
to_upper = RunnableLambda(lambda s: s.upper())
print(to_upper.invoke("select 1;"))                  # SELECT 1;

# 3. ...or just drop the function into the chain. LangChain wraps it for you.
#    We deliberately ASK for markdown here, to prove clean_sql removes it.
prompt = ChatPromptTemplate.from_messages([
    ("system", "Write a SQLite query. Wrap it in a ```sql markdown block."),
    ("human", "{question}"),
])
chain = prompt | llm | StrOutputParser() | clean_sql

sql = chain.invoke({"question": "Count the rows in the Artist table"})
print(repr(sql))     # clean SQL, no ``` fences
```

**What to notice:**
- **Any Python function can be a chain step.** It just needs one input and one output.
- `repr()` shows hidden characters, so you can *see* that the fences and newlines are gone.
- **Test each piece alone before chaining.** When a long chain breaks, finding the bad step is much harder.

---

### Script 11: `learning/phase3/11_parallel_and_assign.py`: keeping data along the way

**The problem:** in a plain chain, each step **only sees the previous step's output**. After the LLM writes SQL, the original **question is lost**. But the final "explain the answer" step needs the question **and** the SQL **and** the result.

```python
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

short_answer = (
    ChatPromptTemplate.from_messages([("human", "Answer in 5 words: {question}")])
    | llm
    | StrOutputParser()
)

# ---------- 1. RunnableParallel: several steps, same input, dict out ----------
parallel = RunnableParallel(
    original=RunnablePassthrough(),     # input passes through untouched
    answer=short_answer,                # the LLM's answer
    length=lambda x: len(x["question"]),
)
print(parallel.invoke({"question": "What is SQL?"}))
# {'original': {'question': 'What is SQL?'}, 'answer': '...', 'length': 12}

# ---------- 2. .assign(): keep everything, ADD new keys step by step ----------
pipeline = (
    RunnablePassthrough.assign(answer=short_answer)                 # adds "answer"
    .assign(word_count=lambda x: len(x["answer"].split()))          # uses "answer"!
)
print(pipeline.invoke({"question": "What is a JOIN?"}))
# {'question': 'What is a JOIN?', 'answer': '...', 'word_count': 5}
```

**What to notice:**
- `RunnablePassthrough()` = "give me the input exactly as it came in."
- `.assign()` **grows** the dict: `{question}` → `{question, answer}` → `{question, answer, word_count}`.
- The second `.assign` **used the result of the first**. That's how data flows forward in Text2SQL.

```
{question}
   │ .assign(answer=...)
   ▼
{question, answer}
   │ .assign(word_count=...)
   ▼
{question, answer, word_count}
```

---

### Script 12: `learning/phase3/12_text2sql_chain.py`: 🎯 the full Text2SQL pipeline

```python
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
db = SQLDatabase.from_uri("sqlite:///text2sql/data/chinook.db")

def clean_sql(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()

# ---------- Chain 1: question → SQL ----------
SQL_SYSTEM = """You are an expert {dialect} developer.
Given the database schema below, write ONE syntactically correct {dialect} query
that answers the user's question.

Rules:
- Only use tables and columns that exist in the schema.
- Only write SELECT queries. Never INSERT, UPDATE, DELETE or DROP.
- Unless the user asks for more, LIMIT results to {top_k} rows.
- Return ONLY the SQL. No explanation, no markdown.

Schema:
{schema}"""

sql_prompt = ChatPromptTemplate.from_messages([
    ("system", SQL_SYSTEM),
    ("human", "{question}"),
]).partial(dialect=db.dialect, schema=db.get_table_info(), top_k=5)

write_sql = sql_prompt | llm | StrOutputParser() | clean_sql

# ---------- Chain 2: (question + SQL + result) → English answer ----------
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You answer questions about a music store database. "
               "Use ONLY the SQL result below to answer. "
               "If the result is empty, say no data was found. Be concise."),
    ("human", "Question: {question}\nSQL query: {query}\nSQL result: {result}\n\nAnswer:"),
])
write_answer = answer_prompt | llm | StrOutputParser()

# ---------- The full pipeline: build up the dict step by step ----------
text2sql = (
    RunnablePassthrough.assign(query=write_sql)                     # + query
    .assign(result=lambda x: db.run(x["query"]))                     # + result
    .assign(answer=write_answer)                                     # + answer
)

# ---------- Try it ----------
out = text2sql.invoke({"question": "Which employee has the most customers?"})
print("Question:", out["question"])
print("SQL:     ", out["query"])
print("Result:  ", out["result"])
print("Answer:  ", out["answer"])

# batch works on the whole pipeline for free
print("\n--- batch ---")
for o in text2sql.batch([
    {"question": "How many tracks are longer than 5 minutes?"},
    {"question": "What are the 3 most expensive tracks?"},
]):
    print(f"\nQ: {o['question']}\nSQL: {o['query']}\nA: {o['answer']}")
```

**What to notice:**
- **Two small chains** (`write_sql`, `write_answer`) plus **one Python step** (`db.run`) make **one pipeline**.
- The dict grows: `{question}` → `+query` → `+result` → `+answer`. The final step sees **everything**.
- We return **all** the fields, not just the answer. That matters for trust: the user can **see the SQL** and check it. Real Text2SQL tools always show the query.
- `.batch()` worked on the entire pipeline without any extra code.

> 🧪 **Experiments:**
> 1. Ask *"How many customers are from Atlantis?"* What does the answer step say when the result is empty?
> 2. Ask a trick question: *"Delete all invoices."* Look at the `query` field. What SQL was generated, and did it run?
> 3. Ask something that makes the LLM invent a column: *"What is each customer's age?"* (There's no age column.) What error do you get, and at which step?
> 4. Change `top_k=5` to `top_k=2` and ask for "the most expensive tracks". What changes?

**Results from our test run:**
```
Q: Which employee has the most customers?
SQL: SELECT e.FirstName, e.LastName FROM Employee e JOIN Customer c ON e.EmployeeId = c.SupportRepId
     GROUP BY e.EmployeeId ORDER BY COUNT(c.CustomerId) DESC LIMIT 1
A: Jane Peacock                                     (about 5.5 seconds total)

Q: How many tracks are longer than 5 minutes?
SQL: SELECT COUNT(*) FROM Track WHERE Milliseconds > 300000;
A: There are 1,069 tracks longer than 5 minutes.
```
Notice that the LLM worked out on its own that `SupportRepId` links customers to employees, and that 5 minutes = 300000 milliseconds.

> ⚠️ **What's still missing (and why we need the next lessons):**
> - If the SQL is wrong, `db.run` **crashes** the whole chain. There's no retry yet → **LangGraph loop (v0.4)**.
> - We still get SQL as loose text. We want a clean object like `{sql, explanation}` → **structured output (Lesson 3.4)**.
> - Nothing in **code** blocks a `DELETE`. We rely on the prompt → **validator + read-only DB (v0.4)**.

---

## 4. 🎯 Points to stress + deeper knowledge

### Points to stress (the "aha" moments)

**1. A chain is just a Runnable made of Runnables.**
That's why `chain.stream()` and `chain.batch()` worked without any extra code. The whole of LangChain is built on this one idea.

**2. Types must line up.**
`prompt` wants a **dict** → gives **messages**. `llm` wants **messages** → gives an **AIMessage**. `parser` wants an **AIMessage** → gives a **str**. When a chain breaks, first ask: *"what type goes in and out of each step?"*

**3. `.assign()` solves the "lost data" problem.**
In a plain pipe, each step forgets everything except the previous output. `.assign()` carries a growing dict forward so the final step sees the question, SQL and result together. **LangGraph's "State" (Phase 5) is the same idea, made official.**

**4. LLM steps and normal code steps mix freely.**
`db.run()` and `clean_sql()` are plain Python, not AI, and they sit in the same pipeline as LLM calls. Real GenAI apps are **mostly normal code** with a few LLM calls in the right places.

**5. Always show your work.**
Returning `query` and `result`, not just `answer`, lets users verify the output. In Text2SQL, **an answer without the SQL can't be trusted**.

### Deeper knowledge (sounds great in interviews)

| Topic | What to know |
|---|---|
| **Why "Expression Language"?** | LCEL is *declarative*: you describe **what** connects to what, and LangChain handles **how** (streaming, batching, async, parallelism). |
| **Automatic parallelism** | Steps inside a `RunnableParallel` (or a dict) run **at the same time**. Two independent LLM calls take the time of one. |
| **Tracing** | Every step of an LCEL chain shows up separately in **LangSmith** (Phase 6), with its input, output and time. That's how you find *which* step broke. |
| **`.with_retry()` / `.with_fallbacks()`** | Work on **any** step or whole chain, e.g. `write_sql.with_retry(stop_after_attempt=3)`. |
| **LCEL vs. LangGraph** | LCEL = straight lines (A → B → C), with optional parallel branches. **No loops or real decisions.** When you need "if the SQL failed, go back and fix it," you need LangGraph. Knowing **when to switch** is a key skill. |
| **Legacy chains** | Old tutorials use classes like `LLMChain` or `SQLDatabaseChain`. Those are **deprecated**. Modern LangChain uses LCEL and LangGraph. If a tutorial uses `LLMChain`, it's outdated. |

---

## 5. Common errors

| Error | Cause | Fix |
|---|---|---|
| `TypeError: Expected mapping type as input to ChatPromptTemplate` | You passed a **string** to a prompt | Prompts need a dict: `{"question": "..."}` |
| `KeyError: 'query'` inside a lambda | The key doesn't exist yet at that step | Check the order of your `.assign()` calls |
| `sqlite3.OperationalError: no such column` | The LLM invented a column | Expected for now. The fix is the retry loop in v0.4. |
| `sqlite3.OperationalError: near "```sql"` | Markdown fences reached the DB | Make sure `clean_sql` is in the chain |
| Lambda gets a dict when you expected a string | After `.assign`, each step receives the **whole dict** | Use `x["key"]` inside the lambda |
| `500 INTERNAL` when the chain's output is sent to Gemini **embeddings** (found in Lesson 4.5) | `StrOutputParser` returns a `TextAccessor`, which the embedding API rejects | Wrap it: `str(chain.invoke(...))` |

---

## 6. Summary: what to tell your friend

1. **LCEL** = connecting Runnables with `|`. Output of the left side → input of the right side.
2. A chain is **itself a Runnable**, so it gets invoke/stream/batch for free.
3. `StrOutputParser` turns an `AIMessage` into a plain string.
4. **Any Python function** can be a chain step (`RunnableLambda`).
5. `RunnableParallel` runs steps side by side. `.assign()` **adds keys to a growing dict**.
6. Our Text2SQL pipeline: `question → +query → +result → +answer`.
7. LCEL can't loop or retry based on the result. **That's why LangGraph exists.**

---

## 7. Quiz

1. In `prompt | llm | StrOutputParser()`, what type comes out of each step?
2. Why did `chain.stream()` work without writing any streaming code?
3. You put a normal Python function in a chain. What does LangChain do with it?
4. What problem does `.assign()` solve in our Text2SQL pipeline?
5. Why do we return the `query` and `result`, and not just the `answer`?
6. The LLM writes SQL with a wrong column name. What happens in our chain, and which future lesson fixes it?

<details>
<summary>Answers</summary>

1. `prompt` → `ChatPromptValue` (messages), `llm` → `AIMessage`, `StrOutputParser` → `str`.
2. A chain is itself a Runnable. Every Runnable gets invoke/stream/batch automatically.
3. It wraps it in a `RunnableLambda` so it becomes a chain step.
4. Normally each step only sees the previous output, so the question would be lost. `.assign()` keeps the original dict and adds new keys, so the answer step sees question + SQL + result.
5. Trust and verification: the user can check the SQL and the raw data behind the answer.
6. `db.run()` raises an error and the whole chain crashes. The LangGraph self-correction loop (v0.4, Phase 5) fixes it by sending the error back to the LLM to retry.
</details>

---

**Next → Lesson 3.4: Structured Output.** Instead of loose text, the LLM returns a clean Python object like `{sql: "...", explanation: "...", tables_used: [...]}` that we can trust and validate.
