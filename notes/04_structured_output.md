# Lesson 3.4: Structured Output

> **Goal of this lesson:** make the LLM return **clean Python objects** with guaranteed fields and types, instead of loose text we have to clean up.
> We finish with a Text2SQL chain where the LLM returns `{is_answerable, tables_used, sql, explanation}`, and our **code** blocks anything that isn't a SELECT. 🎯

**You will create:** 4 scripts in `learning/phase3/` (13 to 16).
**New packages:** none to install. We use **Pydantic**, which came in automatically with LangChain (LangChain is built on it).
**Time:** about 2 hours.

---

## ⭐ Why this lesson matters

**This is the bridge between "AI text" and "normal software."**
Programs need **data**: a boolean to branch on, a list to loop over, a string that is definitely SQL. LLMs produce **text**. Structured output turns the LLM into a function that returns **typed data your code can trust**.

**Compare Lesson 3.3 and this lesson:**
```
Lesson 3.3:  LLM → "```sql\nSELECT ...```"  → clean_sql() with regex → hope it's SQL
Lesson 3.4:  LLM → SQLQuery(is_answerable=True, tables_used=[...], sql="SELECT ...", explanation="...")
```

**In our Text2SQL project, it solves three real problems:**
1. **No more markdown cleanup.** The SQL arrives in its own field, clean.
2. **"I can't answer that" becomes a proper value.** `is_answerable=False` instead of the LLM inventing a query or the chain crashing.
3. **Our first code-level safety check.** A Pydantic **validator** rejects anything that isn't a SELECT, in code rather than just in the prompt.

**Three reasons it's foundational:**
1. **Every agent decision is structured output.** "Which tool? Which route? Retry or stop?" The LangGraph router in Phase 5 is exactly the classifier you'll build in Script 15.
2. **Tool calling (Lesson 3.5) uses the same mechanism.** A tool call *is* structured output: `{tool_name, arguments}`.
3. **Extraction is one of the most common real-world GenAI jobs:** pulling fields out of emails, invoices, resumes and support tickets.

---

## 1. Pydantic in 2 minutes

**Pydantic** lets you describe the **shape** of your data as a Python class. It then **checks and converts** data to match.

```python
from pydantic import BaseModel

class Customer(BaseModel):
    name: str
    total_spent: float
```
- `Customer(name="Helena", total_spent="49.62")` → ✅ works, and `"49.62"` is **converted** to `49.62` (a float)
- `Customer(name="Helena", total_spent="lots")` → ❌ `ValidationError`

> 📋 **Analogy:** a Pydantic model is like a **form with strict boxes**: "Age: numbers only", "Email: must contain @". The LLM fills in the form. Pydantic is the **clerk who checks every box** before accepting it.

### The pieces you'll use

| Piece | Meaning | Example |
|---|---|---|
| `name: str` | Required field of this type | `sql: str` |
| `x: bool = False` | Optional, with a default | `vip: bool = False` |
| `Optional[str] = None` | May be missing | Customer name not mentioned |
| `list[str]` | A list of strings | `tables_used: list[str]` |
| `Literal["a", "b"]` | Must be **exactly one of** these values | `sentiment: Literal["positive", "negative"]` |
| `Field(description="...")` | **Instructions for the LLM** about this field | `Field(description="One SELECT query")` |
| `Field(ge=0)` | Rules: ge = "greater or equal" | `total_spent: float = Field(ge=0)` |
| `@field_validator` | Your own custom check, in code | "SQL must start with SELECT" |
| Docstring `"""..."""` | Description of the whole object, also sent to the LLM | `"""A SQL query that..."""` |

> 🔑 **Key insight:** field names, types, descriptions and the docstring **are sent to the LLM as part of the prompt**. A good `description` is prompt engineering.

---

## 2. How `with_structured_output` works

```python
structured_llm = llm.with_structured_output(SQLQuery)
result = structured_llm.invoke("...")   # → a SQLQuery object, not an AIMessage
```

```
Your Pydantic class
      │  LangChain converts it to a JSON Schema
      ▼
Gemini receives: prompt + "respond matching this schema"
      │  the model's output is constrained to that shape
      ▼
Raw JSON: {"is_answerable": true, "sql": "SELECT ...", ...}
      │  LangChain parses it with Pydantic
      ▼
SQLQuery object ← types checked, validators run
```

> 💡 `with_structured_output()` returns a new **Runnable**, so it slots straight into LCEL: `prompt | llm.with_structured_output(X)`. There's no parser at the end, because the output is already the object.

---

## 3. Hands-on

> Run each script from `F:\Project\Gen_AI` with `(.venv)` active.

### Script 13: `learning/phase3/13_pydantic_basics.py`: Pydantic alone (no LLM)

```python
from pydantic import BaseModel, Field, ValidationError

class Customer(BaseModel):
    name: str
    country: str
    total_spent: float = Field(ge=0)      # must be 0 or more
    vip: bool = False                     # optional, default False

# 1. Valid data. Notice "49.62" (a string) gets CONVERTED to a float.
c = Customer(name="Helena", country="Czech Republic", total_spent="49.62")
print(c)
print("total_spent type:", type(c.total_spent).__name__)   # float
print("As dict:", c.model_dump())

# 2. Invalid data → a clear error instead of bad data sneaking through.
try:
    Customer(name="X", country="Y", total_spent=-5)
except ValidationError as e:
    print("Rejected:", e.errors()[0]["msg"])

# 3. The JSON Schema. This is what LangChain sends to the LLM.
print(Customer.model_json_schema())
```

**What to notice:**
- Pydantic **converts** when it safely can (`"49.62"` → `49.62`) and **rejects** when it can't (`-5` breaks `ge=0`).
- `model_json_schema()` shows the exact "form" the LLM will be asked to fill in.

---

### Script 14: `learning/phase3/14_extraction.py`: pull structured data out of messy text

```python
from typing import Literal, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

class SupportTicket(BaseModel):
    """Information extracted from a customer message."""
    customer_name: Optional[str] = Field(None, description="The customer's name, if mentioned")
    product: str = Field(description="The album, track or product the message is about")
    sentiment: Literal["positive", "neutral", "negative"]
    wants_refund: bool = Field(description="True if the customer asks for their money back")
    summary: str = Field(description="One-sentence summary of the message")

extractor = llm.with_structured_output(SupportTicket)

# 1. Directly on a string
message = ("Hi, this is Ravi. I bought the album 'Let There Be Rock' yesterday "
           "but 3 tracks won't play. Really disappointed, I want my money back.")
ticket = extractor.invoke(message)
print(type(ticket).__name__)            # SupportTicket, a real Python object!
print(ticket)
if ticket.wants_refund:                 # ← normal code can now make decisions
    print("→ Send to the refunds team")

# 2. Inside an LCEL chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract support ticket info from the customer's message."),
    ("human", "{message}"),
])
chain = prompt | extractor
print(chain.invoke({"message": "Loving the new AC/DC playlist, great job!"}))

# 3. Debug view: see the raw reply AND any parsing error
debug = llm.with_structured_output(SupportTicket, include_raw=True).invoke(message)
print("Keys:", list(debug.keys()))      # ['raw', 'parsed', 'parsing_error']
print("Parsing error:", debug["parsing_error"])
```

**Results from our test run:**
```
customer_name='Ravi' product='Let There Be Rock' sentiment='negative' wants_refund=True
customer_name=None   product='AC/DC playlist'    sentiment='positive' wants_refund=False
```

**What to notice:**
- `ticket.wants_refund` is a real `bool`, so `if ticket.wants_refund:` just works. **That's the whole point.**
- `customer_name=None` in the second message: `Optional` let the LLM say "not mentioned" instead of inventing a name.
- `include_raw=True` is your **debugging tool** when parsing fails.

---

### Script 15: `learning/phase3/15_classifier.py`: a router (preview of LangGraph!)

```python
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

class Route(BaseModel):
    category: Literal["database_question", "greeting", "off_topic"] = Field(
        description="database_question = anything answerable from the music store database "
                    "(customers, sales, artists, albums, tracks, employees). "
                    "greeting = hello/thanks/small talk. off_topic = anything else."
    )
    reason: str = Field(description="One short sentence explaining the choice")

router = (
    ChatPromptTemplate.from_messages([
        ("system", "Classify the user's message for a music-store database assistant."),
        ("human", "{question}"),
    ])
    | llm.with_structured_output(Route)
)

for q in ["How many albums does Aerosmith have?",
          "Hi there!",
          "What's the weather in Paris?",
          "Which country buys the most jazz?"]:
    r = router.invoke({"question": q})
    print(f"{q:40} → {r.category:18} ({r.reason})")
```

**Results from our test run:**
```
How many albums does Aerosmith have?  → database_question
Hi there!                             → greeting
What's the weather in Paris?          → off_topic
Which country buys the most jazz?     → database_question
```

**What to notice:**
- `Literal[...]` means the answer **can only be one of 3 values**, so your code can safely write `if r.category == "greeting":`.
- The `description` holds the **definitions** of each category. That's where you tune the router.
- **This is exactly the first node of our LangGraph app in Phase 5:** classify, then take a different path per category.

---

### Script 16: `learning/phase3/16_text2sql_structured.py`: 🎯 Text2SQL with structured output + a code-level guard

```python
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, ValidationError
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
db = SQLDatabase.from_uri("sqlite:///text2sql/data/chinook.db")

FORBIDDEN = ("insert", "update", "delete", "drop", "alter", "create",
             "replace", "truncate", "attach", "pragma")

# ---------- 1. The shape of a good answer ----------
class SQLQuery(BaseModel):
    """A SQL query that answers the user's question."""
    is_answerable: bool = Field(
        description="False if the database does not contain the information needed")
    tables_used: list[str] = Field(
        default_factory=list, description="Tables the query reads from")
    sql: str = Field(
        default="", description="One SQLite SELECT query, no markdown. Empty if not answerable.")
    explanation: str = Field(
        description="One sentence: what the query does, or why it can't be answered")

    # Runs in CODE every time a SQLQuery is created. The LLM can't skip it.
    @field_validator("sql")
    @classmethod
    def must_be_select(cls, v: str) -> str:
        v = v.strip()
        if not v:
            return v                                   # empty = not answerable
        low = v.lower()
        if not (low.startswith("select") or low.startswith("with")):
            raise ValueError("Query must start with SELECT or WITH")
        if any(f" {word} " in f" {low} " for word in FORBIDDEN):
            raise ValueError("Query contains a forbidden keyword")
        return v

# Prove the guard works, with no LLM involved:
try:
    SQLQuery(is_answerable=True, sql="DELETE FROM Customer", explanation="oops")
except ValidationError as e:
    print("BLOCKED:", e.errors()[0]["msg"])

# ---------- 2. Question → SQLQuery object ----------
SQL_SYSTEM = """You are an expert {dialect} developer.
Given the database schema below, write ONE {dialect} SELECT query that answers the user's question.
- Only use tables and columns that exist in the schema.
- Unless the user asks for more, LIMIT results to {top_k} rows.
- If the schema does not contain the needed information, set is_answerable to false and leave sql empty.

Schema:
{schema}"""

sql_prompt = ChatPromptTemplate.from_messages([
    ("system", SQL_SYSTEM),
    ("human", "{question}"),
]).partial(dialect=db.dialect, schema=db.get_table_info(), top_k=5)

write_sql = sql_prompt | llm.with_structured_output(SQLQuery)
# Notice: no StrOutputParser, no clean_sql(). The object IS the output.

# ---------- 3. Run only if answerable ----------
def run_query(x: dict) -> str:
    q: SQLQuery = x["sql_obj"]
    if not q.is_answerable:
        return "NOT_ANSWERABLE: " + q.explanation
    return db.run(q.sql)

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You answer questions about a music store database. Use ONLY the SQL result. "
               "If it says NOT_ANSWERABLE, politely explain the database doesn't have that "
               "information. Be concise."),
    ("human", "Question: {question}\nSQL result: {result}\n\nAnswer:"),
])

text2sql = (
    RunnablePassthrough.assign(sql_obj=write_sql)
    .assign(result=run_query)
    .assign(answer=answer_prompt | llm | StrOutputParser())
)

# ---------- 4. Try it ----------
for q in ["How many invoices were made in 2022?",
          "Which 3 artists have the most tracks?",
          "What is the favourite colour of each customer?"]:
    out = text2sql.invoke({"question": q})
    s = out["sql_obj"]
    print(f"\nQ: {q}")
    print(f"   answerable: {s.is_answerable} | tables: {s.tables_used}")
    print(f"   SQL:    {s.sql or '(none)'}")
    print(f"   Answer: {out['answer']}")
```

**Results from our test run:**
```
BLOCKED: Value error, Query must start with SELECT or WITH

Q: How many invoices were made in 2022?
   SQL:    SELECT COUNT(*) FROM Invoice WHERE InvoiceDate >= '2022-01-01 00:00:00' AND ...
   Answer: There were 83 invoices made in 2022.

Q: Which 3 artists have the most tracks?
   answerable: True | tables: ['Artist', 'Album', 'Track']
   SQL:    SELECT T1.Name FROM Artist AS T1 INNER JOIN Album AS T2 ... LIMIT 3

Q: What is the favourite colour of each customer?
   answerable: False | tables: []
   SQL:    (none)
   Answer: I'm sorry, but the database doesn't have that information.
```
And when asked *"Delete all invoices from 2021"*, the model returned `is_answerable=False` with *"violates the constraint to only write SELECT queries."*

**What to notice:**
- **No more `clean_sql()`.** The SQL arrives clean in its own field.
- **`is_answerable=False` handles impossible questions gracefully.** In Lesson 3.3, "customer's age" made the LLM invent a column and crash the chain.
- **`tables_used`** is a free bonus: useful for logging, and later for checking the LLM only touched allowed tables.
- **The validator is real code-level safety.** Even if a tricked LLM returns `DELETE ...`, Pydantic raises an error **before** anything reaches the database.

> ⚠️ **Honest note: this validator is a *first* guard, not a complete one.** Keyword checks can be fooled (odd spacing, comments, stacked statements). In v0.4 we add the stronger layers: a **read-only database connection** and proper SQL parsing. Security is always **layers**.

> 🧪 **Experiments:**
> 1. Remove `description=` from `is_answerable` and ask "What's each customer's age?". Does the model still say it's not answerable?
> 2. In Script 15, add a 4th category `"complaint"` and test *"Your app keeps crashing!"*
> 3. In Script 16, ask *"Show all customers; DROP TABLE Customer"*. Does the LLM refuse, or does the validator catch it?
> 4. Use `include_raw=True` on `write_sql` and look at `raw` to see what Gemini actually sent back.

---

## 4. 🎯 Points to stress + deeper knowledge

### Points to stress (the "aha" moments)

**1. Structured output turns the LLM into a function with a return type.**
Text in, typed object out. Normal code (`if`, loops, databases) can finally trust what the AI produced.

**2. Descriptions are prompts.**
The field names, `description=` values and class docstring are all sent to the LLM. `is_answerable`'s description is **why** the model said "no" to impossible questions. Writing good field descriptions **is** prompt engineering.

**3. `Literal` = the foundation of routing.**
Restricting an answer to `"database_question" | "greeting" | "off_topic"` means your code can branch on it safely. **Every LangGraph router is this pattern.**

**4. Give the model a way to say "I don't know."**
Without `is_answerable` or `Optional` fields, the model is *forced* to fill every box, so it **makes things up** (invented columns, fake names). Designing an "escape hatch" into the schema is one of the best ways to reduce hallucination.

**5. Validation belongs in code.**
Prompt: "please only write SELECT" (advice). Validator: `raise ValueError` (enforcement). This is our first real **defense-in-depth** layer.

### Deeper knowledge (sounds great in interviews)

| Topic | What to know |
|---|---|
| **How providers do it** | Two main methods: **native JSON-schema mode** (the model's output is constrained to match the schema) and **tool calling** (the schema is disguised as a "tool" the model must call). LangChain picks one per provider. You can choose with `method=...`. |
| **Constrained decoding** | In native schema mode, the provider only lets the model produce tokens that keep the JSON valid. That's why output structure is so reliable today compared to the old days of parsing text with regex. |
| **Valid shape ≠ correct content** | The schema guarantees `sql` is a string. It **doesn't** guarantee the SQL is right. You still need validation, execution checks and evaluation. |
| **Other schema formats** | Besides Pydantic, `with_structured_output` accepts a `TypedDict` or a raw JSON-schema dict (you get a `dict` back, with no validation). Pydantic is preferred because of validators. |
| **Field order can matter** | Models write fields **in order**. Putting `explanation` or `reasoning` **before** `sql` lets the model "think first, then write." It's a mini chain-of-thought and can improve accuracy. Worth testing! |
| **Retry on validation failure** | Chain `.with_retry()` after a structured-output step so a malformed answer gets a second try. In v0.4, LangGraph will send the *validation error message* back to the LLM so it can fix its own mistake. |

---

## 5. Common errors

| Error | Cause | Fix |
|---|---|---|
| `ValidationError: ... Field required` | The LLM left out a required field | Add a clearer `description`, or give the field a default |
| `ValidationError: Value error, Query must start with SELECT` | Your validator blocked the output (working as intended!) | Log it. In v0.4, send it back to the LLM to retry. |
| `ValidationError: Input should be 'positive', 'neutral' or 'negative'` | The model used a value outside the `Literal` | Rare with native schema mode. Clarify the description. |
| `'AIMessage' object has no attribute 'sql'` | You used `llm` instead of `llm.with_structured_output(...)` | Check the chain |
| Returned `None` | Parsing failed silently | Use `include_raw=True` and look at `parsing_error` |

---

## 6. Summary: what to tell your friend

1. **Structured output** = the LLM returns a **typed Python object**, not text.
2. We describe the shape with a **Pydantic** class: types, defaults, `Literal`, `Optional`, `Field(description=...)`.
3. `llm.with_structured_output(Model)` is a Runnable, so it plugs into LCEL.
4. **Descriptions are prompts.** They tell the model how to fill each field.
5. `Literal` enables **routing** (the LangGraph preview).
6. Give the model an **escape hatch** (`is_answerable`, `Optional`) so it doesn't hallucinate.
7. **Validators = safety in code**, our first real guard against dangerous SQL.

---

## 7. Quiz

1. What do you get back from `llm.invoke(...)` vs. `llm.with_structured_output(X).invoke(...)`?
2. Pydantic receives `total_spent="49.62"`. What happens? And with `total_spent=-5` and `Field(ge=0)`?
3. Why did we add `is_answerable` to `SQLQuery`? What went wrong without it in Lesson 3.3?
4. What's the difference between the prompt rule "only write SELECT" and the `must_be_select` validator?
5. How is the `Route` classifier related to LangGraph?
6. The schema guarantees `sql` is a string. Does that mean the SQL is correct?

<details>
<summary>Answers</summary>

1. `llm.invoke` → an `AIMessage` (text). With structured output → an instance of your Pydantic class `X`.
2. `"49.62"` is converted to the float `49.62`. `-5` raises a `ValidationError` because it breaks `ge=0`.
3. It lets the model say "the database doesn't have this" instead of inventing columns. In 3.3, "customer's age" made it invent a column, and `db.run` crashed.
4. The prompt rule is **advice** the LLM may ignore or be tricked out of. The validator is **code** that always runs and raises an error before anything reaches the DB.
5. A router is exactly this: classify with a `Literal`, then take a different path per category. It becomes the first node of our LangGraph app.
6. No. It only guarantees the **shape**. The SQL can still be wrong (wrong JOIN, wrong filter), so we still need execution checks, retries and evaluation.
</details>

---

**Next → Lesson 3.5: Tools / Function Calling.** The LLM **decides** to call *your* Python functions, like `list_tables()`, `get_schema(table)` and `run_query(sql)`, instead of us hard-wiring every step.
