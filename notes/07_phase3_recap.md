# Phase 3 Recap: All 20 Scripts at a Glance

> Use this page to revise, or to teach Phase 3 in one sitting. Each script: **what it does → the key line → the one idea it proves**.

---

## The big picture: how the 20 scripts build on each other

```
3.1  Talk to the model        01 → 02 → 03 → 04
       (one call)  (history) (stream/batch) (give it the schema)
                                               │
3.2  Build prompts properly   05 → 06 → 07 → 08  ← LLM writes SQL for the first time
                                               │
3.3  Connect the steps        09 → 10 → 11 → 12  ← full pipeline: SQL → run → answer
                                               │
3.4  Get typed data back      13 → 14 → 15 → 16  ← SQL as an object + "can't answer"
                                               │
3.5  Let the LLM decide       17 → 18 → 19 → 20  ← an agent that explores the DB itself
                                               │
3.6  Organize it all          text2sql/app/  (v0.1 + v0.2)
```

**One sentence per lesson:**
- **3.1:** An LLM takes messages in and gives an AIMessage back; it remembers nothing.
- **3.2:** Prompts are reusable templates with blanks; what you put in decides quality.
- **3.3:** `|` connects steps into one pipeline; `.assign()` carries data forward.
- **3.4:** The LLM can return typed Python objects that code can trust.
- **3.5:** Give the LLM tools and a loop, and it decides the steps itself.

---

## Lesson 3.1: Chat Models

| # | Script | What it does | Key line | The one idea |
|---|---|---|---|---|
| 01 | `first_call` | Sends one question to Gemini | `response = llm.invoke("What is SQL?")` | The reply is an **AIMessage object** (text + token usage), not a string |
| 02 | `messages` | Sends System + Human + AI + Human messages | `llm.invoke([SystemMessage(...), HumanMessage(...), ...])` | **LLMs have no memory.** "Memory" = resending the whole history |
| 03 | `stream_batch` | Streams one answer; batches 3 questions | `llm.stream(...)`, `llm.batch([...])` | Every Runnable has **invoke / stream / batch** |
| 04 | `describe_chinook` | Reads the DB schema and asks Gemini to explain it | `schema = db.get_table_info()` | Giving the LLM the **schema** lets it understand the database. About 2,500 tokens every time. |

**Flow of 04:**
```
chinook.db → SQLDatabase → get_table_info() → schema text → prompt → Gemini → 5 bullet points
```

---

## Lesson 3.2: Prompt Templates

| # | Script | What it does | Key line | The one idea |
|---|---|---|---|---|
| 05 | `prompt_template` | Fills `{keyword}` and `{level}` blanks; shows the error for a missing one | `PromptTemplate.from_template("... {keyword} ...")` | Templates **fail loudly** on missing variables, unlike f-strings |
| 06 | `chat_prompt` | Same template used as "SQL tutor" and "pirate" | `ChatPromptTemplate.from_messages([("system", ...), ("human", ...)])` | **One template, many uses.** Tuples become real messages. |
| 07 | `partial_and_history` | Pre-fills `role`; inserts chat history; shows `{{ }}` | `.partial(role=...)`, `MessagesPlaceholder("history")` | `partial` = fixed values; `MessagesPlaceholder` = history slot; `{{ }}` = literal brace |
| 08 | `text2sql_prompt` 🎯 | **LLM writes SQL**, and we run it on Chinook | `SYSTEM + few_shot + question`, `.partial(schema=...)` | Schema + rules + **few-shot examples** = correct SQL with JOINs |

**Structure of the 08 prompt (keep this in mind; it's reused everywhere):**
```
[system]  rules + {dialect} + {top_k} + {schema}   ← fixed (partial)
[human]   example question 1                         ← few-shot
[ai]      example SQL 1
[human]   example question 2
[ai]      example SQL 2
[human]   {question}                                  ← the only thing that changes
```

---

## Lesson 3.3: LCEL Chains

| # | Script | What it does | Key line | The one idea |
|---|---|---|---|---|
| 09 | `first_chain` | prompt → llm → parser as one object | `chain = prompt \| llm \| StrOutputParser()` | A chain **is a Runnable**, so stream/batch come free. Output is a string (`TextAccessor`). |
| 10 | `custom_steps` | Adds `clean_sql()` to strip ```` ``` ```` fences | `prompt \| llm \| StrOutputParser() \| clean_sql` | **Any Python function** can be a chain step |
| 11 | `parallel_and_assign` | Runs steps side by side; builds up a dict | `RunnablePassthrough.assign(answer=...).assign(word_count=...)` | `.assign()` **keeps** earlier data and **adds** new keys |
| 12 | `text2sql_chain` 🎯 | Full pipeline: question → SQL → run → English answer | `.assign(query=write_sql).assign(result=...).assign(answer=...)` | Two small chains + one Python step = **one pipeline** |

**Flow of 12 (the dict grows at each step):**
```
{question}
  → {question, query}
  → {question, query, result}
  → {question, query, result, answer}
```

**Types flowing through a chain (the #1 debugging question):**
```
dict ─prompt→ ChatPromptValue ─llm→ AIMessage ─parser→ str
```

---

## Lesson 3.4: Structured Output

| # | Script | What it does | Key line | The one idea |
|---|---|---|---|---|
| 13 | `pydantic_basics` | Creates, converts and rejects data (no LLM) | `class Customer(BaseModel): ... Field(ge=0)` | Pydantic **converts** what it can (`"49.62"` → `49.62`) and **rejects** what it can't (`-5`) |
| 14 | `extraction` | Turns a messy customer message into a `SupportTicket` | `llm.with_structured_output(SupportTicket)` | The LLM returns a **real Python object**, so `if ticket.wants_refund:` works |
| 15 | `classifier` | Sorts messages into database_question / greeting / off_topic | `category: Literal["database_question", "greeting", "off_topic"]` | `Literal` = **routing**. This is the first node of our future LangGraph app. |
| 16 | `text2sql_structured` 🎯 | SQL comes back as a `SQLQuery` object; a validator blocks non-SELECT | `is_answerable: bool`, `@field_validator("sql")` | Give the model an **escape hatch** ("can't answer"), and enforce safety **in code** |

**Before and after 3.4:**
```
3.3:  "```sql\nSELECT ...```"  → regex cleanup → hope
3.4:  SQLQuery(is_answerable=True, tables_used=[...], sql="SELECT ...", explanation="...")
```

---

## Lesson 3.5: Tools / Function Calling

| # | Script | What it does | Key line | The one idea |
|---|---|---|---|---|
| 17 | `first_tool` | Turns functions into tools; inspects them | `@tool def multiply(a: float, b: float)` | **Name + docstring + type hints** = all the LLM knows about a tool |
| 18 | `bind_tools` | The LLM picks tools (or none, or two at once) | `llm.bind_tools([...])` → `msg.tool_calls` | The LLM only **requests** a tool. **Nothing has run yet.** |
| 19 | `tool_loop` ⭐ | The agent loop by hand: 12 × $0.99 → ₹ | `while` tool_calls: run tool → `ToolMessage` → call LLM again | **Agent = Think → Act → Observe → repeat** (ReAct). The most important script. |
| 20 | `text2sql_agent` 🎯 | The agent explores Chinook: list tables → get schema → run query | `list_tables`, `get_schema`, `run_query` tools | The agent fetches **only the tables it needs**. Errors come back as text. Safety sits **inside the tool**. |

**The loop from Script 19 (memorize this shape):**
```
messages = [question]
repeat (max N times):
    ai = llm_with_tools.invoke(messages)        THINK
    if no ai.tool_calls → return ai.text        DONE
    for each call: result = tool(args)          ACT
    messages += ToolMessage(result, call.id)    OBSERVE
```

**Bonus:** `create_agent(...)` does this loop for you, and it returns a **`CompiledStateGraph`**, which is a LangGraph graph. That's the bridge to Phase 5.

---

## How the Text2SQL idea evolved across the scripts

| Script | How SQL was produced | Weakness it exposed |
|---|---|---|
| 04 | We wrote the SQL by hand | The LLM only *described* the DB |
| 08 | LLM writes SQL from template + few-shot | Manual steps; markdown could break `db.run` |
| 12 | Chain: SQL → run → answer | Wrong column → crash; loose text output |
| 16 | Structured `SQLQuery` + validator | Validator error crashes the chain |
| 20 | Agent explores and self-corrects | Less predictable, more LLM calls |
| **app v0.2** | Chain + structured output + read-only DB + never-crash runner | See the "known limitations" list in notes/06 |

Each script fixed a weakness of the one before. **That's the story to tell your friend.**

---

## 🏢 The story version (for teaching)

**Setup:** You run the Chinook music store. Your boss asks *"Which genre earned the most money?"* The answer is in the **archive room** (the database). You hire a **brilliant expert** (the LLM) who sits in a **sealed room** and communicates only by **notes under the door**.

| Lesson | In the story | In the code |
|---|---|---|
| **3.1** | The expert has **amnesia** and forgets every note, so you **staple the whole conversation** each time. They charge **per word**. You can wait for the full reply (invoke), read over their shoulder (stream) or hand notes to 3 copies at once (batch). You slip them a **map of the archive**. | `llm.invoke(messages)`, `AIMessage`, tokens, schema |
| **3.2** | Instead of handwritten notes, you use a **printed form** with blanks. The map and rules are **pre-printed**. **Sample filled-in forms** are attached. The mailroom **rejects** a form with an empty blank. | `ChatPromptTemplate`, `.partial()`, few-shot, `KeyError` |
| **3.3** | A **conveyor belt**: form desk → expert → secretary (opens the envelope) → cleaner → archive clerk → writer. A **folder** travels along and each station **adds a page**. | `\|`, `StrOutputParser`, `RunnableLambda`, `.assign()` (→ LangGraph "State") |
| **3.4** | The expert must reply on a **strict form with boxes**, including a ☐ **"No, the archive doesn't have this"** box. A **checking clerk** inspects every box and tears up any form whose SQL says DELETE. A reception desk sorts mail into **3 trays**. | Pydantic, `is_answerable`, `@field_validator`, `Literal` routing |
| **3.5** | The expert gets a **phone with labeled speed-dial buttons** (list shelves / shelf details / run query). They still can't leave the room: they send **request slips**, **you** run the errand and return a **result slip with the same ticket number**. They decide the next call until they write the final answer. Max 8 errands. The archive clerk refuses destructive requests. | `@tool` + docstring, `bind_tools`, `tool_calls`, `ToolMessage(tool_call_id)`, agent loop, `max_steps`, safety in tools |
| **3.6** | Organize the office: each department in its own room. | `config`, `llm`, `db`, `prompts`, `schemas`, `chains`, `main` |

**The one sentence:**
> *"The LLM is a brilliant expert with amnesia locked in a room. Phase 3 is about writing it good forms, building an assembly line around it, making it fill in strict boxes, and finally giving it a phone so it can ask for what it needs, while we stay in control of what actually happens."*

---

## Self-check: can you answer these without looking?

1. What type does `llm.invoke()` return, and how do you get the text?
2. How does a chatbot "remember" if LLMs are stateless?
3. Name the 3 Runnable methods.
4. What does `.partial()` do, and why do we use it for the schema?
5. What are few-shot examples for?
6. What does `|` do? What does `.assign()` add on top of it?
7. What's the difference between `llm` and `llm.with_structured_output(X)`?
8. Why does `SQLQuery` have an `is_answerable` field?
9. When the LLM returns `tool_calls`, who runs the tool?
10. What are the 4 steps of the agent loop, and why does it need a step limit?

<details>
<summary>Answers</summary>

1. `AIMessage`; use `.text`.
2. The app resends the whole conversation every call.
3. `invoke`, `stream`, `batch`.
4. Pre-fills fixed values. The schema never changes between questions, so each call only needs `question`.
5. They show the LLM what a good answer looks like (format, style, JOIN patterns).
6. `|` passes one step's output as the next step's input. `.assign()` keeps the input dict and adds a new key, so later steps see everything.
7. `llm` returns an `AIMessage` (text). The structured version returns an instance of your Pydantic class `X`.
8. So the model can say "the DB doesn't have this" instead of inventing columns.
9. Your code. The LLM only requests it.
10. Think → Act → Observe → Repeat. Without a limit, a confused agent could loop forever and waste tokens.
</details>
