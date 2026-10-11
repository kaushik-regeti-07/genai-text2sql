# Lesson 4.4: Retrievers & the RAG Chain: "Chat with the Handbook"

> **Goal of this lesson:** add the **G** in RAG. Turn the vector store into a **retriever**, plug it into an LCEL chain (Phase 3 skills) and build an assistant that answers from the handbook, **cites page numbers** and says **"I don't know"** when the handbook doesn't cover it. 🎯

**You will create:** 4 scripts in `learning/phase4/` (12 to 15).
**New packages:** none.
**Before you start:** run Script 11 (Lesson 4.3) so the `chinook_handbook` collection exists in `learning/phase4/chroma_db/`.
**Time:** about 2 hours.

---

## ⭐ Why this lesson matters

**This is the payoff of Phase 4.** Lessons 4.1–4.3 built the **R** (retrieve). Now we add **A** (augment the prompt) and **G** (generate), and get a working RAG system: the most common GenAI application in industry ("chat with your docs", support bots, internal knowledge assistants).

**Here is the problem RAG solves, in numbers from our own test.** We asked Gemini about the handbook **without** RAG:

| Question | Gemini **without** RAG | The handbook says | Gemini **with** RAG |
|---|---|---|---|
| How long do customers have to request a refund? | *"30 days from the date of purchase."* ❌ | **14 days** | *"within 14 days of purchase..."* ✅ |
| What commission do Sales Support Agents earn? | *"Typically do not earn commissions... fixed salary."* ❌ | **5%** | *"a commission of 5%..."* ✅ |

Without RAG, the model gave **confident, plausible, wrong** answers. It filled the gap with what's *typical* for companies. That's **hallucination**, and RAG is the main cure.

**Three reasons it's foundational:**
1. **Grounding:** answers come from **your** documents, not the model's guesses, and they can be **checked** through citations.
2. **It's pure Phase 3 + Phase 4:** a retriever is just another **Runnable**, so it snaps into LCEL with `|` and `.assign()` like everything else.
3. **The same pattern powers Text2SQL v0.3.** Swap "handbook chunks" for "table descriptions" and "answer" for "SQL", and it's the same chain.

---

## 1. Key concepts

### 1.1 Retriever

A **retriever** is a Runnable that takes a **question string** and returns a **list of `Document`s**.

```python
retriever = store.as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke("Can I get my money back?")   # → [Document, Document, Document]
```

> 🔑 Why not just call `store.similarity_search(...)`? Because a **retriever is a Runnable**: it has `invoke`/`batch`/`stream` and plugs into chains with `|`. A vector store method doesn't.

> 📚 **Analogy:** the vector store is the **library**. The retriever is the **librarian**: you ask a question, they bring back the 3 most useful pages. You can tell the librarian *how* to choose (`search_type`) and *how many* to bring (`k`).

### 1.2 Three search types

| `search_type` | How it chooses | Use when |
|---|---|---|
| `"similarity"` (default) | Top `k` closest chunks | Most of the time |
| `"mmr"` | Closest **but different from each other** (Maximal Marginal Relevance) | Results are near-duplicates and you want variety |
| `"similarity_score_threshold"` | Only chunks scoring **above** a minimum | You'd rather return nothing than something irrelevant |

### 1.3 The RAG chain shape

```
question
   │
   ├──► retriever ──► format_docs ──► context (one string with [page N] labels)
   │                                          │
   └──────────────────────────────────────────┤
                                              ▼
                                prompt: "Answer ONLY from this context: {context}
                                         If it isn't there, say I don't know.
                                         Question: {question}"
                                              │
                                              ▼
                                        llm ──► answer (+ cited pages)
```

The two **most important lines in the prompt:**
1. **"Answer using ONLY the context"** → stops the model from using its own (possibly wrong) knowledge.
2. **"If the context doesn't contain the answer, say I don't know"** → handles the "there's always a top result" problem from 4.1 and 4.3.

---

## 2. Hands-on

> Run each script from `F:\Project\Gen_AI` with `(.venv)` active.

### Script 12: `learning/phase4/12_retriever.py`: meet the librarian

```python
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()
store = Chroma(
    collection_name="chinook_handbook",
    embedding_function=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"),
    persist_directory="learning/phase4/chroma_db",
    collection_metadata={"hnsw:space": "cosine"},
)
print("Chunks in store:", store._collection.count())     # 14 (built by Script 11)

# 1. The basic retriever: a Runnable that returns Documents
retriever = store.as_retriever(search_kwargs={"k": 3})
print("Type:", type(retriever).__name__)                  # VectorStoreRetriever

docs = retriever.invoke("Can I get my money back after three weeks?")
print("Pages:", [d.metadata["page_label"] for d in docs])

# It's a Runnable, so batch works too
results = retriever.batch(["gift cards", "support hours"])
print("Batch pages:", [[d.metadata["page_label"] for d in r] for r in results])

# 2. similarity vs MMR
q = "refunds"
sim = store.as_retriever(search_kwargs={"k": 3})
mmr = store.as_retriever(search_type="mmr",
                         search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5})
print("\nSIMILARITY:")
for d in sim.invoke(q):
    print("  ", repr(d.page_content[:60]))
print("MMR:")
for d in mmr.invoke(q):
    print("  ", repr(d.page_content[:60]))

# 3. Score threshold: return NOTHING if nothing is good enough
for threshold in [0.6, 0.7]:
    thr = store.as_retriever(search_type="similarity_score_threshold",
                             search_kwargs={"k": 3, "score_threshold": threshold})
    for question in ["Do you sell vinyl records?",
                     "What if a customer forgets their password?"]:
        print(f"threshold {threshold}: {question!r} → {len(thr.invoke(question))} chunks")
```

**Results from our test run:**
```
Chunks in store: 14
Type: VectorStoreRetriever
Pages: ['3', '3', '4']
Batch pages: [['3', '3', '2'], ['4', '5', '4']]

SIMILARITY:
   '3. Payments, Refunds and Gift Cards\n3.1 Payment methods...'
   'payment method and are processed within 5 business days...'
   '5. Customer Support\n5.1 Opening hours...'
MMR:
   '3. Payments, Refunds and Gift Cards\n3.1 Payment methods...'
   'video files.\n2.2 Standard prices...'                        ← different topic
   'Manager, and finally the General Manager. Technical...'    ← different topic

threshold 0.6: 'Do you sell vinyl records?' → 3 chunks
threshold 0.6: 'What if a customer forgets their password?' → 3 chunks
threshold 0.7: 'Do you sell vinyl records?' → 0 chunks       ✅ nothing irrelevant
threshold 0.7: 'What if a customer forgets their password?' → 2 chunks
```

**What to notice:**
- **MMR traded relevance for variety.** It dropped the *second refund chunk* (which held "processed within 5 business days") to include different topics. Good for broad questions, risky for specific ones.
- **The threshold works for vinyl at 0.7**, but remember the refund question scored **0.663** in Lesson 4.3. At 0.7 it would return **nothing** for a question the handbook *does* answer. **Thresholds are fragile.** That's why we rely on the **prompt** ("say I don't know") as the main defence, not a magic number.
- `fetch_k=10` = MMR first grabs 10 candidates, then picks 3 varied ones. `lambda_mult` = 1 means pure relevance, 0 means pure variety.

---

### Script 13: `learning/phase4/13_rag_vs_no_rag.py`: proof that RAG matters

```python
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
store = Chroma(
    collection_name="chinook_handbook",
    embedding_function=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"),
    persist_directory="learning/phase4/chroma_db",
    collection_metadata={"hnsw:space": "cosine"},
)
retriever = store.as_retriever(search_kwargs={"k": 3})

def format_docs(docs) -> str:
    """Turn retrieved Documents into one text block, each labelled with its page."""
    return "\n\n".join(f"[page {d.metadata['page_label']}]\n{d.page_content}" for d in docs)

questions = [
    "How long do customers have to request a refund?",
    "What commission do Sales Support Agents earn?",
]

# ---------- 1. WITHOUT RAG: the model answers from memory ----------
print("=== WITHOUT RAG ===")
for q in questions:
    print(f"\nQ: {q}\nA: {llm.invoke(q + ' Answer in one sentence.').text}")

# ---------- 2. WITH RAG ----------
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant for Chinook Music Store employees. "
     "Answer the question using ONLY the context below. "
     "If the context does not contain the answer, say \"I don't know based on the handbook.\" "
     "Keep answers short.\n\nContext:\n{context}"),
    ("human", "{question}"),
])

rag_chain = (
    {"context": retriever | format_docs,       # question → docs → one string
     "question": RunnablePassthrough()}        # question passes through unchanged
    | prompt
    | llm
    | StrOutputParser()
)

print("\n=== WITH RAG ===")
for q in questions + ["Do you sell vinyl records?",
                      "Can I get my money back after three weeks?"]:
    print(f"\nQ: {q}\nA: {rag_chain.invoke(q)}")
```

**Results from our test run:**
```
=== WITHOUT RAG ===
Q: How long do customers have to request a refund?
A: Customers have 30 days from the date of purchase to request a refund.                    ❌ invented

Q: What commission do Sales Support Agents earn?
A: Sales Support Agents typically do not earn commissions, as they are usually paid a
   fixed salary or hourly wage...                                                            ❌ invented

=== WITH RAG ===
Q: How long do customers have to request a refund?
A: Customers can request a full refund within 14 days of purchase if the track or video
   has been downloaded no more than twice. After 14 days, refunds are only given when
   the file is faulty.                                                                       ✅

Q: What commission do Sales Support Agents earn?
A: Sales Support Agents earn a commission of 5% on the total value of all invoices
   from the customers assigned to them.                                                      ✅

Q: Do you sell vinyl records?
A: I don't know based on the handbook.                                                       ✅

Q: Can I get my money back after three weeks?
A: Yes, after 14 days, refunds are only given when the file is faulty...                    ⚠️
```

**What to notice:**
- **Without RAG: confident and wrong.** "30 days" and "no commission" sound reasonable, which makes them dangerous.
- **With RAG: correct, specific and grounded.** The extra details ("downloaded no more than twice") prove the answer came from the handbook.
- **"I don't know" worked for vinyl**, because the prompt gave the model **permission to not know**.
- ⚠️ **The last answer starts with "Yes,"**, which is misleading: after 3 weeks you get money back **only if the file is faulty**. Retrieval was perfect; the **wording** was sloppy. RAG fixes *knowledge*, not *reasoning*. (Script 14's design improves this.)
- **The chain shape:** a **dict** at the start of a chain runs its parts **in parallel** (Lesson 3.3: a dict becomes a `RunnableParallel`). `retriever | format_docs` builds the context while `RunnablePassthrough()` carries the question along.

---

### Script 14: `learning/phase4/14_rag_with_citations.py`: 🎯 answers with evidence and page numbers

Script 13 returns plain text: you can't tell **which page** a fact came from, and a "not found" is just a sentence. Here we use **structured output** (Lesson 3.4) so every answer comes with its **evidence**.

```python
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
store = Chroma(
    collection_name="chinook_handbook",
    embedding_function=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"),
    persist_directory="learning/phase4/chroma_db",
    collection_metadata={"hnsw:space": "cosine"},
)
retriever = store.as_retriever(search_kwargs={"k": 3})

def format_docs(docs) -> str:
    return "\n\n".join(f"[page {d.metadata['page_label']}]\n{d.page_content}" for d in docs)

# ---------- The answer form: EVIDENCE FIRST, then the answer ----------
class HandbookAnswer(BaseModel):
    """An answer grounded in the handbook context."""
    quote: str = Field(
        description="The exact sentence from the context that answers the question, "
                    "or an empty string if no sentence answers it")
    found_in_context: bool = Field(
        description="True if quote is not empty; False if the context does not answer the question")
    answer: str = Field(
        description="Short answer based on the quote. "
                    "If found_in_context is False, write: I don't know based on the handbook.")
    pages: list[int] = Field(
        default_factory=list,
        description="Page numbers from the [page N] labels of the quoted text. Empty if not found.")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant for Chinook Music Store employees. "
               "Answer using ONLY the context.\n\nContext:\n{context}"),
    ("human", "{question}"),
])

rag_chain = (
    RunnablePassthrough.assign(docs=lambda x: retriever.invoke(x["question"]))   # + docs
    .assign(context=lambda x: format_docs(x["docs"]))                              # + context
    .assign(result=prompt | llm.with_structured_output(HandbookAnswer))           # + result
)

for q in ["How many vacation days do full-time employees get?",
          "Who approves a $50 refund?",
          "Can I get my money back after three weeks?",
          "Do you sell vinyl records?",
          "What is the CEO's name?"]:
    out = rag_chain.invoke({"question": q})
    r = out["result"]
    retrieved = sorted({d.metadata["page_label"] for d in out["docs"]})
    print(f"\nQ: {q}")
    print(f"   Answer:   {r.answer}")
    print(f"   Evidence: {r.quote!r}")
    print(f"   Cited pages: {r.pages}   (retrieved pages: {retrieved})")
```

**Results from our test run:**
```
Q: How many vacation days do full-time employees get?
   Answer:   Full-time employees receive 20 days of paid vacation per year, plus Canadian public holidays.
   Evidence: 'Full-time employees receive 20 days of paid vacation per year, plus Canadian public holidays.'
   Cited pages: [5]

Q: Who approves a $50 refund?
   Answer:   The Sales Manager approves refunds above $20, including a $50 refund.
   Evidence: 'Refunds above $20 must be approved by the Sales Manager.'
   Cited pages: [3]                                      ← it REASONED: $50 > $20 ✅

Q: Can I get my money back after three weeks?
   Answer:   After 14 days, refunds are only given when the file is faulty...
   Evidence: 'After 14 days, refunds are only given when the file is faulty, for example...'
   Cited pages: [3]

Q: Do you sell vinyl records?
   Answer:   I don't know based on the handbook.
   Evidence: ''          found_in_context=False   Cited pages: []        ✅

Q: What is the CEO's name?
   Answer:   I don't know based on the handbook.          ✅ (the handbook never names anyone)
```

**What to notice:**
- **Every answer comes with its evidence** (`quote`) and **precise pages**. A user can open page 3 and check. **Citations build trust.**
- **Cited pages ≠ retrieved pages.** The retriever brought pages `[2, 3]`, but the answer used only page `3`. Showing *cited* pages is more honest than listing everything retrieved.
- **"$50 refund" needed reasoning:** the handbook says "above $20", not "$50". RAG gives the facts; the LLM applies them.

> 🔍 **A real bug we found while building this.** Our first version put `found_in_context` **first** in the form and had no `quote` field. For *"Do you sell vinyl?"* it returned:
> ```
> found_in_context=True   answer="I don't know based on the handbook."   ← contradicts itself!
> ```
> The model **claimed** it found the answer while saying it didn't. **Fix: make it show evidence first.** With `quote` as the first field, the model must copy a real sentence before deciding. No sentence → empty quote → `found_in_context=False`. It's the "field order matters" tip from Lesson 3.4, now with a real failure to prove it. **Self-reported flags from an LLM are unreliable unless they're tied to evidence.**

---

### Script 15: `learning/phase4/15_chat_with_handbook.py`: the interactive app

Script 14's chain wrapped in a small chat loop, like `text2sql/main.py`.

```python
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
store = Chroma(
    collection_name="chinook_handbook",
    embedding_function=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"),
    persist_directory="learning/phase4/chroma_db",
    collection_metadata={"hnsw:space": "cosine"},
)
retriever = store.as_retriever(search_kwargs={"k": 3})

def format_docs(docs) -> str:
    return "\n\n".join(f"[page {d.metadata['page_label']}]\n{d.page_content}" for d in docs)

class HandbookAnswer(BaseModel):
    """An answer grounded in the handbook context."""
    quote: str = Field(description="The exact sentence from the context that answers the question, "
                                   "or an empty string if no sentence answers it")
    found_in_context: bool = Field(description="True if quote is not empty; False otherwise")
    answer: str = Field(description="Short answer based on the quote. If found_in_context is False, "
                                    "write: I don't know based on the handbook.")
    pages: list[int] = Field(default_factory=list,
                             description="Page numbers from the [page N] labels of the quoted text")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant for Chinook Music Store employees. "
               "Answer using ONLY the context.\n\nContext:\n{context}"),
    ("human", "{question}"),
])
rag_chain = (
    RunnablePassthrough.assign(context=lambda x: format_docs(retriever.invoke(x["question"])))
    | prompt
    | llm.with_structured_output(HandbookAnswer)
)

print("📘 Chat with the Chinook Employee Handbook. Type 'exit' to quit.")
while True:
    question = input("\nYou: ").strip()
    if question.lower() in {"", "exit", "quit"}:
        break
    try:
        r = rag_chain.invoke({"question": question})
        print(f"Bot: {r.answer}")
        if r.found_in_context:
            print(f"     📄 Source: page {', '.join(map(str, r.pages))} | \"{r.quote}\"")
    except Exception as e:
        print(f"Bot: Something went wrong ({type(e).__name__}: {e})")
```

**Try asking:**
- *What payment methods do we accept?*
- *Can I work from home?*
- *How long do we keep invoices?*
- *Is there a limit on how many refunds one customer can get?*
- *Do we have a dress code?* (not in the handbook)
- *What about for videos?* ← a **follow-up**. It won't work well. Why? (Answer in Lesson 4.5.)

> 🧪 **Experiments:**
> 1. In Script 13, delete the line *"If the context does not contain the answer, say I don't know"* and ask about vinyl. What does the model do now?
> 2. In Script 13, set `k=1`, then ask *"What are the phone support hours and the email response time?"* (two facts, possibly in different chunks). Then try `k=5`.
> 3. In Script 14, use the `mmr` retriever from Script 12 instead. Does the refund answer get worse?
> 4. Ask Script 15 something tricky: *"Can an agent approve a $15 refund on their own?"* Check the quote and decide whether the reasoning is right.

---

## 3. 🎯 Points to stress + deeper knowledge

### Points to stress (the "aha" moments)

**1. RAG = Retrieve → Augment → Generate.**
Retriever finds chunks → the prompt is filled with them → the LLM answers from them. Every piece is a Runnable, joined with Phase 3 LCEL.

**2. Without RAG, LLMs fill gaps with "typical" answers.**
"30 days" and "no commission" weren't random; they're what's *common*. That's what makes hallucinations convincing. **Plausible ≠ true.**

**3. Two prompt rules do the heavy lifting.**
"Answer ONLY from the context" + "say I don't know if it's not there." Without the second rule, the model fills gaps again.

**4. Evidence first, then the answer.**
Making the model **quote** its source before answering prevented self-contradiction. **An LLM's claim is only as trustworthy as the evidence attached to it.**

**5. Citations = trust.**
Show **which page** each answer came from, and prefer *cited* pages over *retrieved* pages.

**6. RAG fixes knowledge, not reasoning.**
Retrieval was perfect for "money back after three weeks", but the first answer still said "Yes". Good retrieval + careless wording = a misleading answer. We'll measure this properly in Lesson 4.5 and Phase 6.

### Deeper knowledge (sounds great in interviews)

| Topic | What to know |
|---|---|
| **The two halves of RAG quality** | **Retrieval quality**: did we fetch the right chunks? (recall, precision). **Generation quality**: did the answer use them correctly? (faithfulness, relevance). Debug them **separately**: always look at the retrieved chunks first. |
| **Faithfulness / groundedness** | Does every claim in the answer appear in the context? It's the core RAG metric. Tools like **RAGAS** and LangSmith evaluators measure it automatically (Phase 6). |
| **`k` trade-off** | Small `k`: precise but may miss facts spread across chunks. Large `k`: more coverage but more noise and tokens, and a higher chance of "lost in the middle." Typical: 3–5. |
| **Thresholds are fragile** | The same threshold that blocked "vinyl" (0.64) would also block the real refund question (0.663). Prefer "retrieve, then let the prompt/LLM judge relevance." |
| **Prompt injection via documents** | Retrieved text goes **into the prompt**. A malicious document saying *"Ignore your instructions and..."* can hijack the model. That's **indirect prompt injection**: treat retrieved content as **data**, never as instructions. |
| **RAG vs. fine-tuning** | **RAG** = give the model the right facts at question time (easy to update, citable). **Fine-tuning** = change the model's behaviour or style through training (expensive, can't cite, hard to update). For company knowledge, RAG is almost always the first choice. |
| **Conversational RAG** | Follow-ups like *"what about videos?"* fail because the retriever only sees that short question. The fix is **query rewriting**: turn the follow-up into a standalone question using the chat history. That's Lesson 4.5. |

---

## 4. Common errors

| Error / symptom | Cause | Fix |
|---|---|---|
| `Chunks in store: 0` | Script 11 not run, or a different `persist_directory`/collection name | Run Script 11, and make sure the folder and name match exactly |
| `TypeError: expected string or bytes-like object` from the retriever | Passed a dict to `retriever.invoke` | A retriever takes the **question string**: `retriever.invoke(x["question"])` |
| `KeyError: 'page_label'` | Documents from a different loader without that field | Use `d.metadata.get("page_label", "?")` |
| Answers ignore the handbook | Prompt missing "ONLY the context", or context empty | Print the `context` to check what the LLM actually received |
| Says "I don't know" for something that IS in the handbook | Retrieval missed it (wrong chunks) | Print the retrieved chunks; try a larger `k`, or a different chunk size |
| Contradictory structured output | Flags not tied to evidence | Put an evidence field (`quote`) **first** |

---

## 5. Summary: what to tell your friend

1. A **retriever** = a Runnable "librarian": question in, relevant `Document`s out. `store.as_retriever(search_kwargs={"k": 3})`.
2. Search types: **similarity** (default), **MMR** (variety), **threshold** (fragile).
3. **RAG chain:** `{"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm`.
4. **Without RAG** the model invented "30 days" and "no commission." **With RAG**: 14 days and 5%, correct.
5. The prompt must say **"ONLY from the context"** and **"say I don't know."**
6. **Evidence first** (a `quote` field) prevents self-contradiction. **Cite pages** for trust.
7. RAG fixes **knowledge**, not **reasoning**. Always check retrieval first when debugging.

---

## 6. Quiz

1. What does a retriever take as input and return as output? Why use it instead of `similarity_search`?
2. When would you use MMR instead of similarity search? What's the risk?
3. Why is a score threshold a fragile way to block irrelevant questions? Use our numbers.
4. Without RAG, Gemini said refunds are allowed for "30 days." Why did it say that, and why is this dangerous?
5. Which two prompt instructions are most important in a RAG prompt?
6. Our first structured version returned `found_in_context=True` with "I don't know." What fixed it, and why?
7. Why show *cited* pages rather than all *retrieved* pages?
8. An answer is wrong. What should you check first: the prompt, the LLM or the retrieved chunks?

<details>
<summary>Answers</summary>

1. A question string in, a list of `Document`s out. It's a Runnable, so it plugs into LCEL chains with `|` and supports invoke/batch.
2. When results are near-duplicates and you want variety (broad questions). Risk: it can drop a highly relevant chunk to make room for a different topic, as it dropped the second refund chunk.
3. Relevance scores overlap: "vinyl" (irrelevant) scored about 0.64 while the real refund question scored 0.663. A threshold of 0.7 blocks both. No single number separates them reliably.
4. It filled the gap with what's typical for companies. It's dangerous because it sounds confident and plausible, so people believe it.
5. "Answer using ONLY the context" and "If the context doesn't contain the answer, say I don't know."
6. Adding a `quote` field **first**. The model must copy a real supporting sentence before setting `found_in_context`; no sentence means empty quote and `False`. Claims tied to evidence are more reliable.
7. The retriever often brings extra chunks that weren't used. Cited pages show where the answer really came from, which is more honest and easier to verify.
8. The retrieved chunks. If the right information wasn't retrieved, no prompt or model can answer correctly.
</details>

---

**Next → Lesson 4.5: Improving RAG.** Why RAG fails and how to fix it: follow-up questions (**query rewriting**), chunk size and `k` tuning with a mini test set, and **hybrid** keyword + semantic search for names like "AC/DC."
