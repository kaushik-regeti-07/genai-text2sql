# Lesson 4.5: Improving RAG (Measure, Then Fix)

> **Goal of this lesson:** stop *guessing* whether RAG works and start **measuring** it with a small test set. Then fix the two most common real problems: **follow-up questions** (query rewriting) and **settings** (chunk size and `k`). We also try **hybrid search** and find out honestly whether it helps.

**You will create:** 2 scripts + 1 optional bonus in `learning/phase4/` (16 to 18).
**New package (bonus only):** `rank-bm25`, for keyword search.
**Time:** about 2 hours.

---

## ⭐ Why this lesson matters

**Building RAG is easy. Making it *reliably* good is the real job.** In 4.4 we tested by eyeballing a few answers. That doesn't scale: change `chunk_size` and you don't know whether you fixed 3 questions and broke 5 others.

**This lesson's rule: measure → change one thing → measure again.**

**Everything here came from real experiments while building the lesson:**
| What we did | What we learned |
|---|---|
| Compared 3 chunk sizes × 3 values of `k` on 12 test questions | `k=1` fails often; **800 / k=3** got 12/12 |
| Asked follow-ups like *"Who has to approve it?"* | Retrieval returned the **welcome page**. Query rewriting fixed it. |
| Embedded test questions with the wrong method | Score dropped from **9/12 to 6/12**. The details matter! |
| Added keyword (BM25) + hybrid search | **Didn't help** on this small handbook. Measure before adding complexity. |
| Ran too many embedding calls | Hit the **free-tier limit (100/minute)**. Cache your embeddings. |
| Fed an LLM's output into the embedding model | Got a **server error (500)**. Found a real library bug, fixed with `str()`. |

**Three reasons it's foundational:**
1. **Evaluation is what separates demos from products.** Every serious RAG team keeps a test set and checks it after every change.
2. **Retrieval quality limits answer quality.** If the right chunk isn't retrieved, nothing downstream can fix it, so we measure retrieval **first**.
3. **Query rewriting is the bridge to conversational apps**, and to v0.5's memory in Text2SQL.

---

## 1. Key concepts

### 1.1 Measuring retrieval: hit rate

A **test set** = questions + the fact each one *should* find.
```python
("How long do customers have to ask for their money back?", "within 14 days")
```
**Hit** = the expected fact appears in at least one of the `k` retrieved chunks.
**Hit rate** = hits ÷ questions. Simple, fast and **no LLM needed**.

> 🎯 **Analogy:** a **practice exam** with an answer key. You don't judge a student by one lucky answer; you run the same 12 questions after every study change and compare scores.

> 💡 **Write test questions the way users talk, not the way the document is written.** "Ask for their money back" (user) vs. "request a full refund" (handbook). Copying the document's words makes the test too easy.

### 1.2 The follow-up problem

```
User:  "Can I work from home?"
Bot:   "Yes, up to two days per week with manager approval."
User:  "Who has to approve it?"     ← the retriever only sees THIS
                                       "it" = ??? → retrieves the Welcome page ❌
```
The **LLM** sees the chat history, but the **retriever** only sees the latest message. A follow-up full of pronouns ("it", "that", "what about...") has almost no meaning on its own.

**Fix: query rewriting.** Before retrieving, ask the LLM to turn the follow-up into a **standalone question** using the history:
```
"Who has to approve it?"  ──rewrite with history──►  "Who has to approve working from home?"  ──► retriever ✅
```

### 1.3 Keyword vs. semantic vs. hybrid search

| | Semantic (embeddings) | Keyword (BM25) | Hybrid |
|---|---|---|---|
| Matches | **Meaning** | **Exact words** | Both, merged |
| Good at | Paraphrases ("money back" ↔ "refund") | Codes, emails, IDs, names ("INV-2031", "AC/DC") | Mixed queries |
| Weak at | Rare exact strings, sometimes | Synonyms, rephrasing | Extra complexity |

**BM25** = the classic search-engine scoring: rewards chunks containing the query's words, especially **rare** words. **Reciprocal Rank Fusion (RRF)** merges two ranked lists: a chunk ranked high in **either** list rises to the top.

---

## 2. Hands-on

> Run each script from `F:\Project\Gen_AI` with `(.venv)` active.

### Script 16: `learning/phase4/16_evaluate_retrieval.py`: a practice exam for RAG

Compares **chunk size × k** on 12 test questions. It builds temporary **in-memory** stores (no `persist_directory`), so it doesn't touch your saved data.

```python
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
pages = PyPDFLoader("learning/phase4/data/chinook_handbook.pdf").load()

# Questions phrased like a USER would ask, plus the fact that must be found
TEST_SET = [
    ("How long do customers have to ask for their money back?", "within 14 days"),
    ("What percentage do agents get on sales?",                 "commission of 5%"),
    ("How much time off do full-time staff get each year?",     "20 days of paid vacation"),
    ("What should a customer do if they can't log in?",         "Reset password"),
    ("How much does one song cost?",                            "$0.99"),
    ("When can customers reach us by live chat?",               "8:00 to 18:00"),
    ("How long are invoices stored?",                           "kept for 7 years"),
    ("Can I work remotely?",                                    "two days per week"),
    ("What's the biggest refund an agent can approve alone?",   "up to $20"),
    ("How many songs fit in one playlist?",                     "up to 500 tracks"),
    ("Do gift cards have an expiry date?",                      "never expire"),
    ("What are the password rules for staff?",                  "at least 12 characters"),
]

# Embed every test question ONCE (as a query) and reuse the vectors.
# Free tier = 100 embedding requests per minute. Re-embedding the same questions
# for every experiment would hit that limit quickly.
question_vectors = {q: embeddings.embed_query(q) for q, _ in TEST_SET}

def joined_text(docs) -> str:
    return " ".join(d.page_content.replace("\n", " ") for d in docs)   # PDF line breaks → spaces

for chunk_size, overlap in [(300, 50), (800, 150), (2000, 200)]:
    chunks = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap) \
        .split_documents(pages)
    store = Chroma(
        collection_name=f"eval_{chunk_size}",                 # in-memory: no persist_directory
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )
    store.add_documents(chunks, ids=[f"c{i}" for i in range(len(chunks))])
    avg_len = sum(len(c.page_content) for c in chunks) // len(chunks)

    for k in [1, 3, 5]:
        misses = []
        for question, expected in TEST_SET:
            docs = store.similarity_search_by_vector(question_vectors[question], k=k)
            if expected not in joined_text(docs):
                misses.append(question)
        hits = len(TEST_SET) - len(misses)
        print(f"chunk={chunk_size:4} ({len(chunks):2} chunks)  k={k}:  {hits:2}/{len(TEST_SET)}"
              f"   context≈{avg_len * k:5} chars   misses={misses}")
```

**Results from our test run:**
```
chunk= 300 (35 chunks)  k=1:   9/12   context≈  236 chars   misses=[money back, can't log in, work remotely]
chunk= 300 (35 chunks)  k=3:  11/12   context≈  708 chars   misses=[work remotely]
chunk= 300 (35 chunks)  k=5:  12/12   context≈ 1180 chars
chunk= 800 (14 chunks)  k=1:   9/12   context≈  632 chars   misses=[can't log in, gift cards, password rules]
chunk= 800 (14 chunks)  k=3:  12/12   context≈ 1896 chars   ✅ our choice
chunk= 800 (14 chunks)  k=5:  12/12   context≈ 3160 chars
chunk=2000 ( 7 chunks)  k=1:   8/12   context≈ 1183 chars
chunk=2000 ( 7 chunks)  k=3:  11/12   context≈ 3549 chars   misses=[invoices stored]
chunk=2000 ( 7 chunks)  k=5:  12/12   context≈ 5915 chars
```

**How to read this table:**
- **`k=1` is risky everywhere** (8–9/12). One chunk leaves no margin for error.
- **Small chunks (300) need a bigger `k`**: each chunk holds less, so you need more of them.
- **Big chunks (2000) still missed "invoices stored" at k=3.** The fact was buried among many other topics, so the chunk's "blurry average" meaning didn't match.
- **800 / k=3 = 12/12 with about 1,900 characters of context.** 300 / k=5 also scores 12/12 with even less context. Both are good choices. **The cost column matters as much as the score:** 2000 / k=5 also scores 12/12 but sends **3× more text** to the LLM on every question.
- With only 12 questions, **one question = 8%**. Small test sets are noisy, so treat differences of 1 question carefully.

> 🔍 **A mistake we made (and measured).** To save API calls, we first embedded the test questions with `embed_documents` instead of `embed_query`. Result for 800 / k=1: **6/12 instead of 9/12.** Gemini embeds *questions* and *documents* differently (Lesson 4.1). Using the wrong method silently made retrieval worse, **with no error message**. That's why we test.

---

### Script 17: `learning/phase4/17_query_rewriting.py`: fix follow-up questions

```python
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
store = Chroma(
    collection_name="chinook_handbook",
    embedding_function=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"),
    persist_directory="learning/phase4/chroma_db",
    collection_metadata={"hnsw:space": "cosine"},
)
retriever = store.as_retriever(search_kwargs={"k": 3})

# The conversation so far
history = [
    HumanMessage("Can I work from home?"),
    AIMessage("Yes, employees can work from home up to two days per week with their manager's approval."),
]

# A small chain whose ONLY job is to rewrite, not answer
rewrite_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Rewrite the user's latest question as a standalone question that can be "
                   "understood without the chat history. Keep it short. Do NOT answer it. "
                   "If it is already standalone, return it unchanged."),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ])
    | llm
    | StrOutputParser()
)

def top_chunk(question: str) -> str:
    return retriever.invoke(question)[0].page_content[:70].replace("\n", " ")

for follow_up in ["How many days?",
                  "Who has to approve it?",
                  "And what about vacation?",
                  "How much does a music video cost?"]:      # already standalone
    standalone = str(rewrite_chain.invoke({"history": history, "question": follow_up}))
    #            ↑ str() is REQUIRED here (see the bug note below)
    print(f"\nFollow-up:  {follow_up!r}")
    print(f"  raw → top chunk:       {top_chunk(follow_up)!r}")
    print(f"  rewritten:             {standalone!r}")
    print(f"  rewritten → top chunk: {top_chunk(standalone)!r}")
```

**Results from our test run:**
```
Follow-up:  'How many days?'
  raw → top chunk:       '7. Pay, Commission and Benefits 7.1 Commission...'         (lucky guess)
  rewritten:             'How many days per week can employees work from home?'
  rewritten → top chunk: '7. Pay, Commission and Benefits 7.1 Commission...'         ✅

Follow-up:  'Who has to approve it?'
  raw → top chunk:       'first. This handbook explains how we work...'               ❌ Welcome page
  rewritten:             'Who has to approve working from home?'
  rewritten → top chunk: '7. Pay, Commission and Benefits 7.1 Commission...'         ✅

Follow-up:  'And what about vacation?'
  rewritten:             'What is the vacation policy?'                               ✅

Follow-up:  'How much does a music video cost?'
  rewritten:             'How much does a music video cost?'                          ✅ unchanged
```
(The section 7 chunk contains the work-from-home rule, the manager-approval rule **and** the 20 vacation days. We checked.)

**What to notice:**
- **"Who has to approve it?" alone retrieved the Welcome page.** After rewriting, it found the right section.
- **"How many days?" happened to work raw**, by luck: "days" matched the vacation text. Don't rely on luck. The rewritten version is right *for the right reason*.
- An **already-standalone question passes through unchanged**, thanks to the last instruction in the prompt.
- **Cost:** one extra (fast) LLM call per question. A common optimization is to skip the rewrite on the **first** message, since there's no history yet.
- **How it plugs into a chat app (Script 15):** keep a `history` list → rewrite → retrieve with the **rewritten** question → answer the **original** question with the context + history. In Phase 5, LangGraph will manage that history for us.

> 🐛 **A real bug we found: why `str()` is required.**
> `StrOutputParser` returns LangChain's `TextAccessor` (a subclass of `str`, Lesson 3.3). When we passed that **directly** to Gemini's embedding model, Google's server replied **`500 INTERNAL`** every time, while the *same text* as a plain `str` worked. It's a compatibility bug between the libraries.
> **Rule: when an LLM's text output is passed to an embedding model or retriever, wrap it in `str()`.**
> How we found it: the raw follow-ups always worked and the rewritten ones always failed. Same retriever, same kind of text, **different type**. Comparing what works with what fails is the core debugging skill.

---

### Script 18 (bonus, optional): `learning/phase4/18_hybrid_search.py`: keyword + semantic

Install first:
```powershell
uv pip install rank-bm25
```

```python
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
pages = PyPDFLoader("learning/phase4/data/chinook_handbook.pdf").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150).split_documents(pages)

semantic = Chroma(collection_name="hybrid_demo", embedding_function=embeddings,
                  collection_metadata={"hnsw:space": "cosine"})
semantic.add_documents(chunks, ids=[f"c{i}" for i in range(len(chunks))])
keyword = BM25Retriever.from_documents(chunks, k=5)       # pure word matching, no API calls

def rrf(ranked_lists, top=3, c=60):
    """Reciprocal Rank Fusion: score = sum of 1/(c + rank) across the lists."""
    scores, docs = {}, {}
    for ranked in ranked_lists:
        for rank, d in enumerate(ranked):
            scores[d.page_content] = scores.get(d.page_content, 0) + 1 / (c + rank + 1)
            docs[d.page_content] = d
    return [docs[t] for t in sorted(scores, key=scores.get, reverse=True)[:top]]

def hybrid(question: str):
    return rrf([semantic.similarity_search(question, k=5), keyword.invoke(question)])

for q in ["it@chinookcorp.com", "MPEG-4", "What should a customer do if they can't log in?"]:
    print(f"\nQ: {q!r}")
    print("  semantic:", [d.page_content[:40].replace("\n", " ") for d in semantic.similarity_search(q, k=3)])
    print("  keyword: ", [d.page_content[:40].replace("\n", " ") for d in keyword.invoke(q)[:3]])
    print("  hybrid:  ", [d.page_content[:40].replace("\n", " ") for d in hybrid(q)])
```

**Our honest result** (same 12 paraphrase questions + 5 exact-string questions like emails and addresses):
```
semantic   paraphrase 12/12   exact 5/5
keyword    paraphrase 11/12   exact 5/5     ← missed "can't log in" (the handbook says "Reset password")
hybrid     paraphrase 11/12   exact 5/5     ← no better than semantic alone here
```

**What to notice:**
- **Keyword search failed on the paraphrase** ("can't log in" shares no words with "Reset password"), just as predicted.
- **Hybrid didn't beat semantic** on this handbook. Why? It's small (14 chunks), and every exact string appears in only one chunk, so semantic search already found them.
- **Hybrid shines on large collections** full of IDs, product codes, error codes and names (thousands of similar documents where meaning alone can't separate "INV-2031" from "INV-2013").
- **The real lesson: don't add complexity because a blog said so. Measure it on *your* data.**

> 🧪 **Experiments:**
> 1. In Script 16, add 3 questions of your own (one **not** answerable, e.g. *"Do we sell concert tickets?"*). How should "hit" work for an unanswerable question?
> 2. In Script 16, try `chunk_size=500, chunk_overlap=100`. Where does it land in the table?
> 3. In Script 17, add a third history turn and a follow-up that refers back **two** turns.
> 4. In Script 17, remove `str()` around `rewrite_chain.invoke(...)` and watch the `500 INTERNAL` error. Now you'll recognize it.

---

## 3. 🎯 Points to stress + deeper knowledge

### Points to stress (the "aha" moments)

**1. Measure → change one thing → measure again.**
A 12-question test set showed that `k=1` is risky, that 800/k=3 works and that big chunks bury facts. **You can't see any of that by eyeballing a few answers.**

**2. Retrieval first.**
Hit rate needs no LLM, so it's fast and cheap. If retrieval misses, the answer **can't** be right. Fix retrieval before touching the prompt.

**3. Score vs. cost.**
2000/k=5 and 800/k=3 both score 12/12, but one sends 3× more text per question. Good engineering picks the **cheapest setting that passes**.

**4. Retrievers don't see the conversation.**
Follow-ups need **query rewriting** into standalone questions. *"Who has to approve it?"* went from the Welcome page to the right section.

**5. Silent failures are the dangerous ones.**
The wrong embedding method lost 25% of hits with **no error**. The `TextAccessor` bug at least crashed loudly. Your test set is what catches silent failures.

**6. Real-world limits shape design.**
Free tier: **100 embedding requests per minute**. Cache embeddings, batch requests and add retries for temporary server errors (`.with_retry()`).

### Deeper knowledge (sounds great in interviews)

| Topic | What to know |
|---|---|
| **Retrieval metrics** | **Hit rate / recall@k** (did the right chunk appear in the top k?), **MRR** (how high was it ranked?), **precision@k** (how much of what we retrieved was relevant?). Our hit rate is recall@k with one expected fact per question. |
| **Generation metrics** | **Faithfulness** (are the answer's claims in the context?) and **answer relevance** (does it address the question?). Usually measured with an **LLM-as-judge** (Phase 6, RAGAS / LangSmith). |
| **Golden test sets** | Real teams collect real user questions + verified answers and grow the set whenever a bug is found ("regression tests" for RAG). |
| **Query transformations** | **Rewriting** (follow-ups → standalone), **multi-query** (generate 3 phrasings, retrieve for each, merge) and **HyDE** (have the LLM write a *hypothetical answer*, then search with that, since answers look more like documents than questions do). |
| **Re-ranking** | Retrieve 20 candidates cheaply, then a **re-ranker** (a cross-encoder model or the LLM) re-scores them carefully and keeps the best 3. Often a big quality jump for little cost. |
| **Hybrid search** | BM25 + embeddings merged with **RRF**. Standard in production search (Elasticsearch, OpenSearch, Weaviate and Qdrant support it natively). Helps most with exact identifiers in large collections. |
| **Caching** | Cache embeddings of documents (already done by the vector store) **and** of frequent queries. Also respect rate limits with batching, retries and backoff. |

---

## 4. Common errors

| Error / symptom | Cause | Fix |
|---|---|---|
| `429 RESOURCE_EXHAUSTED ... embed_content_free_tier_requests, limit: 100` | More than 100 embedding requests in a minute | Embed each question **once** and reuse it; wait 1 minute; avoid loops that re-embed |
| `500 INTERNAL` when embedding an LLM's output | `TextAccessor` passed to Gemini embeddings | Wrap in `str(...)` |
| `500 INTERNAL` on random calls | Temporary Google server error | `.with_retry(stop_after_attempt=3)` on the model or retriever |
| Hit rate drops for no visible reason | Questions embedded with `embed_documents` instead of `embed_query` | Use `embed_query` for questions, always |
| `ImportError: Could not import rank_bm25` | Bonus package not installed | `uv pip install rank-bm25` |
| "Hit" never matches although the answer is there | PDF line breaks inside the expected phrase | Replace `\n` with spaces before checking (`joined_text`) |

---

## 5. Summary: what to tell your friend

1. **Measure RAG with a test set:** questions + the fact each should find → **hit rate**. No LLM needed.
2. **Change one thing, measure again.** We found `k=1` risky, 800/k=3 at 12/12, and big chunks bury facts.
3. **Pick the cheapest setting that passes.** Context size = cost on every question.
4. **Follow-ups break retrieval** because the retriever doesn't see the history. **Query rewriting** turns them into standalone questions.
5. **Hybrid search** (keyword + semantic) helps with exact codes in big collections. On our small handbook it **didn't help**, so measure first.
6. **Real-world gotchas:** embed questions with `embed_query`, wrap LLM output in `str()` before embedding, respect the 100/minute limit, add retries.

---

## 6. Quiz

1. What is "hit rate," and why can we measure it without an LLM?
2. Why should test questions be phrased differently from the document's wording?
3. 800/k=3 and 2000/k=5 both scored 12/12. Which would you choose, and why?
4. Why did *"Who has to approve it?"* retrieve the Welcome page? What fixed it?
5. Embedding test questions with `embed_documents` dropped the score from 9/12 to 6/12. Why is this kind of bug dangerous?
6. When does hybrid (keyword + semantic) search help most? Why didn't it help here?
7. What caused the `500 INTERNAL` error in query rewriting, and how did we find it?

<details>
<summary>Answers</summary>

1. The fraction of test questions whose expected fact appears in the top-k retrieved chunks. It only checks retrieval (text matching), so no answer generation is needed: fast and cheap.
2. Users don't use the document's exact words. Copying the wording tests keyword overlap rather than real understanding, which makes the test too easy.
3. 800/k=3: same score with about 3× less context, so it's cheaper and faster on every question, with less noise for the LLM.
4. The retriever only sees the latest message, and "it" carries no meaning. Rewriting with the history turned it into "Who has to approve working from home?", which retrieved the right section.
5. It raises no error. Retrieval just gets quietly worse. Only a test set reveals it.
6. Large collections with exact identifiers (codes, IDs, emails, names) where many documents look alike. Our handbook is tiny, and each exact string is in one chunk, so semantic search already found them.
7. `StrOutputParser` returns a `TextAccessor` (a `str` subclass), which Gemini's embedding API rejects. We found it by noticing that raw follow-ups always worked while rewritten ones always failed, then testing the type difference directly. Fix: `str(...)`.
</details>

---

**Next → Lesson 4.6: 🎯 Text2SQL v0.3.** Everything from Phase 4 comes together in the real project: index the **table descriptions**, retrieve only the relevant tables for each question, **expand with foreign keys** (fixing the AC/DC problem from 4.1), and send a much smaller schema to the SQL prompt. With an evaluation to prove it still works.
