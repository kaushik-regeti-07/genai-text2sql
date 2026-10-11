# Phase 4: RAG (Retrieval-Augmented Generation)

## 🗺️ Phase 4 overview

**The problem RAG solves:** an LLM only knows (1) what it learned during training and (2) what you put in the prompt. It doesn't know **your** documents, **your** company data or anything after its training cutoff. And you can't paste *everything* into the prompt: it's too big, too slow and too expensive.

**RAG = find the relevant pieces first, then give only those to the LLM.**
```
Question ──► RETRIEVE the most relevant pieces ──► AUGMENT the prompt with them ──► GENERATE the answer
```

> 📚 **Analogy:** an **open-book exam**. The student (LLM) is smart but hasn't memorized *your* textbook. RAG is the skill of **quickly flipping to the right pages** and answering from them, instead of guessing or reading the whole book during the exam.

**You've already seen the problem in Text2SQL:** we send all 11 tables (about 2,500 tokens) with every question. For a company database with 500 tables, that breaks. RAG lets us send **only the relevant tables**.

| Lesson | Topic | You build |
|---|---|---|
| **4.1** | **Embeddings**: turning meaning into numbers | Semantic search from scratch with numpy |
| 4.2 | Vector stores (Chroma): storing and searching embeddings | A persistent searchable store |
| 4.3 | Loaders & splitters: getting documents in and cutting them into chunks | Load and chunk a PDF |
| 4.4 | Retrievers & the RAG chain | 📄 **"Chat with a PDF"** with page citations |
| 4.5 | Improving RAG: why it fails and how to fix it | Experiments: chunk size, k, MMR, query rewriting |
| 4.6 | 🎯 **Text2SQL v0.3**: schema RAG | Retrieve only relevant tables + similar example queries |

**New packages in this phase** (installed in Lesson 4.2): `langchain-chroma` (vector store) and `pypdf` (reads PDFs).

---

# Lesson 4.1: Embeddings, Turning Meaning into Numbers

> **Goal of this lesson:** understand what an embedding is, **measure** how similar two sentences are, and build a tiny **semantic search** by hand.
> We finish by asking: *given a question, which Chinook tables are relevant?* That's the core idea of v0.3. 🎯

**You will create:** 4 scripts in `learning/phase4/` (01 to 04).
**New packages:** none. `GoogleGenerativeAIEmbeddings` is in `langchain-google-genai` (already installed), and `numpy` came with LangChain.
**Time:** about 1.5 hours.

---

## ⭐ Why this lesson matters

**Embeddings are the engine inside every RAG system.** Vector stores, retrievers and "chat with your PDF" are all built on one operation: *turn text into numbers, then find the numbers that are closest.* If you understand this lesson, the rest of Phase 4 is plumbing.

**Why normal keyword search isn't enough:**
```
Question:  "How long until I get my money back?"
Document:  "Refunds are processed within 5 business days."
Shared words: NONE   ← keyword search finds nothing
Embedding similarity: 0.748 (best match) ← finds it
```
Embeddings match by **meaning**, not by spelling.

**In our Text2SQL project:** a user asks *"Which genre earned the most money?"* and embeddings tell us the relevant tables are `Genre`, `InvoiceLine` and `Track`, **before** we call the LLM. That's how v0.3 will shrink the prompt.

**Three reasons it's foundational:**
1. **RAG = embeddings + search + a prompt.** You already know the prompt part (Phase 3).
2. **Same idea, many uses:** search, recommendations ("similar songs"), duplicate detection, clustering and choosing few-shot examples (dynamic few-shot, from Lesson 3.2).
3. **Knowing the limits** (we'll see some surprising ones) is what separates people who *use* RAG from people who can *debug* it.

---

## 1. What is an embedding?

An **embedding** is a list of numbers (a **vector**) that represents the **meaning** of a piece of text.

```
"Which genre sells the most?"  ──embedding model──►  [-0.0032, 0.0088, 0.0023, -0.0603, ... ]  (3,072 numbers)
```

**The key property:** texts with **similar meaning** get vectors that are **close together**.

> 🗺️ **Analogy: a map of meanings.** Imagine every sentence placed as a dot on a giant map. "Customers", "clients" and "buyers" are neighbours in one area; "album", "track" and "song" are in another. An embedding is the sentence's **GPS coordinates** on that map. Real embeddings use **3,072 directions** instead of 2, which is impossible to picture but works the same way.

### Embedding model vs. chat model

| | Chat model (Phase 3) | Embedding model (Phase 4) |
|---|---|---|
| Input | Messages | Text |
| Output | **New text** | **A list of numbers** |
| Job | Generate, answer, reason | Represent meaning for comparison |
| Our model | `gemini-3.5-flash-lite` | `gemini-embedding-001` |
| LangChain class | `ChatGoogleGenerativeAI` | `GoogleGenerativeAIEmbeddings` |

> 🔑 This is the reason we chose Gemini back in setup: **one key gives you both.** Groq has no embedding models.

### Two methods

| Method | Use it for | Returns |
|---|---|---|
| `embed_query(text)` | The **user's question** | One vector |
| `embed_documents([texts])` | The **documents** you'll search | A list of vectors |

(Why two? Some models embed questions and documents slightly differently to improve search. Always use the matching method.)

---

## 2. Measuring similarity: cosine similarity

To compare two vectors we use **cosine similarity**: it measures the **angle** between two arrows.

```
      ▲ B                  ▲ B
     /                     │
    / small angle          │  90° angle        A ◄────────► B
   /                       │                      opposite
  ●────► A                 ●────► A
 similarity ≈ 1         similarity ≈ 0         similarity ≈ -1
 (same meaning)         (unrelated)            (opposite)
```

**Formula** (you don't need to memorize it; numpy does it):
```
cosine(A, B) = (A · B) / (|A| × |B|)
```
- `A · B` (dot product): multiply matching numbers and add them all up
- `|A|` (length / norm): how long the arrow is

> 💡 Gemini's embeddings already have length 1 (we checked: `norm = 1.0`), so for them the cosine is just the dot product.

---

## 3. Hands-on

> Create the folder `learning/phase4/`. Run each script from `F:\Project\Gen_AI` with `(.venv)` active.

### Script 1: `learning/phase4/01_first_embedding.py`: see an embedding

```python
import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

vector = embeddings.embed_query("Which genre sells the most?")

print("Type:        ", type(vector).__name__)            # list
print("Length:      ", len(vector))                      # 3072
print("First 5:     ", [round(x, 4) for x in vector[:5]])
print("Vector norm: ", round(float(np.linalg.norm(vector)), 4))   # 1.0
```

**Results from our test run:**
```
Type:         list
Length:       3072
First 5:      [-0.0032, 0.0088, 0.0023, -0.0603, 0.0109]
Vector norm:  1.0
```

**What to notice:**
- A sentence became **3,072 numbers**. Individual numbers mean nothing to humans; only **comparisons** between vectors are meaningful.
- **Every** text gets the same length, whether it's one word or a paragraph.
- The call took under a second, and embedding is **much cheaper** than a chat call.

---

### Script 2: `learning/phase4/02_similarity.py`: measure meaning

```python
import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

def cosine(a, b) -> float:
    """1 = same direction (similar meaning), 0 = unrelated."""
    a, b = np.array(a), np.array(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))   # @ = dot product

pairs = [
    ("How many customers do we have?", "What is the total number of clients?"),   # same meaning
    ("How many customers do we have?", "Count the rows in the Customer table"),   # related
    ("How many customers do we have?", "Which rock album is the longest?"),       # different topic
    ("I love this song",               "This track is amazing"),                  # same feeling
    ("I love this song",               "I hate this song"),                       # OPPOSITE feeling
    ("bank of a river",                "bank that gives loans"),                  # same word, different meaning
]

# Embed all 12 sentences in ONE call (faster and cheaper than 12 calls)
flat = [sentence for pair in pairs for sentence in pair]
vectors = embeddings.embed_documents(flat)

for i, (a, b) in enumerate(pairs):
    score = cosine(vectors[2 * i], vectors[2 * i + 1])
    print(f"{score:.3f}   {a!r}  vs  {b!r}")
```

**Results from our test run:**
```
0.912   'How many customers do we have?'  vs  'What is the total number of clients?'
0.874   'How many customers do we have?'  vs  'Count the rows in the Customer table'
0.735   'How many customers do we have?'  vs  'Which rock album is the longest?'
0.936   'I love this song'                vs  'This track is amazing'
0.893   'I love this song'                vs  'I hate this song'          ← 😮
0.783   'bank of a river'                 vs  'bank that gives loans'
```

**What to notice (these are important!):**
1. ✅ **"customers" ≈ "clients"** (0.912) with totally different words. That's semantic similarity.
2. ⚠️ **Even unrelated sentences score about 0.73.** Scores are **not percentages**. Only the **ranking** matters: 0.912 > 0.874 > 0.735. Never use a fixed rule like "above 0.8 = relevant" without testing on your own data.
3. 😮 **"I love this song" vs. "I hate this song" = 0.893.** Embeddings capture **what the text is about** (a song, feelings about it) much more than **negation or opinion**. So RAG can retrieve a document saying the **opposite** of what you need. The LLM reading it must still think.
4. ✅ **"bank" (river) vs. "bank" (money)** scored lower than real paraphrases. The model uses **context**, not just the word.

---

### Script 3: `learning/phase4/03_semantic_search.py`: search by meaning

This is **a complete search engine** in about 20 lines. It's what a vector store does for you (Lesson 4.2).

```python
import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

def cosine(a, b) -> float:
    a, b = np.array(a), np.array(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

# 1. Our "knowledge base": store policies
docs = [
    "Customers can pay by credit card or PayPal.",
    "Refunds are processed within 5 business days.",
    "We sell music tracks in MP3 and AAC formats.",
    "Our support team is available Monday to Friday.",
    "Playlists can contain up to 500 tracks.",
    "Employees earn a commission on every sale.",
]

# 2. INDEX: embed every document ONCE, ahead of time
doc_vectors = embeddings.embed_documents(docs)

# 3. SEARCH: embed the question, compare with every document, keep the best
def search(question: str, k: int = 2):
    q_vec = embeddings.embed_query(question)
    scored = [(cosine(q_vec, d_vec), doc) for d_vec, doc in zip(doc_vectors, docs)]
    scored.sort(reverse=True)            # highest score first
    return scored[:k]

for q in ["How long until I get my money back?",
          "What audio file types do you offer?",
          "Do staff get paid extra for selling?"]:
    print(f"\nQ: {q}")
    for score, doc in search(q):
        print(f"   {score:.3f}  {doc}")
```

**Results from our test run:**
```
Q: How long until I get my money back?
   0.748  Refunds are processed within 5 business days.
   0.639  Our support team is available Monday to Friday.

Q: What audio file types do you offer?
   0.768  We sell music tracks in MP3 and AAC formats.
   0.623  Customers can pay by credit card or PayPal.

Q: Do staff get paid extra for selling?
   0.790  Employees earn a commission on every sale.
   0.605  Our support team is available Monday to Friday.
```

**What to notice:**
- **All 3 best matches share ZERO words with the question.** money back ↔ Refunds, audio file types ↔ MP3/AAC, paid extra ↔ commission. Keyword search would have found **nothing**.
- The two phases of every search system:
  - **Indexing** (once, ahead of time): embed all the documents
  - **Querying** (each question): embed the question and compare
- `k` = how many results to return. You'll tune this in RAG.
- This compares the question with **every** document, which is fine for 6 but slow for 1 million. **Vector stores** (next lesson) solve that with smart indexes.

> 🧪 **Now add the "G" (generation):** take the top result and put it into a prompt: `"Answer using this info: {doc}\n\nQuestion: {q}"`, then send it to the chat model. **That's RAG!** Retrieve → Augment → Generate, which you can build with Phase 3 skills.

---

### Script 4: `learning/phase4/04_find_tables.py`: 🎯 which tables does a question need?

This is the **core idea of Text2SQL v0.3**: search **table descriptions** instead of store policies.

```python
import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

def cosine(a, b) -> float:
    a, b = np.array(a), np.array(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

# A plain-English description of each table. Written by us, for the embedding model.
TABLES = {
    "Artist":        "Musicians and bands. Columns: ArtistId, Name.",
    "Album":         "Music albums, each made by one artist. Columns: AlbumId, Title, ArtistId.",
    "Track":         "Individual songs: name, album, genre, media type, composer, "
                     "length in milliseconds, size, unit price.",
    "Genre":         "Music genres such as Rock, Jazz, Metal, Latin.",
    "MediaType":     "File formats of tracks such as MPEG audio or AAC.",
    "Playlist":      "Named playlists created in the store.",
    "PlaylistTrack": "Which tracks belong to which playlist (link table).",
    "Customer":      "Store customers: name, company, address, city, country, phone, email, "
                     "and their support rep (employee).",
    "Employee":      "Store staff: name, job title, manager (ReportsTo), hire date, "
                     "address, phone, email.",
    "Invoice":       "A customer purchase: date, billing address and country, total amount paid.",
    "InvoiceLine":   "Each track bought in an invoice, with unit price and quantity.",
}

names = list(TABLES.keys())
table_vectors = embeddings.embed_documents(list(TABLES.values()))   # index once

def relevant_tables(question: str, k: int = 5):
    q_vec = embeddings.embed_query(question)
    ranked = sorted(zip((cosine(q_vec, v) for v in table_vectors), names), reverse=True)
    return ranked[:k]

for q in ["Which genre earned the most money?",
          "Who is Jane Peacock's manager?",
          "Which country has the most customers?",
          "Longest songs by AC/DC"]:
    print(f"\nQ: {q}")
    print("   " + ", ".join(f"{name} ({score:.3f})" for score, name in relevant_tables(q)))
```

**Results from our test run:**
```
Q: Which genre earned the most money?
   Genre (0.630), InvoiceLine (0.575), Track (0.573), Album (0.561), MediaType (0.555)     ✅ top 3 = exactly right

Q: Who is Jane Peacock's manager?
   Employee (0.576), Customer (0.563), Artist (0.552), Genre (0.551), Playlist (0.539)   ✅ Employee first

Q: Which country has the most customers?
   Customer (0.694), Invoice (0.677), Employee (0.599), InvoiceLine (0.585), Artist (...) ✅

Q: Longest songs by AC/DC
   Track (0.597), Genre (0.581), MediaType (0.570), InvoiceLine (0.563), Playlist (...)  ❌ Artist and Album MISSING!
```

**What to notice:**
- ✅ It mostly works, **without any LLM call**. For the genre question, the top 3 are exactly the tables the agent in Lesson 3.5 chose.
- ❌ **The AC/DC question fails.** It needs `Track` → `Album` → `Artist` (to find AC/DC), but "AC/DC" is a **data value**, not a description, so nothing points to `Artist`. That's a real RAG weakness: **the question mentions a value, not the concept.**
- **The fix (v0.3):** after retrieving tables, **also add the tables they connect to** through foreign keys (`Track` → `Album` → `Artist`). Combining search with **known structure** is a common pro technique.
- **Descriptions matter.** We wrote them by hand. Better descriptions mean better retrieval. That's *context engineering* again, like field descriptions in Lesson 3.4.

> 🧪 **Experiments:**
> 1. In Script 2, add your own pairs: a Hindi sentence and its English translation. (Gemini embeddings are multilingual. What score do you get?)
> 2. In Script 3, ask something **no document answers**: *"Do you sell concert tickets?"* What's the top score, and why is that dangerous?
> 3. In Script 4, change the `Artist` description to *"Musicians and bands, e.g. AC/DC, Aerosmith, Queen. Columns: ArtistId, Name."* Does the AC/DC question improve?
> 4. In Script 4, ask *"best-selling artist in Brazil"* and count how many tables it needs vs. how many were retrieved.

---

## 4. 🎯 Points to stress + deeper knowledge

### Points to stress (the "aha" moments)

**1. An embedding is the GPS coordinates of meaning.**
Similar meaning → nearby coordinates. Search = "find the nearest dots."

**2. Semantic search beats keyword search at understanding.**
All three policy searches matched with **zero shared words**. That's why RAG uses embeddings.

**3. Scores are relative, not absolute.**
Unrelated sentences still scored about 0.73. **The ranking is what matters.** And there's **always** a "top result", even when nothing is relevant (experiment 2), so RAG can confidently retrieve the wrong thing.

**4. Embeddings capture topic, not truth or opinion.**
"love" vs. "hate" = 0.893. Retrieval finds what's *about* the same thing, which could say the opposite. The LLM must still read critically.

**5. Index once, query many times.**
Documents are embedded **ahead of time**. Each question needs only **one** embedding call. That's why RAG is fast and cheap per question.

**6. Retrieval can miss things that aren't described.**
"AC/DC" didn't point to `Artist`. Good descriptions and **structure-aware expansion** (foreign keys) fix a lot of this.

### Deeper knowledge (sounds great in interviews)

| Topic | What to know |
|---|---|
| **Dimensions** | `gemini-embedding-001` gives 3,072 numbers. More dimensions capture more nuance but cost more storage. Many models let you request **fewer** dimensions (e.g. 768) to trade a little quality for speed and space. |
| **Normalized vectors** | Gemini vectors have length 1, so cosine similarity = dot product, which is faster to compute. Vector databases use this trick. |
| **Same model for both sides** | Documents and questions **must** be embedded with the **same model**. Vectors from different models live on different "maps" and can't be compared. Changing models means **re-embedding everything**. |
| **Query vs. document embeddings** | Some models are trained so a *question* lands near its *answer*, not near other questions. That's why there are two methods. |
| **Hybrid search** | Real systems often combine **keyword search** (great for exact names, IDs, codes like "AC/DC" or "INV-2031") with **semantic search** (great for meaning). The weakness in Script 4 is exactly where keywords would help. |
| **Other uses** | Recommendations ("similar tracks"), duplicate detection, clustering customer feedback, **dynamic few-shot** (retrieve the most similar example questions: part of v0.3). |
| **Cost** | Embedding calls are much cheaper than chat calls, and indexing is a one-time cost. |

---

## 5. Common errors

| Error | Cause | Fix |
|---|---|---|
| `API key required` | `.env` not found | Run from `F:\Project\Gen_AI` |
| `404 ... model not found` | Wrong embedding model name | Use `gemini-embedding-001` |
| `429 RESOURCE_EXHAUSTED` | Too many embedding calls too fast | Embed in batches with `embed_documents([...])`, not one by one in a loop |
| `ValueError: shapes (3072,) and (768,) not aligned` | Comparing vectors from different models or dimension settings | Use the same model and settings for documents and queries |
| Scores all look similar (0.5–0.7) | That's normal for this model | Compare **rankings**, not raw scores |

---

## 6. Summary: what to tell your friend

1. **RAG** = retrieve relevant pieces → put them in the prompt → generate. An open-book exam.
2. An **embedding** = a list of numbers (3,072 for Gemini) that captures **meaning**.
3. **Similar meaning → nearby vectors.** We measure with **cosine similarity** (1 = same direction).
4. **Semantic search:** embed documents once, embed each question, return the closest documents.
5. It finds matches with **no shared words**, but scores are **relative**, it captures topic rather than opinion, and there's always a "top result."
6. For Text2SQL: embed **table descriptions** → retrieve relevant tables → smaller prompt. Missing JOIN tables are fixed with foreign-key expansion.

---

## 7. Quiz

1. What does an embedding model output, and how is that different from a chat model?
2. What does a cosine similarity of about 1 mean? About 0?
3. Unrelated sentences scored 0.735. Why is it wrong to treat that as "73% similar"?
4. "I love this song" vs. "I hate this song" scored 0.893. What does that teach us about RAG?
5. Why do we embed documents **once** ahead of time?
6. Why did the AC/DC question fail to retrieve `Artist`, and how will v0.3 fix it?
7. Can you compare a vector from Gemini with a vector from a different embedding model?

<details>
<summary>Answers</summary>

1. A list of numbers (a vector) representing meaning. A chat model outputs new text.
2. About 1 = pointing the same direction (very similar meaning). About 0 = unrelated.
3. Scores aren't percentages or probabilities. For this model even unrelated text scores about 0.7. Only the ranking between candidates is meaningful.
4. Embeddings capture what the text is *about*, not whether it agrees or disagrees. Retrieval can bring back text that says the opposite, so the LLM must read it critically.
5. Indexing is a one-time cost. Each question then needs only one embedding call plus fast comparisons.
6. "AC/DC" is a data value, not a concept in any description, so nothing pointed to `Artist`. v0.3 will add connected tables via foreign keys (Track → Album → Artist) and use better descriptions.
7. No. Each model has its own "map." Vectors from different models (or dimension settings) can't be compared. Use the same model for documents and queries.
</details>

---

**Next → Lesson 4.2: Vector Stores (Chroma).** Script 3 compared the question with *every* document in a Python list. A **vector store** saves the embeddings to disk, searches millions of them quickly and stores metadata (like page numbers) alongside them. We'll install `langchain-chroma`.
