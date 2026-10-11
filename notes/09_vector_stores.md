# Lesson 4.2: Vector Stores (Chroma)

> **Goal of this lesson:** replace the hand-made Python-list search from Lesson 4.1 with a real **vector database** that saves embeddings to disk, searches fast, stores **metadata** and lets you **filter**.
> We finish by putting the Chinook table descriptions into a persistent store, the index that v0.3 will search. 🎯

**You will create:** 3 scripts in `learning/phase4/` (05 to 07).
**New package:** `langchain-chroma` (brings in `chromadb`, the actual database engine).
**Time:** about 1.5 hours.

---

## 0. Install

From `F:\Project\Gen_AI` with `(.venv)` active:
```powershell
uv pip install langchain-chroma
```
(It's also now listed in `requirements.txt`.) It installs quite a few dependencies; that's normal for a database engine.

| Package | What it is |
|---|---|
| `chromadb` | The vector database itself: stores vectors, builds a search index, saves to disk |
| `langchain-chroma` | The LangChain adapter, so Chroma works like any other LangChain vector store |

> 💡 Same pattern as Lesson 3.1: LangChain doesn't *include* databases, it **connects** to them. Swap Chroma for Pinecone, FAISS or PGVector later by changing the adapter; your search code barely changes.

---

## ⭐ Why this lesson matters

**Script 3 in Lesson 4.1 had three problems:**
| Problem in 4.1 | What a vector store does |
|---|---|
| Embeddings lived in a Python list and **vanished** when the script ended. Every run **re-embedded everything** (time + API calls). | **Saves to disk.** Embed once, reopen instantly (0.01 s in our test). |
| Compared the question with **every** document, one by one. Fine for 6, hopeless for 1 million. | Builds a **search index** (HNSW) that finds near neighbours **without checking everything**. |
| Just text. No way to say *"which page?"* or *"only search the HR docs."* | Stores **metadata** next to each vector (source, page, topic) and supports **filters**. |

**In our Text2SQL project:** the table descriptions get embedded **once** into a store on disk. Every question afterwards costs just **one** embedding call (the question) plus a millisecond search.

**Three reasons it's foundational:**
1. **Every production RAG system has one.** Chroma, Pinecone, Weaviate, Qdrant, PGVector, FAISS: same idea, different scale.
2. **Metadata is how you get citations.** "Answer from page 12 of policy.pdf" is only possible because the page number is stored **next to** the vector. That's Lesson 4.4.
3. **The `Document` object** introduced here is LangChain's standard unit of text. Loaders (4.3), splitters (4.3) and retrievers (4.4) all speak `Document`.

---

## 1. Key concepts

### 1.1 The `Document`: LangChain's standard unit of text

```python
from langchain_core.documents import Document

doc = Document(
    page_content="Refunds are processed within 5 business days.",   # the text (gets embedded)
    metadata={"topic": "billing", "source": "policy.txt"},         # labels (stored, NOT embedded)
)
```

> 📇 **Analogy:** a library **index card**. The front (`page_content`) is the text that gets "fingerprinted" for search. The back (`metadata`) has labels like shelf, page and author, used for **filtering and citing**, not for meaning.

### 1.2 What a vector store holds

For every document, Chroma saves **four things together**:
```
┌──────────┬────────────────────────────────┬──────────────────────┬─────────────────────────┐
│ id       │ text (page_content)            │ metadata             │ vector (3,072 numbers)  │
├──────────┼────────────────────────────────┼──────────────────────┼─────────────────────────┤
│ doc1     │ Refunds are processed within…  │ {topic: billing, …}  │ [-0.003, 0.008, …]      │
│ doc2     │ We sell music tracks in MP3…   │ {topic: catalog, …}  │ [0.011, -0.042, …]      │
└──────────┴────────────────────────────────┴──────────────────────┴─────────────────────────┘
```
**One folder (`persist_directory`) can hold many collections**, like one database holding many tables.

### 1.3 Distance vs. similarity (read this, it confuses everyone)

Chroma returns a **distance** (lower = closer), not a similarity (higher = closer).

| Setting | What `similarity_search_with_score` returns | Best match in our test |
|---|---|---|
| Default (`l2`, squared Euclidean) | Squared distance, **lower is better** | 0.5048 |
| **`cosine`** (we use this) | `1 − cosine similarity`, **lower is better** | **0.2524** |
| Any setting + `..._with_relevance_scores` | A 0–1 relevance score, **higher is better** | 0.7476 (with cosine) |

With cosine, `1 − 0.2524 = 0.7476`, **exactly the 0.748 you calculated with numpy in Lesson 4.1, Script 3.** Chroma does the same maths, just faster and saved to disk.

> 💡 **We always create collections with `collection_metadata={"hnsw:space": "cosine"}`** so the scores match what you learned in 4.1 and are easier to reason about. (For normalized vectors like Gemini's, L2 and cosine give the **same ranking**; only the numbers differ.)

---

## 2. Hands-on

> Run each script from `F:\Project\Gen_AI` with `(.venv)` active. They save data to `learning/phase4/chroma_db/` (already in `.gitignore`: it's generated data, not code).

### Script 5: `learning/phase4/05_create_store.py`: build and search a store

```python
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# 1. Documents = text + metadata labels
docs = [
    Document("Customers can pay by credit card or PayPal.",     metadata={"topic": "billing", "source": "policy.txt"}),
    Document("Refunds are processed within 5 business days.",  metadata={"topic": "billing", "source": "policy.txt"}),
    Document("We sell music tracks in MP3 and AAC formats.",   metadata={"topic": "catalog", "source": "faq.txt"}),
    Document("Our support team is available Monday to Friday.", metadata={"topic": "support", "source": "faq.txt"}),
    Document("Playlists can contain up to 500 tracks.",        metadata={"topic": "catalog", "source": "faq.txt"}),
    Document("Employees earn a commission on every sale.",     metadata={"topic": "staff",   "source": "hr.txt"}),
]

# 2. Open (or create) a collection saved to disk
store = Chroma(
    collection_name="store_policies",                 # 3+ characters, letters/digits/._-
    embedding_function=embeddings,                    # Chroma calls this to embed text
    persist_directory="learning/phase4/chroma_db",    # where it's saved
    collection_metadata={"hnsw:space": "cosine"},     # use cosine distance
)

# 3. Add documents WITH ids. Re-running the script replaces them instead of duplicating.
ids = [f"policy_{i}" for i in range(len(docs))]
store.add_documents(docs, ids=ids)
print("Documents in store:", store._collection.count())     # 6

question = "How long until I get my money back?"

# 4a. Plain search: returns Documents
print("\n--- similarity_search ---")
for doc in store.similarity_search(question, k=2):
    print(" ", doc.page_content, "|", doc.metadata)

# 4b. With DISTANCE (lower = closer)
print("\n--- with distance (lower is better) ---")
for doc, distance in store.similarity_search_with_score(question, k=3):
    print(f"  {distance:.4f}  {doc.page_content}")

# 4c. With RELEVANCE (higher = closer)
print("\n--- with relevance (higher is better) ---")
for doc, score in store.similarity_search_with_relevance_scores(question, k=3):
    print(f"  {score:.4f}  {doc.page_content}")
```

**Results from our test run:**
```
Documents in store: 6

--- similarity_search ---
  Refunds are processed within 5 business days. | {'source': 'policy.txt', 'topic': 'billing'}
  Our support team is available Monday to Friday. | {'source': 'faq.txt', 'topic': 'support'}

--- with distance (lower is better) ---
  0.2524  Refunds are processed within 5 business days.
  0.3608  Our support team is available Monday to Friday.
  0.3897  Customers can pay by credit card or PayPal.

--- with relevance (higher is better) ---
  0.7476  Refunds are processed within 5 business days.     ← same as numpy in 4.1!
  0.6392  Our support team is available Monday to Friday.
  0.6103  Customers can pay by credit card or PayPal.
```

**What to notice:**
- We never called `embed_documents` ourselves. **Chroma calls the embedding model for us** when adding documents and when searching.
- **Results come back as `Document`s with their metadata**, so you know where each answer came from.
- **Why `ids`?** Without them, Chroma gives each document a random id, so **every re-run adds 6 more copies** and your search results fill up with duplicates. With fixed ids, re-adding the same id **replaces** it (an "upsert").
- Look in `learning/phase4/chroma_db/`: you'll see `chroma.sqlite3` plus a folder. That's the database on disk.

---

### Script 6: `learning/phase4/06_reopen_and_filter.py`: persistence, filters, updates

Run this **after** Script 5. It opens the saved store **without re-embedding anything**.

```python
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# 1. REOPEN the store from disk. Same name + same folder = same data.
store = Chroma(
    collection_name="store_policies",
    embedding_function=embeddings,          # still needed: to embed new QUESTIONS
    persist_directory="learning/phase4/chroma_db",
    collection_metadata={"hnsw:space": "cosine"},
)
print("Loaded from disk:", store._collection.count(), "documents")   # 6, no re-embedding!

# 2. METADATA FILTERS: search only part of the store
print("\n--- only catalog docs ---")
for doc in store.similarity_search("What can I buy?", k=3, filter={"topic": "catalog"}):
    print(" ", doc.page_content)

print("\n--- only policy.txt or hr.txt ---")
for doc in store.similarity_search("money", k=5,
                                   filter={"source": {"$in": ["policy.txt", "hr.txt"]}}):
    print(" ", doc.page_content, "|", doc.metadata["source"])

# 3. READ by id
print("\n--- get by id ---")
print(store.get(ids=["policy_1"])["documents"])

# 4. UPDATE: adding with an EXISTING id replaces it ("upsert")
store.add_documents(
    [Document("Gift cards never expire.", metadata={"topic": "billing", "source": "policy.txt"})],
    ids=["policy_1"],
)
print("After upsert:", store.get(ids=["policy_1"])["documents"], "| count:", store._collection.count())

# 5. DELETE by id
store.delete(ids=["policy_5"])
print("After delete, count:", store._collection.count())

print("\nSearch for the updated doc:",
      store.similarity_search("Do gift cards expire?", k=1)[0].page_content)
```

**Results from our test run:**
```
Loaded from disk: 6 documents

--- only catalog docs ---
  We sell music tracks in MP3 and AAC formats.
  Playlists can contain up to 500 tracks.             ← only 2 results: only 2 catalog docs exist

--- only policy.txt or hr.txt ---
  Refunds are processed within 5 business days. | policy.txt
  Customers can pay by credit card or PayPal. | policy.txt
  Employees earn a commission on every sale. | hr.txt

--- get by id ---
['Refunds are processed within 5 business days.']
After upsert: ['Gift cards never expire.'] | count: 6
After delete, count: 5

Search for the updated doc: Gift cards never expire.
```

**What to notice:**
- **Persistence:** the data survived the script ending. The store reopened in about 0.01 s with **zero embedding calls** for the documents.
- `embedding_function` is still needed when reopening, but **only to embed new questions**.
- **Filters run *before* the similarity ranking.** `filter={"topic": "catalog"}` searched only 2 documents, so asking for `k=3` returned 2.
- **The full CRUD set:** **C**reate (`add_documents`), **R**ead (`get`), **U**pdate (add with the same id), **D**elete (`delete`). This is how you keep a store in sync when documents change.
- Run Script 5 again: it **resets** the 6 originals (same ids), and "Gift cards" is replaced by "Refunds…" again.

**Filter operators you can use:**
| Filter | Meaning |
|---|---|
| `{"topic": "billing"}` | equals |
| `{"topic": {"$ne": "staff"}}` | not equal |
| `{"source": {"$in": ["a.txt", "b.txt"]}}` | one of |
| `{"page": {"$gte": 10}}` | greater or equal (also `$gt`, `$lt`, `$lte`) |
| `{"$and": [{"topic": "billing"}, {"source": "policy.txt"}]}` | both conditions |

---

### Script 7: `learning/phase4/07_table_store.py`: 🎯 the Chinook tables in a vector store

This turns Script 4 from Lesson 4.1 into a **persistent index**, which is exactly what v0.3 will use.

```python
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

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

store = Chroma(
    collection_name="chinook_tables",              # a 2nd collection in the SAME folder
    embedding_function=embeddings,
    persist_directory="learning/phase4/chroma_db",
    collection_metadata={"hnsw:space": "cosine"},
)

# Build the index ONLY if it's empty: embed once, reuse forever.
if store._collection.count() == 0:
    print("Building table index (embedding 11 descriptions)...")
    docs = [Document(desc, metadata={"table": name}) for name, desc in TABLES.items()]
    store.add_documents(docs, ids=list(TABLES.keys()))      # table name = id
else:
    print("Index already exists, loaded", store._collection.count(), "tables from disk")

def relevant_tables(question: str, k: int = 3) -> list[str]:
    results = store.similarity_search_with_relevance_scores(question, k=k)
    for doc, score in results:
        print(f"     {doc.metadata['table']:14} {score:.3f}")
    return [doc.metadata["table"] for doc, _ in results]

for q in ["Which genre earned the most money?",
          "Who is Jane Peacock's manager?",
          "Longest songs by AC/DC"]:
    print(f"\nQ: {q}")
    print("   →", relevant_tables(q))
```

**Results from our test run** (first run builds; later runs say *"Index already exists"*):
```
Q: Which genre earned the most money?
     Genre          0.477
     InvoiceLine    0.399
     Track          0.397
   → ['Genre', 'InvoiceLine', 'Track']        ✅

Q: Who is Jane Peacock's manager?
   → ['Employee', 'Customer', 'Artist']       ✅ Employee first

Q: Longest songs by AC/DC
   → ['Track', 'Genre', 'MediaType']          ❌ still no Artist/Album (fixed in v0.3)
```

**What to notice:**
- **The same ranking as Lesson 4.1, Script 4.** Only the storage and search engine changed, not the maths.
- These relevance numbers look lower than in Script 5. **Never compare scores across different collections or questions.** Compare only rankings within one search.
- **`if count == 0` → build once.** The first run embeds 11 descriptions; every later run loads from disk. In v0.3 we'll add a **rebuild** option for when the descriptions change.
- **Table name as id + metadata** gives clean results (`doc.metadata["table"]`), with no string parsing.
- **Two collections, one folder:** `store_policies` and `chinook_tables` live side by side in `chroma_db/`.

> 🧪 **Experiments:**
> 1. Run Script 5 **without** `ids=ids` three times, then check the count. (Then delete the `chroma_db` folder and run Script 5 normally.)
> 2. Remove `collection_metadata={"hnsw:space": "cosine"}`, use a **new** collection name, and compare the distance numbers with Script 5's. Is the **ranking** the same?
> 3. In Script 6, combine a filter and a question: only `source = faq.txt` + *"Can I make a really long playlist?"*
> 4. In Script 7, change the Artist description to mention *"e.g. AC/DC, Aerosmith"*. ⚠️ It won't change anything until you **rebuild**. Why? (Hint: `if count == 0`.) Delete the collection with `store.delete_collection()`, run again, and compare.

---

## 3. 🎯 Points to stress + deeper knowledge

### Points to stress (the "aha" moments)

**1. A vector store = a database where you search by meaning.**
It stores **id + text + metadata + vector** together, saves them to disk and finds nearest neighbours fast.

**2. Embed once, search forever.**
Indexing is a one-time cost. After that, each question costs one embedding call plus a fast search. Reopening took 0.01 s.

**3. Metadata is not searched by meaning, but it's how you filter and cite.**
`page_content` gets embedded; `metadata` is the label on the back of the card. "Only HR docs" and "page 12 of policy.pdf" both come from metadata.

**4. Always use ids.**
No ids → duplicates on every re-run → the same chunk shows up 3 times in your results. Fixed ids give clean **upserts**.

**5. Distance vs. similarity: check which one you're getting.**
`with_score` → distance (**lower** is better). `with_relevance_scores` → relevance (**higher** is better). Mixing them up is a classic bug.

**6. A stale index is a silent bug.**
Change the source documents (or the table descriptions) without re-indexing, and search keeps returning the **old** content with no error. Index management is real engineering work.

### Deeper knowledge (sounds great in interviews)

| Topic | What to know |
|---|---|
| **ANN search (HNSW)** | Vector stores don't check every vector. They use **Approximate Nearest Neighbour** indexes like **HNSW** (a layered graph you "hop" through toward the closest points). It's very fast, and *approximately* right, which is a deliberate trade-off. |
| **Choosing a vector store** | **Chroma / FAISS**: local, great for learning and prototypes. **PGVector**: vectors inside PostgreSQL, nice if you already use Postgres. **Pinecone / Weaviate / Qdrant**: managed or scalable services for production. The LangChain interface is almost identical across them. |
| **Embedding model lock-in** | A store is tied to the embedding model that built it. Switching models = **re-embedding everything** (vectors from different models can't be compared). |
| **Upsert & sync** | Real systems track document versions or content hashes so only **changed** documents get re-embedded. Ids are the foundation of that. |
| **Metadata filtering for security** | In multi-user apps, filters enforce access control: `filter={"department": user.department}` so people only retrieve documents they're allowed to see. **Never rely on the LLM to hide restricted text.** Don't retrieve it in the first place. |
| **Scores aren't comparable across collections** | Relevance depends on the collection's contents and the question. Thresholds must be tuned **per collection** on real data. |

---

## 4. Common errors

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'langchain_chroma'` | Not installed, or `.venv` not active | `uv pip install langchain-chroma` |
| `InvalidArgumentError: Expected a name containing 3-512 characters...` | Collection name too short or has spaces | Use 3+ characters: letters, digits, `.`, `_`, `-` |
| Same result appears 2–3 times | Added documents without `ids`, multiple runs | Always pass `ids=`, then delete the `chroma_db` folder and rebuild |
| Search returns old content | Index built before you changed the text (`if count == 0` skipped the rebuild) | `store.delete_collection()` or delete the folder, then rebuild |
| Fewer results than `k` | A filter matched fewer documents than `k` | Expected: the filter runs before ranking |
| Telemetry or other warning lines in the output | Chroma's own logging | Harmless. Ignore them. |
| Scores all "wrong way round" | Mixed up distance (lower=better) and relevance (higher=better) | Check which method you called |

---

## 5. Summary: what to tell your friend

1. A **vector store** saves **id + text + metadata + vector** to disk and searches by meaning, fast.
2. A **`Document`** = `page_content` (gets embedded) + `metadata` (labels for filtering and citing).
3. **Chroma** is a local vector database. `langchain-chroma` connects it to LangChain.
4. Chroma **embeds for you** when adding and searching.
5. **Always use `ids`** (no duplicates, clean updates) and **cosine** distance (scores match 4.1).
6. `with_score` = distance (lower = better). `with_relevance_scores` = relevance (higher = better).
7. **Filters** limit the search to some documents. **Persistence** means embed once, reopen instantly.
8. For Text2SQL: the 11 table descriptions now live in a **persistent index**, ready for v0.3.

---

## 6. Quiz

1. What 4 things does a vector store keep for each document?
2. In a `Document`, which part gets embedded and which part is used for filtering?
3. Why must you pass `ids` when adding documents?
4. `similarity_search_with_score` returned 0.25 and 0.36. Which is the better match? What about `with_relevance_scores` returning 0.75 and 0.64?
5. When you reopen a saved store, why do you still need to pass `embedding_function`?
6. You edited the Artist description, but search results didn't change. Why?
7. How can metadata filters help with security in a company app?

<details>
<summary>Answers</summary>

1. An id, the text (`page_content`), the metadata and the vector.
2. `page_content` is embedded (searched by meaning). `metadata` is stored for filtering and citing, not embedded.
3. Without ids, every re-run adds duplicate copies. With fixed ids, adding the same id replaces it (upsert).
4. Distance: 0.25 is better (lower = closer). Relevance: 0.75 is better (higher = closer).
5. To embed new **questions** at search time. The stored documents are already embedded.
6. The index was built earlier and `if count == 0` skipped rebuilding, so the store still has the old vector. You must delete or rebuild the collection after changing source text.
7. Filter by the user's permissions (e.g. department) so restricted documents are never retrieved. That's safer than asking the LLM not to reveal them.
</details>

---

**Next → Lesson 4.3: Loaders & Splitters.** Real documents aren't one-sentence `Document`s. They're PDFs with 50 pages. We'll **load** a PDF (installing `pypdf`) and **split** it into chunks with page-number metadata, ready for "chat with a PDF."
