# Lesson 4.3: Loaders & Splitters

> **Goal of this lesson:** get a **real document** (a 5-page PDF) into LangChain, cut it into **chunks** of the right size, and index those chunks in Chroma **with page numbers**.
> We finish with a searchable handbook index, ready for "chat with a PDF" in Lesson 4.4. 🎯

**You will create:** 4 scripts in `learning/phase4/` (08 to 11).
**New package:** `pypdf` (reads PDF files).
**Practice document:** `learning/phase4/data/chinook_handbook.pdf` (already created for you; see below).
**Time:** about 1.5 hours.

---

## 0. Setup

**Install:**
```powershell
uv pip install pypdf
```

| Package | What it is |
|---|---|
| `pypdf` | A pure-Python library that opens PDF files and extracts their text. LangChain's `PyPDFLoader` uses it underneath. |

**The practice PDF:** `learning/phase4/data/chinook_handbook.pdf` is a **made-up "Chinook Music Store Employee Handbook"** (5 pages: a cover + 9 sections about prices, refunds, gift cards, support hours, commission, security...). It was written as HTML (`chinook_handbook.html`, same folder) and printed to PDF with Microsoft Edge. Open the PDF and skim it so you know what's inside. You'll be asking questions about it.

> 💡 Why a made-up document? The LLM has **never seen it** during training, so if it answers *"refunds within 14 days"* correctly, that knowledge **must** have come from retrieval, not from memory. That's the cleanest way to test RAG.

---

## ⭐ Why this lesson matters

**Real knowledge doesn't come as neat one-sentence `Document`s.** It comes as PDFs, web pages, Word files and wikis, often dozens of pages long. Before RAG can work, two jobs must happen:

```
📄 PDF file ──LOAD──► 5 page Documents ──SPLIT──► 14 chunk Documents ──EMBED + STORE──► Chroma
             (4.3)      (text + page no.)   (4.3)    (~600 chars each)        (4.2)
```

**Why not just embed whole pages, or the whole PDF?**
| Embed... | Problem |
|---|---|
| The whole PDF as one vector | One vector for 9 different topics = a blurry "average" meaning. Search can't tell the refund part from the commission part. And the whole thing goes into the prompt every time. |
| Whole pages | A page mixes several topics (page 3 has payments **and** refunds **and** gift cards **and** passwords). Still blurry. |
| **Chunks of a few paragraphs** ✅ | Each vector represents **one focused idea**, so search finds it precisely and the prompt gets only what's needed. |

> ✂️ **Analogy:** making **revision flashcards** from a textbook. One card per idea, small enough to read quickly but big enough to make sense alone. Each card has the **page number** written in the corner (metadata) so you can check the book. A card that's too small ("...within 14 days.") is useless; one that's too big (a whole chapter) isn't a flashcard anymore.

**Chunking is often the #1 reason RAG gives bad answers.** Get it wrong and the right information is either split across two chunks or buried in a big one. That's why it gets its own lesson.

---

## 1. Key concepts

### 1.1 Document loaders

A **loader** reads a source and returns a **list of `Document`s** (the same `Document` from Lesson 4.2).

| Loader | Reads | Package |
|---|---|---|
| `PyPDFLoader` | PDF files, **one Document per page** | `langchain-community` + `pypdf` |
| `TextLoader` | `.txt` files, one Document per file | `langchain-community` |
| `CSVLoader` | CSV, one Document per row | `langchain-community` |
| `WebBaseLoader` | Web pages | `langchain-community` + `beautifulsoup4` |
| `Docx2txtLoader` | Word files | `langchain-community` + `docx2txt` |

All loaders have the same interface: `.load()` → `list[Document]`, and `.lazy_load()` → one at a time (saves memory for huge files).

### 1.2 Text splitters

A **splitter** cuts long text into **chunks** with two settings:

| Setting | Meaning | Our value |
|---|---|---|
| `chunk_size` | Maximum characters per chunk | 800 |
| `chunk_overlap` | Characters repeated at the start of the next chunk | 150 |

**`RecursiveCharacterTextSplitter`** (the one everybody uses) tries to cut at the **most natural boundary** first:
```
1st choice:  "\n\n"   paragraph breaks
2nd choice:  "\n"     line breaks
3rd choice:  " "      spaces (between words)
last resort: ""       anywhere (mid-word)
```
It only falls back to a smaller boundary when a piece is still too big. That's why it's called "recursive": it keeps text that belongs together **together**.

### 1.3 Why overlap?

```
Without overlap:                         With overlap (the end of chunk 1 repeats):
[...refund within 14 days. Refunds are]  [...refund within 14 days. Refunds are]
[processed within 5 business days.]       [Refunds are processed within 5 business days.]
     ↑ "processed" what? Context lost.        ↑ Makes sense on its own ✅
```
Overlap is a **safety margin** so a sentence cut at a boundary still appears whole in at least one chunk.

---

## 2. Hands-on

> Run each script from `F:\Project\Gen_AI` with `(.venv)` active.

### Script 8: `learning/phase4/08_load_pdf.py`: load a PDF

```python
from langchain_community.document_loaders import PyPDFLoader

PDF_PATH = "learning/phase4/data/chinook_handbook.pdf"

loader = PyPDFLoader(PDF_PATH)
pages = loader.load()                       # one Document per page

print("Number of pages:", len(pages))       # 5
print("Type:", type(pages[0]).__name__)     # Document

# Look at page 2 (index 1)
page = pages[1]
print("\nSource:     ", page.metadata["source"])
print("page:       ", page.metadata["page"])         # 1  ← counts from 0!
print("page_label: ", page.metadata["page_label"])   # '2' ← the human page number
print("total_pages:", page.metadata["total_pages"])
print("\nText preview:\n", page.page_content[:300])

# How much text is on each page?
for p in pages:
    print(f"page_label {p.metadata['page_label']}: {len(p.page_content)} characters")
```

**Results from our test run:**
```
Number of pages: 5
Type: Document

Source:      learning/phase4/data/chinook_handbook.pdf
page:        1
page_label:  2
total_pages: 5

Text preview:
 1. Welcome to Chinook
Chinook is an online music store that sells individual tracks, full albums and music videos to
customers in 24 countries. The company was founded in 2009 in Calgary, Canada...

page_label 1: 175 characters     ← the cover
page_label 2: 2048 characters
page_label 3: 2196 characters
page_label 4: 1586 characters
page_label 5: 1977 characters
```

**What to notice:**
- A loader turns a file into the **same `Document` objects** you used in 4.2: text + metadata.
- ⚠️ **`page` counts from 0, `page_label` from 1.** Use `page_label` (or `page + 1`) when showing citations to humans.
- The metadata also contains PDF details like `producer`, `creator` and `creationdate`. Usually you only need `source` and the page.
- Extracted text has `\n` where the PDF had **line breaks**, not just paragraph ends. PDFs store text **by position on the page**, not as paragraphs, so extraction is never perfect.

---

### Script 9: `learning/phase4/09_split_text.py`: see how splitting works

Small text, small chunks, so you can **see** every cut.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = ("Customers may request a full refund within 14 days of purchase. "
        "Refunds are processed within 5 business days.\n\n"
        "Gift cards never expire. Gift cards cannot be exchanged for cash.\n\n"
        "Support is available Monday to Friday. Phone support starts at 9:00.")

# Tiny chunks on purpose, to make the behaviour visible
splitter = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=20)
print("Separators tried, in order:", splitter._separators)

print("\n--- WITH overlap (20) ---")
for i, chunk in enumerate(splitter.split_text(text)):
    print(f"[{i}] ({len(chunk)} chars) {chunk!r}")

print("\n--- WITHOUT overlap ---")
no_overlap = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=0)
for i, chunk in enumerate(no_overlap.split_text(text)):
    print(f"[{i}] ({len(chunk)} chars) {chunk!r}")
```

**Results from our test run:**
```
Separators tried, in order: ['\n\n', '\n', ' ', '']

--- WITH overlap (20) ---
[0] (75 chars) 'Customers may request a full refund within 14 days of purchase. Refunds are'
[1] (45 chars) 'Refunds are processed within 5 business days.'                 ← "Refunds are" repeated ✅
[2] (65 chars) 'Gift cards never expire. Gift cards cannot be exchanged for cash.'
[3] (68 chars) 'Support is available Monday to Friday. Phone support starts at 9:00.'

--- WITHOUT overlap ---
[0] (75 chars) 'Customers may request a full refund within 14 days of purchase. Refunds are'
[1] ...        'processed within 5 business days.'                           ← "processed" what? ❌
[2] (65 chars) 'Gift cards never expire. Gift cards cannot be exchanged for cash.'
[3] (68 chars) 'Support is available Monday to Friday. Phone support starts at 9:00.'
```

**What to notice:**
- **Paragraphs stayed whole.** The gift card and support paragraphs each became one clean chunk, because the splitter cut at `\n\n` first.
- The first paragraph was **too long** (110 chars > 80), so it was cut at a **space**, never mid-word.
- **Overlap rescued the meaning:** with overlap, chunk [1] starts with "Refunds are", so it can stand alone. Without it, chunk [1] starts with "processed...", and retrieved alone it doesn't say *what* is processed.
- Overlap is applied **only where a cut happens inside a paragraph**, not between cleanly separated paragraphs.

---

### Script 10: `learning/phase4/10_split_pdf.py`: chunk the handbook

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

pages = PyPDFLoader("learning/phase4/data/chinook_handbook.pdf").load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,          # max characters per chunk
    chunk_overlap=150,       # safety margin between chunks
    add_start_index=True,    # record where in the page each chunk starts
)
chunks = splitter.split_documents(pages)      # split_DOCUMENTS keeps the metadata!

print(f"{len(pages)} pages → {len(chunks)} chunks")
sizes = [len(c.page_content) for c in chunks]
print("Chunk size  min / avg / max:", min(sizes), sum(sizes) // len(sizes), max(sizes))

# Every chunk still knows where it came from
for c in chunks[:4]:
    print(f"\npage_label={c.metadata['page_label']}  start_index={c.metadata['start_index']}")
    print("  ", repr(c.page_content[:80]))

# Find the chunk about refunds
print("\n--- chunk containing '14 days' ---")
for c in chunks:
    if "14 days" in c.page_content:
        print(f"page_label={c.metadata['page_label']}:", c.page_content[:300])

# How does chunk_size change the count?
print("\n--- chunk_size experiment ---")
for size, overlap in [(300, 50), (800, 150), (2000, 200)]:
    n = len(RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
            .split_documents(pages))
    print(f"chunk_size={size:5}  overlap={overlap:3}  → {n} chunks")
```

**Results from our test run:**
```
5 pages → 14 chunks
Chunk size  min / avg / max: 175 632 792

page_label=1  start_index=0
   'Chinook Music Store\nEmployee Handbook\nVersion 3.2 · Effective 1 January 2026...'
page_label=2  start_index=0
   '1. Welcome to Chinook\nChinook is an online music store that sells individual...'
page_label=2  start_index=602        ← starts BEFORE the previous chunk ended (overlap!)
   'first. This handbook explains how we work, what customers can expect from us...'

--- chunk containing '14 days' ---
page_label=3: 3. Payments, Refunds and Gift Cards
3.1 Payment methods
Customers can pay by credit card (Visa, Mastercard, American Express)...

--- chunk_size experiment ---
chunk_size=  300  overlap= 50  → 35 chunks
chunk_size=  800  overlap=150  → 14 chunks
chunk_size= 2000  overlap=200  → 7 chunks
```

**What to notice:**
- **`split_documents`** (not `split_text`) **copies each page's metadata into its chunks**, so every chunk still knows its page. That's what makes citations possible.
- **`start_index=602`** for a chunk whose previous chunk was about 708 characters long: the overlap in action.
- **No chunk crosses a page boundary**, because each page is split separately. Usually fine, but a sentence that continues onto the next page gets cut.
- **The trade-off in numbers:** small chunks (300) → 35 precise but context-poor pieces. Large chunks (2000) → 7 context-rich but blurry pieces. 800/150 is a common, sensible start. **You tune it by testing** (Lesson 4.5).

---

### Script 11: `learning/phase4/11_index_handbook.py`: 🎯 load → split → index

The full **indexing pipeline**. After this, the handbook is searchable, ready for 4.4.

```python
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# 1. LOAD
pages = PyPDFLoader("learning/phase4/data/chinook_handbook.pdf").load()

# 2. SPLIT
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150, add_start_index=True)
chunks = splitter.split_documents(pages)

# 3. EMBED + STORE
store = Chroma(
    collection_name="chinook_handbook",
    embedding_function=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"),
    persist_directory="learning/phase4/chroma_db",
    collection_metadata={"hnsw:space": "cosine"},
)
store.reset_collection()      # start clean, so changing chunk_size never leaves old chunks behind
ids = [f"handbook_p{c.metadata['page_label']}_c{i}" for i, c in enumerate(chunks)]
store.add_documents(chunks, ids=ids)
print(f"Indexed {store._collection.count()} chunks from {len(pages)} pages")

# 4. TEST RETRIEVAL
for q in ["Can I get my money back after three weeks?",
          "How much commission do agents earn?",
          "What if a customer forgets their password?",
          "Do you sell vinyl records?"]:                 # ← NOT in the handbook!
    print(f"\nQ: {q}")
    for doc, score in store.similarity_search_with_relevance_scores(q, k=2):
        print(f"   {score:.3f}  [page {doc.metadata['page_label']}]  {doc.page_content[:90]!r}")
```

**Results from our test run:**
```
Indexed 14 chunks from 5 pages

Q: Can I get my money back after three weeks?
   0.663  [page 3]  '3. Payments, Refunds and Gift Cards\n3.1 Payment methods\nCustomers can pay by...'
   0.644  [page 3]  'payment method and are processed within 5 business days of approval...'

Q: How much commission do agents earn?
   0.644  [page 5]  '7. Pay, Commission and Benefits\n7.1 Commission\nSales Support Agents earn a commission of 5%...'
   0.594  [page 4]  'Team Roles\nManagement General Manager\nSales Sales Manager and three Sales...'

Q: What if a customer forgets their password?
   0.782  [page 3]  "For security reasons, employees can never see, set or send a customer's password..."
   0.708  [page 3]  'playlists, and each playlist can contain up to 500 tracks...'

Q: Do you sell vinyl records?
   0.642  [page 2]  'video files.\n2.2 Standard prices\nProduct Price (USD)\nStandard audio track $0.99...'
   0.630  [page 2]  'first. This handbook explains how we work...'
```

**What to notice:**
- ✅ **Correct chunks with correct pages.** Refunds → page 3, commission → page 5, password → page 3. Retrieval works **with no shared words** ("money back" ↔ "Refunds").
- ✅ **The page number travelled all the way:** PDF page → `Document` → chunk metadata → Chroma → search result. That's the citation chain for 4.4.
- ⚠️ **"Vinyl records" isn't in the handbook, yet search still returned results** with scores (0.64) **as high as real matches**. Remember 4.1: *there's always a top result.* In 4.4 the **prompt** must tell the LLM: *"if the context doesn't answer the question, say you don't know."*
- **`reset_collection()`** empties the collection before re-adding. Without it, changing `chunk_size` would leave **old chunks** mixed in with the new ones (the stale index bug from 4.2).
- Notice how the **embedding (search) step** and the **reading (answer) step** are separate. We have the first; Lesson 4.4 adds the second.

> 🧪 **Experiments:**
> 1. In Script 11, change to `chunk_size=300, chunk_overlap=50`, run again, and compare the refund results. Are they more precise? Do they still contain "14 days"?
> 2. Try `chunk_size=2000`. What happens to the password question's top result?
> 3. In Script 10, set `chunk_overlap=0` and look at a chunk boundary inside page 3. Is any sentence cut?
> 4. Put your own PDF (e.g. college notes) in `learning/phase4/data/`, change the path in Script 8 and look at the extracted text. Is it clean? (Scanned PDFs return **empty text**. Why?)

---

## 3. 🎯 Points to stress + deeper knowledge

### Points to stress (the "aha" moments)

**1. Indexing = Load → Split → Embed → Store.**
This pipeline runs **once** (or whenever documents change). Questions come later, against the stored chunks.

**2. Chunk = one focused idea.**
One vector per idea gives precise search. Too small → no context ("processed within 5 days" of *what*?). Too big → blurry meaning and wasted prompt space.

**3. Overlap is a safety margin.**
It makes sure a sentence cut at a boundary appears whole in at least one chunk.

**4. Metadata must survive the whole journey.**
`split_documents` copies page numbers into chunks → Chroma stores them → search returns them. Lose them anywhere and **you can't cite sources**.

**5. Chunking is the #1 RAG tuning knob.**
The *same* question can succeed or fail purely because of `chunk_size`. There's no universal best value: test on **your** documents and questions.

**6. Garbage in, garbage out.**
PDF text extraction is imperfect (odd line breaks, tables flattened, headers and footers mixed in). **Scanned PDFs are images** and return no text at all without OCR. **Always look at the extracted text** before blaming the LLM.

### Deeper knowledge (sounds great in interviews)

| Topic | What to know |
|---|---|
| **Characters vs. tokens** | `chunk_size` here counts **characters**. LLM limits are in **tokens** (about 4 characters per token in English). `RecursiveCharacterTextSplitter.from_tiktoken_encoder(...)` splits by tokens when that matters. |
| **Structure-aware splitting** | `MarkdownHeaderTextSplitter` and HTML header splitters cut at **headings** and store the heading in metadata ("Section 3.2 Refund policy"). That's often much better than blind character counts for well-structured docs. |
| **Semantic chunking** | Advanced: embed each sentence and cut where the **meaning changes**. More expensive, sometimes better. |
| **Parent-document retrieval** | Search with **small** chunks (precise) but give the LLM the **bigger** surrounding section (context). It gets the best of both sizes. |
| **Contextual chunk headers** | Prefix each chunk with its document title and section ("Chinook Handbook > 3.2 Refunds: ...") before embedding, so even small chunks carry their context. |
| **Tables & scanned PDFs** | Tables lose their structure when extracted (see the price table on page 2). Scanned PDFs need **OCR**. Specialized parsers (e.g. `unstructured`, Docling, LlamaParse) handle complex layouts. |
| **Incremental indexing** | Production systems re-index only **changed** documents (tracked by content hashes or modification dates) instead of resetting everything. |

---

## 4. Common errors

| Error | Cause | Fix |
|---|---|---|
| `ImportError: pypdf package not found` | `pypdf` not installed | `uv pip install pypdf` |
| `FileNotFoundError` / `ValueError: File path ... is not a valid file` | Wrong path or wrong working directory | Run from `F:\Project\Gen_AI`, and check that the PDF exists in `learning/phase4/data/` |
| Pages load with **empty** text | Scanned PDF (images, no text layer) | Needs OCR. Use a PDF with real text. |
| Citations off by one | Used `page` (0-based) instead of `page_label` | Show `page_label` or `page + 1` |
| Chunks have no page metadata | Used `split_text` on strings instead of `split_documents` | Use `split_documents(pages)` |
| Old chunks mixed with new results | Re-indexed after changing `chunk_size` without clearing | `store.reset_collection()` before adding |
| `UnicodeDecodeError` with `TextLoader` | File isn't UTF-8 / default encoding | `TextLoader(path, encoding="utf-8")` |

---

## 5. Summary: what to tell your friend

1. **Loaders** turn files (PDF, text, web, CSV...) into `Document`s. `PyPDFLoader` gives **one Document per page**, with page metadata.
2. **Splitters** cut long text into **chunks**. `RecursiveCharacterTextSplitter` cuts at paragraphs first, then lines, then words.
3. **`chunk_size`** = max length; **`chunk_overlap`** = safety margin so sentences aren't lost at cuts.
4. **`split_documents`** keeps metadata, so every chunk still knows its page.
5. **Indexing pipeline:** Load → Split → Embed → Store, done once.
6. Retrieval found the right chunks and pages, but **also returned results for a question the PDF can't answer**. The prompt must handle that (4.4).
7. Chunk size is the **#1 tuning knob**. There's no perfect value; test it.

---

## 6. Quiz

1. What does `PyPDFLoader(...).load()` return for a 5-page PDF?
2. What's the difference between `page` and `page_label` in the metadata?
3. Why don't we embed each whole page as one vector?
4. What does `RecursiveCharacterTextSplitter` try first when cutting, and why?
5. What is `chunk_overlap` for? Give an example of what goes wrong without it.
6. Why use `split_documents` instead of `split_text` for the PDF?
7. "Do you sell vinyl records?" returned chunks with score 0.64. Is that a problem? How will we handle it?
8. Why does Script 11 call `reset_collection()` before adding chunks?

<details>
<summary>Answers</summary>

1. A list of 5 `Document`s, one per page, each with the page text and metadata (source, page, page_label, total_pages...).
2. `page` counts from 0 (internal index). `page_label` is the human page number, counting from 1. Show `page_label` in citations.
3. A page mixes several topics, so its single vector is a blurry average and search can't pinpoint one idea. It would also put a lot of irrelevant text into the prompt.
4. Paragraph breaks (`\n\n`), then line breaks, then spaces, and only then anywhere. That keeps text that belongs together in the same chunk.
5. A safety margin: the end of one chunk repeats at the start of the next. Without it, "Refunds are" / "processed within 5 business days" splits into a chunk that doesn't say *what* is processed.
6. `split_documents` keeps each page's metadata (page number, source) on every chunk, which is needed for citations. `split_text` returns plain strings with no metadata.
7. Yes. Search always returns a "top result," even when nothing is relevant, and the scores can look as high as real matches. The answer prompt in 4.4 must tell the LLM to say "I don't know" when the context doesn't contain the answer.
8. So re-running with a different `chunk_size` doesn't leave old chunks in the collection mixed with the new ones (a stale index).
</details>

---

**Next → Lesson 4.4: Retrievers & the RAG chain.** We add the **G** in RAG: turn the store into a **retriever**, plug it into an LCEL chain (Phase 3 skills!) and build **"Chat with the Handbook"**, with answers that **cite their page numbers** and say *"I don't know"* when the handbook doesn't cover it.
