# Phase 4 Recap: RAG, as a Story + All 18 Scripts

> Use this page to revise, or to teach Phase 4 in one sitting. It continues the Phase 3 story (notes/07).

---

## 🏢 The story: the expert gets a library

**Previously (Phase 3):** the Chinook music store hired a brilliant **expert with amnesia** in a **sealed room** who communicates only through **notes under the door**. We gave them forms, an assembly line, strict answer boxes and a phone to the **archive room** (the database).

**The new problem:** employees ask *"Can a customer get their money back after three weeks?"* The answer is in the **Employee Handbook**, which the expert has **never read**. So they guess: *"30 days"*. ❌ (The handbook says **14 days**.) Not a lie, but the **typical** answer, said with confidence. That's **hallucination**.

| Lesson | In the story | In the code |
|---|---|---|
| **4.1 Embeddings** | A **librarian** gives every sentence **GPS coordinates on a map of meaning**. "Money back" lands next to "refunds". Quirks: "love" and "hate" sit on the same street; there's **always** a nearest page; "AC/DC" is a name, not a topic. | Embedding vectors (3,072 numbers), cosine similarity |
| **4.2 Vector stores** | A **filing cabinet** of numbered **index cards**: front = text (mapped), back = labels (page, topic). Map once, open instantly the next day. Drawers = filters. A card left over from an old policy = a stale index. | Chroma, `persist_directory`, metadata, filters, `ids`, upsert |
| **4.3 Loaders & splitters** | **Photocopy** the handbook page by page, **cut it into flashcards** (one idea each), write the **page number in the corner**, and repeat the last line on the next card. Too small = no context; too big = blurry. | `PyPDFLoader`, `RecursiveCharacterTextSplitter`, `chunk_size`, `chunk_overlap`, `split_documents` |
| **4.4 RAG chain** | The librarian slides the **3 best cards under the door with the question** and a rule: *"answer ONLY from these cards, or say I don't know."* The expert must **copy the exact line first** and **cite the page**. | `as_retriever`, `format_docs`, the RAG prompt, `quote` field first, page citations |
| **4.5 Improving RAG** | **Practice exams** with an answer key to tune card size and number of cards. A **receptionist** rewrites follow-ups ("Who approves *it*?" → "...working from home?"). Agency limit: 100 mappings a minute. A keyword index didn't help in a small library. | Test set + hit rate, query rewriting, rate limits, hybrid search (BM25 + RRF) |
| **4.6 Text2SQL v0.3** | Back to the archive: instead of sliding the map of **all 11 shelves** under the door, the librarian picks the **3 relevant shelves** by their **label cards** and follows **signposts** to connected shelves (Track → Album → Artist). **Better labels beat more signposts.** The "90’s Music" mystery: a curly apostrophe fooled the expert **and** our answer key. | Table descriptions, `find_tables`, foreign-key expansion, `eval_schema_rag.py` |

**Phase 3 vs. Phase 4:**
| | Phase 3 | Phase 4 |
|---|---|---|
| The expert's problem | Doesn't know **how** to work | Doesn't know **your facts** |
| What we gave them | Forms, assembly line, answer boxes, phone | Librarian, filing cabinet, flashcards |
| Main idea | Control **how** the LLM works | Control **what** the LLM knows |

**The one sentence:**
> *"Our expert is brilliant but has never read our documents, so they guess. RAG hires a librarian who finds the right pages by meaning and slides them under the door, and the expert has to answer only from those pages, quoting the line and citing the page."*

---

## 🗺️ How the 18 scripts build on each other

```
4.1  Meaning as numbers   01 → 02 → 03 → 04   ← semantic search by hand; find tables
4.2  Store it properly    05 → 06 → 07        ← Chroma, filters, persistent table index
4.3  Real documents       08 → 09 → 10 → 11   ← PDF → chunks with pages → indexed
4.4  Add generation       12 → 13 → 14 → 15   ← retriever, RAG vs. no RAG, citations, chat app
4.5  Measure and fix      16 → 17 → (18)      ← evaluation, query rewriting, hybrid (bonus)
4.6  Real project         text2sql v0.3       ← schema RAG + FK expansion + eval
```

| # | Script | The one idea |
|---|---|---|
| 01 | `first_embedding` | A sentence becomes **3,072 numbers**; only comparisons are meaningful |
| 02 | `similarity` | "customers" ≈ "clients" (0.91); **"love" ≈ "hate" (0.89)**; scores are relative |
| 03 | `semantic_search` | A search engine in 20 lines: matches with **zero shared words** |
| 04 | `find_tables` | Which Chinook tables a question needs. **AC/DC fails** (a name, not a topic) |
| 05 | `create_store` | Chroma: add **with ids**, search three ways; cosine **0.7476 = numpy's 0.748** |
| 06 | `reopen_and_filter` | Reopen from disk (no re-embedding), **filters**, add/read/update/delete |
| 07 | `table_store` | A persistent table index, built once (`if count == 0`) |
| 08 | `load_pdf` | PDF → one Document per page. `page` counts from 0, `page_label` from 1 |
| 09 | `split_text` | See every cut; **overlap** keeps "Refunds are..." meaningful |
| 10 | `split_pdf` | 5 pages → 14 chunks, each keeps its page; 300 → 35 chunks, 2000 → 7 |
| 11 | `index_handbook` | Load → split → index; "vinyl" still returns results (score 0.64!) |
| 12 | `retriever` | The retriever as a Runnable; **MMR** = variety; thresholds are fragile |
| 13 | `rag_vs_no_rag` | No RAG: "30 days", "no commission" ❌ → RAG: "14 days", "5%" ✅ |
| 14 | `rag_with_citations` | **Quote first**, then answer + page. Fixed the self-contradiction bug |
| 15 | `chat_with_handbook` | An interactive app; follow-ups fail (sets up 4.5) |
| 16 | `evaluate_retrieval` | Practice exam: chunk size × k. **800 / k=3 = 12/12** |
| 17 | `query_rewriting` | "Who has to approve it?": Welcome page ❌ → rewritten → right section ✅ |
| 18 | `hybrid_search` (bonus) | Keyword + meaning (RRF). **Didn't help** on a small handbook. Measure first. |

**Text2SQL v0.3 (notes/13):** `schema_index.py` with table descriptions + foreign-key expansion → **12/12 tables found, 48% smaller schema, 11/12 SQL correct (same as v0.2)**.

---

## 🐛 Real bugs we found in Phase 4 (each one is a lesson)

| Bug | Lesson |
|---|---|
| A structured answer said `found_in_context=True` **and** "I don't know" | Put **evidence first** (a `quote` field) |
| Free tier: **100 embedding requests per minute** | Cache embeddings; embed each question once |
| Questions embedded with `embed_documents` → **9/12 dropped to 6/12**, no error | Use `embed_query` for questions; silent failures need test sets |
| A `TextAccessor` sent to Gemini embeddings → `500 INTERNAL` | Wrap LLM output in `str()` before embedding |
| Chroma name `"g"` rejected | Collection names need 3+ characters |
| "90’s Music" (curly apostrophe): the answer key **and** the LLM both wrong → fake pass | Check suspicious numbers; compare exact values; "0 rows" is a warning sign |

---

## Self-check: can you answer these without looking?

1. What problem does RAG solve? What did Gemini say about refunds **without** RAG?
2. What is an embedding, and what does cosine similarity measure?
3. Why do "I love this song" and "I hate this song" score as similar?
4. What 4 things does a vector store keep for each item? Why use `ids`?
5. Why split documents into chunks instead of embedding whole pages?
6. What are the two most important rules in a RAG prompt?
7. Why should the model quote its evidence **before** answering?
8. How do you measure retrieval without using an LLM?
9. Why do follow-up questions break retrieval, and what fixes it?
10. In v0.3, how does the app find the `Artist` table for an AC/DC question?

<details>
<summary>Answers</summary>

1. LLMs don't know your private documents, so they guess plausible answers (hallucination). Without RAG, Gemini said customers have "30 days" (the handbook says 14) and that agents earn "no commission" (it's 5%).
2. A list of numbers representing meaning. Cosine similarity measures the angle between two vectors: about 1 = similar meaning, about 0 = unrelated.
3. Embeddings capture what text is *about* (feelings about a song), not whether it's positive or negative.
4. An id, the text, the metadata and the vector. Ids stop re-runs from creating duplicates (re-adding the same id replaces it).
5. A page mixes topics, so its vector is a blurry average. Chunks give one focused idea per vector, so search is precise and the prompt gets only what's needed.
6. "Answer using ONLY the context" and "If the context doesn't contain the answer, say I don't know."
7. Claims tied to evidence are more reliable. Quoting first stopped the model from claiming it found an answer it didn't have.
8. A test set of questions + expected facts, and the **hit rate**: does the expected fact appear in the top-k retrieved chunks?
9. The retriever only sees the latest message, and pronouns like "it" carry no meaning. Query rewriting turns follow-ups into standalone questions using the chat history.
10. Semantic search finds `Track` and `Album` (with good descriptions), then foreign-key expansion follows `Album → Artist`.
</details>
