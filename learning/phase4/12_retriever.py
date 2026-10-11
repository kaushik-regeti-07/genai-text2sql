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