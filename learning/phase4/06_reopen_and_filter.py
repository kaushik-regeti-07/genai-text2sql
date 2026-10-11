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