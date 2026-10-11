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