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