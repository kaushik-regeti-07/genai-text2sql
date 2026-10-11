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