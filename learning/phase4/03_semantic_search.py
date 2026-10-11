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