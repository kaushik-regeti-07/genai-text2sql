from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
pages = PyPDFLoader("learning/phase4/data/chinook_handbook.pdf").load()

# Questions phrased like a USER would ask, plus the fact that must be found
TEST_SET = [
    ("How long do customers have to ask for their money back?", "within 14 days"),
    ("What percentage do agents get on sales?",                 "commission of 5%"),
    ("How much time off do full-time staff get each year?",     "20 days of paid vacation"),
    ("What should a customer do if they can't log in?",         "Reset password"),
    ("How much does one song cost?",                            "$0.99"),
    ("When can customers reach us by live chat?",               "8:00 to 18:00"),
    ("How long are invoices stored?",                           "kept for 7 years"),
    ("Can I work remotely?",                                    "two days per week"),
    ("What's the biggest refund an agent can approve alone?",   "up to $20"),
    ("How many songs fit in one playlist?",                     "up to 500 tracks"),
    ("Do gift cards have an expiry date?",                      "never expire"),
    ("What are the password rules for staff?",                  "at least 12 characters"),
]

# Embed every test question ONCE (as a query) and reuse the vectors.
# Free tier = 100 embedding requests per minute. Re-embedding the same questions
# for every experiment would hit that limit quickly.
question_vectors = {q: embeddings.embed_query(q) for q, _ in TEST_SET}

def joined_text(docs) -> str:
    return " ".join(d.page_content.replace("\n", " ") for d in docs)   # PDF line breaks → spaces

for chunk_size, overlap in [(300, 50), (800, 150), (2000, 200)]:
    chunks = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap) \
        .split_documents(pages)
    store = Chroma(
        collection_name=f"eval_{chunk_size}",                 # in-memory: no persist_directory
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )
    store.add_documents(chunks, ids=[f"c{i}" for i in range(len(chunks))])
    avg_len = sum(len(c.page_content) for c in chunks) // len(chunks)

    for k in [1, 3, 5]:
        misses = []
        for question, expected in TEST_SET:
            docs = store.similarity_search_by_vector(question_vectors[question], k=k)
            if expected not in joined_text(docs):
                misses.append(question)
        hits = len(TEST_SET) - len(misses)
        print(f"chunk={chunk_size:4} ({len(chunks):2} chunks)  k={k}:  {hits:2}/{len(TEST_SET)}"
              f"   context≈{avg_len * k:5} chars   misses={misses}")