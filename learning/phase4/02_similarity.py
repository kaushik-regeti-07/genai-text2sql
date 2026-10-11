import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

def cosine(a, b) -> float:
    """1 = same direction (similar meaning), 0 = unrelated."""
    a, b = np.array(a), np.array(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))   # @ = dot product

pairs = [
    ("How many customers do we have?", "What is the total number of clients?"),   # same meaning
    ("How many customers do we have?", "Count the rows in the Customer table"),   # related
    ("How many customers do we have?", "Which rock album is the longest?"),       # different topic
    ("I love this song",               "This track is amazing"),                  # same feeling
    ("I love this song",               "I hate this song"),                       # OPPOSITE feeling
    ("bank of a river",                "bank that gives loans"),                  # same word, different meaning
]

# Embed all 12 sentences in ONE call (faster and cheaper than 12 calls)
flat = [sentence for pair in pairs for sentence in pair]
vectors = embeddings.embed_documents(flat)

for i, (a, b) in enumerate(pairs):
    score = cosine(vectors[2 * i], vectors[2 * i + 1])
    print(f"{score:.3f}   {a!r}  vs  {b!r}")