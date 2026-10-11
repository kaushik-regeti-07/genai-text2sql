import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

def cosine(a, b) -> float:
    a, b = np.array(a), np.array(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

# A plain-English description of each table. Written by us, for the embedding model.
TABLES = {
    "Artist":        "Musicians and bands. Columns: ArtistId, Name.",
    "Album":         "Music albums, each made by one artist. Columns: AlbumId, Title, ArtistId.",
    "Track":         "Individual songs: name, album, genre, media type, composer, "
                     "length in milliseconds, size, unit price.",
    "Genre":         "Music genres such as Rock, Jazz, Metal, Latin.",
    "MediaType":     "File formats of tracks such as MPEG audio or AAC.",
    "Playlist":      "Named playlists created in the store.",
    "PlaylistTrack": "Which tracks belong to which playlist (link table).",
    "Customer":      "Store customers: name, company, address, city, country, phone, email, "
                     "and their support rep (employee).",
    "Employee":      "Store staff: name, job title, manager (ReportsTo), hire date, "
                     "address, phone, email.",
    "Invoice":       "A customer purchase: date, billing address and country, total amount paid.",
    "InvoiceLine":   "Each track bought in an invoice, with unit price and quantity.",
}

names = list(TABLES.keys())
table_vectors = embeddings.embed_documents(list(TABLES.values()))   # index once

def relevant_tables(question: str, k: int = 5):
    q_vec = embeddings.embed_query(question)
    ranked = sorted(zip((cosine(q_vec, v) for v in table_vectors), names), reverse=True)
    return ranked[:k]

for q in ["Which genre earned the most money?",
          "Who is Jane Peacock's manager?",
          "Which country has the most customers?",
          "Longest songs by AC/DC"]:
    print(f"\nQ: {q}")
    print("   " + ", ".join(f"{name} ({score:.3f})" for score, name in relevant_tables(q)))