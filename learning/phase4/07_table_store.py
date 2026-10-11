from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

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

store = Chroma(
    collection_name="chinook_tables",              # a 2nd collection in the SAME folder
    embedding_function=embeddings,
    persist_directory="learning/phase4/chroma_db",
    collection_metadata={"hnsw:space": "cosine"},
)

# Build the index ONLY if it's empty: embed once, reuse forever.
if store._collection.count() == 0:
    print("Building table index (embedding 11 descriptions)...")
    docs = [Document(desc, metadata={"table": name}) for name, desc in TABLES.items()]
    store.add_documents(docs, ids=list(TABLES.keys()))      # table name = id
else:
    print("Index already exists, loaded", store._collection.count(), "tables from disk")

def relevant_tables(question: str, k: int = 3) -> list[str]:
    results = store.similarity_search_with_relevance_scores(question, k=k)
    for doc, score in results:
        print(f"     {doc.metadata['table']:14} {score:.3f}")
    return [doc.metadata["table"] for doc, _ in results]

for q in ["Which genre earned the most money?",
          "Who is Jane Peacock's manager?",
          "Longest songs by AC/DC"]:
    print(f"\nQ: {q}")
    print("   →", relevant_tables(q))