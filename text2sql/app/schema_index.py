"""Schema RAG (v0.3): find which tables a question needs, instead of sending all of them."""
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document

from text2sql.app.config import FK_HOPS, INDEX_DIR, TABLE_K
from text2sql.app.db import get_foreign_keys
from text2sql.app.llm import get_embeddings

# Written for the embedding model: use the words USERS say ("sales", "band", "songs").
TABLE_DESCRIPTIONS = {
    "Artist": "Artists: musicians and bands who made the albums, e.g. AC/DC, Aerosmith, Queen, "
              "Iron Maiden. Use for any question naming a band or singer. Columns: ArtistId, Name.",
    "Album": "Albums: each album belongs to one artist. Links Track to Artist. "
             "Columns: AlbumId, Title, ArtistId.",
    "Track": "Tracks (songs): name, composer, length in milliseconds, file size, unit price, "
             "and which album, genre and media type each song has.",
    "Genre": "Genres of music, e.g. Rock, Jazz, Metal, Latin, Blues. Columns: GenreId, Name.",
    "MediaType": "Media types / file formats of tracks, e.g. MPEG audio file, AAC audio file, "
                 "protected video.",
    "Playlist": "Playlists: named collections of tracks, e.g. Music, 90's Music, Grunge, Classical.",
    "PlaylistTrack": "Link table saying which tracks are in which playlist. "
                     "Needed to count or list tracks in a playlist.",
    "Customer": "Customers who buy music: name, company, address, city, state, country, phone, "
                "email, and their support rep (sales agent employee).",
    "Employee": "Employees / staff: name, job title (e.g. Sales Support Agent, manager), who they "
                "report to (ReportsTo = their manager), hire date, contact details. "
                "Sales agents support customers.",
    "Invoice": "Invoices = sales / purchases / orders: customer, invoice date, billing address and "
               "country, and the total amount paid. Use for revenue, spending and sales over time.",
    "InvoiceLine": "Invoice lines: each track sold on an invoice, with unit price and quantity. "
                   "Use for sales or revenue per track, album, artist or genre.",
}


@lru_cache
def get_table_index() -> Chroma:
    """Open the table index, rebuilding it if any description was added, removed or edited."""
    store = Chroma(
        collection_name="chinook_tables",
        embedding_function=get_embeddings(),
        persist_directory=str(INDEX_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )
    saved = store.get()
    if dict(zip(saved["ids"], saved["documents"])) != TABLE_DESCRIPTIONS:  # stale or empty
        store.reset_collection()
        store.add_documents(
            [Document(desc, metadata={"table": name}) for name, desc in TABLE_DESCRIPTIONS.items()],
            ids=list(TABLE_DESCRIPTIONS),
        )
    return store


def expand_with_foreign_keys(tables: set[str], hops: int = FK_HOPS) -> set[str]:
    """Add the tables these tables point to (Track -> Album -> Artist), `hops` steps deep."""
    fks = get_foreign_keys()
    selected, frontier = set(tables), set(tables)
    for _ in range(hops):
        frontier = {ref for t in frontier for ref in fks.get(t, [])} - selected
        selected |= frontier
    return selected


def find_tables(question: str, k: int = TABLE_K) -> list[str]:
    """The tables a question probably needs: top-k by meaning, plus foreign-key neighbours."""
    hits = get_table_index().similarity_search(str(question), k=k)  # str(): see Lesson 4.5
    return sorted(expand_with_foreign_keys({d.metadata["table"] for d in hits}))
