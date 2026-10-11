from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

pages = PyPDFLoader("learning/phase4/data/chinook_handbook.pdf").load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,          # max characters per chunk
    chunk_overlap=150,       # safety margin between chunks
    add_start_index=True,    # record where in the page each chunk starts
)
chunks = splitter.split_documents(pages)      # split_DOCUMENTS keeps the metadata!

print(f"{len(pages)} pages → {len(chunks)} chunks")
sizes = [len(c.page_content) for c in chunks]
print("Chunk size  min / avg / max:", min(sizes), sum(sizes) // len(sizes), max(sizes))

# Every chunk still knows where it came from
for c in chunks[:4]:
    print(f"\npage_label={c.metadata['page_label']}  start_index={c.metadata['start_index']}")
    print("  ", repr(c.page_content[:80]))

# Find the chunk about refunds
print("\n--- chunk containing '14 days' ---")
for c in chunks:
    if "14 days" in c.page_content:
        print(f"page_label={c.metadata['page_label']}:", c.page_content[:300])

# How does chunk_size change the count?
print("\n--- chunk_size experiment ---")
for size, overlap in [(300, 50), (800, 150), (2000, 200)]:
    n = len(RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
            .split_documents(pages))
    print(f"chunk_size={size:5}  overlap={overlap:3}  → {n} chunks")