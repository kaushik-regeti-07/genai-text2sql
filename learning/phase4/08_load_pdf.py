from langchain_community.document_loaders import PyPDFLoader

PDF_PATH = "learning/phase4/data/chinook_handbook.pdf"

loader = PyPDFLoader(PDF_PATH)
pages = loader.load()                       # one Document per page

print("Number of pages:", len(pages))       # 5
print("Type:", type(pages[0]).__name__)     # Document

# Look at page 2 (index 1)
page = pages[1]
print("\nSource:     ", page.metadata["source"])
print("page:       ", page.metadata["page"])         # 1  ← counts from 0!
print("page_label: ", page.metadata["page_label"])   # '2' ← the human page number
print("total_pages:", page.metadata["total_pages"])
print("\nText preview:\n", page.page_content[:300])

# How much text is on each page?
for p in pages:
    print(f"page_label {p.metadata['page_label']}: {len(p.page_content)} characters")