from langchain_text_splitters import RecursiveCharacterTextSplitter

text = ("Customers may request a full refund within 14 days of purchase. "
        "Refunds are processed within 5 business days.\n\n"
        "Gift cards never expire. Gift cards cannot be exchanged for cash.\n\n"
        "Support is available Monday to Friday. Phone support starts at 9:00.")

# Tiny chunks on purpose, to make the behaviour visible
splitter = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=20)
print("Separators tried, in order:", splitter._separators)

print("\n--- WITH overlap (20) ---")
for i, chunk in enumerate(splitter.split_text(text)):
    print(f"[{i}] ({len(chunk)} chars) {chunk!r}")

print("\n--- WITHOUT overlap ---")
no_overlap = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=0)
for i, chunk in enumerate(no_overlap.split_text(text)):
    print(f"[{i}] ({len(chunk)} chars) {chunk!r}")