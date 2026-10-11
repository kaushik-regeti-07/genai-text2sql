from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
store = Chroma(
    collection_name="chinook_handbook",
    embedding_function=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"),
    persist_directory="learning/phase4/chroma_db",
    collection_metadata={"hnsw:space": "cosine"},
)
retriever = store.as_retriever(search_kwargs={"k": 3})

def format_docs(docs) -> str:
    """Turn retrieved Documents into one text block, each labelled with its page."""
    return "\n\n".join(f"[page {d.metadata['page_label']}]\n{d.page_content}" for d in docs)

questions = [
    "How long do customers have to request a refund?",
    "What commission do Sales Support Agents earn?",
]

# ---------- 1. WITHOUT RAG: the model answers from memory ----------
print("=== WITHOUT RAG ===")
for q in questions:
    print(f"\nQ: {q}\nA: {llm.invoke(q + ' Answer in one sentence.').text}")

# ---------- 2. WITH RAG ----------
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant for Chinook Music Store employees. "
     "Answer the question using ONLY the context below. "
     "If the context does not contain the answer, say \"I don't know based on the handbook.\" "
     "Keep answers short.\n\nContext:\n{context}"),
    ("human", "{question}"),
])

rag_chain = (
    {"context": retriever | format_docs,       # question → docs → one string
     "question": RunnablePassthrough()}        # question passes through unchanged
    | prompt
    | llm
    | StrOutputParser()
)

print("\n=== WITH RAG ===")
for q in questions + ["Do you sell vinyl records?",
                      "Can I get my money back after three weeks?"]:
    print(f"\nQ: {q}\nA: {rag_chain.invoke(q)}")