from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
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
    return "\n\n".join(f"[page {d.metadata['page_label']}]\n{d.page_content}" for d in docs)

# ---------- The answer form: EVIDENCE FIRST, then the answer ----------
class HandbookAnswer(BaseModel):
    """An answer grounded in the handbook context."""
    quote: str = Field(
        description="The exact sentence from the context that answers the question, "
                    "or an empty string if no sentence answers it")
    found_in_context: bool = Field(
        description="True if quote is not empty; False if the context does not answer the question")
    answer: str = Field(
        description="Short answer based on the quote. "
                    "If found_in_context is False, write: I don't know based on the handbook.")
    pages: list[int] = Field(
        default_factory=list,
        description="Page numbers from the [page N] labels of the quoted text. Empty if not found.")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant for Chinook Music Store employees. "
               "Answer using ONLY the context.\n\nContext:\n{context}"),
    ("human", "{question}"),
])

rag_chain = (
    RunnablePassthrough.assign(docs=lambda x: retriever.invoke(x["question"]))   # + docs
    .assign(context=lambda x: format_docs(x["docs"]))                              # + context
    .assign(result=prompt | llm.with_structured_output(HandbookAnswer))           # + result
)

for q in ["How many vacation days do full-time employees get?",
          "Who approves a $50 refund?",
          "Can I get my money back after three weeks?",
          "Do you sell vinyl records?",
          "What is the CEO's name?"]:
    out = rag_chain.invoke({"question": q})
    r = out["result"]
    retrieved = sorted({d.metadata["page_label"] for d in out["docs"]})
    print(f"\nQ: {q}")
    print(f"   Answer:   {r.answer}")
    print(f"   Evidence: {r.quote!r}")
    print(f"   Cited pages: {r.pages}   (retrieved pages: {retrieved})")