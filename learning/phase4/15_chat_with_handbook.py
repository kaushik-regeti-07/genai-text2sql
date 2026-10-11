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

class HandbookAnswer(BaseModel):
    """An answer grounded in the handbook context."""
    quote: str = Field(description="The exact sentence from the context that answers the question, "
                                   "or an empty string if no sentence answers it")
    found_in_context: bool = Field(description="True if quote is not empty; False otherwise")
    answer: str = Field(description="Short answer based on the quote. If found_in_context is False, "
                                    "write: I don't know based on the handbook.")
    pages: list[int] = Field(default_factory=list,
                             description="Page numbers from the [page N] labels of the quoted text")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant for Chinook Music Store employees. "
               "Answer using ONLY the context.\n\nContext:\n{context}"),
    ("human", "{question}"),
])
rag_chain = (
    RunnablePassthrough.assign(context=lambda x: format_docs(retriever.invoke(x["question"])))
    | prompt
    | llm.with_structured_output(HandbookAnswer)
)

print("📘 Chat with the Chinook Employee Handbook. Type 'exit' to quit.")
while True:
    question = input("\nYou: ").strip()
    if question.lower() in {"", "exit", "quit"}:
        break
    try:
        r = rag_chain.invoke({"question": question})
        print(f"Bot: {r.answer}")
        if r.found_in_context:
            print(f"     📄 Source: page {', '.join(map(str, r.pages))} | \"{r.quote}\"")
    except Exception as e:
        print(f"Bot: Something went wrong ({type(e).__name__}: {e})")