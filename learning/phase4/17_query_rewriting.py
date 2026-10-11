from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
store = Chroma(
    collection_name="chinook_handbook",
    embedding_function=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"),
    persist_directory="learning/phase4/chroma_db",
    collection_metadata={"hnsw:space": "cosine"},
)
retriever = store.as_retriever(search_kwargs={"k": 3})

# The conversation so far
history = [
    HumanMessage("Can I work from home?"),
    AIMessage("Yes, employees can work from home up to two days per week with their manager's approval."),
]

# A small chain whose ONLY job is to rewrite, not answer
rewrite_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Rewrite the user's latest question as a standalone question that can be "
                   "understood without the chat history. Keep it short. Do NOT answer it. "
                   "If it is already standalone, return it unchanged."),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ])
    | llm
    | StrOutputParser()
)

def top_chunk(question: str) -> str:
    return retriever.invoke(question)[0].page_content[:70].replace("\n", " ")

for follow_up in ["How many days?",
                  "Who has to approve it?",
                  "And what about vacation?",
                  "How much does a music video cost?"]:      # already standalone
    standalone = str(rewrite_chain.invoke({"history": history, "question": follow_up}))
    #            ↑ str() is REQUIRED here (see the bug note below)
    print(f"\nFollow-up:  {follow_up!r}")
    print(f"  raw → top chunk:       {top_chunk(follow_up)!r}")
    print(f"  rewritten:             {standalone!r}")
    print(f"  rewritten → top chunk: {top_chunk(standalone)!r}")