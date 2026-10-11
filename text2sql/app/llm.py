"""Creates the AI models. The only file that knows which provider we use."""
from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from text2sql.app.config import EMBEDDING_MODEL, MODEL_NAME


@lru_cache  # create the model once, reuse it everywhere
def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=MODEL_NAME)


@lru_cache
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
