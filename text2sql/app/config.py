"""Central settings for the Text2SQL app. Change things here, not all over the code."""
from pathlib import Path

from dotenv import load_dotenv

# This file is Gen_AI/text2sql/app/config.py, so parents[2] is the Gen_AI folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

DB_PATH = PROJECT_ROOT / "text2sql" / "data" / "chinook.db"
MODEL_NAME = "gemini-3.5-flash-lite"
TOP_K = 5  # default row limit for queries

# Schema RAG (v0.3)
EMBEDDING_MODEL = "gemini-embedding-001"
INDEX_DIR = PROJECT_ROOT / "text2sql" / "data" / "chroma_db"  # vector index of table descriptions
TABLE_K = 3   # how many tables to retrieve by meaning
FK_HOPS = 1   # then add tables they reference through foreign keys, this many steps away

# LangGraph (v0.4 / v0.5)
MAX_ATTEMPTS = 3  # how many times the graph may write SQL for one question (1 try + 2 fixes)
