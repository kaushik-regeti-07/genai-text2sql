from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_community.utilities import SQLDatabase

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
db = SQLDatabase.from_uri("sqlite:///text2sql/data/chinook.db")

# ---------- 1. Few-shot examples: "here's how a good answer looks" ----------
examples = [
    {"question": "How many artists are there?",
     "sql": "SELECT COUNT(*) FROM Artist;"},
    {"question": "List all albums by AC/DC.",
     "sql": "SELECT a.Title FROM Album a JOIN Artist ar ON a.ArtistId = ar.ArtistId "
            "WHERE ar.Name = 'AC/DC';"},
]
# How ONE example is shown: the user asks, then the AI answers with SQL.
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{question}"),
    ("ai", "{sql}"),
])
few_shot = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# ---------- 2. The full Text2SQL prompt ----------
SYSTEM = """You are an expert {dialect} developer.
Given the database schema below, write ONE syntactically correct {dialect} query
that answers the user's question.

Rules:
- Only use tables and columns that exist in the schema.
- Only write SELECT queries. Never INSERT, UPDATE, DELETE or DROP.
- Unless the user asks for more, LIMIT results to {top_k} rows.
- Return ONLY the SQL. No explanation, no markdown.

Schema:
{schema}"""

text2sql_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    few_shot,                 # the examples go here, as human/ai message pairs
    ("human", "{question}"),
]).partial(
    dialect=db.dialect,           # "sqlite"
    schema=db.get_table_info(),   # read once, reused for every question
    top_k=5,
)
print("Only needs:", text2sql_prompt.input_variables)   # ['question']

# ---------- 3. Ask questions ----------
questions = [
    "How many customers are from Brazil?",
    "Which 3 genres have the most tracks?",
    "Who are the top 5 customers by total spending?",
]
for q in questions:
    filled = text2sql_prompt.invoke({"question": q})
    sql = llm.invoke(filled).text.strip()
    print("\nQ:  ", q)
    print("SQL:", sql)
    print("Result:", db.run(sql))    # ← running the LLM's SQL on the real database