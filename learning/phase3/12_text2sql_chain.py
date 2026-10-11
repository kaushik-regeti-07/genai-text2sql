import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
db = SQLDatabase.from_uri("sqlite:///text2sql/data/chinook.db")

def clean_sql(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()

# ---------- Chain 1: question → SQL ----------
SQL_SYSTEM = """You are an expert {dialect} developer.
Given the database schema below, write ONE syntactically correct {dialect} query
that answers the user's question.

Rules:
- Only use tables and columns that exist in the schema.
- Only write SELECT queries. Never INSERT, UPDATE, DELETE or DROP.
- Unless the user asks for more, LIMIT results to {top_k} rows.
- Return ONLY the SQL. No explanation, no markdown.

Schema:
{schema}"""

sql_prompt = ChatPromptTemplate.from_messages([
    ("system", SQL_SYSTEM),
    ("human", "{question}"),
]).partial(dialect=db.dialect, schema=db.get_table_info(), top_k=5)

write_sql = sql_prompt | llm | StrOutputParser() | clean_sql

# ---------- Chain 2: (question + SQL + result) → English answer ----------
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You answer questions about a music store database. "
               "Use ONLY the SQL result below to answer. "
               "If the result is empty, say no data was found. Be concise."),
    ("human", "Question: {question}\nSQL query: {query}\nSQL result: {result}\n\nAnswer:"),
])
write_answer = answer_prompt | llm | StrOutputParser()

# ---------- The full pipeline: build up the dict step by step ----------
text2sql = (
    RunnablePassthrough.assign(query=write_sql)                     # + query
    .assign(result=lambda x: db.run(x["query"]))                     # + result
    .assign(answer=write_answer)                                     # + answer
)

# ---------- Try it ----------
out = text2sql.invoke({"question": "Which employee has the most customers?"})
print("Question:", out["question"])
print("SQL:     ", out["query"])
print("Result:  ", out["result"])
print("Answer:  ", out["answer"])

# batch works on the whole pipeline for free
print("\n--- batch ---")
for o in text2sql.batch([
    {"question": "How many tracks are longer than 5 minutes?"},
    {"question": "What are the 3 most expensive tracks?"},
]):
    print(f"\nQ: {o['question']}\nSQL: {o['query']}\nA: {o['answer']}")