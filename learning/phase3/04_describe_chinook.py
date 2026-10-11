from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.utilities import SQLDatabase

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

# 1. Connect to the database.
#    "sqlite:///" + path. The path is relative to where you RUN the script (the project root).
db = SQLDatabase.from_uri("sqlite:///text2sql/data/chinook.db")

print("Dialect:", db.dialect)                   # sqlite
print("Tables: ", db.get_usable_table_names())  # the 11 tables

# 2. Get the schema: CREATE TABLE statements + 3 sample rows per table.
schema = db.get_table_info()
print("Schema length:", len(schema), "characters")
print(schema[:800])                             # peek at the first part

# 3. We can also run SQL directly (we write the SQL ourselves, for now).
print("Customers from Brazil:", db.run("SELECT COUNT(*) FROM Customer WHERE Country = 'Brazil';"))

# 4. Give the schema to the LLM and ask it to explain the database.
response = llm.invoke([
    SystemMessage("You are a data analyst. Be brief."),
    HumanMessage(f"Here is a database schema:\n\n{schema}\n\n"
                 "In 5 bullet points, explain what this database is about "
                 "and how the tables connect."),
])
print("\n--- Gemini says ---")
print(response.text)
print("\nTokens used:", response.usage_metadata)