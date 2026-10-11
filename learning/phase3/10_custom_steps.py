import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

# 1. A normal Python function: str in, str out.
def clean_sql(text: str) -> str:
    """Remove ```sql ... ``` markdown fences around a query."""
    text = text.strip()
    text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)  # opening fence
    text = re.sub(r"\s*```$", "", text)                                # closing fence
    return text.strip()

# Test the function alone first (always test pieces separately!)
print(repr(clean_sql("```sql\nSELECT 1;\n```")))    # 'SELECT 1;'

# 2. Explicitly wrapped with RunnableLambda...
to_upper = RunnableLambda(lambda s: s.upper())
print(to_upper.invoke("select 1;"))                  # SELECT 1;

# 3. ...or just drop the function into the chain. LangChain wraps it for you.
#    We deliberately ASK for markdown here, to prove clean_sql removes it.
prompt = ChatPromptTemplate.from_messages([
    ("system", "Write a SQLite query. Wrap it in a ```sql markdown block."),
    ("human", "{question}"),
])
chain = prompt | llm | StrOutputParser() | clean_sql

sql = chain.invoke({"question": "Count the rows in the Artist table"})
print(repr(sql))     # clean SQL, no ``` fences