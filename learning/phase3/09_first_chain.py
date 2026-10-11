from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a SQL tutor. Answer in one sentence."),
    ("human", "{question}"),
])

# 1. Build the chain: prompt → model → parser
chain = prompt | llm | StrOutputParser()
print("Chain type:", type(chain).__name__)     # RunnableSequence

# 2. invoke: dict in, string out (no more .text!)
answer = chain.invoke({"question": "What is a primary key?"})
print("Type:", type(answer).__name__)           # str
print(answer)

# 3. stream: works automatically on the whole chain
print("\nSTREAM:")
for piece in chain.stream({"question": "What is a database index?"}):
    print(piece, end="", flush=True)
print()

# 4. batch: also automatic
answers = chain.batch([
    {"question": "What is NULL in SQL?"},
    {"question": "What is a SQL view?"},
])
for a in answers:
    print("-", a)