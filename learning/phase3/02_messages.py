from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

messages = [
    SystemMessage("You are a SQL tutor. Answer in one short sentence."),
    HumanMessage("What does JOIN do?"),
    AIMessage("JOIN combines rows from two tables using a related column."),
    HumanMessage("Give me a one-line example."),   # "example" of what? The history tells it.
]

response = llm.invoke(messages)
print(response.text)