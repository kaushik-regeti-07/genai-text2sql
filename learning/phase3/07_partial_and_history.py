from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

# ---------- Feature 1: partial() = pre-fill some blanks ----------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Answer in one sentence."),
    ("human", "{question}"),
])
tutor_prompt = prompt.partial(role="SQL tutor")     # {role} is now fixed
print("Still needed:", tutor_prompt.input_variables)  # ['question']
print(llm.invoke(tutor_prompt.invoke({"question": "What is a foreign key?"})).text)

# ---------- Feature 2: MessagesPlaceholder = a slot for chat history ----------
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a SQL tutor. Answer in one short sentence."),
    MessagesPlaceholder("history"),     # ← a whole LIST of messages goes here
    ("human", "{question}"),
])

history = [
    HumanMessage("What does JOIN do?"),
    AIMessage("JOIN combines rows from two tables using a related column."),
]
filled = chat_prompt.invoke({"history": history, "question": "Give me a one-line example."})
print("\nMessages sent:", len(filled.to_messages()))   # 1 system + 2 history + 1 question = 4
print(llm.invoke(filled).text)

# ---------- Gotcha: literal curly braces ----------
# { } means "variable", so to write a real brace you double it: {{ }}
json_prompt = ChatPromptTemplate.from_messages([
    ("human", 'Reply as JSON like {{"answer": "..."}} to: {q}'),
])
print("\n" + json_prompt.invoke({"q": "hi"}).to_messages()[0].content)