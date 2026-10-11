from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

# 1. A chat template: a list of (role, text) pairs with blanks.
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Answer in one sentence."),
    ("human", "{question}"),
])

# 2. Fill it in. The result is a list of real message objects.
filled = prompt.invoke({"role": "SQL tutor", "question": "What is a primary key?"})
print("Type:", type(filled).__name__)       # ChatPromptValue
for msg in filled.to_messages():
    print(" ", type(msg).__name__, "->", msg.content)

# 3. Send it to the model. The filled prompt goes straight into llm.invoke().
response = llm.invoke(filled)
print("\nAnswer:", response.text)

# 4. Reuse the SAME template with different values.
filled2 = prompt.invoke({"role": "pirate", "question": "What is a primary key?"})
print("Pirate:", llm.invoke(filled2).text)