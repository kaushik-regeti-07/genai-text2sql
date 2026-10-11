from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

# --- stream: pieces arrive one by one ---
print("STREAM:")
for chunk in llm.stream("Explain what a database index is in 3 sentences."):
    print(chunk.text, end="", flush=True)   # end="" keeps it on one line
print("\n")

# --- batch: several questions in parallel ---
questions = [
    "What does SELECT do in SQL? One sentence.",
    "What does GROUP BY do in SQL? One sentence.",
    "What does ORDER BY do in SQL? One sentence.",
]
answers = llm.batch(questions)
for q, a in zip(questions, answers):
    print(f"Q: {q}\nA: {a.text}\n")