from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

@tool
def get_track_price_in_inr(price_usd: float) -> str:
    """Convert a track price from US dollars to Indian rupees."""
    return f"₹{price_usd * 88:.2f}"

# Give the LLM the "menu" of tools.
llm_with_tools = llm.bind_tools([multiply, get_track_price_in_inr])

questions = [
    "What is 23.5 times 4?",                                   # → multiply
    "How much is a 0.99 dollar track in rupees?",              # → currency tool
    "What is SQL? One sentence.",                              # → no tool needed
    "What is 3 times 4, and what is 1.99 dollars in rupees?",  # → BOTH tools at once
]
for q in questions:
    msg = llm_with_tools.invoke(q)
    print(f"\nQ: {q}")
    print("   tool_calls:", msg.tool_calls)
    print("   text:      ", repr(msg.text[:80]))