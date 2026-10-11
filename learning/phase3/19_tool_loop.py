from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

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

tools = [multiply, get_track_price_in_inr]
tools_by_name = {t.name: t for t in tools}     # lookup: "multiply" → the tool
llm_with_tools = llm.bind_tools(tools)

# The conversation. It grows as we go.
messages = [HumanMessage("A customer buys 12 tracks at 0.99 dollars each. "
                         "What's the total in rupees?")]

# ---------------- THE AGENT LOOP ----------------
for step in range(5):                              # safety limit: max 5 rounds
    ai_msg = llm_with_tools.invoke(messages)       # 1. THINK: LLM decides
    messages.append(ai_msg)

    if not ai_msg.tool_calls:                      # 2. DONE? No tool calls = final answer
        print("\nFINAL ANSWER:", ai_msg.text)
        break

    for call in ai_msg.tool_calls:                 # 3. ACT: run each requested tool
        tool_fn = tools_by_name[call["name"]]
        result = tool_fn.invoke(call["args"])
        print(f"Step {step}: {call['name']}({call['args']}) → {result}")

        messages.append(ToolMessage(               # 4. OBSERVE: send the result back
            content=str(result),
            tool_call_id=call["id"],
        ))
# ------------------------------------------------

print("\nFull conversation:", [type(m).__name__ for m in messages])