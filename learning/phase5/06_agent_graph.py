"""Lesson 5.3 - The Lesson 3.5 agent loop (Script 19), rebuilt as a graph."""
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

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
llm_with_tools = llm.bind_tools(tools)


def think(state: MessagesState) -> dict:          # THINK: the LLM decides (answer or tool call)
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


builder = StateGraph(MessagesState)               # MessagesState = {"messages": [...]} with add_messages
builder.add_node("think", think)
builder.add_node("tools", ToolNode(tools))        # ACT: runs every requested tool → ToolMessages
builder.add_edge(START, "think")
builder.add_conditional_edges("think", tools_condition)   # tool calls? → "tools", otherwise → END
builder.add_edge("tools", "think")                # OBSERVE: results go back to the LLM (the loop!)
graph = builder.compile()

question = "A customer buys 12 tracks at 0.99 dollars each. What's the total in rupees?"
result = graph.invoke({"messages": [HumanMessage(question)]}, {"recursion_limit": 10})
for m in result["messages"]:
    calls = getattr(m, "tool_calls", None)
    print(f"{type(m).__name__:12}", calls if calls else m.text)   # .text, not .content (Lesson 3.1)

print("\n" + graph.get_graph().draw_mermaid())
