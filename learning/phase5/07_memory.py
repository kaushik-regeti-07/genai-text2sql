"""Lesson 5.4 - Memory: a checkpointer saves the state after every step, per conversation (thread)."""
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")


def chat(state: MessagesState) -> dict:
    system = SystemMessage("You are a friendly assistant. Answer in one short sentence.")
    return {"messages": [llm.invoke([system] + state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("chat", chat)
builder.add_edge(START, "chat")
graph = builder.compile(checkpointer=InMemorySaver())     # ← memory switched on

ravi = {"configurable": {"thread_id": "ravi"}}             # one thread = one conversation
priya = {"configurable": {"thread_id": "priya"}}


def say(text: str, thread: dict) -> None:
    out = graph.invoke({"messages": [HumanMessage(text)]}, thread)   # send ONLY the new message
    print(f"[{thread['configurable']['thread_id']}] You: {text}\n         Bot: {out['messages'][-1].text}")


say("Hi, my name is Ravi and I love AC/DC.", ravi)
say("Hi, I'm Priya and I prefer Jazz.", priya)
say("What's my name and favourite music?", ravi)          # remembers Ravi
say("What's my name and favourite music?", priya)         # remembers Priya, separately

# Look inside the saved memory
saved = graph.get_state(ravi).values["messages"]
print(f"\nRavi's thread holds {len(saved)} messages:")
for m in saved:
    print(f"   {type(m).__name__:12} {m.text[:70]}")
