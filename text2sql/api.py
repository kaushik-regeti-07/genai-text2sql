"""Text2SQL as a web API (v1.0). Run from the Gen_AI folder:

    uvicorn text2sql.api:app --reload

Then open http://127.0.0.1:8000/docs for an interactive page to try every endpoint.
Same LangGraph chat graph and saved memory as the CLI and the Streamlit app.
"""
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from text2sql.app.config import MEMORY_DB_PATH
from text2sql.app.graph import build_chat_graph, list_threads, make_checkpointer

app = FastAPI(title="Text2SQL API", version="1.0",
              description="Ask the Chinook music store database questions in plain English.")

checkpointer = make_checkpointer(MEMORY_DB_PATH)
graphs = {False: build_chat_graph(checkpointer), True: build_chat_graph(checkpointer, approval=True)}


# ---------- request / response shapes (Pydantic, like Lesson 3.4) ----------
class ChatRequest(BaseModel):
    message: str = Field(examples=["How many tracks are in the 90's Music playlist?"])
    thread_id: str = Field(default="api-default", description="Conversation id: same id = same memory")
    approval: bool = Field(default=False, description="Pause for /approve before running the SQL")


class ApproveRequest(BaseModel):
    thread_id: str
    decision: Literal["y", "n"]


class ChatResponse(BaseModel):
    status: Literal["answered", "needs_approval"]
    thread_id: str
    answer: Optional[str] = None
    question: Optional[str] = None          # the standalone (rewritten) question
    route: Optional[str] = None
    sql: Optional[str] = None
    tables: list[str] = []
    fix_log: list[str] = []
    columns: list[str] = []
    rows: list[list] = []


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id},
            "run_name": "text2sql-chat", "tags": ["api"],         # labels for LangSmith traces
            "metadata": {"thread_id": thread_id}}


def _response(thread_id: str, out: dict) -> ChatResponse:
    """Turn the graph's final state (or a pause) into the API response."""
    if "__interrupt__" in out:
        pending = out["__interrupt__"][0].value
        return ChatResponse(status="needs_approval", thread_id=thread_id, question=pending["question"],
                            sql=pending["sql"], tables=pending["tables"], route="database_question")
    turn = (out.get("turns") or [{}])[-1]
    return ChatResponse(status="answered", thread_id=thread_id, answer=turn.get("answer"),
                        question=turn.get("question"), route=turn.get("route"), sql=turn.get("sql"),
                        tables=turn.get("tables", []), fix_log=turn.get("fix_log", []),
                        columns=turn.get("columns", []), rows=turn.get("rows", []))


# ---------- endpoints ----------
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Ask a question. Follow-ups work if you reuse the same thread_id."""
    try:
        out = graphs[req.approval].invoke({"messages": [HumanMessage(req.message)]}, _config(req.thread_id))
    except Exception as e:  # e.g. the LLM provider is down or rate-limited
        raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}")
    return _response(req.thread_id, out)


@app.post("/approve", response_model=ChatResponse)
def approve(req: ApproveRequest) -> ChatResponse:
    """Answer a pending approval ('y' runs the SQL, 'n' cancels it)."""
    graph = graphs[True]
    snapshot = graph.get_state(_config(req.thread_id))
    if not (snapshot.next and "approve" in snapshot.next):
        raise HTTPException(status_code=409, detail="Nothing is waiting for approval in this thread.")
    try:
        out = graph.invoke(Command(resume=req.decision), _config(req.thread_id))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}")
    return _response(req.thread_id, out)


@app.get("/threads")
def threads() -> list[str]:
    """Saved conversation ids, most recent first."""
    return list_threads(checkpointer)


@app.get("/threads/{thread_id}")
def thread_history(thread_id: str) -> list[dict]:
    """Every question and answer in one conversation."""
    turns = graphs[False].get_state(_config(thread_id)).values.get("turns")
    if turns is None:
        raise HTTPException(status_code=404, detail="No such conversation.")
    return turns


@app.delete("/threads/{thread_id}")
def delete_thread(thread_id: str) -> dict:
    checkpointer.delete_thread(thread_id)
    return {"deleted": thread_id}
