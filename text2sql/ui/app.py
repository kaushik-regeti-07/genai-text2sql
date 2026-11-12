"""Text2SQL web app (v1.0). Run from the Gen_AI folder:

    streamlit run text2sql/ui/app.py

Same LangGraph chat graph as the CLI; this file only draws it in a browser.
"""
import os
import sys
import uuid
from pathlib import Path

# `streamlit run` puts THIS file's folder on the import path, not the project root,
# so add the Gen_AI folder to be able to `import text2sql`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.types import Command  # noqa: E402

from text2sql.app.config import MEMORY_DB_PATH  # noqa: E402
from text2sql.app.graph import build_chat_graph, list_threads, make_checkpointer  # noqa: E402

EXAMPLES = [
    "Which genre earned the most money?",
    "How many tracks are in the 90's Music playlist?",
    "How many of those are by Nirvana?",
    "Who is Jane Peacock's manager?",
    "Top 5 countries by total sales",
]

st.set_page_config(page_title="Text2SQL · Chinook", page_icon="🎵", layout="wide")


# ---------- shared resources: created once per server, reused by every page refresh ----------
@st.cache_resource
def get_checkpointer():
    return make_checkpointer(MEMORY_DB_PATH)          # conversations survive restarts


@st.cache_resource
def get_graph(approval: bool):
    return build_chat_graph(get_checkpointer(), approval=approval)


# ---------- optional password (set APP_PASSWORD before deploying publicly) ----------
def password_ok() -> bool:
    expected = os.getenv("APP_PASSWORD")
    if not expected or st.session_state.get("authed"):
        return True
    st.title("🎵 Text2SQL")
    entered = st.text_input("Password", type="password")
    if entered == expected:
        st.session_state.authed = True
        st.rerun()
    elif entered:
        st.error("Wrong password.")
    return False


# ---------- drawing helpers ----------
def show_table(columns: list[str], rows: list[list]) -> None:
    """A single number becomes a big metric; small label+number tables also get a bar chart."""
    if not rows:
        return
    df = pd.DataFrame(rows, columns=columns)
    if df.shape == (1, 1):
        value = df.iat[0, 0]
        label = columns[0] if columns[0].isidentifier() else "Result"   # "COUNT(*)" → "Result"
        st.metric(label, f"{value:,}" if pd.api.types.is_number(value) else str(value))
        return
    st.dataframe(df, hide_index=True)
    numbers = [c for c in df.select_dtypes("number").columns if not c.lower().endswith("id")]
    labels = [c for c in df.columns if c not in df.select_dtypes("number").columns]
    if numbers and labels and 2 <= len(df) <= 25:
        st.bar_chart(df, x=labels[0], y=numbers[-1], sort=False)   # keep the SQL's ORDER BY


def show_turn(turn: dict) -> None:
    with st.chat_message("user"):
        st.write(turn["user"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        if turn["route"] != "database_question":
            return
        show_table(turn["columns"], turn["rows"])
        with st.expander("🔍 How I got this answer"):
            if turn["question"] != turn["user"]:
                st.markdown(f"**Understood as:** {turn['question']}")
            st.markdown(f"**Tables used:** {', '.join(turn['tables']) or '-'}")
            for i, fix in enumerate(turn["fix_log"], start=1):
                st.markdown(f"🔧 **Fix {i}:** {fix}")
            if turn["sql"]:
                st.code(turn["sql"], language="sql")
            st.caption(f"Raw result: {str(turn['result'])[:300]}")


def run(graph, config: dict, payload) -> None:
    """Invoke the graph; errors are shown instead of crashing the page."""
    try:
        with st.spinner("Thinking…"):
            graph.invoke(payload, config)
    except Exception as e:
        st.error(f"Something went wrong: {type(e).__name__}: {e}")


# ======================================================================
if not password_ok():
    st.stop()

checkpointer = get_checkpointer()
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"chat-{uuid.uuid4().hex[:8]}"

# ---------- sidebar ----------
with st.sidebar:
    st.header("💬 Conversations")
    if st.button("➕ New conversation", width="stretch"):
        st.session_state.thread_id = f"chat-{uuid.uuid4().hex[:8]}"
        st.rerun()
    saved = list_threads(checkpointer)
    options = [st.session_state.thread_id] + [t for t in saved if t != st.session_state.thread_id]
    chosen = st.selectbox("Open", options, index=0)
    if chosen != st.session_state.thread_id:
        st.session_state.thread_id = chosen
        st.rerun()
    if chosen in saved and st.button("🗑️ Delete this conversation", width="stretch"):
        checkpointer.delete_thread(chosen)
        st.session_state.thread_id = f"chat-{uuid.uuid4().hex[:8]}"
        st.rerun()

    st.divider()
    approval = st.toggle("✋ Ask before running SQL", value=False)
    st.divider()
    st.subheader("Try asking")
    for example in EXAMPLES:
        if st.button(example, width="stretch"):
            st.session_state.queued = example
    st.divider()
    if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
        st.caption(f"📡 LangSmith tracing on · project `{os.getenv('LANGSMITH_PROJECT', 'default')}`")
    st.caption("Chinook music store · Gemini · LangGraph")

# ---------- main area ----------
config = {"configurable": {"thread_id": st.session_state.thread_id}}
approval_graph = get_graph(True)
snapshot = approval_graph.get_state(config)

st.title("🎵 Text2SQL: ask the Chinook music store")
st.caption("Plain-English questions → checked SQL → answers. Follow-up questions work.")

for turn in snapshot.values.get("turns", []):
    show_turn(turn)

# A query waiting for approval? (Saved in memory, so it survives a page refresh.)
if snapshot.next and "approve" in snapshot.next:
    pending = snapshot.tasks[0].interrupts[0].value
    with st.chat_message("assistant"):
        st.warning("I'd like to run this query. Approve?")
        st.markdown(f"**Question:** {pending['question']}  \n**Tables:** {', '.join(pending['tables'])}")
        st.code(pending["sql"], language="sql")
        yes, no = st.columns(2)
        if yes.button("✅ Run it", width="stretch"):
            run(approval_graph, config, Command(resume="y"))
            st.rerun()
        if no.button("❌ Don't run", width="stretch"):
            run(approval_graph, config, Command(resume="n"))
            st.rerun()
    st.stop()                                        # finish the pending query first

question = st.chat_input("Ask about customers, sales, artists, tracks…") or st.session_state.pop("queued", None)
if question:
    run(get_graph(approval), config, {"messages": [HumanMessage(question)]})
    st.rerun()
