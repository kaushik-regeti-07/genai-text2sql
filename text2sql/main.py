"""Command-line app (v1.0: a LangGraph conversation with saved memory). Run from the Gen_AI folder:

    python -m text2sql.main                         (chat; continues the "cli-session" conversation)
    python -m text2sql.main --thread ravi           (continue / start the conversation named "ravi")
    python -m text2sql.main --list                  (show saved conversations)
    python -m text2sql.main --approve               (ask before running each query)
    python -m text2sql.main --no-save               (don't save this conversation)
    python -m text2sql.main "How many artists?"     (one question, then exit)
"""
import argparse

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from text2sql.app.config import MEMORY_DB_PATH
from text2sql.app.graph import build_chat_graph, list_threads, make_checkpointer


def show(out: dict) -> None:
    """Print what happened in this turn."""
    if out.get("route") == "database_question":
        print(f"  Question: {out['question']}")             # the standalone version
        print(f"  Tables:   {', '.join(out.get('tables') or [])}")
        for i, problem in enumerate(out.get("fix_log") or [], start=1):
            print(f"  🔧 Fix {i}: {problem}")
        if out.get("sql_query") is not None:
            print(f"  SQL:      {out['sql_query'].sql or '(none)'}")
        print(f"  Result:   {out.get('result')}")
    print(f"  Answer:   {out.get('answer')}")


def ask(graph, config: dict, text: str) -> None:
    """Run one turn. If the graph pauses for approval, ask the human and resume."""
    try:
        out = graph.invoke({"messages": [HumanMessage(text)]}, config)
        while "__interrupt__" in out:                        # paused at the approve node
            info = out["__interrupt__"][0].value
            print(f"  Tables:   {', '.join(info['tables'])}")
            print(f"  SQL:      {info['sql']}")
            decision = input("  Run this query? (y/n): ").strip()
            out = graph.invoke(Command(resume=decision), config)
        show(out)
    except Exception as e:                                   # e.g. network error or rate limit
        print(f"  Something went wrong: {type(e).__name__}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the Chinook music store database.")
    parser.add_argument("--approve", action="store_true", help="ask before running each SQL query")
    parser.add_argument("--thread", default="cli-session", help="conversation name (memory is per name)")
    parser.add_argument("--list", action="store_true", help="list saved conversations and exit")
    parser.add_argument("--no-save", action="store_true", help="keep memory in RAM only")
    parser.add_argument("question", nargs="*", help="a single question to answer, then exit")
    args = parser.parse_args()

    checkpointer = make_checkpointer(None if args.no_save else MEMORY_DB_PATH)
    if args.list:
        threads = list_threads(checkpointer)
        print("Saved conversations:", ", ".join(threads) if threads else "(none)")
        return

    graph = build_chat_graph(checkpointer=checkpointer, approval=args.approve)
    config = {"configurable": {"thread_id": args.thread}}   # one conversation = one thread

    if args.question:
        ask(graph, config, " ".join(args.question))
        return

    previous = graph.get_state(config).values.get("turns") or []
    print(f"Text2SQL - conversation '{args.thread}'"
          + (f" ({len(previous)} earlier questions remembered)" if previous else "")
          + ". Follow-up questions work! Type 'exit' to quit.")
    while True:
        text = input("\nYou: ").strip()
        if text.lower() in {"", "exit", "quit"}:
            break
        ask(graph, config, text)


if __name__ == "__main__":
    main()
