"""Command-line app (v0.5: a LangGraph conversation). Run from the Gen_AI folder:

    python -m text2sql.main                         (interactive chat with memory)
    python -m text2sql.main --approve               (ask before running each query)
    python -m text2sql.main "How many artists?"     (one question, then exit)
"""
import argparse

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from text2sql.app.graph import build_chat_graph, make_checkpointer


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
    parser.add_argument("question", nargs="*", help="a single question to answer, then exit")
    args = parser.parse_args()

    graph = build_chat_graph(checkpointer=make_checkpointer(), approval=args.approve)
    config = {"configurable": {"thread_id": "cli-session"}}   # one conversation = one thread

    if args.question:
        ask(graph, config, " ".join(args.question))
        return

    print("Text2SQL - chat about the Chinook music store. Follow-up questions work! Type 'exit' to quit.")
    while True:
        text = input("\nYou: ").strip()
        if text.lower() in {"", "exit", "quit"}:
            break
        ask(graph, config, text)


if __name__ == "__main__":
    main()
