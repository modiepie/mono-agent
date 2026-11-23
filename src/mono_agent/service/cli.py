"""Command line: `mono-agent ask "question"` or `mono-agent chat`."""

import argparse
import uuid

from dotenv import load_dotenv
from langgraph.types import Command

from mono_agent.app import build_app
from mono_agent.config import get_settings


def run_turn(app, question: str, thread: str) -> str:
    config = {"configurable": {"thread_id": thread}}
    result = app.invoke({"messages": [("user", question)]}, config)
    # Human-in-the-loop: the export node pauses the graph and waits for a yes/no.
    while interrupts := result.get("__interrupt__"):
        info = interrupts[0].value
        reply = input(f"Export a report with {info['tool_results']} result(s)? [y/N] ")
        result = app.invoke(
            Command(resume=reply.strip().lower() in ("y", "yes", "т", "так")), config
        )
    return result["messages"][-1].content


def main() -> None:
    load_dotenv()  # provider SDKs read API keys from the environment
    parser = argparse.ArgumentParser(prog="mono-agent")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ask = sub.add_parser("ask", help="ask one question")
    ask.add_argument("question")
    ask.add_argument("--thread", default=None, help="continue a conversation")
    chat = sub.add_parser("chat", help="interactive conversation")
    chat.add_argument("--thread", default=None)
    sub.add_parser("index", help="index data/docs into Qdrant (both chunking strategies)")
    args = parser.parse_args()

    if args.cmd == "index":
        from mono_agent.rag import index_docs

        for name, n in index_docs(get_settings(), ["fixed", "structure"]).items():
            print(f"{name}: {n} chunks")
        return

    app = build_app(get_settings())
    thread = args.thread or uuid.uuid4().hex[:8]
    if args.cmd == "ask":
        print(run_turn(app, args.question, thread))
        print(f"\n(thread {thread})")
        return
    print(f"thread {thread}; empty line to quit")
    while question := input("> ").strip():
        print(run_turn(app, question, thread), end="\n\n")


if __name__ == "__main__":
    main()
