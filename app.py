"""
app.py - command-line entry point for the Mood Check-In Agent.

Examples:
  python app.py "Oh great, another Monday"          # analyze one post
  python app.py --trace "this burger slaps fr"      # also show every agent step
  python app.py                                     # interactive mode
  python app.py --offline "..."                     # never call Gemini
"""

import argparse
import json

from agent import MoodAgent, setup_logging
from llm_client import get_llm


def show(result, show_trace: bool) -> None:
    if result.status == "rejected":
        print(f"REJECTED: {result.reason}")
        return
    label = result.label or "-"
    print(f"Mood:       {label}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Status:     {result.status}")
    print(f"Source:     {result.source}")
    print(f"Why:        {result.reason}")
    if result.status == "needs_human_review":
        print(">> Flagged for a human reviewer.")
    if show_trace:
        print("Trace:")
        for step in result.trace:
            print("  " + json.dumps(step, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Mood Check-In Agent")
    parser.add_argument("text", nargs="*", help="post to analyze (omit for interactive mode)")
    parser.add_argument("--offline", action="store_true", help="do not call the LLM")
    parser.add_argument("--trace", action="store_true", help="print the agent's steps")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logs on console")
    args = parser.parse_args()

    setup_logging(args.verbose)
    llm = get_llm(offline=args.offline)
    agent = MoodAgent(llm=llm, save_traces=True)
    print(f"[mode: {'Gemini + retrieval' if llm else 'offline (rules + retrieval)'}]")

    if args.text:
        show(agent.run(" ".join(args.text)), args.trace)
        return

    print("Type a post (empty line or 'quit' to exit).")
    while True:
        try:
            text = input("\nPost: ")
        except (EOFError, KeyboardInterrupt):
            break
        if text.strip().lower() in ("", "quit"):
            break
        show(agent.run(text), args.trace)


if __name__ == "__main__":
    main()
