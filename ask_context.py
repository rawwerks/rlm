"""Simple CLI helper to run RLM_REPL against a text or markdown file."""
from __future__ import annotations

import argparse
from pathlib import Path

from rlm.rlm_repl import RLM_REPL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ask a question about the contents of a text/markdown file using RLM_REPL.",
    )
    parser.add_argument(
        "context",
        type=Path,
        help="Path to the input file containing source text (e.g. .txt or .md).",
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Question or instruction for the model, phrased as if speaking to the author.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5",
        help="Root model name passed to RLM_REPL (default: gpt-5).",
    )
    parser.add_argument(
        "--recursive-model",
        default=None,
        help=(
            "Recursive model used for REPL tool calls. Defaults to the same value as --model "
            "when omitted."
        ),
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Maximum outer-loop iterations before forcing a final answer (default: 20).",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Enable verbose logging of the REPL loop.",
    )
    parser.add_argument(
        "--single-turn",
        action="store_true",
        help="Run a single question/answer turn instead of starting interactive chat.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    context_path: Path = args.context
    if not context_path.exists():
        parser.error(f"Context file '{context_path}' does not exist.")

    text = context_path.read_text(encoding="utf-8")

    recursive_model = args.recursive_model or args.model
    rlm = RLM_REPL(
        model=args.model,
        recursive_model=recursive_model,
        enable_logging=args.log,
        max_iterations=args.max_iterations,
    )

    if args.single_turn:
        if args.query is None:
            parser.error("query is required when --single-turn is specified.")
        answer = rlm.completion(context=text, query=args.query)
        print(answer)
        return

    initial_query = args.query
    if not initial_query:
        try:
            initial_query = input("Enter your first question: ").strip()
        except EOFError:
            initial_query = ""
    if not initial_query:
        parser.error("An initial question is required to start chat mode.")

    answer = rlm.completion(context=text, query=initial_query)
    print(answer)

    while True:
        try:
            follow_up = input("Follow-up (press Enter to exit): ")
        except EOFError:
            break

        if not follow_up.strip():
            break

        answer = rlm.ask_followup(follow_up)
        print(answer)


if __name__ == "__main__":
    main()
