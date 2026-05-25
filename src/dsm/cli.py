from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dsm.memory import DynamicSegmentedMemory


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    memory = DynamicSegmentedMemory(args.store)

    if args.command == "write":
        segments = memory.write(
            args.text,
            category_path=args.category,
            importance=args.importance,
            metadata=parse_metadata(args.metadata),
        )
        memory.save()
        print(json.dumps({"written": [segment.id for segment in segments]}, indent=2))
        return 0

    if args.command == "skill":
        trajectory = args.step or []
        kernel = memory.crystallize_skill(
            args.goal,
            trajectory,
            args.outcome,
            name=args.name,
            metadata=parse_metadata(args.metadata),
        )
        memory.save()
        print(json.dumps({"skill_kernel": kernel.to_dict()}, indent=2, ensure_ascii=False))
        return 0

    if args.command == "query":
        context = memory.active_context(args.query, k=args.k, skill_k=args.skill_k)
        print(context.context_text)
        return 0

    if args.command == "stats":
        print(json.dumps(memory.stats(), indent=2, ensure_ascii=False))
        return 0

    if args.command == "demo":
        run_demo(memory)
        memory.save()
        return 0

    parser.print_help()
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dsm",
        description="Dynamic Segmented Memory + Skill Crystallization MVP",
    )
    parser.add_argument(
        "--store",
        default=".dsm/memory.json",
        type=Path,
        help="Path to the persistent JSON memory store.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    write = subparsers.add_parser("write", help="Persist a semantic memory segment.")
    write.add_argument("text")
    write.add_argument("--category", default=None)
    write.add_argument("--importance", type=float, default=0.5)
    write.add_argument("--metadata", default=None, help="JSON object stored with the segment.")

    skill = subparsers.add_parser("skill", help="Crystallize a successful trajectory.")
    skill.add_argument("--goal", required=True)
    skill.add_argument("--step", action="append", required=True)
    skill.add_argument("--outcome", required=True)
    skill.add_argument("--name", default=None)
    skill.add_argument("--metadata", default=None, help="JSON object stored with the kernel.")

    query = subparsers.add_parser("query", help="Retrieve active context.")
    query.add_argument("query")
    query.add_argument("-k", type=int, default=3)
    query.add_argument("--skill-k", type=int, default=2)

    subparsers.add_parser("stats", help="Print memory statistics.")
    subparsers.add_parser("demo", help="Run a local PCS demonstration.")
    return parser


def parse_metadata(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise argparse.ArgumentTypeError("metadata must be a JSON object")
    return data


def run_demo(memory: DynamicSegmentedMemory) -> None:
    memory.write(
        "DSM routes memory through semantic segments, a hierarchy tree and graph traversal.",
        category_path="AI → Memory → DSM",
        importance=0.9,
    )
    memory.write(
        "Skill Crystallization turns successful reasoning traces into reusable kernels.",
        category_path="AI → Agents → Skill Crystallization",
        importance=0.9,
    )
    memory.crystallize_skill(
        "Repair a failing software agent workflow",
        [
            "Capture the failing command, logs and environment assumptions.",
            "Route through prior repo facts and related debugging kernels.",
            "Apply the smallest patch and rerun the failing verification.",
        ],
        "The workflow is repaired and the strategy is reusable.",
        name="Software Agent Repair Kernel",
    )
    context = memory.active_context("How does persistent cognition debug agent failures?", k=3)
    print(context.context_text)


if __name__ == "__main__":
    raise SystemExit(main())
