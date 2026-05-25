from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dsm.memory import DynamicSegmentedMemory

JSON = dict[str, Any]
PROTOCOL_VERSION = "2024-11-05"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: JSON
    handler: Callable[[JSON], JSON]


class DsmMcpServer:
    def __init__(self, store: str | Path = ".dsm/mcp-memory.json"):
        self.store = Path(store)
        self.memory = DynamicSegmentedMemory(self.store)
        self.tools = self._build_tools()

    def handle(self, message: JSON) -> JSON | None:
        request_id = message.get("id")
        method = message.get("method")
        params = ensure_object(message.get("params", {}), "params")

        try:
            if method == "initialize":
                return self.response(request_id, self.initialize(params))
            if method == "notifications/initialized":
                return None
            if method == "ping":
                return self.response(request_id, {})
            if method == "tools/list":
                return self.response(request_id, self.list_tools())
            if method == "tools/call":
                return self.response(request_id, self.call_tool(params))
            return self.error(request_id, -32601, f"unknown method: {method}")
        except Exception as exc:
            return self.error(
                request_id,
                -32603,
                str(exc),
                {"traceback": traceback.format_exc(limit=6)},
            )

    def initialize(self, params: JSON) -> JSON:
        client = ensure_object(params.get("clientInfo", {}), "clientInfo")
        return {
            "protocolVersion": params.get("protocolVersion", PROTOCOL_VERSION),
            "capabilities": {"tools": {}},
            "serverInfo": {
                "name": "dsm-persistent-cognition",
                "version": "1.0.0",
                "client": client.get("name", "unknown"),
            },
        }

    def list_tools(self) -> JSON:
        return {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.input_schema,
                }
                for tool in self.tools.values()
            ]
        }

    def call_tool(self, params: JSON) -> JSON:
        name = require_str(params, "name")
        arguments = ensure_object(params.get("arguments", {}), "arguments")
        tool = self.tools.get(name)
        if tool is None:
            raise ValueError(f"unknown tool: {name}")
        result = tool.handler(arguments)
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(result, ensure_ascii=False, indent=2),
                }
            ],
            "isError": False,
        }

    def _build_tools(self) -> dict[str, ToolSpec]:
        tools = [
            ToolSpec(
                name="dsm_write",
                description="Store text as DSM semantic memory segments.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "category_path": {"type": "string"},
                        "importance": {"type": "number", "default": 0.5},
                        "metadata": {"type": "object"},
                    },
                    "required": ["text"],
                },
                handler=self.tool_write,
            ),
            ToolSpec(
                name="dsm_query",
                description="Retrieve active PCS context containing DSM segments and Skill Kernels.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "k": {"type": "integer", "default": 5},
                        "skill_k": {"type": "integer", "default": 2},
                        "token_budget": {"type": "integer"},
                    },
                    "required": ["query"],
                },
                handler=self.tool_query,
            ),
            ToolSpec(
                name="dsm_crystallize_skill",
                description="Convert a successful agent trajectory into a persistent Skill Kernel.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "goal": {"type": "string"},
                        "trajectory": {"type": "array", "items": {"type": "string"}},
                        "outcome": {"type": "string"},
                        "name": {"type": "string"},
                        "metadata": {"type": "object"},
                    },
                    "required": ["goal", "trajectory", "outcome"],
                },
                handler=self.tool_crystallize_skill,
            ),
            ToolSpec(
                name="dsm_route_skills",
                description="Retrieve only procedural Skill Kernels for a task.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "k": {"type": "integer", "default": 3},
                    },
                    "required": ["query"],
                },
                handler=self.tool_route_skills,
            ),
            ToolSpec(
                name="dsm_update_from_interaction",
                description="Persist a query/answer interaction as durable DSM memory.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "answer": {"type": "string"},
                        "importance": {"type": "number", "default": 0.6},
                        "metadata": {"type": "object"},
                    },
                    "required": ["query", "answer"],
                },
                handler=self.tool_update_from_interaction,
            ),
            ToolSpec(
                name="dsm_stats",
                description="Return DSM/PCS memory statistics.",
                input_schema={"type": "object", "properties": {}},
                handler=self.tool_stats,
            ),
            ToolSpec(
                name="dsm_save",
                description="Flush memory to the configured JSON store.",
                input_schema={"type": "object", "properties": {}},
                handler=self.tool_save,
            ),
        ]
        return {tool.name: tool for tool in tools}

    def tool_write(self, args: JSON) -> JSON:
        segments = self.memory.write(
            require_str(args, "text"),
            category_path=optional_str(args, "category_path"),
            importance=float(args.get("importance", 0.5)),
            metadata=ensure_object(args.get("metadata", {}), "metadata"),
        )
        self.memory.save()
        return {
            "stored": len(segments),
            "segment_ids": [segment.id for segment in segments],
            "stats": self.memory.stats(),
        }

    def tool_query(self, args: JSON) -> JSON:
        context = self.memory.active_context(
            require_str(args, "query"),
            k=int(args.get("k", self.memory.active_segment_limit)),
            skill_k=int(args.get("skill_k", 2)),
            token_budget=optional_int(args, "token_budget"),
        )
        return {
            "query": context.query,
            "context_text": context.context_text,
            "segment_ids": context.segment_ids,
            "skill_ids": context.skill_ids,
            "estimated_tokens": context.estimated_tokens,
            "global_summary": context.global_summary,
        }

    def tool_crystallize_skill(self, args: JSON) -> JSON:
        trajectory = ensure_str_list(args.get("trajectory"), "trajectory")
        kernel = self.memory.crystallize_skill(
            require_str(args, "goal"),
            trajectory,
            require_str(args, "outcome"),
            name=optional_str(args, "name"),
            metadata=ensure_object(args.get("metadata", {}), "metadata"),
        )
        self.memory.save()
        return {"skill_kernel": kernel.to_dict(), "stats": self.memory.stats()}

    def tool_route_skills(self, args: JSON) -> JSON:
        results = self.memory.route_skills(
            require_str(args, "query"),
            k=int(args.get("k", 3)),
        )
        return {
            "skills": [
                {
                    "id": item.kernel.id,
                    "name": item.kernel.name,
                    "description": item.kernel.description,
                    "triggers": list(item.kernel.triggers),
                    "procedure": list(item.kernel.procedure),
                    "score": item.total_score,
                }
                for item in results
            ]
        }

    def tool_update_from_interaction(self, args: JSON) -> JSON:
        segments = self.memory.update_from_interaction(
            require_str(args, "query"),
            require_str(args, "answer"),
            importance=float(args.get("importance", 0.6)),
            metadata=ensure_object(args.get("metadata", {}), "metadata"),
        )
        self.memory.save()
        return {
            "segment_ids": [segment.id for segment in segments],
            "stats": self.memory.stats(),
        }

    def tool_stats(self, args: JSON) -> JSON:
        return self.memory.stats()

    def tool_save(self, args: JSON) -> JSON:
        self.memory.save()
        return {"saved": True, "store": str(self.store)}

    def response(self, request_id: Any, result: JSON) -> JSON:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def error(self, request_id: Any, code: int, message: str, data: JSON | None = None) -> JSON:
        error: JSON = {"code": code, "message": message}
        if data is not None:
            error["data"] = data
        return {"jsonrpc": "2.0", "id": request_id, "error": error}


def serve_stdio(server: DsmMcpServer) -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            response = server.error(None, -32700, f"parse error: {exc}")
        else:
            response = server.handle(ensure_object(message, "message"))
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DSM Persistent Cognition MCP server")
    parser.add_argument("--store", default=".dsm/mcp-memory.json", type=Path)
    args = parser.parse_args(argv)
    serve_stdio(DsmMcpServer(args.store))
    return 0


def require_str(data: JSON, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def optional_str(data: JSON, key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def optional_int(data: JSON, key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    return int(value)


def ensure_object(value: Any, name: str) -> JSON:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def ensure_str_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{name} must contain non-empty strings")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
