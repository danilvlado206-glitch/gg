from pathlib import Path

from mcp.dsm_server import DsmMcpServer


def test_mcp_initialize_and_tool_list(tmp_path: Path) -> None:
    server = DsmMcpServer(tmp_path / "memory.json")

    init = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "clientInfo": {"name": "pytest", "version": "1.0"},
            },
        }
    )
    tools = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})

    assert init is not None
    assert init["result"]["capabilities"]["tools"] == {}
    assert tools is not None
    names = {tool["name"] for tool in tools["result"]["tools"]}
    assert {"dsm_write", "dsm_query", "dsm_crystallize_skill", "dsm_stats"} <= names


def test_mcp_write_crystallize_and_query(tmp_path: Path) -> None:
    server = DsmMcpServer(tmp_path / "memory.json")

    write = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "dsm_write",
                "arguments": {
                    "text": "MCP lets Claude Code call DSM persistent memory tools.",
                    "category_path": "AI → Agents → MCP",
                    "importance": 0.9,
                },
            },
        }
    )
    skill = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "dsm_crystallize_skill",
                "arguments": {
                    "goal": "Give an MCP agent persistent cognition",
                    "trajectory": [
                        "Write important task facts into DSM.",
                        "Crystallize successful task traces into kernels.",
                        "Recall context before solving related tasks.",
                    ],
                    "outcome": "The agent retrieves facts and reusable procedures.",
                },
            },
        }
    )
    query = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "dsm_query",
                "arguments": {"query": "How can Claude Code use persistent cognition?"},
            },
        }
    )

    assert write is not None
    assert "segment_ids" in write["result"]["content"][0]["text"]
    assert skill is not None
    assert "skill_kernel" in skill["result"]["content"][0]["text"]
    assert query is not None
    query_text = query["result"]["content"][0]["text"]
    assert "SKILL KERNEL" in query_text
    assert "context_text" in query_text


def test_mcp_unknown_tool_returns_tool_error(tmp_path: Path) -> None:
    server = DsmMcpServer(tmp_path / "memory.json")

    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "missing_tool", "arguments": {}},
        }
    )

    assert response is not None
    assert response["error"]["code"] == -32603
    assert "unknown tool" in response["error"]["message"]
