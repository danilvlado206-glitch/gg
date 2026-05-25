from dsm import AgentMemory


memory = AgentMemory.open(".dsm/agent-example.json")
memory.remember(
    "Claude Code and other agents can use DSM through the MCP server.",
    category_path="AI → Agents → MCP",
    importance=0.9,
)
memory.crystallize(
    "Use persistent cognition from an agent",
    [
        "Write relevant task facts into DSM memory.",
        "Crystallize successful execution paths as Skill Kernels.",
        "Recall active context before planning the next similar task.",
    ],
    "The agent retrieves both facts and reusable procedures.",
    name="Agent Persistent Cognition Loop",
)

context = memory.recall("How should an agent use persistent cognition?", k=2, skill_k=1)
print(context.context_text)
