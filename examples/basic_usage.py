from dsm import DynamicSegmentedMemory


memory = DynamicSegmentedMemory(".dsm/example.json", active_segment_limit=3)

memory.write(
    "Rust websocket services need bounded channels, ping timeouts and cancellation-safe tasks.",
    category_path="Programming → Rust → Async",
    importance=0.9,
)
memory.write(
    "MemoryOS and HMT organize agent memory into short, medium and long-term layers.",
    category_path="Science → AI → Memory",
    importance=0.8,
)
memory.write(
    "A personal preference can be stored as a high-priority long-term segment.",
    category_path="Personal → Preferences",
    importance=0.7,
)
memory.crystallize_skill(
    "Repair a Rust websocket timeout",
    [
        "Reproduce the timeout and capture the failing boundary conditions.",
        "Inspect bounded channels, ping intervals and cancellation-safe task shutdown.",
        "Patch the smallest failing path, then rerun the regression test.",
    ],
    "The websocket flow remains stable under backpressure.",
    name="Rust Websocket Debugging",
)

context = memory.active_context("Как исправить websocket timeout в Rust?", k=2)
print(context.context_text)
memory.save()
