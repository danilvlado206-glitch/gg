from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dsm.memory import DynamicSegmentedMemory
from dsm.models import ActiveContext
from dsm.skill import SkillKernel


@dataclass(slots=True)
class AgentMemory:
    memory: DynamicSegmentedMemory

    @classmethod
    def open(cls, store: str | Path = ".dsm/agent-memory.json") -> "AgentMemory":
        return cls(DynamicSegmentedMemory(store))

    def remember(
        self,
        text: str,
        *,
        category_path: str | None = None,
        importance: float = 0.5,
    ) -> list[str]:
        segments = self.memory.write(text, category_path=category_path, importance=importance)
        self.memory.save()
        return [segment.id for segment in segments]

    def recall(self, query: str, *, k: int = 5, skill_k: int = 2) -> ActiveContext:
        return self.memory.active_context(query, k=k, skill_k=skill_k)

    def crystallize(
        self,
        goal: str,
        trajectory: list[str],
        outcome: str,
        *,
        name: str | None = None,
    ) -> SkillKernel:
        kernel = self.memory.crystallize_skill(goal, trajectory, outcome, name=name)
        self.memory.save()
        return kernel
