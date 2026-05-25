from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from dsm.embedding import EmbeddingModel, cosine, tokenize, top_terms
from dsm.models import PriorityVector, now_ts


@dataclass(slots=True)
class ReasoningTrace:
    goal: str
    trajectory: list[str]
    outcome: str
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        steps = "\n".join(f"{index}. {step}" for index, step in enumerate(self.trajectory, 1))
        return f"Goal: {self.goal}\nSteps:\n{steps}\nOutcome: {self.outcome}"


@dataclass(slots=True)
class SkillKernel:
    name: str
    description: str
    triggers: tuple[str, ...]
    procedure: tuple[str, ...]
    embedding: list[float]
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    category_path: tuple[str, ...] = ("Skills", "Crystallized")
    evidence: tuple[str, ...] = ()
    priorities: PriorityVector = field(default_factory=lambda: PriorityVector(importance=0.82))
    created_at: float = field(default_factory=now_ts)
    updated_at: float = field(default_factory=now_ts)
    last_accessed_at: float = field(default_factory=now_ts)
    success_count: int = 1
    use_count: int = 0
    source_segment_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def estimated_tokens(self) -> int:
        return max(1, len(self.to_context_text().split()))

    def touch(self, similarity: float) -> None:
        self.last_accessed_at = now_ts()
        self.use_count += 1
        self.priorities.touch(similarity, boost=0.12)

    def reinforce(self, evidence: str | None = None) -> None:
        self.success_count += 1
        self.updated_at = now_ts()
        self.priorities.importance = min(1.0, self.priorities.importance + 0.04)
        if evidence:
            self.evidence = tuple(dict.fromkeys((*self.evidence, evidence)))

    def to_context_text(self) -> str:
        triggers = ", ".join(self.triggers) or "general"
        procedure = "\n".join(f"{index}. {step}" for index, step in enumerate(self.procedure, 1))
        evidence = "\n".join(f"- {item}" for item in self.evidence)
        parts = [
            f"Skill Kernel: {self.name}",
            f"Description: {self.description}",
            f"Triggers: {triggers}",
            "Procedure:",
            procedure,
        ]
        if evidence:
            parts.extend(["Evidence:", evidence])
        return "\n".join(parts)

    def to_segment_text(self) -> str:
        return self.to_context_text()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "triggers": list(self.triggers),
            "procedure": list(self.procedure),
            "embedding": self.embedding,
            "category_path": list(self.category_path),
            "evidence": list(self.evidence),
            "priorities": self.priorities.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_accessed_at": self.last_accessed_at,
            "success_count": self.success_count,
            "use_count": self.use_count,
            "source_segment_ids": list(self.source_segment_ids),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillKernel":
        return cls(
            id=str(data["id"]),
            name=str(data.get("name", "Skill Kernel")),
            description=str(data.get("description", "")),
            triggers=tuple(str(item) for item in data.get("triggers", ())),
            procedure=tuple(str(item) for item in data.get("procedure", ())),
            embedding=[float(value) for value in data.get("embedding", [])],
            category_path=tuple(str(item) for item in data.get("category_path", ("Skills",))),
            evidence=tuple(str(item) for item in data.get("evidence", ())),
            priorities=PriorityVector.from_dict(data.get("priorities")),
            created_at=float(data.get("created_at", now_ts())),
            updated_at=float(data.get("updated_at", now_ts())),
            last_accessed_at=float(data.get("last_accessed_at", now_ts())),
            success_count=int(data.get("success_count", 1)),
            use_count=int(data.get("use_count", 0)),
            source_segment_ids=tuple(str(item) for item in data.get("source_segment_ids", ())),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class SkillRouteResult:
    kernel: SkillKernel
    similarity: float
    priority_score: float
    total_score: float


class SkillCrystallizer:
    """Turns successful trajectories into reusable Skill Kernels."""

    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model

    def crystallize(
        self,
        trace: ReasoningTrace,
        *,
        name: str | None = None,
        category_path: tuple[str, ...] = ("Skills", "Crystallized"),
    ) -> SkillKernel:
        if not trace.success:
            raise ValueError("only successful traces can be crystallized")
        if not trace.trajectory:
            raise ValueError("trajectory must contain at least one step")

        procedure = tuple(compress_step(step) for step in trace.trajectory if step.strip())
        evidence = tuple(item for item in (trace.goal, trace.outcome) if item.strip())
        triggers = tuple(keyword_terms(trace.to_text(), limit=8))
        description = build_description(trace.goal, trace.outcome)
        kernel_name = name or build_name(trace.goal, triggers)
        kernel_text = "\n".join((kernel_name, description, *triggers, *procedure, trace.outcome))

        return SkillKernel(
            name=kernel_name,
            description=description,
            triggers=triggers,
            procedure=procedure,
            embedding=self.embedding_model.encode(kernel_text),
            category_path=category_path,
            evidence=evidence,
            metadata=dict(trace.metadata),
        )


def route_skill_kernels(
    kernels: dict[str, SkillKernel],
    query: str,
    embedding_model: EmbeddingModel,
    *,
    k: int,
    similarity_floor: float = -1.0,
) -> list[SkillRouteResult]:
    query_embedding = embedding_model.encode(query)
    current_time = now_ts()
    scored: list[SkillRouteResult] = []

    for kernel in kernels.values():
        similarity = cosine(query_embedding, kernel.embedding)
        if similarity < similarity_floor:
            continue
        age_seconds = current_time - kernel.last_accessed_at
        priority_score = kernel.priorities.total(similarity, age_seconds)
        success_bonus = min(0.12, kernel.success_count * 0.02)
        total_score = 0.72 * similarity + 0.20 * priority_score + success_bonus
        scored.append(
            SkillRouteResult(
                kernel=kernel,
                similarity=similarity,
                priority_score=priority_score,
                total_score=total_score,
            )
        )

    scored.sort(reverse=True, key=lambda item: item.total_score)
    selected = scored[: max(0, k)]
    for item in selected:
        item.kernel.touch(item.similarity)
    return selected


def build_description(goal: str, outcome: str) -> str:
    compact_goal = compact_sentence(goal)
    compact_outcome = compact_sentence(outcome)
    if compact_outcome:
        return f"Reusable strategy for {compact_goal}; proven outcome: {compact_outcome}"
    return f"Reusable strategy for {compact_goal}"


def build_name(goal: str, triggers: tuple[str, ...]) -> str:
    if triggers:
        label = " ".join(term.title() for term in triggers[:4])
        return f"{label} Skill"
    words = compact_sentence(goal).split()
    return f"{' '.join(words[:4]).title()} Skill" if words else "Crystallized Skill"


def compress_step(step: str, max_words: int = 22) -> str:
    compact = compact_sentence(step)
    words = compact.split()
    if len(words) <= max_words:
        return compact
    return f"{' '.join(words[:max_words])}…"


def compact_sentence(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def keyword_terms(text: str, limit: int) -> list[str]:
    stop_words = {
        "and",
        "are",
        "before",
        "for",
        "from",
        "into",
        "the",
        "then",
        "this",
        "under",
        "with",
    }
    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    for token in tokenize(text):
        term = token.strip(".,:;!?()[]{}")
        if len(term) < 3 or term in stop_words:
            continue
        if term not in counts:
            first_seen[term] = len(first_seen)
        counts[term] = counts.get(term, 0) + 1
    if counts:
        ordered_terms = sorted(counts, key=lambda term: (-counts[term], first_seen[term]))
        return ordered_terms[:limit]
    return top_terms(text, limit=limit)
