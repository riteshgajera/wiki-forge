"""
Pipeline engine: executes agents in dependency order (topological sort).
Supports incremental processing and human-in-the-loop checkpoints.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from kb.agents.base import AgentInput, AgentOutput, BaseAgent
from kb.utils.logging import get_logger

logger = get_logger("pipelines.engine")


@dataclass
class PipelineStep:
    name: str
    agent: BaseAgent
    depends_on: list[str] = field(default_factory=list)
    skip_on_low_confidence: bool = True
    confidence_threshold: float = 0.5  # Below this, pipeline halts


@dataclass
class PipelineResult:
    doc_id: str
    outputs: dict[str, AgentOutput] = field(default_factory=dict)
    halted: bool = False
    halt_reason: str = ""
    needs_review: bool = False
    overall_confidence: float = 0.0

    @property
    def success(self) -> bool:
        return not self.halted and bool(self.outputs)

    def get_output(self, step: str) -> dict[str, Any]:
        out = self.outputs.get(step)
        return out.result if out else {}


ReviewCallback = Callable[[str, AgentOutput], bool]  # (doc_id, output) -> approved


class Pipeline:
    """
    Dependency-aware agent pipeline.

    Steps run in topological order. Each step receives:
    - Original AgentInput
    - All prior step outputs (injected into metadata["prior_results"])

    If a step's confidence is below threshold, the pipeline halts and
    triggers the review_callback (if set).
    """

    def __init__(
        self,
        steps: list[PipelineStep],
        review_callback: ReviewCallback | None = None,
        auto_approve_threshold: float = 0.85,
    ) -> None:
        self.steps = {s.name: s for s in steps}
        self.review_callback = review_callback
        self.auto_approve_threshold = auto_approve_threshold
        self._order = self._topological_sort()

    def run(self, inp: AgentInput) -> PipelineResult:
        """Execute all pipeline steps in dependency order."""
        result = PipelineResult(doc_id=inp.doc_id)
        prior_results: dict[str, Any] = {}

        logger.info("pipeline_start", doc_id=inp.doc_id, steps=self._order)

        for step_name in self._order:
            step = self.steps[step_name]

            # Inject prior outputs
            enriched_inp = AgentInput(
                doc_id=inp.doc_id,
                content=inp.content,
                metadata={**inp.metadata, "prior_results": prior_results},
            )

            logger.info("step_start", doc_id=inp.doc_id, step=step_name)
            output = step.agent.run(enriched_inp)
            result.outputs[step_name] = output

            if output.success:
                prior_results[step_name] = output.result

            # Check for low confidence
            if step.skip_on_low_confidence and output.confidence < step.confidence_threshold:
                if output.error:
                    result.halted = True
                    result.halt_reason = f"Step '{step_name}' failed: {output.error}"
                    logger.error("pipeline_halted", **{"doc_id": inp.doc_id, "step": step_name,
                                                       "reason": result.halt_reason})
                    break
                else:
                    # Low confidence but not an error — flag for review
                    result.needs_review = True
                    logger.warning("pipeline_low_confidence",
                                   doc_id=inp.doc_id, step=step_name,
                                   confidence=output.confidence)

            # Human-in-the-loop checkpoint for low-confidence outputs
            if output.needs_review and self.review_callback:
                approved = self.review_callback(inp.doc_id, output)
                if not approved:
                    result.halted = True
                    result.halt_reason = f"Human rejected step '{step_name}'"
                    break

        # Calculate overall confidence (min of all step confidences)
        confidences = [o.confidence for o in result.outputs.values() if o.success]
        result.overall_confidence = min(confidences) if confidences else 0.0
        result.needs_review = result.needs_review or result.overall_confidence < self.auto_approve_threshold

        logger.info(
            "pipeline_complete",
            doc_id=inp.doc_id,
            steps_run=len(result.outputs),
            confidence=result.overall_confidence,
            needs_review=result.needs_review,
            halted=result.halted,
        )
        return result

    def _topological_sort(self) -> list[str]:
        """Return step names in dependency-respecting order."""
        visited: set[str] = set()
        order: list[str] = []

        def visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            step = self.steps.get(name)
            if step:
                for dep in step.depends_on:
                    if dep in self.steps:
                        visit(dep)
            order.append(name)

        for name in self.steps:
            visit(name)
        return order
