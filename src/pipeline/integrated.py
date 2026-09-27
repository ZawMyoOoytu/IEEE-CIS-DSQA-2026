from __future__ import annotations

from dataclasses import dataclass

from src.llm.base import (
    LLMBackend,
    LLMGenerationResult,
)

from src.llm.mock import (
    MockLLMBackend,
)

from src.pipeline.reasoning import (
    ReasoningContext,
    ReasoningResult,
    build_reasoning_prompt,
)

from src.retrieval.trust_aware import (
    TrustAwareRetrievalResult,
    retrieve_trust_aware,
)

from src.verification.answer_verification import (
    VerificationResult,
    verify_answer,
)


@dataclass(frozen=True)
class IntegratedPipelineResult:
    """
    End-to-end result for the DSQA reasoning pipeline.

    Pipeline:

        Question
            ↓
        Trust-Aware Retrieval
            ↓
        Reasoning Context
            ↓
        LLM Backend
            ↓
        Generated Answer
            ↓
        Answer Verification
            ↓
        Final Result
    """

    question: str

    retrieval: TrustAwareRetrievalResult

    reasoning_context: ReasoningContext

    reasoning_prompt: str

    generation: LLMGenerationResult

    reasoning_result: ReasoningResult

    verification: VerificationResult

    final_answer: str

    accepted: bool


def build_reasoning_result(
    context: ReasoningContext,
    generation: LLMGenerationResult,
) -> ReasoningResult:
    """
    Convert a standardized LLM generation result into
    the project's ReasoningResult representation.
    """

    return ReasoningResult(
        question=context.question,
        answer=generation.answer,
        evidence_count=context.evidence_count,
        verification_required=context.verification_required,
        model_name=generation.model_name,
        reasoning_status=(
            "generated"
            if generation.success
            else "failed"
        ),
    )


def run_integrated_pipeline(
    question: str,
    llm_backend: LLMBackend | None = None,
) -> IntegratedPipelineResult:
    """
    Execute the complete DSQA pipeline.

    Parameters
    ----------
    question:
        User question.

    llm_backend:
        LLM backend implementation.

        If omitted, MockLLMBackend is used. This keeps
        local development deterministic while allowing
        future OLMo-compatible backends to be injected
        without changing the retrieval or verification
        layers.

    Returns
    -------
    IntegratedPipelineResult
        Complete retrieval, reasoning, generation,
        verification, and final acceptance state.
    """

    # -------------------------------------------------
    # 1. Trust-aware retrieval
    # -------------------------------------------------

    retrieval = retrieve_trust_aware(
        question
    )

    # -------------------------------------------------
    # 2. Build reasoning context
    # -------------------------------------------------

    reasoning_context = (
        retrieval.to_reasoning_context()
    )

    # -------------------------------------------------
    # 3. Build grounded reasoning prompt
    # -------------------------------------------------

    reasoning_prompt = (
        build_reasoning_prompt(
            reasoning_context
        )
    )

    # -------------------------------------------------
    # 4. Select LLM backend
    # -------------------------------------------------

    backend = (
        llm_backend
        if llm_backend is not None
        else MockLLMBackend()
    )

    # -------------------------------------------------
    # 5. Backend health check
    # -------------------------------------------------

    if not backend.health_check():
        raise RuntimeError(
            "LLM backend health check failed: "
            f"{backend.backend_name}"
        )

    # -------------------------------------------------
    # 6. Generate answer
    # -------------------------------------------------

    generation = backend.generate(
        context=reasoning_context,
        prompt=reasoning_prompt,
    )

    # -------------------------------------------------
    # 7. Validate generation
    # -------------------------------------------------

    if not generation.success:
        raise RuntimeError(
            "LLM generation failed: "
            f"{generation.error or 'unknown error'}"
        )

    answer = str(
        generation.answer
    ).strip()

    if not answer:
        raise RuntimeError(
            "LLM backend returned an empty answer."
        )

    # -------------------------------------------------
    # 8. Convert to ReasoningResult
    # -------------------------------------------------

    reasoning_result = (
        build_reasoning_result(
            reasoning_context,
            generation,
        )
    )

    # -------------------------------------------------
    # 9. Verify generated answer
    # -------------------------------------------------

    verification = verify_answer(
        answer=answer,
        evidence=list(
            retrieval.results
        ),
        require_citation=False,
    )

    # -------------------------------------------------
    # 10. Final acceptance
    # -------------------------------------------------

    accepted = bool(
        verification.verified
    )

    return IntegratedPipelineResult(
        question=question,
        retrieval=retrieval,
        reasoning_context=reasoning_context,
        reasoning_prompt=reasoning_prompt,
        generation=generation,
        reasoning_result=reasoning_result,
        verification=verification,
        final_answer=answer,
        accepted=accepted,
    )