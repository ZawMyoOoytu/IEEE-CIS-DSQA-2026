from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.llm.mock import (
    MockLLMBackend,
)

from src.pipeline.integrated import (
    run_integrated_pipeline,
)


QUESTION = (
    "What is IFMIF-DONES?"
)


def print_result(
    result,
) -> None:
    print()
    print(
        f"Question              : "
        f"{result.question}"
    )

    print(
        f"Question Type         : "
        f"{result.retrieval.analysis.question_type}"
    )

    print(
        f"Difficulty             : "
        f"{result.retrieval.analysis.difficulty}"
    )

    print(
        f"Recommended Top-K      : "
        f"{result.retrieval.analysis.recommended_top_k}"
    )

    print(
        f"Retrieved Evidence     : "
        f"{result.retrieval.retrieved_count}"
    )

    print(
        f"Highest Trust          : "
        f"{result.retrieval.highest_trust_score:.4f}"
    )

    print(
        f"Average Trust          : "
        f"{result.retrieval.average_trust_score:.4f}"
    )

    print(
        f"Reasoning Evidence     : "
        f"{result.reasoning_context.evidence_count}"
    )

    print(
        f"Verification Required  : "
        f"{result.reasoning_context.verification_required}"
    )

    print(
        f"Backend                : "
        f"{result.generation.backend_name}"
    )

    print(
        f"Model                  : "
        f"{result.generation.model_name}"
    )

    print(
        f"Generation Success     : "
        f"{result.generation.success}"
    )

    print(
        f"Prompt Tokens          : "
        f"{result.generation.prompt_tokens}"
    )

    print(
        f"Completion Tokens      : "
        f"{result.generation.completion_tokens}"
    )

    print(
        f"Total Tokens           : "
        f"{result.generation.total_tokens}"
    )

    print(
        f"Generation Time        : "
        f"{result.generation.generation_time_seconds:.6f}s"
    )

    print()
    print(
        "Generated Answer:"
    )

    print(
        result.final_answer
    )

    print()
    print(
        f"Evidence Coverage      : "
        f"{result.verification.evidence_coverage:.4f}"
    )

    print(
        f"Citation Coverage      : "
        f"{result.verification.citation_coverage:.4f}"
    )

    print(
        f"Trust Coverage         : "
        f"{result.verification.trust_coverage:.4f}"
    )

    print(
        f"Unsupported Claims     : "
        f"{result.verification.unsupported_claim_ratio:.4f}"
    )

    print(
        f"Verification Score     : "
        f"{result.verification.verification_score:.4f}"
    )

    print(
        f"Accepted               : "
        f"{result.accepted}"
    )


def main() -> None:
    print("=" * 78)
    print(
        "IEEE CIS DSQA 2026 — "
        "Phase 4.1 Backend Injection Test"
    )
    print("=" * 78)

    # -------------------------------------------------
    # Test 1: Explicit MockLLMBackend injection
    # -------------------------------------------------

    print()
    print(
        "Test 1: Explicit LLM backend injection"
    )
    print("-" * 78)

    backend = MockLLMBackend()

    result = run_integrated_pipeline(
        question=QUESTION,
        llm_backend=backend,
    )

    print_result(
        result
    )

    if (
        result.generation.backend_name
        != "mock"
    ):
        raise RuntimeError(
            "Injected backend was not used."
        )

    if (
        result.generation.model_name
        != "mock-grounded-reasoner"
    ):
        raise RuntimeError(
            "Unexpected model name."
        )

    if not result.generation.success:
        raise RuntimeError(
            "Injected backend generation failed."
        )

    if not result.final_answer:
        raise RuntimeError(
            "Generated answer is empty."
        )

    if not result.verification.verified:
        raise RuntimeError(
            "Grounded answer failed verification."
        )

    if not result.accepted:
        raise RuntimeError(
            "Grounded answer was not accepted."
        )

    print()
    print(
        "Result                : PASS"
    )

    # -------------------------------------------------
    # Test 2: Default backend
    # -------------------------------------------------

    print()
    print(
        "Test 2: Default backend fallback"
    )
    print("-" * 78)

    default_result = (
        run_integrated_pipeline(
            question=QUESTION
        )
    )

    print(
        f"Backend               : "
        f"{default_result.generation.backend_name}"
    )

    print(
        f"Model                 : "
        f"{default_result.generation.model_name}"
    )

    print(
        f"Accepted              : "
        f"{default_result.accepted}"
    )

    if (
        default_result.generation.backend_name
        != "mock"
    ):
        raise RuntimeError(
            "Default backend is not MockLLMBackend."
        )

    if not default_result.accepted:
        raise RuntimeError(
            "Default backend result was rejected."
        )

    print()
    print(
        "Result                : PASS"
    )

    # -------------------------------------------------
    # Test 3: Hallucination rejection
    # -------------------------------------------------

    print()
    print(
        "Test 3: Verification still rejects "
        "unsupported answers"
    )
    print("-" * 78)

    class HallucinationBackend(
        MockLLMBackend
    ):
        """
        Test-only backend that deliberately produces
        an unsupported answer.
        """

        def generate(
            self,
            context,
            prompt,
        ):
            from src.llm.base import (
                LLMGenerationResult,
            )

            return LLMGenerationResult(
                answer=(
                    "IFMIF-DONES is a commercial "
                    "spacecraft manufacturing facility "
                    "located on Mars."
                ),
                model_name=(
                    "mock-hallucination-test"
                ),
                prompt_tokens=10,
                completion_tokens=12,
                total_tokens=22,
                generation_time_seconds=0.0,
                backend_name=(
                    "mock-hallucination"
                ),
                success=True,
                error=None,
            )

    hallucination_result = (
        run_integrated_pipeline(
            question=QUESTION,
            llm_backend=(
                HallucinationBackend()
            ),
        )
    )

    print(
        f"Backend               : "
        f"{hallucination_result.generation.backend_name}"
    )

    print(
        f"Generated Answer      : "
        f"{hallucination_result.final_answer}"
    )

    print(
        f"Evidence Coverage     : "
        f"{hallucination_result.verification.evidence_coverage:.4f}"
    )

    print(
        f"Unsupported Claims    : "
        f"{hallucination_result.verification.unsupported_claim_ratio:.4f}"
    )

    print(
        f"Accepted              : "
        f"{hallucination_result.accepted}"
    )

    if hallucination_result.accepted:
        raise RuntimeError(
            "Unsupported answer was incorrectly accepted."
        )

    print()
    print(
        "Result                : PASS"
    )

    # -------------------------------------------------
    # Final
    # -------------------------------------------------

    print()
    print("=" * 78)
    print(
        "Phase 4.1 backend injection test: PASS"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()