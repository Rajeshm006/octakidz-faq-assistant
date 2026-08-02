from .config import Settings
from .schemas import FinalOutput, GuardrailOutput
from .tools import guardrail_check
from .workflow import (
    classify_message,
    draft_response,
    make_decision,
    retrieve_faq,
    run_optional_live_crewai,
)


REFUSAL_RESPONSE = (
    "I’m sorry, but I can’t help with that request. "
    "For a legitimate OctaKidz question, I’m happy to help with information from our FAQ."
)


class FAQAssistantService:
    """Application service: strict four handoffs followed by a deterministic finalizer."""

    def __init__(self, faq_dataset, settings: Settings):
        self.faq_dataset = faq_dataset
        self.settings = settings
        self._session_memory: dict[str, dict[str, str]] = {}
        self._response_cache: dict[tuple[str, str], FinalOutput] = {}

    def process_message(self, user_message: str, session_id: str | None = None) -> FinalOutput:
        normalized = user_message.strip().lower()
        cache_key = (session_id or "", normalized)
        if cache_key in self._response_cache:
            return self._response_cache[cache_key]

        guardrails = guardrail_check(user_message)
        # The exact four sequential handoffs are locally validated Pydantic models.
        classification = classify_message(self.faq_dataset, user_message, guardrails)
        retrieval = retrieve_faq(self.faq_dataset, user_message, classification)
        draft = draft_response(retrieval)
        decision = make_decision(classification, retrieval, guardrails, user_message)
        # An optional CrewAI pass may assist operations, but cannot override this finalizer.
        run_optional_live_crewai(self.settings, self.faq_dataset, user_message)
        result = self._finalize(classification, retrieval, draft, decision, guardrails)

        if session_id:
            self._session_memory.setdefault(session_id, {})["last_message"] = user_message
        self._response_cache[cache_key] = result
        return result

    @staticmethod
    def _finalize(classification, retrieval, draft, decision, guardrails: GuardrailOutput) -> FinalOutput:
        """Final safety layer; client-visible facts come only from retrieval.answer."""
        if guardrails.should_refuse:
            return FinalOutput(
                final_response=REFUSAL_RESPONSE,
                category=classification.category,
                confidence="none",
                escalated=False,
                lead_capture_requested=False,
                internal_note="Security/professional-claim refusal: " + ", ".join(guardrails.flags),
            )
        if not retrieval.matched:
            return FinalOutput(
                final_response=(
                    "I want to make sure you get the right information. "
                    "The OctaKidz team should help with this."
                ),
                category=classification.category,
                confidence="low",
                escalated=True,
                lead_capture_requested=True,
                internal_note=decision.internal_reason,
            )

        final_response = draft.draft_response
        if decision.escalated:
            final_response += " The OctaKidz team should help with the next steps."
        return FinalOutput(
            final_response=final_response,
            category=retrieval.category,
            confidence="high" if retrieval.confidence >= 0.60 else "medium",
            escalated=decision.escalated,
            lead_capture_requested=decision.lead_capture_requested,
            internal_note=decision.internal_reason,
        )
