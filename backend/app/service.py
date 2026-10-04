import re

from .config import Settings
from .schemas import FinalOutput, GuardrailOutput
from .tools import guardrail_check
from .workflow import (
    classify_message,
    draft_response,
    make_decision,
    retrieve_faq,
    run_live_crewai,
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
        self._session_history: dict[str, list[dict[str, str]]] = {}
        self._response_cache: dict[tuple[str, str, str], FinalOutput] = {}

    def process_message(self, user_message: str, session_id: str | None = None) -> FinalOutput:
        normalized = user_message.strip().lower()

        guardrails = guardrail_check(user_message)
        session: dict[str, str] = {}
        if session_id:
            session = self._session_memory.setdefault(session_id, {})
            self._remember_user_facts(user_message, session)
            memory_response = self._answer_from_session_memory(normalized, session)
            if memory_response is not None and not guardrails.should_refuse:
                session["last_message"] = user_message
                self._record_history(session_id, user_message, memory_response.final_response)
                return memory_response

        small_talk_response = self._answer_small_talk(normalized)
        if small_talk_response is not None and not guardrails.should_refuse:
            if session_id:
                session["last_message"] = user_message
                self._record_history(session_id, user_message, small_talk_response.final_response)
            return small_talk_response

        facts_fingerprint = "|".join(
            f"{key}={session[key]}" for key in ("name", "child_age") if session.get(key)
        )
        cache_key = (session_id or "", normalized, facts_fingerprint)
        if cache_key in self._response_cache:
            return self._response_cache[cache_key]

        faq_message = self._contextualize_faq_question(user_message, normalized, session)
        if self.settings.enable_live_crewai and not guardrails.should_refuse:
            try:
                result = run_live_crewai(
                    self.settings,
                    self.faq_dataset,
                    faq_message,
                    self._known_context(session),
                    self._history_text(session_id),
                )
                if session_id:
                    session["last_message"] = user_message
                    self._record_history(session_id, user_message, result.final_response)
                self._response_cache[cache_key] = result
                return result
            except Exception:
                # A model outage must not break the FAQ assistant. The validated,
                # deterministic pipeline below remains the safe fallback.
                pass

        # The exact four sequential handoffs are locally validated Pydantic models.
        classification = classify_message(self.faq_dataset, faq_message, guardrails)
        retrieval = retrieve_faq(self.faq_dataset, faq_message, classification)
        draft = draft_response(retrieval)
        decision = make_decision(classification, retrieval, guardrails, user_message)
        result = self._finalize(classification, retrieval, draft, decision, guardrails)

        if session_id:
            self._session_memory.setdefault(session_id, {})["last_message"] = user_message
            self._record_history(session_id, user_message, result.final_response)
        self._response_cache[cache_key] = result
        return result

    @staticmethod
    def _known_context(session: dict[str, str]) -> str:
        labels = {"name": "Parent name", "child_age": "Child age"}
        parts = [
            f"{labels[key]}: {session[key]}"
            for key in labels
            if session.get(key)
        ]
        return "; ".join(parts) if parts else "No known parent or child details yet."

    def _history_text(self, session_id: str | None) -> str:
        if not session_id:
            return "(no prior messages in this conversation)"
        recent = self._session_history.get(session_id, [])[-6:]
        if not recent:
            return "(no prior messages in this conversation)"
        return "\n".join(
            ("Parent" if turn["role"] == "user" else "Assistant") + ": " + turn["text"]
            for turn in recent
        )

    def _record_history(self, session_id: str, user_message: str, assistant_message: str) -> None:
        history = self._session_history.setdefault(session_id, [])
        history.extend(
            [
                {"role": "user", "text": user_message},
                {"role": "assistant", "text": assistant_message},
            ]
        )
        del history[:-12]

    @staticmethod
    def _remember_user_facts(user_message: str, session: dict[str, str]) -> None:
        """Keep a few user-provided facts in process memory for this session only."""
        name_marker = re.search(r"\bmy\s+name\s+is\s+", user_message, flags=re.IGNORECASE)
        if name_marker:
            name_tail = user_message[name_marker.end():]
            candidate = re.split(
                r"[,.;!?]|\b(?:and|but|i\s+have|i've)\b",
                name_tail,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip()
            if re.fullmatch(r"[A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*){0,2}", candidate):
                session["name"] = " ".join(part[:1].upper() + part[1:] for part in candidate.split())

        age_match = re.search(
            r"\b(\d{1,2})\s*(?:years?|yrs?|year|yr)\s*old\b",
            user_message,
            flags=re.IGNORECASE,
        )
        if age_match and (
            re.search(r"\b(?:child|kid|kis|son|daughter)\b", user_message, flags=re.IGNORECASE)
            or (
                "octakidz" in user_message.lower()
                and re.search(
                    r"\b(?:join|use|suitable|eligible|participate)\b",
                    user_message,
                    flags=re.IGNORECASE,
                )
            )
        ):
            session["child_age"] = age_match.group(1)

    @staticmethod
    def _answer_small_talk(normalized_message: str) -> FinalOutput | None:
        greeting = re.sub(r"[^a-z\s]", "", normalized_message).strip()
        if greeting in {"hi", "hello", "hey", "hi there", "hello there", "hey there"}:
            return FinalOutput(
                final_response=(
                    "Hi! I’m the OctaKidz Parent Assistant. Ask me about activities, "
                    "eligibility, books, reports, pricing or onboarding."
                ),
                category="Conversation",
                confidence="high",
                escalated=False,
                lead_capture_requested=False,
                internal_note="Handled as conversational greeting.",
            )
        return None

    @staticmethod
    def _contextualize_faq_question(
        user_message: str, normalized_message: str, session: dict[str, str]
    ) -> str:
        asks_about_eligibility = (
            "octakidz" in normalized_message
            and bool(re.search(
                r"\b(?:join|use|suitable|eligible|participate)\b",
                normalized_message,
            ))
        )
        if asks_about_eligibility and (
            session.get("child_age")
            or re.search(r"\b\d{1,2}\s*(?:years?|yrs?|year|yr)\s*old\b", normalized_message)
        ):
            return "What age group is OctaKidz for?"
        return user_message

    @staticmethod
    def _answer_from_session_memory(
        normalized_message: str, session: dict[str, str]
    ) -> FinalOutput | None:
        asks_for_name = bool(re.search(
            r"\b(?:what(?:'s| is) my name|do you remember my name|who am i)\b",
            normalized_message,
        ))
        if asks_for_name and session.get("name"):
            return FinalOutput(
                final_response=f"Your name is {session['name']}.",
                category="Conversation memory",
                confidence="high",
                escalated=False,
                lead_capture_requested=False,
                internal_note="Answered from user-provided session memory.",
            )

        asks_for_child_age = bool(re.search(
            r"\bhow old is my (?:child|kid|son|daughter)\b",
            normalized_message,
        ))
        if asks_for_child_age and session.get("child_age"):
            return FinalOutput(
                final_response=f"You told me your child is {session['child_age']} years old.",
                category="Conversation memory",
                confidence="high",
                escalated=False,
                lead_capture_requested=False,
                internal_note="Answered from user-provided session memory.",
            )

        return None

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
