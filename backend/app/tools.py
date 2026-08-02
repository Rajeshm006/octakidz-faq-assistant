import re
from typing import Optional

from .schemas import FAQDataset, GuardrailOutput, LookupOutput


STOP_WORDS = {
    "a", "an", "and", "are", "can", "do", "does", "for", "how", "i", "is", "it",
    "me", "my", "of", "or", "the", "this", "to", "what", "will", "with", "you",
    "your", "please", "about", "tell",
}
LOW_CONFIDENCE_THRESHOLD = 0.28


def token_set(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in STOP_WORDS and len(token) > 1
    }


def faq_lookup(
    faq_dataset: FAQDataset, user_message: str, category: Optional[str] = None
) -> LookupOutput:
    """Transparent token-overlap lookup over the validated local FAQ list only."""
    message_tokens = token_set(user_message)
    candidates = [
        entry for entry in faq_dataset.questions if not category or entry.category == category
    ] or faq_dataset.questions

    best_entry = None
    best_score = 0.0
    for entry in candidates:
        question_tokens = token_set(entry.question)
        score = len(message_tokens & question_tokens) / max(len(question_tokens), 1)
        if score > best_score:
            best_entry, best_score = entry, score

    if best_entry is None or best_score < LOW_CONFIDENCE_THRESHOLD:
        return LookupOutput(matched=False, confidence=round(best_score, 2))

    return LookupOutput(
        matched=True,
        faq_id=best_entry.id,
        category=best_entry.category,
        answer=best_entry.answer,
        link_text=best_entry.link_text,
        url=best_entry.url,
        confidence=round(best_score, 2),
    )


def guardrail_check(user_message: str) -> GuardrailOutput:
    """Detect refusal requests without relying on an LLM."""
    text = user_message.lower()
    flags: list[str] = []

    if any(term in text for term in (
        "ignore previous", "ignore your", "hidden prompt", "system prompt",
        "change your rules", "reveal instructions",
    )):
        flags.append("prompt_injection")
    if any(term in text for term in (
        "password", "api key", "credential", "admin access", "admin password",
        "private system", "database access",
    )):
        flags.append("credentials_or_internal_access")
    if (
        any(term in text for term in ("another customer", "other customer", "other user's"))
        and any(term in text for term in ("phone", "email", "contact", "record", "account"))
    ):
        flags.append("private_data_request")
    if any(term in text for term in (
        "bypass security", "scrape private", "attack system", "malicious code", "hack ",
    )):
        flags.append("harmful_security_request")
    if any(term in text for term in (
        "guarantee", "medical", "clinical", "psychological", "legal advice", "financial advice",
    )):
        flags.append("professional_or_guarantee_claim")
    if any(term in text for term in ("competitor", "better than", "worse than")):
        flags.append("unsupported_competitor_claim")

    should_refuse = bool(flags)
    return GuardrailOutput(
        flags=flags,
        should_refuse=should_refuse,
        recommended_action="refuse_without_lead_capture" if should_refuse else "continue",
    )


def escalation_decision(
    category: str, confidence: float, guardrail_flags: list[str], user_message: str
) -> dict[str, object]:
    """Apply the OctaKidz escalation policy deterministically."""
    text = user_message.lower()
    security_flags = {
        "prompt_injection",
        "credentials_or_internal_access",
        "private_data_request",
        "harmful_security_request",
    }
    if security_flags.intersection(guardrail_flags):
        return {
            "escalated": False,
            "lead_capture_requested": False,
            "reason": "security refusal",
        }

    triggers = {
        "low confidence": confidence < LOW_CONFIDENCE_THRESHOLD,
        "payment or account-access issue": "paid" in text and any(
            word in text for word in ("access", "login", "account")
        ),
        "damaged or missing delivery": any(
            phrase in text for phrase in ("damaged", "missing", "not received", "did not receive")
        ),
        "privacy or account-specific help": any(
            phrase in text
            for phrase in ("delete my data", "deletion", "privacy", "my account", "my order")
        ),
        "sensitive or personalized advice": any(
            phrase in text
            for phrase in ("clinical", "special needs", "legal", "personalized advice")
        ),
        "commercial exception": any(
            phrase in text
            for phrase in ("refund", "discount", "negotiate", "exception", "custom arrangement")
        ),
        "urgent complaint": any(
            phrase in text for phrase in ("urgent", "complaint", "dissatisfied", "unhappy")
        ),
    }
    reasons = [reason for reason, active in triggers.items() if active]
    escalated = bool(reasons)
    return {
        "escalated": escalated,
        "lead_capture_requested": escalated,
        "reason": "; ".join(reasons) if reasons else "no escalation trigger",
    }

