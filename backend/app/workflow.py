from __future__ import annotations

import json
from typing import TYPE_CHECKING

from .schemas import (
    ClassificationOutput,
    DecisionOutput,
    DraftOutput,
    GuardrailOutput,
    RetrievalOutput,
)
from .tools import escalation_decision, faq_lookup, guardrail_check

if TYPE_CHECKING:
    from .config import Settings
    from .schemas import FAQDataset


def classify_message(
    faq_dataset: "FAQDataset", user_message: str, guardrails: GuardrailOutput
) -> ClassificationOutput:
    """Agent 1 handoff: classifies only; it does not write a customer response."""
    rough_match = faq_lookup(faq_dataset, user_message)
    return ClassificationOutput(
        category=rough_match.category if rough_match.matched else "Uncategorized",
        intent="faq_question" if rough_match.matched else "support_or_unknown",
        urgency=(
            "urgent"
            if any(word in user_message.lower() for word in ("urgent", "complaint", "dissatisfied"))
            else "normal"
        ),
        possible_guardrail_flags=guardrails.flags,
    )


def retrieve_faq(
    faq_dataset: "FAQDataset", user_message: str, classification: ClassificationOutput
) -> RetrievalOutput:
    """Agent 2 handoff: retrieves only from the validated local FAQ data."""
    category = classification.category if classification.category != "Uncategorized" else None
    return RetrievalOutput.model_validate(
        faq_lookup(faq_dataset, user_message, category).model_dump()
    )


def draft_response(retrieval: RetrievalOutput) -> DraftOutput:
    """Agent 3 handoff: drafts only using the retrieved answer and supplied URL."""
    if retrieval.matched and retrieval.answer:
        response = retrieval.answer
        if retrieval.link_text and retrieval.url:
            response += f" {retrieval.link_text}: {retrieval.url}"
        return DraftOutput(draft_response=response, used_faq_id=retrieval.faq_id)
    return DraftOutput(
        draft_response=(
            "I want to make sure you get the right information. "
            "The OctaKidz team should help with this."
        )
    )


def make_decision(
    classification: ClassificationOutput,
    retrieval: RetrievalOutput,
    guardrails: GuardrailOutput,
    user_message: str,
) -> DecisionOutput:
    """Agent 4 handoff: packages deterministic escalation/refusal information."""
    data = escalation_decision(
        classification.category, retrieval.confidence, guardrails.flags, user_message
    )
    return DecisionOutput(
        escalated=bool(data["escalated"]),
        lead_capture_requested=bool(data["lead_capture_requested"]),
        internal_reason=str(data["reason"]),
        refusal=guardrails.should_refuse,
    )


def build_crewai_workflow(settings: "Settings", faq_dataset: "FAQDataset"):
    """Build exactly four sequential CrewAI agents; this function makes no API call."""
    from crewai import Agent, Crew, Process, Task
    from crewai.tools import tool

    @tool("faq_lookup")
    def faq_lookup_tool(user_message: str, category: str = "") -> str:
        """Retrieve validated local FAQ data with transparent token-overlap matching."""
        return faq_lookup(faq_dataset, user_message, category or None).model_dump_json()

    @tool("guardrail_check")
    def guardrail_tool(user_message: str) -> str:
        """Ask the server's deterministic safety checker to inspect a message."""
        return guardrail_check(user_message).model_dump_json()

    @tool("escalation_decision")
    def escalation_tool(category: str, confidence: float, flags: str, user_message: str) -> str:
        """Request a deterministic escalation decision from local server policy."""
        parsed_flags = json.loads(flags) if flags else []
        return json.dumps(escalation_decision(category, confidence, parsed_flags, user_message))

    classifier = Agent(
        role="Classification agent",
        goal="Classify category, intent, urgency, and guardrail risks; never answer the customer.",
        backstory="You route parent questions carefully.",
        llm=settings.model_name,
        verbose=False,
    )
    retriever = Agent(
        role="FAQ retrieval agent",
        goal="Retrieve only validated FAQ information and report confidence truthfully.",
        backstory="You never invent a FAQ match.",
        tools=[faq_lookup_tool],
        llm=settings.model_name,
        verbose=False,
    )
    drafter = Agent(
        role="Response drafting agent",
        goal="Write concise, warm, parent-friendly replies from retrieved FAQ information only.",
        backstory="You do not guess when retrieval is uncertain.",
        llm=settings.model_name,
        verbose=False,
    )
    decider = Agent(
        role="Escalation and refusal agent",
        goal="Use local safety and escalation outcomes without exposing internal details.",
        backstory="You preserve the server's safety policy.",
        tools=[guardrail_tool, escalation_tool],
        llm=settings.model_name,
        verbose=False,
    )
    tasks = [
        Task(
            description="Classify {user_message}. Do not answer the customer.",
            expected_output="Classification JSON.",
            agent=classifier,
            output_pydantic=ClassificationOutput,
        ),
        Task(
            description="Retrieve a FAQ match for {user_message} using the local tool only.",
            expected_output="Retrieval JSON.",
            agent=retriever,
            output_pydantic=RetrievalOutput,
        ),
        Task(
            description="Draft a response based only on retrieved FAQ content.",
            expected_output="Draft JSON.",
            agent=drafter,
            output_pydantic=DraftOutput,
        ),
        Task(
            description="Apply safety and escalation policy for {user_message}.",
            expected_output="Decision JSON.",
            agent=decider,
            output_pydantic=DecisionOutput,
        ),
    ]
    return Crew(
        agents=[classifier, retriever, drafter, decider],
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
    )


def run_optional_live_crewai(
    settings: "Settings", faq_dataset: "FAQDataset", user_message: str
) -> None:
    """Optional model pass. Deterministic Python still supplies every final decision."""
    if settings.enable_live_crewai:
        build_crewai_workflow(settings, faq_dataset).kickoff(inputs={"user_message": user_message})
