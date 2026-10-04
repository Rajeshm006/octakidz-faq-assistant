from __future__ import annotations

import json
from typing import TYPE_CHECKING

from .schemas import (
    ClassificationOutput,
    DecisionOutput,
    DraftOutput,
    FinalOutput,
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
    """Build the notebook's four-agent workflow for one isolated API request."""
    from crewai import Agent, Crew, LLM, Process, Task
    from crewai.tools import tool

    llm = LLM(model=settings.model_name)

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
        role="Parent Query Classification Agent",
        goal=(
            "Classify every parent message into the correct OctaKidz FAQ category and detect "
            "guardrail signals before any answer is generated."
        ),
        backstory=(
            "You are the first point of contact for the OctaKidz Parent Assistant. You triage "
            "messages and use conversation context, but never answer the parent yourself."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
    retriever = Agent(
        role="Knowledge & Policy Reasoning Agent",
        goal=(
            "Retrieve the best matching answer from the curated OctaKidz FAQ and report "
            "confidence honestly rather than guessing."
        ),
        backstory="You trust only the supplied FAQ tool and never invent product facts.",
        tools=[faq_lookup_tool],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
    drafter = Agent(
        role="Response Drafting Agent",
        goal=(
            "Turn the retrieved FAQ answer into a warm, concise, parent-friendly reply without "
            "adding facts that are absent from the FAQ result."
        ),
        backstory=(
            "You write in the OctaKidz voice: simple, supportive and never over-promising."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
    decider = Agent(
        role="Escalation & Lead Decision Agent",
        goal=(
            "Apply the deterministic escalation decision and package the final parent-facing "
            "response without exposing internal details."
        ),
        backstory="You are the workflow's safety net and must follow tool policy exactly.",
        tools=[guardrail_tool, escalation_tool],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
    tasks = [
        Task(
            description=(
                "KNOWN CONTEXT: {known_context}\n\nRECENT CONVERSATION:\n{history}\n\n"
                "Classify the parent's new message, {user_message}, into the best FAQ category. "
                "Call the guardrail tool on the raw message, use context to resolve references, "
                "and return only the structured classification. Do not answer the parent."
            ),
            expected_output="Classification JSON.",
            agent=classifier,
            output_pydantic=ClassificationOutput,
        ),
        Task(
            description=(
                "Using the category from the classification and the original message "
                "{user_message}, call the FAQ lookup tool. Return its result faithfully. "
                "Never invent an answer or claim a stronger confidence than the tool reports."
            ),
            expected_output="Retrieval JSON.",
            agent=retriever,
            context=[],
            output_pydantic=RetrievalOutput,
        ),
        Task(
            description=(
                "KNOWN CONTEXT: {known_context}\n\nDraft a warm, concise reply to {user_message} "
                "using only the retrieved FAQ answer and its optional link. Personalize with known "
                "context only when relevant. If no FAQ matched, use the standard team-help fallback."
            ),
            expected_output="Draft JSON.",
            agent=drafter,
            context=[],
            output_pydantic=DraftOutput,
        ),
        Task(
            description=(
                "For {user_message}, call the guardrail and escalation tools using the previous "
                "outputs. Package the draft into the exact FinalOutput schema. Preserve the FAQ "
                "category and confidence, refuse guardrail violations, and request lead capture "
                "only when the escalation tool requires it. Do not add new product facts."
            ),
            expected_output="FinalOutput JSON.",
            agent=decider,
            context=[],
            output_pydantic=FinalOutput,
        ),
    ]
    tasks[1].context = [tasks[0]]
    tasks[2].context = [tasks[0], tasks[1]]
    tasks[3].context = [tasks[0], tasks[1], tasks[2]]
    return Crew(
        agents=[classifier, retriever, drafter, decider],
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
    )


def run_live_crewai(
    settings: "Settings",
    faq_dataset: "FAQDataset",
    user_message: str,
    known_context: str,
    history: str,
) -> FinalOutput:
    """Run the notebook-derived LLM workflow and validate its final structured output."""
    result = build_crewai_workflow(settings, faq_dataset).kickoff(
        inputs={
            "user_message": user_message,
            "known_context": known_context,
            "history": history,
        }
    )
    if result.pydantic is not None:
        return FinalOutput.model_validate(result.pydantic)
    return FinalOutput.model_validate_json(result.raw)
