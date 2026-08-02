from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FAQEntry(BaseModel):
    """A validated FAQ entry. Only these answers may provide business facts."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    link_text: Optional[str] = None
    url: Optional[str] = None


class FAQDataset(BaseModel):
    """The source dataset permits top-level metadata but requires questions."""

    model_config = ConfigDict(extra="allow")
    questions: list[FAQEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def ids_must_be_unique(self) -> "FAQDataset":
        ids = [entry.id for entry in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("FAQ entry ids must be unique.")
        return self


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=2000)
    session_id: Optional[str] = Field(default=None, max_length=128)


class GuardrailOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flags: list[str]
    should_refuse: bool
    recommended_action: str


class LookupOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    matched: bool
    faq_id: Optional[str] = None
    category: str = "Uncategorized"
    answer: Optional[str] = None
    link_text: Optional[str] = None
    url: Optional[str] = None
    confidence: float = Field(ge=0, le=1)


class ClassificationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    intent: str
    urgency: str
    possible_guardrail_flags: list[str]


class RetrievalOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    matched: bool
    faq_id: Optional[str] = None
    category: str
    answer: Optional[str] = None
    link_text: Optional[str] = None
    url: Optional[str] = None
    confidence: float = Field(ge=0, le=1)


class DraftOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    draft_response: str
    used_faq_id: Optional[str] = None


class DecisionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    escalated: bool
    lead_capture_requested: bool
    internal_reason: str
    refusal: bool


class FinalOutput(BaseModel):
    """The exact safe response contract returned by POST /api/chat."""

    model_config = ConfigDict(extra="forbid")

    final_response: str
    category: str
    confidence: str = Field(pattern="^(high|medium|low|none)$")
    escalated: bool
    lead_capture_requested: bool
    internal_note: str

