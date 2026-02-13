"""Pydantic structured output models for agent responses."""

from pydantic import BaseModel


class ArticleCitation(BaseModel):
    article: str
    title: str
    relevance: str


class RiskClassification(BaseModel):
    risk_level: str  # "unacceptable", "high", "limited", "minimal"
    confidence: str  # "high", "medium", "low"
    reasoning: str
    citations: list[ArticleCitation]
    prohibited_check: str
    high_risk_check: str
    transparency_check: str


class Obligation(BaseModel):
    obligation: str
    article: str
    description: str
    priority: str  # "critical", "important", "recommended"


class ObligationChecklist(BaseModel):
    role: str  # "provider", "deployer", "importer", "distributor"
    risk_level: str
    obligations: list[Obligation]
    summary: str
    citations: list[ArticleCitation]


class RegulationAnswer(BaseModel):
    answer: str
    citations: list[ArticleCitation]
    confidence: str


class OrchestratorReport(BaseModel):
    system_description: str
    classification: RiskClassification | None = None
    obligations: ObligationChecklist | None = None
    additional_qa: list[RegulationAnswer] | None = None
    executive_summary: str
