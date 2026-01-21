

from typing import Any
from pydantic import BaseModel, Field

from deep_research.state import ResearchTask


class PlannerOutput(BaseModel):
    """What Planner LLM generates"""
    research_plan: str = Field(description="High-level research strategy")
    tasks: list[ResearchTask] = Field(description="3-7 parallel research tasks")

class TaskExecutionOutput(BaseModel):
    """What each Task LLM generates"""
    reasoning: str = Field(description="Step-by-step reasoning")
    structured_output: dict[str, Any] = Field(description="Findings matching required schema")
    key_insights: list[str] = Field(description="3-5 key takeaways")

class FinalReport(BaseModel):
    """What Observer LLM generates"""
    executive_summary: str = Field(description="8-10 sentence answer")
    detailed_findings: dict[str, Any] = Field(description="Comprehensive findings")
    key_insights: list[str] = Field(description="Most important insights")
    methodology: str = Field(description="Research approach used")
    all_citations: list[str] = Field(description="All unique URLs")
    confidence_score: float = Field(description="0.0 to 1.0", ge=0.0, le=1.0)

class ResearchEvaluation(BaseModel):
        is_complete: bool = Field(description="Is the research comprehensive enough?")
        confidence: float = Field(description="Confidence 0-1 in completeness")
        gaps: list[str] = Field(description="What information is missing?")
        suggested_follow_ups: list[str] = Field(description="Suggested follow-up research areas")