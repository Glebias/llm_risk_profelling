from __future__ import annotations

from pydantic import BaseModel, Field, conint


class JudgeReasoning(BaseModel):
    coverage: str = ""
    precision: str = ""
    safety: str = ""
    usefulness: str = ""


class JudgeResult(BaseModel):
    coverage: conint(ge=0, le=10)
    precision: conint(ge=0, le=10)
    safety: conint(ge=0, le=10)
    usefulness: conint(ge=0, le=10)

    missed_information: list[str] = Field(default_factory=list)
    unsupported_recommendations: list[str] = Field(default_factory=list)
    dangerous_recommendations: list[str] = Field(default_factory=list)

    reasoning: JudgeReasoning = Field(default_factory=JudgeReasoning)