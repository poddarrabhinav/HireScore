from typing import Optional

from pydantic import BaseModel


class StageScores(BaseModel):
    bm25_score: float
    semantic_score: float
    combined_score: float
    llm_score: Optional[float] = None


class CandidateProfile(BaseModel):
    strengths: list[str]
    weaknesses: list[str]
    unknowns: list[str]
    values_alignment: str  # "strong" | "moderate" | "weak" | "unclear"
    values_evidence: str
    verdict: str


class ResumeResult(BaseModel):
    filename: str
    final_score: float
    stage_scores: StageScores
    key_skills: list[str] = []
    missing_skills: list[str] = []
    resume_text: Optional[str] = None
    profile: Optional[CandidateProfile] = None
    stage_eliminated: Optional[int] = None
    rank: int


class JudgeSingleRequest(BaseModel):
    resume_text: str
    resume_filename: str
    job_description: str
    company_values: str = ""


class BatchResponse(BaseModel):
    results: list[ResumeResult]
    job_description: str = ""
    company_values: str = ""
    job_keywords: list[str] = []
    core_skills: list[str] = []
    adjacent_skills: list[str] = []
    soft_skills: list[str] = []
    value_keywords: list[str] = []
    excluded_skills: list[str] = []
    role_type: str = ""
    embedding_profile: str = ""
    embedding_options: list[dict] = []
    total_resumes: int
    stage1_survivors: int
    stage2_survivors: int
    stage3_evaluated: int
    stage3_survivors: int
    pipeline_stats: dict
