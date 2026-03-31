import logging
import json
import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import yaml

from app.config import get_config, get_skill_expansions
from app.models import BatchResponse, CandidateProfile, JudgeSingleRequest, ResumeResult, StageScores
from app.parser import ParseError, extract_resume_text, extract_zip_resumes
from app.pipeline import agent_judge, get_embedding_profile_options, get_llm_client, init_pipeline, load_openai_client, run_pipeline
from app.pipeline.stage2_sbert import get_default_embedding_profile, resolve_embedding_profile

load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s — %(message)s")
logger = logging.getLogger(__name__)
_CONFIG = get_config()
_SCORING = _CONFIG["scoring"]

_FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
_FRONTEND     = _FRONTEND_DIR / "index.html"
_SETTINGS_FRONTEND = _FRONTEND_DIR / "settings.html"
_EVAL_FRONTEND = _FRONTEND_DIR / "eval.html"
_SKILL_EXPANSIONS_PATH = Path(__file__).parent.parent / _CONFIG.get("skill_matching", {}).get("expansion_file", "skill_expansions.yaml")


class SkillExpansionPayload(BaseModel):
    content: str


def _clamp_unit_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _clamp_threshold(value: float, default: float, *, max_value: float = 0.95) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(max_value, numeric))


def _parse_eval_targets(payload: object) -> dict[str, float]:
    if isinstance(payload, dict):
        if "scores" in payload:
            return _parse_eval_targets(payload["scores"])
        if "labels" in payload:
            return _parse_eval_targets(payload["labels"])
        if "items" in payload:
            return _parse_eval_targets(payload["items"])
        parsed: dict[str, float] = {}
        for key, value in payload.items():
            if isinstance(value, (int, float)):
                parsed[str(key)] = _clamp_unit_score(value)
        if parsed:
            return parsed

    if isinstance(payload, list):
        parsed = {}
        for item in payload:
            if not isinstance(item, dict):
                continue
            filename = item.get("filename") or item.get("name") or item.get("resume")
            score = item.get("score", item.get("target_score", item.get("final_score", item.get("label"))))
            if filename and isinstance(score, (int, float)):
                parsed[str(filename)] = _clamp_unit_score(score)
        if parsed:
            return parsed

    raise ValueError("Labels JSON must be a filename->score mapping or a list of objects with filename and score.")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Initializing pipeline…")
    init_pipeline()
    logger.info("Pipeline ready.")
    yield


app = FastAPI(title="Resume Scorer", version="3.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000", "http://localhost:8080"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Serve frontend/css and frontend/js as static assets
app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    return HTMLResponse(content=_FRONTEND.read_text(encoding="utf-8"))


@app.get("/settings", response_class=HTMLResponse)
async def settings_page() -> HTMLResponse:
    return HTMLResponse(content=_SETTINGS_FRONTEND.read_text(encoding="utf-8"))


@app.get("/eval", response_class=HTMLResponse)
async def eval_page() -> HTMLResponse:
    return HTMLResponse(content=_EVAL_FRONTEND.read_text(encoding="utf-8"))


@app.get("/api/settings")
async def settings_data() -> dict:
    return {
        "stage1_threshold": _SCORING["stage1_threshold"],
        "stage2_threshold": _SCORING["stage2_threshold"],
        "stage3_threshold": _SCORING["stage3_threshold"],
        "stage1_mode": _SCORING.get("stage1_mode", "skill_match"),
        "use_llm": True,
        "embedding_profile": get_default_embedding_profile(),
        "embedding_options": get_embedding_profile_options(),
    }


@app.get("/api/skill-expansions")
async def skill_expansions_data() -> dict:
    content = _SKILL_EXPANSIONS_PATH.read_text(encoding="utf-8") if _SKILL_EXPANSIONS_PATH.exists() else ""
    return {
        "path": str(_SKILL_EXPANSIONS_PATH),
        "content": content,
    }


@app.post("/api/skill-expansions")
async def save_skill_expansions(payload: SkillExpansionPayload) -> dict:
    try:
        parsed = yaml.safe_load(payload.content) if payload.content.strip() else {}
    except yaml.YAMLError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid YAML: {exc}") from exc

    if parsed is None:
        parsed = {}
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=422, detail="Skill expansions must be a YAML mapping of skill -> list[str].")

    normalized: dict[str, list[str]] = {}
    for key, values in parsed.items():
        if not isinstance(key, str):
            raise HTTPException(status_code=422, detail="All skill expansion keys must be strings.")
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise HTTPException(status_code=422, detail=f"Skill '{key}' must map to a list of strings.")
        normalized[key] = values

    _SKILL_EXPANSIONS_PATH.write_text(
        yaml.safe_dump(normalized, sort_keys=True, allow_unicode=False),
        encoding="utf-8",
    )
    get_skill_expansions.cache_clear()
    return {
        "status": "ok",
        "path": str(_SKILL_EXPANSIONS_PATH),
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "llm_configured": "true" if (get_llm_client() or load_openai_client()) else "false",
        "default_embedding_profile": get_default_embedding_profile(),
    }


@app.post("/score/batch", response_model=BatchResponse)
async def score_batch(
    job_description: str = Form(default=""),
    company_values: str = Form(default=""),
    use_llm: bool = Form(default=True),
    stage1_threshold: float = Form(default=_SCORING["stage1_threshold"]),
    stage2_threshold: float = Form(default=_SCORING["stage2_threshold"]),
    stage3_threshold: float = Form(default=_SCORING["stage3_threshold"]),
    stage1_mode: str = Form(default=_SCORING.get("stage1_mode", "skill_match")),
    embedding_profile: str = Form(default=get_default_embedding_profile()),
    excluded_skills_json: str = Form(default="[]"),
    resumes: list[UploadFile] = File(...),
    jd_file: Optional[UploadFile] = File(default=None),
    values_file: Optional[UploadFile] = File(default=None),
) -> BatchResponse:
    stage1_threshold = max(0.0, min(10.0, float(stage1_threshold)))
    stage2_threshold = _clamp_threshold(stage2_threshold, _SCORING["stage2_threshold"])
    stage3_threshold = _clamp_threshold(stage3_threshold, _SCORING["stage3_threshold"])
    if stage1_mode not in ("skill_match", "ats"):
        stage1_mode = _SCORING.get("stage1_mode", "skill_match")

    try:
        excluded_skills = json.loads(excluded_skills_json or "[]")
        if not isinstance(excluded_skills, list):
            raise ValueError("excluded_skills_json must be a JSON array")
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid excluded_skills_json: {exc}") from exc

    # Resolve JD
    if jd_file and jd_file.filename:
        jd_bytes = await jd_file.read()
        try:
            job_description = extract_resume_text(jd_bytes, jd_file.filename)
        except ParseError as exc:
            raise HTTPException(status_code=422, detail=f"JD file error: {exc}") from exc

    if len(job_description.strip()) < 50:
        raise HTTPException(status_code=422, detail="Job description must be at least 50 characters.")

    # Resolve company values
    if values_file and values_file.filename:
        val_bytes = await values_file.read()
        try:
            company_values = extract_resume_text(val_bytes, values_file.filename)
        except ParseError as exc:
            raise HTTPException(status_code=422, detail=f"Company values file error: {exc}") from exc

    # Parse resumes — support individual files and ZIP archives
    parsed: list[dict] = []
    for upload in resumes:
        content = await upload.read()
        filename = upload.filename or "unknown"
        if Path(filename).suffix.lower() == ".zip":
            try:
                for fname, text in extract_zip_resumes(content):
                    parsed.append({"filename": fname, "text": text})
            except ParseError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        else:
            try:
                text = extract_resume_text(content, filename)
                parsed.append({"filename": filename, "text": text})
            except ParseError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not parsed:
        raise HTTPException(status_code=422, detail="No readable resume files found.")

    pipeline_output = await run_pipeline(
        jd_text=job_description,
        company_values=company_values,
        resumes=parsed,
        stage1_threshold=stage1_threshold,
        stage2_threshold=stage2_threshold,
        stage3_threshold=stage3_threshold,
        use_llm=use_llm,
        embedding_profile=embedding_profile,
        excluded_skills=excluded_skills,
        stage1_mode=stage1_mode,
    )

    results = [_build_result(r) for r in pipeline_output["results"]]
    stats = pipeline_output["stats"]

    return BatchResponse(
        results=results,
        job_description=job_description,
        company_values=company_values,
        total_resumes=stats["total"],
        stage1_survivors=stats["stage1_survivors"],
        stage2_survivors=stats["stage2_survivors"],
        stage3_evaluated=stats["stage3_evaluated"],
        stage3_survivors=stats["stage3_survivors"],
        job_keywords=stats.get("job_keywords", []),
        core_skills=stats.get("core_skills", []),
        adjacent_skills=stats.get("adjacent_skills", []),
        soft_skills=stats.get("soft_skills", []),
        value_keywords=stats.get("value_keywords", []),
        excluded_skills=stats.get("excluded_skills", []),
        role_type=stats.get("role_type", ""),
        embedding_profile=stats.get("embedding_profile", resolve_embedding_profile(embedding_profile)),
        embedding_options=stats.get("embedding_options", get_embedding_profile_options()),
        pipeline_stats=stats,
    )


@app.post("/score/judge-single")
async def judge_single(req: JudgeSingleRequest) -> dict:
    client = get_llm_client() or load_openai_client()
    if not client:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured - LLM judge unavailable.")
    resume = {"filename": req.resume_filename, "text": req.resume_text}
    profile = await agent_judge(client, req.job_description, req.company_values, resume)
    if "llm_score" in profile:
        profile["llm_score"] = _clamp_unit_score(profile["llm_score"])
    return profile


@app.post("/api/eval/run")
async def run_eval(
    labels_file: UploadFile = File(...),
    resumes_zip: UploadFile = File(...),
    job_description: str = Form(default=""),
    jd_file: Optional[UploadFile] = File(default=None),
    company_values: str = Form(default=""),
    values_file: Optional[UploadFile] = File(default=None),
) -> dict:
    has_jd_text = bool(job_description.strip())
    has_jd_file = bool(jd_file and jd_file.filename)
    if has_jd_text and has_jd_file:
        raise HTTPException(status_code=422, detail="Provide either JD text or a JD file for evaluation, not both.")
    if not has_jd_text and not has_jd_file:
        raise HTTPException(status_code=422, detail="Provide either JD text or a JD file for evaluation.")

    labels_bytes = await labels_file.read()
    try:
        labels_payload = json.loads(labels_bytes.decode("utf-8"))
        target_scores = _parse_eval_targets(labels_payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid labels JSON: {exc}") from exc

    resumes_zip_bytes = await resumes_zip.read()
    try:
        parsed_resumes = [
            {"filename": filename, "text": text}
            for filename, text in extract_zip_resumes(resumes_zip_bytes)
        ]
    except ParseError as exc:
        raise HTTPException(status_code=422, detail=f"Resume ZIP error: {exc}") from exc

    if jd_file and jd_file.filename:
        jd_bytes = await jd_file.read()
        try:
            job_description = extract_resume_text(jd_bytes, jd_file.filename)
        except ParseError as exc:
            raise HTTPException(status_code=422, detail=f"JD file error: {exc}") from exc
    if len(job_description.strip()) < 50:
        raise HTTPException(status_code=422, detail="Job description must be at least 50 characters.")

    if values_file and values_file.filename:
        values_bytes = await values_file.read()
        try:
            company_values = extract_resume_text(values_bytes, values_file.filename)
        except ParseError as exc:
            raise HTTPException(status_code=422, detail=f"Company values file error: {exc}") from exc

    pipeline_output = await run_pipeline(
        jd_text=job_description,
        company_values=company_values,
        resumes=parsed_resumes,
        stage1_threshold=_SCORING["stage1_threshold"],
        stage2_threshold=_SCORING["stage2_threshold"],
        stage3_threshold=_SCORING["stage3_threshold"],
        use_llm=True,
        embedding_profile=get_default_embedding_profile(),
        excluded_skills=[],
    )

    comparisons = []
    squared_errors: list[float] = []
    absolute_errors: list[float] = []
    matched_count = 0
    for result in pipeline_output["results"]:
        filename = result["filename"]
        if filename not in target_scores:
            continue
        predicted = float(result.get("final_score", 0.0))
        target = float(target_scores[filename])
        error = abs(predicted - target)
        matched_count += 1
        absolute_errors.append(error)
        squared_errors.append(error ** 2)
        comparisons.append({
            "filename": filename,
            "predicted_score": round(predicted, 4),
            "target_score": round(target, 4),
            "absolute_error": round(error, 4),
            "rank": result.get("rank"),
            "stage_eliminated": result.get("stage_eliminated"),
        })

    comparisons.sort(key=lambda item: item["absolute_error"])
    mae = sum(absolute_errors) / matched_count if matched_count else None
    rmse = math.sqrt(sum(squared_errors) / matched_count) if matched_count else None

    return {
        "status": "ok",
        "matched_count": matched_count,
        "target_count": len(target_scores),
        "resume_count": len(parsed_resumes),
        "metrics": {
            "mae": round(mae, 4) if mae is not None else None,
            "rmse": round(rmse, 4) if rmse is not None else None,
            "within_0_10": sum(1 for error in absolute_errors if error <= 0.10),
            "within_0_20": sum(1 for error in absolute_errors if error <= 0.20),
        },
        "comparisons": comparisons,
        "pipeline_stats": pipeline_output["stats"],
    }


def _build_result(r: dict) -> ResumeResult:
    raw_profile = r.get("profile")
    profile = None
    if raw_profile and isinstance(raw_profile, dict) and raw_profile.get("verdict"):
        profile = CandidateProfile(
            strengths=raw_profile.get("strengths", []),
            weaknesses=raw_profile.get("weaknesses", []),
            unknowns=raw_profile.get("unknowns", []),
            values_alignment=raw_profile.get("values_alignment", "unclear"),
            values_evidence=raw_profile.get("values_evidence", ""),
            verdict=raw_profile.get("verdict", ""),
        )

    return ResumeResult(
        filename=r["filename"],
        final_score=round(float(r.get("final_score", 0.0)), 4),
        stage_scores=StageScores(
            bm25_score=round(float(r.get("bm25_score", 0.0)), 4),
            semantic_score=round(float(r.get("semantic_score", 0.0)), 4),
            combined_score=round(float(r.get("combined_score", 0.0)), 4),
            llm_score=round(float(r["llm_score"]), 4) if r.get("llm_score") is not None else None,
        ),
        key_skills=r.get("key_skills", []),
        missing_skills=r.get("missing_skills", []),
        resume_text=r.get("text"),
        profile=profile,
        stage_eliminated=r.get("stage_eliminated"),
        rank=r["rank"],
    )
