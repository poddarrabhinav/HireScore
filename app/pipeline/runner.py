import asyncio
import logging

from app.config import get_config

from .stage1_skill_match import compute_ats_scores, compute_bm25_scores
from .stage2_sbert import (
    compute_sbert_scores,
    get_default_embedding_profile,
    get_embedding_profiles,
    resolve_embedding_profile,
)
from .stage3_agent import extract_keywords, extract_resume_keywords, load_openai_client, score_stage3_batch

logger = logging.getLogger(__name__)

_llm_client = None


def _build_semantic_query_text(
    jd_text: str,
    core_skills: list[str],
    adjacent_skills: list[str],
    soft_skills: list[str],
    role_type: str,
) -> str:
    parts = [
        f"Role type: {role_type}",
        f"Core skills: {', '.join(core_skills[:12])}" if core_skills else "",
        f"Adjacent skills: {', '.join(adjacent_skills[:10])}" if adjacent_skills else "",
        f"Soft skills: {', '.join(soft_skills[:8])}" if soft_skills else "",
        f"Job summary: {jd_text[:1200]}",
    ]
    return "\n".join(part for part in parts if part)


def get_llm_client():
    return _llm_client


def get_embedding_profile_options() -> list[dict[str, str]]:
    profiles = get_embedding_profiles()
    return [
        {
            "value": name,
            "label": profile.get("label", name.title()),
            "model_name": profile.get("model_name", ""),
        }
        for name, profile in profiles.items()
    ]


def init_pipeline() -> None:
    """Initialize pipeline at startup (called from FastAPI lifespan)."""
    global _llm_client
    _llm_client = load_openai_client()
    if _llm_client:
        logger.info("LLM judge + OpenAI embeddings enabled.")
    else:
        logger.info("OPENAI_API_KEY not set — LLM judge and OpenAI embeddings disabled.")


async def run_pipeline(
    jd_text: str,
    company_values: str,
    resumes: list[dict],
    stage1_threshold: float | None = None,
    stage2_threshold: float | None = None,
    stage3_threshold: float | None = None,
    use_llm: bool = True,
    embedding_profile: str | None = None,
    excluded_skills: list[str] | None = None,
    stage1_mode: str | None = None,
) -> dict:
    """
    Three-stage resume scoring pipeline.

    The pipeline is async and uses asyncio for network-bound LLM work.
    Local skill matching and embedding calculations are executed in-process without a thread pool.
    """
    config = get_config()
    scoring = config["scoring"]
    selected_profile = resolve_embedding_profile(embedding_profile)
    stage1_threshold = scoring["stage1_threshold"] if stage1_threshold is None else stage1_threshold
    resolved_stage1_mode = stage1_mode if stage1_mode in ("skill_match", "ats") else scoring.get("stage1_mode", "skill_match")
    stage2_threshold = scoring["stage2_threshold"] if stage2_threshold is None else stage2_threshold
    stage3_threshold = scoring["stage3_threshold"] if stage3_threshold is None else stage3_threshold

    bm25_weight = float(scoring["bm25_weight"])
    semantic_weight = float(scoring["semantic_weight"])
    alpha = float(scoring["alpha"])
    beta = float(scoring["beta"])

    total_stage_weight = bm25_weight + semantic_weight
    if total_stage_weight <= 0:
        bm25_weight, semantic_weight = 0.4, 0.6
        total_stage_weight = 1.0

    total_final_weight = alpha + beta
    if total_final_weight <= 0:
        alpha, beta = 0.4, 0.6
        total_final_weight = 1.0

    bm25_weight /= total_stage_weight
    semantic_weight /= total_stage_weight
    alpha /= total_final_weight
    beta /= total_final_weight

    total = len(resumes)

    extracted_keywords = await extract_keywords(_llm_client, jd_text, company_values)
    role_type = extracted_keywords.get("role_type", "technical")
    excluded_skills_normalized = {
        skill.strip().lower()
        for skill in (excluded_skills or [])
        if skill and skill.strip()
    }
    filtered_core_skills = [
        skill for skill in extracted_keywords.get("core_skills", [])
        if skill.strip().lower() not in excluded_skills_normalized
    ]
    filtered_adjacent_skills = [
        skill for skill in extracted_keywords.get("adjacent_skills", [])
        if skill.strip().lower() not in excluded_skills_normalized
    ]
    filtered_soft_skills = [
        skill for skill in extracted_keywords.get("soft_skills", [])
        if skill.strip().lower() not in excluded_skills_normalized
    ]
    if role_type == "technical":
        filtered_adjacent_skills = []
        filtered_soft_skills = []
    extracted_resume_keywords = await asyncio.gather(*[
        extract_resume_keywords(resume)
        for resume in resumes
    ])
    resume_keywords_map = {
        resume["filename"]: keywords
        for resume, keywords in zip(resumes, extracted_resume_keywords)
    }

    stage1_scorer = compute_ats_scores if resolved_stage1_mode == "ats" else compute_bm25_scores
    bm25_scores, key_skills = stage1_scorer(
        jd_text,
        resumes,
        core_skills=filtered_core_skills,
        adjacent_skills=filtered_adjacent_skills,
        soft_skills=filtered_soft_skills,
        role_type=role_type,
        resume_keywords_map=resume_keywords_map,
        embedding_profile=selected_profile,
        skill_expansions=extracted_keywords.get("skill_expansions", {}),
    )
    semantic_query_text = _build_semantic_query_text(
        jd_text,
        filtered_core_skills,
        filtered_adjacent_skills,
        filtered_soft_skills,
        role_type,
    )
    sbert_scores = await compute_sbert_scores(semantic_query_text, resumes, selected_profile)

    scored = []
    for r in resumes:
        fn = r["filename"]
        bm25 = bm25_scores.get(fn, 0.0)
        semantic = sbert_scores.get(fn, 0.0)
        combined = bm25_weight * bm25 + semantic_weight * semantic
        skills_data = key_skills.get(fn, {})
        scored.append({
            **r,
            "bm25_score": bm25,
            "semantic_score": semantic,
            "combined_score": combined,
            "key_skills": skills_data.get("matched", []),
            "missing_skills": skills_data.get("missing", []),
        })

    s1_pass = [r for r in scored if r["bm25_score"] >= stage1_threshold]
    s1_elim = [{**r, "stage_eliminated": 1} for r in scored if r["bm25_score"] < stage1_threshold]
    logger.info("Stage 1 Skill Match (threshold=%.2f): %d -> %d survivors", stage1_threshold, total, len(s1_pass))

    s2_pass = [r for r in s1_pass if r["semantic_score"] >= stage2_threshold]
    s2_elim = [{**r, "stage_eliminated": 2} for r in s1_pass if r["semantic_score"] < stage2_threshold]
    logger.info("Stage 2 Embeddings (threshold=%.2f): %d -> %d survivors", stage2_threshold, len(s1_pass), len(s2_pass))

    llm_available = use_llm and _llm_client is not None
    s3_pass: list[dict] = []
    s3_elim: list[dict] = []
    stage3_evaluated = 0

    if llm_available and s2_pass:
        s3_results = await score_stage3_batch(
            _llm_client,
            jd_text,
            company_values,
            s2_pass,
            alpha=alpha,
            beta=beta,
        )
        stage3_evaluated = len(s3_results)
        for r in s3_results:
            if r.get("llm_score", 0.0) >= stage3_threshold:
                s3_pass.append(r)
            else:
                s3_elim.append({**r, "stage_eliminated": 3})
        logger.info("Stage 3 LLM (threshold=%.2f): %d evaluated -> %d survivors", stage3_threshold, stage3_evaluated, len(s3_pass))
    else:
        for r in s2_pass:
            s3_pass.append({**r, "final_score": r["combined_score"], "profile": None})

    for r in s1_elim + s2_elim + s3_elim:
        r["final_score"] = r.get("combined_score", 0.0)

    all_results = s3_pass + s3_elim + s2_elim + s1_elim
    # Passed candidates always rank above eliminated ones; within each group sort by score
    all_results.sort(key=lambda x: (0 if not x.get("stage_eliminated") else 1, -x.get("final_score", 0.0)))
    for i, r in enumerate(all_results):
        r["rank"] = i + 1

    selected_profile_config = get_embedding_profiles()[selected_profile]
    return {
        "results": all_results,
        "stats": {
            "total": total,
            "stage1_survivors": len(s1_pass),
            "stage2_survivors": len(s2_pass),
            "stage3_evaluated": stage3_evaluated,
            "stage3_survivors": len(s3_pass),
            "job_keywords": extracted_keywords.get("job_keywords", []),
            "core_skills": filtered_core_skills,
            "adjacent_skills": filtered_adjacent_skills,
            "soft_skills": filtered_soft_skills,
            "skill_expansions": extracted_keywords.get("skill_expansions", {}),
            "value_keywords": extracted_keywords.get("value_keywords", []),
            "excluded_skills": sorted(excluded_skills_normalized),
            "role_type": role_type,
            "stage1_mode": resolved_stage1_mode,
            "bm25_weight": bm25_weight,
            "semantic_weight": semantic_weight,
            "alpha": alpha,
            "beta": beta,
            "embedding_profile": selected_profile,
            "embedding_model": selected_profile_config.get("model_name", ""),
            "semantic_scoring_summary": (
                f"Semantic scoring uses a focused query built from role type, extracted JD skills, and a condensed JD summary "
                f"against each full resume text "
                f"with OpenAI {selected_profile_config.get('model_name', '')}."
            ),
            "semantic_query_preview": semantic_query_text[:280],
            "embedding_options": get_embedding_profile_options(),
        },
    }
