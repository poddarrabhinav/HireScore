import logging
import re
from difflib import SequenceMatcher

import numpy as np
from nltk.corpus import stopwords as nltk_stopwords
from rank_bm25 import BM25Okapi

from app.config import get_config, get_skill_expansions

logger = logging.getLogger(__name__)

_FALLBACK_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "that", "this", "it", "its", "your",
}


def _get_stopwords() -> set[str]:
    try:
        return set(nltk_stopwords.words("english"))
    except LookupError:
        return _FALLBACK_STOPWORDS


def _normalize_skill(keyword: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s+#.-]+", " ", keyword.lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _compress_keywords(keywords: list[str], limit: int) -> list[str]:
    compressed: list[str] = []
    seen: set[str] = set()
    for keyword in keywords:
        cleaned = _normalize_skill(keyword)
        if not cleaned or cleaned in seen:
            continue
        if any(cleaned in existing or existing in cleaned for existing in seen):
            continue
        seen.add(cleaned)
        compressed.append(cleaned)
        if len(compressed) >= limit:
            break
    return compressed


def _expand_keywords(keywords: list[str], llm_expansions: dict[str, list[str]] | None = None) -> list[str]:
    configured_expansions = get_skill_expansions()
    use_llm_expansions = bool(get_config().get("skill_matching", {}).get("use_llm_generated_expansions", True))
    expanded: list[str] = []
    seen: set[str] = set()
    for keyword in keywords:
        normalized = _normalize_skill(keyword)
        candidates = [normalized, *configured_expansions.get(normalized, [])]
        if use_llm_expansions and llm_expansions:
            candidates.extend(llm_expansions.get(normalized, []))
        for candidate in candidates:
            cleaned = _normalize_skill(candidate)
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            expanded.append(cleaned)
    return expanded


def _phrase_weight(skill: str, bucket: str, role_type: str) -> float:
    if bucket == "core":
        return 1.0 if role_type == "technical" else 0.8
    if bucket == "adjacent":
        return 0.7 if role_type == "technical" else 0.5
    return 0.0


def _best_fuzzy_match(skill: str, candidates: list[str]) -> float:
    if not candidates:
        return 0.0
    scores = [SequenceMatcher(None, skill, candidate).ratio() for candidate in candidates]
    return max(scores, default=0.0)


def _normalize_score_components(exact_scores: list[float], fuzzy_scores: list[float], weights: list[float]) -> float:
    """Normalize exact + fuzzy scores to 0-1 range."""
    total_weight = sum(weights)
    if total_weight <= 0:
        return 0.0

    exact = sum(exact_scores) / total_weight
    fuzzy = sum(fuzzy_scores) / total_weight

    scoring = get_config()["scoring"]
    exact_w = float(scoring.get("exact_skill_weight", 0.5))
    fuzzy_w = float(scoring.get("fuzzy_skill_weight", 0.2))
    total_w = exact_w + fuzzy_w
    if total_w <= 0:
        return exact
    return (exact_w * exact + fuzzy_w * fuzzy) / total_w


def compute_bm25_scores(
    jd_text: str,
    resumes: list[dict],
    core_skills: list[str] | None = None,
    adjacent_skills: list[str] | None = None,
    soft_skills: list[str] | None = None,
    role_type: str = "technical",
    resume_keywords_map: dict[str, dict] | None = None,
    embedding_profile: str | None = None,
    skill_expansions: dict[str, list[str]] | None = None,
) -> tuple[dict[str, float], dict[str, list[str]]]:
    """
    Stage 1 structured skill match score.

    Uses:
    - canonical skill expansion
    - exact overlap
    - fuzzy overlap
    """
    if not resumes:
        return {}, {}

    core_skills = _compress_keywords(core_skills or [], limit=12)
    adjacent_skills = _compress_keywords(adjacent_skills or [], limit=10)
    hard_skills = [(skill, "core") for skill in core_skills] + [(skill, "adjacent") for skill in adjacent_skills]
    if not hard_skills:
        stop_words = _get_stopwords()
        fallback_tokens = [
            token
            for token in re.findall(r"\b[a-z][a-z0-9+\-/]{2,}\b", jd_text.lower())
            if token not in stop_words
        ]
        fallback = _compress_keywords(fallback_tokens, limit=12)
        hard_skills = [(skill, "core") for skill in fallback]

    score_map: dict[str, float] = {}
    skills_map: dict[str, dict[str, list[str]]] = {}
    jd_skill_terms = {
        skill: _expand_keywords([skill], skill_expansions)
        for skill, _bucket in hard_skills
    }

    for resume in resumes:
        fn = resume["filename"]
        extracted_resume_keywords = (resume_keywords_map or {}).get(fn, {})
        resume_keywords = _compress_keywords(
            extracted_resume_keywords.get("skills_keywords", [])
            + extracted_resume_keywords.get("project_keywords", [])
            + extracted_resume_keywords.get("inferred_related_skills", []),
            limit=36,
        )
        expanded_resume_keywords = _expand_keywords(resume_keywords)
        resume_lower = resume["text"].lower()

        exact_components: list[float] = []
        fuzzy_components: list[float] = []
        weights: list[float] = []
        matched: list[str] = []
        missing: list[str] = []

        for skill, bucket in hard_skills:
            expanded_skill_terms = jd_skill_terms.get(skill, [skill])
            weight = _phrase_weight(skill, bucket, role_type)
            weights.append(weight)

            exact_hit = any(term in expanded_resume_keywords or term in resume_lower for term in expanded_skill_terms)
            exact_components.append(weight if exact_hit else 0.0)

            fuzzy_ratio = max(_best_fuzzy_match(term, expanded_resume_keywords) for term in expanded_skill_terms) if expanded_resume_keywords else 0.0
            fuzzy_components.append(weight * fuzzy_ratio)

            if exact_hit or fuzzy_ratio >= 0.86:
                matched.append(skill)
            else:
                missing.append(skill)

        combined = _normalize_score_components(exact_components, fuzzy_components, weights)
        score_map[fn] = float(np.clip(combined, 0.0, 1.0))
        skills_map[fn] = {"matched": matched, "missing": missing}

    logger.info("Stage 1 skill matcher scored %d resumes", len(resumes))
    return score_map, skills_map


def _tokenize(text: str) -> list[str]:
    stop_words = _get_stopwords()
    return [
        token
        for token in re.findall(r"\b[a-z][a-z0-9+#.\-/]{1,}\b", text.lower())
        if token not in stop_words
    ]


def compute_ats_scores(
    jd_text: str,
    resumes: list[dict],
    core_skills: list[str] | None = None,
    adjacent_skills: list[str] | None = None,
    soft_skills: list[str] | None = None,
    role_type: str = "technical",
    resume_keywords_map: dict[str, dict] | None = None,
    embedding_profile: str | None = None,
    skill_expansions: dict[str, list[str]] | None = None,
) -> tuple[dict[str, float], dict[str, list[str]]]:
    """
    Stage 1 ATS scoring using BM25Okapi (rank_bm25).

    Builds a BM25 corpus from all resume texts, then scores each resume against
    a query composed of the extracted JD skill terms. Scores are normalized to
    0-1 via min-max scaling. Also reports per-resume matched/missing skills via
    a simple keyword presence check for UI display.
    """
    if not resumes:
        return {}, {}

    # Build the JD query from extracted skills (or fall back to JD tokens)
    core_skills = _compress_keywords(core_skills or [], limit=12)
    adjacent_skills = _compress_keywords(adjacent_skills or [], limit=10)
    hard_skills = [(skill, "core") for skill in core_skills] + [(skill, "adjacent") for skill in adjacent_skills]
    if not hard_skills:
        stop_words = _get_stopwords()
        fallback_tokens = [
            token
            for token in re.findall(r"\b[a-z][a-z0-9+\-/]{2,}\b", jd_text.lower())
            if token not in stop_words
        ]
        fallback = _compress_keywords(fallback_tokens, limit=12)
        hard_skills = [(skill, "core") for skill in fallback]

    jd_skill_terms = {
        skill: _expand_keywords([skill], skill_expansions)
        for skill, _bucket in hard_skills
    }

    # Build BM25 corpus — one tokenized doc per resume
    tokenized_corpus = [_tokenize(r["text"]) for r in resumes]
    bm25 = BM25Okapi(tokenized_corpus)

    # Query = all expanded JD skill tokens, weighted by bucket
    query_tokens: list[str] = []
    for skill, bucket in hard_skills:
        weight = _phrase_weight(skill, bucket, role_type)
        reps = max(1, round(weight * 3))  # repeat tokens proportional to weight
        for term in jd_skill_terms.get(skill, [skill]):
            query_tokens.extend(_tokenize(term) * reps)

    raw_scores = bm25.get_scores(query_tokens) if query_tokens else np.zeros(len(resumes))

    # Min-max normalize to 0-1
    score_min, score_max = raw_scores.min(), raw_scores.max()
    if score_max > score_min:
        normalized = (raw_scores - score_min) / (score_max - score_min)
    else:
        normalized = np.zeros(len(resumes))

    # Per-resume skill presence for UI (simple exact match)
    score_map: dict[str, float] = {}
    skills_map: dict[str, dict[str, list[str]]] = {}
    for i, resume in enumerate(resumes):
        fn = resume["filename"]
        resume_lower = resume["text"].lower()
        matched: list[str] = []
        missing: list[str] = []
        for skill, bucket in hard_skills:
            expanded_terms = jd_skill_terms.get(skill, [skill])
            if any(term in resume_lower for term in expanded_terms):
                matched.append(skill)
            else:
                missing.append(skill)
        score_map[fn] = float(np.clip(normalized[i], 0.0, 1.0))
        skills_map[fn] = {"matched": matched, "missing": missing}

    logger.info("Stage 1 BM25 ATS scored %d resumes", len(resumes))
    return score_map, skills_map
