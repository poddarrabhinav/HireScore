import asyncio
import json
import logging
import os
import re
from pathlib import Path

from nltk.corpus import stopwords as nltk_stopwords
from jinja2 import Environment, FileSystemLoader

from app.config import get_config, get_skill_expansions

logger = logging.getLogger(__name__)

_NEUTRAL_PROFILE = {
    "strengths": [],
    "weaknesses": [],
    "unknowns": ["Evaluation failed - manual review required"],
    "values_alignment": "unclear",
    "values_evidence": "",
    "llm_score": 0.5,
    "verdict": "Automated evaluation failed.",
}

_FALLBACK_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "that", "this", "it", "its", "your",
}

_GENERIC_JD_LABELS = {
    "job",
    "description",
    "job description",
    "responsibilities",
    "responsibility",
    "qualifications",
    "qualification",
    "requirements",
    "requirement",
    "summary",
    "about us",
    "preferred qualifications",
    "nice to have",
}


def _get_stopwords() -> set[str]:
    try:
        return set(nltk_stopwords.words("english"))
    except LookupError:
        return _FALLBACK_STOPWORDS


def _get_prompt_environment() -> Environment:
    config = get_config()
    prompts_dir = Path(__file__).parent.parent.parent / config["prompts"]["directory"]
    return Environment(loader=FileSystemLoader(prompts_dir), autoescape=False, trim_blocks=True, lstrip_blocks=True)


def _render_prompt(template_name: str, **context) -> str:
    env = _get_prompt_environment()
    return env.get_template(template_name).render(**context)


def _clamp_unit_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _extract_json_content(content: str) -> dict:
    raw = content.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) >= 2:
            raw = parts[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def _fallback_keywords(text: str, limit: int = 15) -> list[str]:
    stop_words = _get_stopwords()
    tokens = re.findall(r"\b[a-z][a-z0-9+\-/]{2,}\b", text.lower())
    unique: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in stop_words or token in seen:
            continue
        seen.add(token)
        unique.append(token)
        if len(unique) >= limit:
            break
    return unique


def _fallback_soft_skills(text: str, limit: int = 8) -> list[str]:
    known_soft_skills = [
        "communication",
        "collaboration",
        "leadership",
        "ownership",
        "problem solving",
        "stakeholder management",
        "teamwork",
        "mentorship",
        "adaptability",
        "time management",
        "cross-functional collaboration",
    ]
    lower_text = text.lower()
    matches = [skill for skill in known_soft_skills if skill in lower_text]
    return matches[:limit]


def _dedupe_keywords(*groups: list[str], limit: int = 20) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for keyword in group:
            cleaned = keyword.strip().lower()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            merged.append(cleaned)
            if len(merged) >= limit:
                return merged
    return merged


def _keyword_key(keyword: str) -> str:
    normalized = re.sub(r"[^a-z0-9\s]+", " ", keyword.lower()).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    parts = []
    for token in normalized.split():
        if token.endswith("s") and len(token) > 4:
            token = token[:-1]
        parts.append(token)
    return " ".join(parts)


def _compress_keywords(keywords: list[str], limit: int) -> list[str]:
    compressed: list[str] = []
    seen_keys: set[str] = set()
    for keyword in keywords:
        cleaned = keyword.strip().lower()
        if not cleaned:
            continue
        key = _keyword_key(cleaned)
        if not key or key in seen_keys:
            continue
        if any(key in existing or existing in key for existing in seen_keys):
            continue
        seen_keys.add(key)
        compressed.append(cleaned)
        if len(compressed) >= limit:
            break
    return compressed


def _remove_generic_labels(keywords: list[str]) -> list[str]:
    filtered: list[str] = []
    for keyword in keywords:
        key = _keyword_key(keyword)
        if not key or key in _GENERIC_JD_LABELS:
            continue
        if key.endswith("description") or key.endswith("responsibilities") or key.endswith("qualifications"):
            continue
        filtered.append(keyword)
    return filtered


def _filter_keywords_by_word_count(keywords: list[str], min_words: int = 1, max_words: int = 2) -> list[str]:
    filtered: list[str] = []
    for keyword in keywords:
        words = [word for word in keyword.split() if word.strip()]
        if min_words <= len(words) <= max_words:
            filtered.append(keyword)
    return filtered


_RESUME_SECTION_HINTS = {
    "skills": ("skills", "technical skills", "tech stack", "stack", "tools"),
    "projects": ("projects", "project", "selected projects"),
    "experience": ("experience", "work experience", "professional experience", "employment"),
}


def _split_resume_sections(text: str) -> dict[str, str]:
    sections = {"skills": "", "projects": "", "experience": "", "full_text": text}

    if "\n" in text:
        current = "full_text"
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            lowered = line.lower().rstrip(":")
            matched = next(
                (name for name, hints in _RESUME_SECTION_HINTS.items() if lowered in hints),
                None,
            )
            if matched:
                current = matched
                continue
            sections[current] = f"{sections[current]}\n{line}".strip()
        if any(sections[name] for name in ("skills", "projects", "experience")):
            return sections

    flattened = re.sub(r"\s+", " ", text)
    header_patterns = {
        "experience": r"\b(?:work experience|professional experience|experience)\b",
        "projects": r"\b(?:selected projects|projects|project)\b",
        "skills": r"\b(?:technical skills|skills|tech stack)\b",
    }
    positions: list[tuple[int, str]] = []
    for name, pattern in header_patterns.items():
        match = re.search(pattern, flattened, flags=re.IGNORECASE)
        if match:
            positions.append((match.start(), name))

    if not positions:
        return sections

    positions.sort()
    for index, (start, name) in enumerate(positions):
        end = positions[index + 1][0] if index + 1 < len(positions) else len(flattened)
        sections[name] = flattened[start:end].strip()
    return sections


def _keyword_candidates_from_text(text: str, limit: int) -> list[str]:
    stop_words = _get_stopwords()
    phrases = re.findall(r"\b[a-zA-Z][a-zA-Z0-9+#./-]{1,}(?:\s+[a-zA-Z0-9+#./-]{2,}){0,2}\b", text)
    filtered: list[str] = []
    for phrase in phrases:
        cleaned = phrase.strip().lower()
        if not cleaned:
            continue
        tokens = cleaned.split()
        if all(token in stop_words for token in tokens):
            continue
        filtered.append(cleaned)
    return _compress_keywords(filtered, limit=limit)


def _infer_related_resume_skills(*keyword_groups: list[str], limit: int = 12) -> list[str]:
    expansions = get_skill_expansions()
    inferred: list[str] = []
    seen: set[str] = set()
    for group in keyword_groups:
        for keyword in group:
            normalized = _compress_keywords([keyword], limit=1)
            if not normalized:
                continue
            for expansion in expansions.get(normalized[0], []):
                cleaned = _compress_keywords([expansion], limit=1)
                if not cleaned:
                    continue
                value = cleaned[0]
                if value in seen:
                    continue
                seen.add(value)
                inferred.append(value)
                if len(inferred) >= limit:
                    return inferred
    return inferred


def load_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import AsyncOpenAI
        return AsyncOpenAI(api_key=api_key)
    except ImportError:
        logger.warning("openai not installed - LLM judge disabled.")
        return None


async def _chat_json(client, prompt: str, model: str) -> dict:
    response = await client.chat.completions.create(
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You return strict JSON only."},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content or "{}"
    return _extract_json_content(content)


async def extract_keywords(client, jd_text: str, company_values: str) -> dict:
    config = get_config()
    prompt_config = config["prompts"]
    openai_config = config["openai"]

    core_fallback = _fallback_keywords(jd_text, limit=12)
    soft_fallback = _fallback_soft_skills(jd_text, limit=8)
    values_fallback = _fallback_keywords(company_values) if company_values.strip() else []

    if client is None:
        return {
            "role_type": "technical",
            "core_skills": core_fallback,
            "adjacent_skills": [],
            "skill_expansions": {},
            "soft_skills": soft_fallback,
            "job_keywords": _dedupe_keywords(core_fallback, soft_fallback, limit=20),
            "value_keywords": values_fallback,
            "all_keywords": _dedupe_keywords(core_fallback, soft_fallback, values_fallback),
        }

    prompt = _render_prompt(
        prompt_config["keyword_template"],
        jd=jd_text[:5000],
        values=company_values[:2500],
    )
    try:
        payload = await _chat_json(client, prompt, openai_config["keyword_model"])
        role_type = payload.get("role_type", "technical")
        if role_type not in {"technical", "non_technical"}:
            role_type = "technical"
        core_skills = payload.get("core_skills", [])
        adjacent_skills = payload.get("adjacent_skills", [])
        skill_expansions = payload.get("skill_expansions", {})
        soft_skills = payload.get("soft_skills", [])
        value_keywords = payload.get("value_keywords", [])
        if not isinstance(core_skills, list):
            core_skills = []
        if not isinstance(adjacent_skills, list):
            adjacent_skills = []
        if not isinstance(skill_expansions, dict):
            skill_expansions = {}
        if not isinstance(soft_skills, list):
            soft_skills = []
        if not isinstance(value_keywords, list):
            value_keywords = []
        core_skills = _remove_generic_labels(_compress_keywords(_dedupe_keywords(core_skills, core_fallback, limit=20), limit=12))
        adjacent_skills = _remove_generic_labels(_compress_keywords(_dedupe_keywords(adjacent_skills, limit=16), limit=10))
        cleaned_expansions: dict[str, list[str]] = {}
        for key, values in skill_expansions.items():
            if not isinstance(key, str) or not isinstance(values, list):
                continue
            cleaned_key_list = _compress_keywords([key], limit=1)
            cleaned_key = cleaned_key_list[0] if cleaned_key_list else ""
            if not cleaned_key:
                continue
            cleaned_values = _compress_keywords([value for value in values if isinstance(value, str)], limit=6)
            if cleaned_values:
                cleaned_expansions[cleaned_key] = cleaned_values
        soft_skills = _remove_generic_labels(_compress_keywords(_dedupe_keywords(soft_skills, soft_fallback, limit=16), limit=8))
        value_keywords = _remove_generic_labels(_compress_keywords(_dedupe_keywords(value_keywords, values_fallback, limit=20), limit=15))
        if role_type == "technical":
            core_skills = _filter_keywords_by_word_count(core_skills, max_words=2)
            adjacent_skills = []
            soft_skills = []
        return {
            "role_type": role_type,
            "core_skills": core_skills,
            "adjacent_skills": adjacent_skills,
            "skill_expansions": cleaned_expansions,
            "soft_skills": soft_skills,
            "job_keywords": _dedupe_keywords(core_skills, adjacent_skills, soft_skills, limit=24),
            "value_keywords": value_keywords,
            "all_keywords": _dedupe_keywords(core_skills, adjacent_skills, soft_skills, value_keywords, limit=30),
        }
    except Exception as exc:
        logger.warning("Keyword extraction failed: %s", exc)
        return {
            "role_type": "technical",
            "core_skills": _remove_generic_labels(core_fallback),
            "adjacent_skills": [],
            "skill_expansions": {},
            "soft_skills": _remove_generic_labels(soft_fallback),
            "job_keywords": _dedupe_keywords(_remove_generic_labels(core_fallback), _remove_generic_labels(soft_fallback), limit=20),
            "value_keywords": _remove_generic_labels(values_fallback),
            "all_keywords": _dedupe_keywords(_remove_generic_labels(core_fallback), _remove_generic_labels(soft_fallback), _remove_generic_labels(values_fallback)),
        }


async def extract_resume_keywords(resume: dict) -> dict[str, list[str]]:
    sections = _split_resume_sections(resume["text"])
    fallback_skills = _fallback_keywords(sections.get("skills") or resume["text"], limit=15)
    fallback_projects = _fallback_keywords(
        "\n".join(
            part for part in (
                sections.get("projects", ""),
                sections.get("experience", ""),
            ) if part
        ) or resume["text"],
        limit=15,
    )
    explicit_skills = _keyword_candidates_from_text(sections.get("skills") or resume["text"], limit=20)
    project_keywords = _keyword_candidates_from_text(
        "\n".join(
            part for part in (
                sections.get("projects", ""),
                sections.get("experience", ""),
            ) if part
        ) or resume["text"],
        limit=20,
    )
    skills_keywords = _compress_keywords(_dedupe_keywords(explicit_skills, fallback_skills, limit=24), limit=15)
    project_keywords = _compress_keywords(_dedupe_keywords(project_keywords, fallback_projects, limit=24), limit=15)
    inferred_related_skills = _infer_related_resume_skills(skills_keywords, project_keywords, limit=12)
    return {
        "skills_keywords": skills_keywords,
        "project_keywords": project_keywords,
        "inferred_related_skills": inferred_related_skills,
        "all_keywords": _dedupe_keywords(skills_keywords, project_keywords, inferred_related_skills, limit=35),
    }


async def agent_judge(client, jd_text: str, company_values: str, resume: dict) -> dict:
    """Run LLM judge on a single resume. Returns structured profile dict."""
    config = get_config()
    prompt = _render_prompt(
        config["prompts"]["judge_template"],
        jd=jd_text,
        values=company_values,
        resume=resume["text"],
    )
    try:
        return await _chat_json(client, prompt, config["openai"]["model"])
    except Exception as exc:
        logger.warning("OpenAI judge failed for '%s': %s", resume.get("filename"), exc)
        return {**_NEUTRAL_PROFILE, "verdict": f"Automated evaluation failed: {exc}"}


async def score_stage3_batch(
    client,
    jd_text: str,
    company_values: str,
    resumes: list[dict],
    alpha: float,
    beta: float,
) -> list[dict]:
    """Run LLM judge on all stage 2 survivors. Returns scored resume dicts."""
    config = get_config()
    max_concurrency = max(1, int(config["openai"].get("max_concurrency", 5)))
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _score_resume(resume: dict) -> dict:
        async with semaphore:
            profile = await agent_judge(client, jd_text, company_values, resume)
        combined = resume.get("combined_score", 0.0)
        llm = _clamp_unit_score(profile.get("llm_score", 0.5))
        final = (alpha * combined) + (beta * llm)
        return {**resume, "llm_score": llm, "final_score": final, "profile": profile}

    results = await asyncio.gather(*[_score_resume(resume) for resume in resumes])
    logger.info("Stage 3 LLM judge: evaluated %d resumes", len(results))
    return results
