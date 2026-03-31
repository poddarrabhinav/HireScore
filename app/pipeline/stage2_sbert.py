import logging
import os
import re

import numpy as np

from app.config import get_config

logger = logging.getLogger(__name__)


def get_embedding_profiles() -> dict[str, dict]:
    return get_config()["embeddings"]["profiles"]


def get_default_embedding_profile() -> str:
    config = get_config()["embeddings"]
    default_profile = config["default_profile"]
    if default_profile in config["profiles"]:
        return default_profile
    return next(iter(config["profiles"]))


def resolve_embedding_profile(profile: str | None) -> str:
    profiles = get_embedding_profiles()
    if profile and profile in profiles:
        return profile
    return get_default_embedding_profile()


def _load_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import AsyncOpenAI
        return AsyncOpenAI(api_key=api_key)
    except ImportError:
        logger.warning("openai not installed - OpenAI embeddings unavailable.")
        return None


def _normalize_scores(raw_scores: np.ndarray) -> np.ndarray:
    config = get_config()["scoring"]
    center = float(config.get("semantic_min_cosine", 0.20))
    sharpness = 10.0
    scores = np.nan_to_num(raw_scores, nan=0.0, posinf=0.0, neginf=0.0)
    scores = np.clip(scores, -1.0, 1.0)
    normalized = 1.0 / (1.0 + np.exp(-sharpness * (scores - center)))
    return np.clip(normalized, 0.0, 1.0)


async def _compute_openai_semantic_scores(
    jd_text: str,
    resumes: list[dict],
    profile_name: str,
) -> dict[str, float]:
    client = _load_openai_client()
    if client is None:
        raise RuntimeError(
            f"Embedding profile '{profile_name}' requires OPENAI_API_KEY but it is not set."
        )

    config = get_embedding_profiles()[profile_name]
    model_name = config["model_name"]

    def _truncate(text: str, max_words: int = 6000) -> str:
        tokens = text.split()
        return " ".join(tokens[:max_words]) if len(tokens) > max_words else text

    texts_to_embed = [_truncate(jd_text)] + [_truncate(r["text"]) for r in resumes]
    response = await client.embeddings.create(input=texts_to_embed, model=model_name)
    embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

    jd_emb = np.array(embeddings[0], dtype=float)
    resume_embs = np.array(embeddings[1:], dtype=float)

    jd_norm = np.linalg.norm(jd_emb)
    resume_norms = np.linalg.norm(resume_embs, axis=1, keepdims=True)
    resume_norms = np.where(resume_norms == 0, 1e-10, resume_norms)
    raw_scores = (resume_embs @ jd_emb) / (resume_norms.squeeze() * (jd_norm if jd_norm else 1e-10))

    scores = _normalize_scores(raw_scores)
    logger.info(
        "Stage 2 OpenAI embeddings scored %d resumes using model '%s'",
        len(resumes), model_name,
    )
    return {r["filename"]: float(scores[i]) for i, r in enumerate(resumes)}


async def compute_sbert_scores(
    jd_text: str,
    resumes: list[dict],
    embedding_profile: str | None = None,
) -> dict[str, float]:
    profile_name = resolve_embedding_profile(embedding_profile)
    return await _compute_openai_semantic_scores(jd_text, resumes, profile_name)
