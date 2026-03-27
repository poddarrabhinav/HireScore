import logging
import re
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

from app.config import get_config

logger = logging.getLogger(__name__)

_models: dict[str, SentenceTransformer] = {}


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


def get_embedding_device() -> str:
    configured = str(get_config()["embeddings"].get("device", "auto")).lower()
    if configured != "auto":
        return configured
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_model(profile: str | None = None) -> SentenceTransformer:
    profile_name = resolve_embedding_profile(profile)
    if profile_name in _models:
        return _models[profile_name]

    config = get_config()["embeddings"]
    profile_config = config["profiles"][profile_name]
    cache_dir = Path(config["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_name = profile_config["model_name"]
    device = get_embedding_device()
    logger.info(
        "Loading embedding model '%s' for profile '%s' on device '%s' (cache=%s)",
        model_name, profile_name, device, cache_dir,
    )
    _models[profile_name] = SentenceTransformer(
        model_name,
        cache_folder=str(cache_dir),
        device=device,
    )
    return _models[profile_name]


def _chunk_text(text: str, chunk_size: int = 140, overlap: int = 35) -> list[str]:
    tokens = re.findall(r"\S+", text)
    if not tokens:
        return [text]
    if len(tokens) <= chunk_size:
        return [" ".join(tokens)]

    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(tokens), step):
        chunk = tokens[start:start + chunk_size]
        if not chunk:
            continue
        chunks.append(" ".join(chunk))
        if start + chunk_size >= len(tokens):
            break
    return chunks


def _normalize_scores(raw_scores: np.ndarray) -> np.ndarray:
    config = get_config()["scoring"]
    center = float(config.get("semantic_min_cosine", 0.20))
    sharpness = 10.0
    scores = np.nan_to_num(raw_scores, nan=0.0, posinf=0.0, neginf=0.0)
    scores = np.clip(scores, -1.0, 1.0)
    normalized = 1.0 / (1.0 + np.exp(-sharpness * (scores - center)))
    return np.clip(normalized, 0.0, 1.0)


def embed_texts(texts: list[str], embedding_profile: str | None = None):
    if not texts:
        return None
    model = get_model(embedding_profile)
    return model.encode(texts, convert_to_tensor=True, show_progress_bar=False)


def _compute_hybrid_semantic_scores(
    jd_text: str,
    resumes: list[dict],
    embedding_profile: str | None = None,
) -> dict[str, float]:
    if not resumes:
        return {}

    model = get_model(embedding_profile)
    jd_emb = model.encode(jd_text, convert_to_tensor=True)

    full_resume_texts = [r["text"] for r in resumes]
    full_resume_embs = model.encode(full_resume_texts, convert_to_tensor=True, show_progress_bar=False)
    full_scores = util.cos_sim(jd_emb, full_resume_embs)[0].cpu().numpy()

    hybrid_scores: list[float] = []
    for i, resume in enumerate(resumes):
        chunks = _chunk_text(resume["text"])
        if chunks:
            chunk_embs = model.encode(chunks, convert_to_tensor=True, show_progress_bar=False)
            chunk_scores = util.cos_sim(jd_emb, chunk_embs)[0].cpu().numpy()
            best_chunk_score = float(chunk_scores.max(initial=full_scores[i]))
        else:
            best_chunk_score = float(full_scores[i])

        hybrid = (0.35 * float(full_scores[i])) + (0.65 * best_chunk_score)
        hybrid_scores.append(hybrid)

    scores = _normalize_scores(np.array(hybrid_scores, dtype=float))
    logger.info(
        "Stage 2 embeddings scored %d resumes using profile '%s' with hybrid chunk/full metric on device '%s'",
        len(resumes),
        resolve_embedding_profile(embedding_profile),
        get_embedding_device(),
    )
    return {r["filename"]: float(scores[i]) for i, r in enumerate(resumes)}


async def compute_sbert_scores(
    jd_text: str,
    resumes: list[dict],
    embedding_profile: str | None = None,
) -> dict[str, float]:
    return _compute_hybrid_semantic_scores(jd_text, resumes, embedding_profile)
