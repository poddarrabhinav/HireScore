from functools import lru_cache
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
ROOT_PATH = Path(__file__).parent.parent

DEFAULT_CONFIG = {
    "openai": {
        "model": "gpt-4o-mini",
        "keyword_model": "gpt-4o-mini",
        "max_concurrency": 5,
    },
    "embeddings": {
        "default_profile": "small",
        "cache_dir": ".cache/sentence-transformers",
        "device": "auto",
        "profiles": {
            "small": {
                "label": "Small",
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            },
            "medium": {
                "label": "Medium",
                "model_name": "sentence-transformers/all-mpnet-base-v2",
            },
        },
    },
    "scoring": {
        "stage1_threshold": 0.3,
        "stage2_threshold": 0.3,
        "stage3_threshold": 0.3,
        "semantic_min_cosine": 0.25,
        "exact_skill_weight": 0.5,
        "fuzzy_skill_weight": 0.2,
        "embedding_skill_weight": 0.3,
        "bm25_weight": 0.25,
        "semantic_weight": 0.75,
        "alpha": 0.4,
        "beta": 0.6,
    },
    "prompts": {
        "directory": "prompts",
        "judge_template": "judge_prompt.jinja",
        "keyword_template": "keyword_prompt.jinja",
    },
    "skill_matching": {
        "expansion_file": "skill_expansions.yaml",
        "use_llm_generated_expansions": True,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@lru_cache(maxsize=1)
def get_config() -> dict:
    if not CONFIG_PATH.exists():
        return DEFAULT_CONFIG

    data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    return _deep_merge(DEFAULT_CONFIG, data)


@lru_cache(maxsize=1)
def get_skill_expansions() -> dict[str, list[str]]:
    config = get_config()
    expansion_file = config.get("skill_matching", {}).get("expansion_file", "skill_expansions.yaml")
    path = ROOT_PATH / expansion_file
    if not path.exists():
        return {}

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return {}

    normalized: dict[str, list[str]] = {}
    for key, values in data.items():
        if not isinstance(key, str) or not isinstance(values, list):
            continue
        cleaned_values = [value for value in values if isinstance(value, str) and value.strip()]
        if cleaned_values:
            normalized[key.strip().lower()] = cleaned_values
    return normalized
