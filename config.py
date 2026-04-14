"""
config.py
---------
Shared configuration, Spark session factory, LLM client, and common types.

Config is loaded from a JSON file rather than hardcoded here.
The default config path is resolved in this order (first match wins):

  1. Argument passed to PipelineConfig.from_json(path)
  2. Environment variable  PIPELINE_CONFIG_PATH
  3. Same directory as this file → pipeline_config.json

JDBC credentials are never stored in the JSON.
Each JDBC source entry specifies:
  "user_env": "MY_USER_ENV_VAR"
  "pass_env": "MY_PASS_ENV_VAR"
and the values are resolved from the environment at load time.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Default config path resolution
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).parent


def _default_config_path() -> Path:
    env_path = os.getenv("PIPELINE_CONFIG_PATH")
    if env_path:
        return Path(env_path)
    return _THIS_DIR / "pipeline_config.json"


# ---------------------------------------------------------------------------
# PipelineConfig — loaded from JSON
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    # ── Volumes ───────────────────────────────────────────────────────────
    volume_paths:            list[str]
    volume_file_extensions:  tuple[str, ...]

    # ── JDBC sources (credentials resolved from env) ──────────────────────
    jdbc_sources: list[dict[str, str]]

    # ── Delta output ──────────────────────────────────────────────────────
    output_catalog: str
    output_schema:  str
    output_tables:  dict[str, str]

    # ── Profiling ─────────────────────────────────────────────────────────
    profiling_sample_rows:   int
    profiling_sample_values: int

    # ── Relationship discovery ────────────────────────────────────────────
    relationship_overlap_threshold:  float
    relationship_max_cols_per_table: int

    # ── Confidence thresholds ─────────────────────────────────────────────
    auto_accept_confidence:  float
    human_review_confidence: float

    # ── MLflow ────────────────────────────────────────────────────────────
    mlflow_experiment_name: str

    # ── LLM ───────────────────────────────────────────────────────────────
    llm_provider: str

    # ── Source path (for logging/audit) ───────────────────────────────────
    config_path: str = ""

    # -----------------------------------------------------------------------
    # Factory
    # -----------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: str | Path | None = None) -> "PipelineConfig":
        """
        Load configuration from a JSON file.

        Args:
            path: explicit path to the JSON file.
                  Falls back to PIPELINE_CONFIG_PATH env var,
                  then to pipeline_config.json beside this module.
        """
        resolved = Path(path) if path else _default_config_path()

        if not resolved.exists():
            raise FileNotFoundError(
                f"Pipeline config not found: {resolved}\n"
                f"Set PIPELINE_CONFIG_PATH or pass the path explicitly."
            )

        with open(resolved, encoding="utf-8") as fh:
            raw = json.load(fh)

        # Remove comment key if present (JSON doesn't support comments natively)
        raw.pop("_comment", None)

        # Resolve JDBC credentials from environment variables
        jdbc_sources = _resolve_jdbc_credentials(raw.get("jdbc_sources", []))

        return cls(
            volume_paths            = raw.get("volume_paths", []),
            volume_file_extensions  = tuple(raw.get("volume_file_extensions", [".parquet", ".csv"])),
            jdbc_sources            = jdbc_sources,
            output_catalog          = raw["output_catalog"],
            output_schema           = raw["output_schema"],
            output_tables           = raw["output_tables"],
            profiling_sample_rows   = int(raw.get("profiling_sample_rows", 10_000)),
            profiling_sample_values = int(raw.get("profiling_sample_values", 10)),
            relationship_overlap_threshold  = float(raw.get("relationship_overlap_threshold", 0.80)),
            relationship_max_cols_per_table = int(raw.get("relationship_max_cols_per_table", 50)),
            auto_accept_confidence  = float(raw.get("auto_accept_confidence", 0.85)),
            human_review_confidence = float(raw.get("human_review_confidence", 0.50)),
            mlflow_experiment_name  = raw.get("mlflow_experiment_name", "/Shared/agentic_data_discovery"),
            llm_provider            = raw.get("llm_provider", "stub"),
            config_path             = str(resolved),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a sanitised dict safe for logging (no credentials)."""
        return {
            "config_path":                    self.config_path,
            "volume_paths":                   self.volume_paths,
            "volume_file_extensions":         list(self.volume_file_extensions),
            "jdbc_source_names":              [j["name"] for j in self.jdbc_sources],
            "jdbc_source_count":              len(self.jdbc_sources),
            "output_catalog":                 self.output_catalog,
            "output_schema":                  self.output_schema,
            "output_tables":                  self.output_tables,
            "profiling_sample_rows":          self.profiling_sample_rows,
            "profiling_sample_values":        self.profiling_sample_values,
            "relationship_overlap_threshold": self.relationship_overlap_threshold,
            "relationship_max_cols_per_table":self.relationship_max_cols_per_table,
            "auto_accept_confidence":         self.auto_accept_confidence,
            "human_review_confidence":        self.human_review_confidence,
            "mlflow_experiment_name":         self.mlflow_experiment_name,
            "llm_provider":                   self.llm_provider,
        }


# ---------------------------------------------------------------------------
# JDBC credential resolver
# ---------------------------------------------------------------------------

def _resolve_jdbc_credentials(sources: list[dict]) -> list[dict]:
    """
    For each JDBC source, resolve user/password from environment variables.

    JSON specifies:
      "user_env": "MY_JDBC_USER_VAR"
      "pass_env": "MY_JDBC_PASS_VAR"

    The resolved dict exposes "user" and "password" keys so the rest of
    the codebase remains unchanged.
    """
    resolved = []
    for src in sources:
        entry = dict(src)  # shallow copy — don't mutate the original

        user_env = entry.pop("user_env", None)
        pass_env = entry.pop("pass_env", None)

        if user_env:
            entry["user"] = os.getenv(user_env, "")
            if not entry["user"]:
                print(f"  [WARN] JDBC source '{entry.get('name')}': "
                      f"env var '{user_env}' is not set.")

        if pass_env:
            entry["password"] = os.getenv(pass_env, "")
            if not entry["password"]:
                print(f"  [WARN] JDBC source '{entry.get('name')}': "
                      f"env var '{pass_env}' is not set.")

        resolved.append(entry)

    return resolved


# ---------------------------------------------------------------------------
# Spark session factory
# ---------------------------------------------------------------------------

def get_spark():
    """
    Return the active SparkSession when running on Databricks.
    Falls back gracefully for off-cluster unit testing.
    """
    try:
        from pyspark.sql import SparkSession
        return SparkSession.builder.getOrCreate()
    except Exception as exc:
        raise RuntimeError(
            "Could not obtain a SparkSession. "
            "Make sure this runs on a Databricks cluster."
        ) from exc


# ---------------------------------------------------------------------------
# LLM client — stub active by default, swap via llm_provider in JSON
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Thin wrapper so every agent calls the same interface regardless of backend.
    Swap llm_provider in pipeline_config.json to change the backend.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def call(self, system_prompt: str, user_prompt: str) -> str:
        provider = self.config.llm_provider
        if provider == "stub":
            return self._stub(user_prompt)
        elif provider == "dbrx":
            return self._dbrx(system_prompt, user_prompt)
        elif provider == "llama3":
            return self._llama3(system_prompt, user_prompt)
        elif provider == "openai":
            return self._openai(system_prompt, user_prompt)
        else:
            raise ValueError(f"Unknown llm_provider: {provider!r}")

    def _stub(self, user_prompt: str) -> str:
        return (
            "STUB RESPONSE | provider=stub | "
            f"prompt_chars={len(user_prompt)} | "
            "Set llm_provider in pipeline_config.json to activate a real model."
        )

    def _dbrx(self, system_prompt: str, user_prompt: str) -> str:
        import mlflow.deployments
        client = mlflow.deployments.get_deploy_client("databricks")
        response = client.predict(
            endpoint="databricks-dbrx-instruct",
            inputs={
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                "max_tokens": 512,
                "temperature": 0.0,
            },
        )
        return response["choices"][0]["message"]["content"]

    def _llama3(self, system_prompt: str, user_prompt: str) -> str:
        import mlflow.deployments
        client = mlflow.deployments.get_deploy_client("databricks")
        response = client.predict(
            endpoint="databricks-meta-llama-3-1-70b-instruct",
            inputs={
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                "max_tokens": 512,
                "temperature": 0.0,
            },
        )
        return response["choices"][0]["message"]["content"]

    def _openai(self, system_prompt: str, user_prompt: str) -> str:
        import openai
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=512,
            temperature=0.0,
        )
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Shared result type
# ---------------------------------------------------------------------------

@dataclass
class DatasetMeta:
    source_type: str
    source_name: str
    table_name:  str
    columns:     list[str]
    row_count:   int
    raw_schema:  dict[str, str]
    extra:       dict[str, Any] = field(default_factory=dict)
