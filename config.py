"""
config.py
---------
Shared configuration, Spark session factory, LLM stub, and common types.
Swap out the LLM stub when you are ready to wire in DBRX / Llama 3.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Pipeline configuration — edit these before running
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    # ── Volumes (Parquet / CSV) ────────────────────────────────────────────
    volume_paths: list[str] = field(default_factory=lambda: [
        "/Volumes/my_catalog/my_schema/raw_data",   # adjust to your mount
    ])
    volume_file_extensions: tuple[str, ...] = (".parquet", ".csv")

    # ── JDBC sources ──────────────────────────────────────────────────────
    jdbc_sources: list[dict[str, str]] = field(default_factory=lambda: [
        # Example — add / remove entries as needed
        {
            "name":     "source_postgres",
            "url":      "jdbc:postgresql://host:5432/mydb",
            "driver":   "org.postgresql.Driver",
            "user":     os.getenv("JDBC_USER", ""),
            "password": os.getenv("JDBC_PASSWORD", ""),
            # Leave table blank to auto-discover from information_schema
            "tables":   "",
        },
    ])

    # ── Delta output (results land here) ─────────────────────────────────
    output_catalog:  str = "my_catalog"
    output_schema:   str = "agent_results"

    output_tables: dict[str, str] = field(default_factory=lambda: {
        "discovery":    "discovery_results",
        "profiling":    "profiling_results",
        "relationship": "relationship_candidates",
        "semantic":     "semantic_results",
    })

    # ── Profiling ─────────────────────────────────────────────────────────
    profiling_sample_rows:   int   = 10_000   # rows sampled per dataset
    profiling_sample_values: int   = 10       # distinct values shown per col

    # ── Relationship discovery ────────────────────────────────────────────
    # A column pair is a FK candidate when value-overlap ratio exceeds this
    relationship_overlap_threshold:  float = 0.80
    # Max columns per table to compare (guards against O(n²) explosion)
    relationship_max_cols_per_table: int   = 50

    # ── Confidence thresholds ─────────────────────────────────────────────
    auto_accept_confidence:  float = 0.85
    human_review_confidence: float = 0.50   # below this → discard

    # ── MLflow experiment ─────────────────────────────────────────────────
    mlflow_experiment_name: str = "/Shared/agentic_data_discovery"

    # ── LLM stub toggle ───────────────────────────────────────────────────
    llm_provider: str = "stub"   # "stub" | "dbrx" | "llama3" | "openai"


# ---------------------------------------------------------------------------
# Spark session factory
# ---------------------------------------------------------------------------

def get_spark():
    """
    Return the active SparkSession when running on Databricks.
    Falls back to a local session for unit-testing off-cluster.
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
# LLM stub — replace the body of `call_llm` when you pick a provider
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Thin wrapper so every agent calls the same interface regardless of backend.

    Usage:
        llm = LLMClient(config)
        response = llm.call(system_prompt, user_prompt)
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

    # ── Stub (always available, no credentials needed) ────────────────────
    def _stub(self, user_prompt: str) -> str:
        return (
            "STUB RESPONSE | provider=stub | "
            f"prompt_chars={len(user_prompt)} | "
            "Replace LLMClient._stub() with a real provider when ready."
        )

    # ── DBRX via Databricks Model Serving ────────────────────────────────
    def _dbrx(self, system_prompt: str, user_prompt: str) -> str:
        import mlflow.deployments
        client = mlflow.deployments.get_deploy_client("databricks")
        response = client.predict(
            endpoint="databricks-dbrx-instruct",
            inputs={
                "messages": [
                    {"role": "system",  "content": system_prompt},
                    {"role": "user",    "content": user_prompt},
                ],
                "max_tokens": 512,
                "temperature": 0.0,
            },
        )
        return response["choices"][0]["message"]["content"]

    # ── Llama 3 via Databricks Model Serving ─────────────────────────────
    def _llama3(self, system_prompt: str, user_prompt: str) -> str:
        import mlflow.deployments
        client = mlflow.deployments.get_deploy_client("databricks")
        response = client.predict(
            endpoint="databricks-meta-llama-3-1-70b-instruct",
            inputs={
                "messages": [
                    {"role": "system",  "content": system_prompt},
                    {"role": "user",    "content": user_prompt},
                ],
                "max_tokens": 512,
                "temperature": 0.0,
            },
        )
        return response["choices"][0]["message"]["content"]

    # ── OpenAI (external call — ensure egress is allowed) ─────────────────
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
# Shared result types (plain dataclasses — easy to serialise to Delta)
# ---------------------------------------------------------------------------

@dataclass
class DatasetMeta:
    source_type:  str          # "volume" | "jdbc"
    source_name:  str          # logical name / path
    table_name:   str          # table or file name
    columns:      list[str]
    row_count:    int
    raw_schema:   dict[str, str]   # col_name → spark dtype string
    extra:        dict[str, Any] = field(default_factory=dict)
