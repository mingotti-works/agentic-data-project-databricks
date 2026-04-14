"""
orchestrator.py
---------------
Orchestrator — runs the full agentic data discovery pipeline.

Execution order:
  1. Discovery   → finds all datasets
  2. Profiling   → statistical profiling of every column
  3. Relationship → FK/PK candidate detection
  4. Semantic    → LLM-based column meaning inference

Everything runs inside a single MLflow run so every decision,
confidence score, and agent output is fully auditable.

Usage (Databricks notebook or Job):
  %run ./orchestrator

Or import and call:
  from orchestrator import run_pipeline
  run_pipeline()
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any

import mlflow

import agent_discovery
import agent_profiling
import agent_relationship
import agent_semantic
from config import PipelineConfig, get_spark


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_pipeline(config: PipelineConfig | None = None) -> dict[str, Any]:
    """
    Execute the full pipeline. Returns a summary dict.
    Wraps everything in an MLflow run for audit and traceability.
    """
    if config is None:
        config = PipelineConfig()

    spark = get_spark()

    # ── Ensure output schema exists ────────────────────────────────────
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {config.output_catalog}")
    spark.sql(
        f"CREATE SCHEMA IF NOT EXISTS "
        f"{config.output_catalog}.{config.output_schema}"
    )

    mlflow.set_experiment(config.mlflow_experiment_name)

    summary: dict[str, Any] = {}

    with mlflow.start_run(run_name=f"discovery_pipeline_{datetime.utcnow():%Y%m%d_%H%M%S}"):

        # ── Log pipeline config (sanitised — no passwords) ─────────────
        safe_config = _sanitise_config(config)
        mlflow.log_params({k: str(v)[:250] for k, v in safe_config.items()})

        # ── PHASE 1: Discovery ─────────────────────────────────────────
        t0 = time.time()
        datasets = agent_discovery.run(config)
        t_discovery = round(time.time() - t0, 1)

        mlflow.log_metrics({
            "discovery_dataset_count": len(datasets),
            "discovery_duration_sec":  t_discovery,
        })

        if not datasets:
            print("\n[ORCHESTRATOR] No datasets discovered — stopping pipeline.")
            mlflow.log_param("pipeline_status", "stopped_no_data")
            return {"status": "stopped_no_data"}

        # ── PHASE 2: Profiling ─────────────────────────────────────────
        t0 = time.time()
        profiles = agent_profiling.run(config, datasets)
        t_profiling = round(time.time() - t0, 1)

        pk_candidates = [p for p in profiles if p.get("pk_candidate")]
        mlflow.log_metrics({
            "profiling_column_count":  len(profiles),
            "profiling_pk_candidates": len(pk_candidates),
            "profiling_duration_sec":  t_profiling,
        })

        # ── PHASE 3: Relationship discovery ───────────────────────────
        t0 = time.time()
        relationships = agent_relationship.run(config, datasets, profiles)
        t_relationship = round(time.time() - t0, 1)

        auto_accepted = [r for r in relationships if r.get("status") == "auto_accepted"]
        needs_review  = [r for r in relationships if r.get("status") == "needs_review"]
        discarded     = [r for r in relationships if r.get("status") == "discarded"]

        mlflow.log_metrics({
            "relationship_total":        len(relationships),
            "relationship_auto_accepted":len(auto_accepted),
            "relationship_needs_review": len(needs_review),
            "relationship_discarded":    len(discarded),
            "relationship_duration_sec": t_relationship,
        })

        # ── PHASE 4: Semantic inference ───────────────────────────────
        t0 = time.time()
        semantics = agent_semantic.run(config, profiles)
        t_semantic = round(time.time() - t0, 1)

        sem_auto     = [s for s in semantics if s.get("status") == "auto_accepted"]
        sem_review   = [s for s in semantics if s.get("status") == "needs_review"]
        sem_pii      = [s for s in semantics if s.get("sensitivity") == "pii"]

        mlflow.log_metrics({
            "semantic_total":          len(semantics),
            "semantic_auto_accepted":  len(sem_auto),
            "semantic_needs_review":   len(sem_review),
            "semantic_pii_columns":    len(sem_pii),
            "semantic_duration_sec":   t_semantic,
        })

        # ── Log top relationship candidates as MLflow artefact ────────
        if relationships:
            top_rels = relationships[:20]  # top 20 by confidence
            mlflow.log_dict({"top_relationship_candidates": top_rels},
                            "relationship_candidates.json")

        # ── Log PII summary as artefact ───────────────────────────────
        if sem_pii:
            pii_summary = [
                {k: s[k] for k in ("source_name", "table_name", "column_name",
                                   "business_label", "confidence")}
                for s in sem_pii
            ]
            mlflow.log_dict({"pii_columns": pii_summary}, "pii_summary.json")

        # ── Build summary ─────────────────────────────────────────────
        total_duration = round(t_discovery + t_profiling + t_relationship + t_semantic, 1)

        summary = {
            "status":               "complete",
            "run_ts":               datetime.utcnow().isoformat(),
            "total_duration_sec":   total_duration,
            "datasets_found":       len(datasets),
            "columns_profiled":     len(profiles),
            "pk_candidates":        len(pk_candidates),
            "relationship_candidates": {
                "auto_accepted": len(auto_accepted),
                "needs_review":  len(needs_review),
                "discarded":     len(discarded),
            },
            "semantic_annotations": {
                "auto_accepted": len(sem_auto),
                "needs_review":  len(sem_review),
                "pii_columns":   len(sem_pii),
            },
        }

        mlflow.log_dict(summary, "pipeline_summary.json")
        mlflow.log_param("pipeline_status", "complete")

        _print_summary(summary, auto_accepted, needs_review, sem_pii)

    return summary


# ---------------------------------------------------------------------------
# Human-review helper — call this after the pipeline to inspect what needs
# manual approval before proceeding to modelling
# ---------------------------------------------------------------------------

def get_review_queue(config: PipelineConfig | None = None) -> dict[str, Any]:
    """
    Read the Delta tables and return everything flagged 'needs_review'.
    Intended for use in a Databricks review notebook or dashboard.
    """
    if config is None:
        config = PipelineConfig()

    spark = get_spark()

    def read(table_key: str):
        full_name = (
            f"{config.output_catalog}.{config.output_schema}"
            f".{config.output_tables[table_key]}"
        )
        try:
            return spark.read.table(full_name).filter("status = 'needs_review'")
        except Exception:
            return spark.createDataFrame([], schema="status string")

    rel_df = read("relationship")
    sem_df = read("semantic")

    return {
        "relationships_pending_review": rel_df.count(),
        "semantics_pending_review":     sem_df.count(),
        "relationship_df":              rel_df,
        "semantic_df":                  sem_df,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitise_config(config: PipelineConfig) -> dict[str, Any]:
    """Return a flat config dict safe to log (no passwords)."""
    return {
        "volume_paths":                 str(config.volume_paths),
        "jdbc_source_count":            len(config.jdbc_sources),
        "jdbc_source_names":            [j["name"] for j in config.jdbc_sources],
        "output_catalog":               config.output_catalog,
        "output_schema":                config.output_schema,
        "profiling_sample_rows":        config.profiling_sample_rows,
        "relationship_overlap_threshold": config.relationship_overlap_threshold,
        "auto_accept_confidence":       config.auto_accept_confidence,
        "human_review_confidence":      config.human_review_confidence,
        "llm_provider":                 config.llm_provider,
    }


def _print_summary(
    summary: dict[str, Any],
    auto_accepted: list,
    needs_review: list,
    pii_cols: list,
):
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE — SUMMARY")
    print("=" * 60)
    print(f"  Duration          : {summary['total_duration_sec']}s")
    print(f"  Datasets found    : {summary['datasets_found']}")
    print(f"  Columns profiled  : {summary['columns_profiled']}")
    print(f"  PK candidates     : {summary['pk_candidates']}")
    print()
    print("  Relationship candidates:")
    print(f"    Auto-accepted   : {summary['relationship_candidates']['auto_accepted']}")
    print(f"    Needs review    : {summary['relationship_candidates']['needs_review']}")
    print(f"    Discarded       : {summary['relationship_candidates']['discarded']}")
    print()
    print("  Semantic annotations:")
    print(f"    Auto-accepted   : {summary['semantic_annotations']['auto_accepted']}")
    print(f"    Needs review    : {summary['semantic_annotations']['needs_review']}")
    print(f"    PII columns     : {summary['semantic_annotations']['pii_columns']}")
    print()

    if needs_review:
        print("  TOP RELATIONSHIP CANDIDATES FOR REVIEW:")
        for r in needs_review[:5]:
            print(
                f"    {r['source_table']}.{r['source_col']}"
                f" → {r['target_table']}.{r['target_col']}"
                f"  conf={r['confidence']:.2f}"
            )

    if pii_cols:
        print("\n  PII COLUMNS DETECTED:")
        for s in pii_cols[:5]:
            print(
                f"    {s['source_name']}.{s['table_name']}.{s['column_name']}"
                f" ({s['business_label']}, conf={s['confidence']:.2f})"
            )

    print("=" * 60)


# ---------------------------------------------------------------------------
# Run when executed directly (notebook top-level or spark-submit)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = PipelineConfig()
    run_pipeline(cfg)
