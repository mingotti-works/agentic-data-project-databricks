# Databricks notebook source

# COMMAND ----------
# MAGIC %md
# MAGIC # Agentic Data Discovery — Pipeline Trigger
# MAGIC
# MAGIC This notebook is intentionally thin. All configuration lives in
# MAGIC `pipeline_config.json`. The orchestrator reads it automatically.
# MAGIC
# MAGIC **To run with the default config** (pipeline_config.json beside this notebook):
# MAGIC Just run all cells — no edits needed.
# MAGIC
# MAGIC **To run with a different config** (e.g. a prod vs dev environment):
# MAGIC Set the `config_path` widget below, or pass it as a Job parameter.

# COMMAND ----------

# Optional: override config path via a Databricks widget
# Leave blank to use pipeline_config.json in the same directory,
# or set PIPELINE_CONFIG_PATH as a cluster/job environment variable.

dbutils.widgets.text(
    "config_path",
    defaultValue="",
    label="Config path (leave blank for default)",
)

# COMMAND ----------

import sys
import os

# Add the directory containing the agent files to sys.path.
# Adjust this if your repo layout puts agents in a subdirectory.
AGENTS_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "/Workspace/Repos/your-repo/databricks_agents"
if AGENTS_DIR not in sys.path:
    sys.path.insert(0, AGENTS_DIR)

# COMMAND ----------

from orchestrator import run_pipeline

# Resolve config path: widget → env var → default (auto-resolved inside from_json)
config_path = dbutils.widgets.get("config_path").strip() or None

summary = run_pipeline(config_path=config_path)

# COMMAND ----------

# MAGIC %md ## Pipeline summary

# COMMAND ----------

import json
print(json.dumps(summary, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Human review queue
# MAGIC
# MAGIC Run the cell below **after** the pipeline completes to see what
# MAGIC needs manual approval before proceeding to data modelling.

# COMMAND ----------

from orchestrator import get_review_queue

queue = get_review_queue()

print(f"Relationships pending review : {queue['relationships_pending_review']}")
print(f"Semantic labels pending review: {queue['semantics_pending_review']}")

# COMMAND ----------

# Inspect relationship candidates that need a human decision
display(queue["relationship_df"].orderBy("confidence", ascending=False))

# COMMAND ----------

# Inspect semantic annotations that need a human decision
display(queue["semantic_df"].orderBy("confidence", ascending=False))
