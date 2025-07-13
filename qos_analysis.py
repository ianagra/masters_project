# network_qos_llm.py
"""Automatic QoS analysis pipeline using DeepSeek‑R1‑14B (Ollama)

This script implements a fully automated, multi‑stage workflow that analyses
network QoS datasets with the help of a Large Language Model.  It follows the
specification provided by the user and leverages advanced, externalised prompt
templates (see *prompt_templates.py*).
"""
from __future__ import annotations

import json
import os
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd
from langchain_ollama import ChatOllama  # pip install langchain‑ollama

from prompt_templates import PROMPTS  # ← externalised templates

###############################################################################
# Configuration
###############################################################################

# Ollama endpoint running DeepSeek‑R1‑14B (change if different)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL_NAME = "deepseek-r1:14b"
MODEL_TEMPERATURE = 0.7
LLM_CTX_SIZE = 8192

# Where to save artefacts
LOG_DIR = Path("logs")
REPORT_DIR = Path("reports")
LOG_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# Timestamp used for this run
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"qos_analysis_{RUN_TS}.txt"

###############################################################################
# Helper classes
###############################################################################

class PromptLogger:
    """Append‑only TXT logger for prompts & responses."""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        filepath.write_text("# QoS‑LLM log file\n")  # truncate or create

    def log(self, title: str, content: str | Dict[str, Any] | List[Any]):
        delim = "=" * 80
        body = json.dumps(content, indent=2) if isinstance(content, (dict, list)) else content
        with self.filepath.open("a", encoding="utf‑8") as f:
            f.write(f"\n{delim}\n{title}\n{delim}\n{body}\n")


class LLM:
    """Wrapper around ChatOllama for brevity."""

    def __init__(self):
        self.chat = ChatOllama(
            model=LLM_MODEL_NAME,
            base_url=OLLAMA_BASE_URL,
            temperature=MODEL_TEMPERATURE,
            num_ctx=LLM_CTX_SIZE,
        )

    def __call__(self, prompt: str) -> str:
        return self.chat.invoke(prompt).content.strip()


###############################################################################
# Core analysis workflow
###############################################################################

class QoSAnalyzer:
    """Runs the 5‑stage analysis for both metrics and produces reports."""

    METRICS = {
        "throughput_download": {
            "cluster_stats": "clusters_stats_throughput_download.csv",
            "coefficients": "coefficients_odds_ratio_throughput_download.csv",
            "survival": "dataset_survival_throughput_download.csv",
        },
        "rtt_upload": {
            "cluster_stats": "clusters_stats_rtt_upload.csv",
            "coefficients": "coefficients_odds_ratio_rtt_upload.csv",
            "survival": "dataset_survival_rtt_upload.csv",
        },
    }

    def __init__(self, llm: LLM, logger: PromptLogger):
        self.llm = llm
        self.logger = logger
        self.system_prompt = PROMPTS["system"]  # reusable preamble

    # --------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------
    def run(self):
        approach_reports: Dict[str, str] = {}

        for metric, paths in self.METRICS.items():
            self.logger.log("INFO", f"Starting analysis for metric: {metric}")
            approach_reports[metric] = self._analyze_metric(metric, paths)

        # Consolidation
        consolidated_report = self._consolidate_reports(approach_reports)
        report_path = REPORT_DIR / f"qos_consolidated_{RUN_TS}.txt"
        report_path.write_text(consolidated_report)
        self.logger.log("REPORT – Consolidated QoS", consolidated_report)
        print(f"✅ Consolidated report saved to: {report_path}")

    # --------------------------------------------------------------
    # Metric‑specific workflow
    # --------------------------------------------------------------
    def _analyze_metric(self, metric: str, paths: Dict[str, str]) -> str:
        # Step 1 – Cluster description
        clusters_df = pd.read_csv(paths["cluster_stats"])
        clusters_json = clusters_df.to_dict(orient="records")
        prompt1 = self._wrap_prompt(
            PROMPTS["cluster_analysis"].format(
                metric=metric,
                clusters_json=json.dumps(clusters_json)[:4000],
            )
        )
        self.logger.log(f"PROMPT 1 – {metric}", prompt1)
        cluster_desc = self.llm(prompt1)
        self.logger.log(f"RESPONSE 1 – {metric}", cluster_desc)

        # Step 2 – Critical entities
        coef_df = pd.read_csv(paths["coefficients"])
        coef_json = coef_df.to_dict(orient="records")
        prompt2 = self._wrap_prompt(
            PROMPTS["critical_entities"].format(
                metric=metric,
                cluster_description=cluster_desc,
                coefficients_json=json.dumps(coef_json)[:4000],
            )
        )
        self.logger.log(f"PROMPT 2 – {metric}", prompt2)
        critical_raw = self.llm(prompt2)
        self.logger.log(f"RESPONSE 2 – {metric}", critical_raw)
        critical_entities = self._parse_list(critical_raw)

        # Step 3 – Interval analysis per entity
        surv_df = pd.read_csv(paths["survival"])
        entity_reports: List[str] = []
        for entity in critical_entities:
            intervals_json = self._extract_intervals_json(surv_df, entity)
            prompt3 = self._wrap_prompt(
                PROMPTS["element_diagnosis"].format(
                    metric=metric,
                    entity_id=entity,
                    intervals_json=json.dumps(intervals_json)[:4000],
                )
            )
            self.logger.log(f"PROMPT 3 – {metric} – {entity}", prompt3)
            entity_report = self.llm(prompt3)
            self.logger.log(f"RESPONSE 3 – {metric} – {entity}", entity_report)
            entity_reports.append(entity_report)

        # Step 4 – Approach‑level report
        prompt4 = self._wrap_prompt(
            PROMPTS["metric_report"].format(
                metric=metric,
                cluster_description=cluster_desc,
                entities_reports="\n\n".join(entity_reports),
            )
        )
        self.logger.log(f"PROMPT 4 – {metric}", prompt4)
        approach_report = self.llm(prompt4)
        self.logger.log(f"RESPONSE 4 – {metric}", approach_report)

        # Persist
        path = REPORT_DIR / f"qos_{metric}_{RUN_TS}.txt"
        path.write_text(approach_report)
        self.logger.log("INFO", f"Saved approach report to {path}")

        return approach_report

    # --------------------------------------------------------------
    # Prompt consolidation
    # --------------------------------------------------------------
    def _consolidate_reports(self, reports: Dict[str, str]) -> str:
        prompt5 = self._wrap_prompt(
            PROMPTS["consolidated_report"].format(
                report_throughput=reports["throughput_download"],
                report_rtt=reports["rtt_upload"],
            )
        )
        self.logger.log("PROMPT 5 – Consolidation", prompt5)
        consolidated = self.llm(prompt5)
        self.logger.log("RESPONSE 5 – Consolidation", consolidated)
        return consolidated

    # --------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------
    def _wrap_prompt(self, body: str) -> str:
        """Prepend the system prompt to the body."""
        return f"{self.system_prompt}\n\n{body}"

    @staticmethod
    def _parse_list(raw: str) -> List[str]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return [x.strip() for x in raw.split(",") if x.strip()]

    @staticmethod
    def _extract_intervals_json(df: pd.DataFrame, entity: str) -> List[Dict[str, Any]]:
        mask = (df["client"] == entity) | (df["server"] == entity)
        subset = df.loc[mask].copy()
        cols = [
            "client",
            "server",
            "timestamp_start",
            "timestamp_end",
            "time",
            "cluster",
            "event",
            "throughput_download",
            "throughput_upload",
            "rtt_download",
            "rtt_upload",
        ]
        return subset[cols].to_dict(orient="records")

###############################################################################
# Entry point
###############################################################################

def main():
    logger = PromptLogger(LOG_FILE)
    llm = LLM()
    analyzer = QoSAnalyzer(llm, logger)
    analyzer.run()


if __name__ == "__main__":
    main()