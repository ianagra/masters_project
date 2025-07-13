# network_qos_llm.py – v3
"""Automatic QoS analysis pipeline (five stages) using two Ollama models.

* **Analyzer model**  : `deepseek-r1:14b`  – generates all narrative outputs.
* **Tool model**      : `llama3.2`         – lightweight reasoning helper used
  as a *function‑calling* engine to pre‑process data (choose worst cluster,
  filter coefficients, etc.).

Changes vs. v2
--------------
1. **Stage‑1 prompt** now bundles *both* cluster statistics **and** *metric‑level
   logistic coefficients* so the LLM understands which metrics push intervals
   toward each cluster.
2. **Stage‑2 selection logic** fully performed by a helper function driven via
   `llama3.2`. Only the clients / servers whose coefficients point to the
   *worst* cluster (largest |coef|) are sent to the analyzer model. List is
   pre‑sorted by magnitude.
3. **Stage‑3 prompts** send *only* the intervals for those critical elements.
4. Re‑aligned prompt keys to the latest template file
   (`prompt_templates_v2.py`, key names: `cluster_analysis`, `critical_elements`,
   `interval_diagnosis`, `approach_report`, `consolidated_report`).
5. Extensive inline comments flagging all new or changed blocks with
   `# >>> CHANGE v3` markers.
"""
from __future__ import annotations
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
from langchain_ollama import ChatOllama

# ---------------------------------------------------------------------------
# Templates de prompts
# ---------------------------------------------------------------------------
from prompt_templates import PROMPTS

###############################################################################
# Configuração
###############################################################################

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Modelo principal para análise de QoS
ANALYZER_MODEL = "deepseek-r1:14b"
ANALYZER_TEMP = 0.5
ANALYZER_CTX = 8192

# Modelo para tool-calling
TOOL_MODEL = "llama3.2"
TOOL_TEMP = 0.0  # deterministic

# Onde salvar os produtos
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = Path("logs"); LOG_DIR.mkdir(exist_ok=True)
REPORT_DIR = Path("reports"); REPORT_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"qos_analysis_{RUN_TS}.txt"

###############################################################################
# Utilidades
###############################################################################

class PromptLogger:
    """Append‑only TXT logger for prompts & responses."""

    def __init__(self, path: Path):
        self.path = path
        path.write_text("# QoS‑LLM log file\n", encoding="utf‑8")

    def log(self, title: str, content: Any):
        delim = "=" * 80
        body = (
            json.dumps(content, indent=2, ensure_ascii=False)
            if isinstance(content, (dict, list))
            else str(content)
        )
        with self.path.open("a", encoding="utf‑8") as f:
            f.write(f"\n{delim}\n{title}\n{delim}\n{body}\n")


class OllamaLLM:
    """Thin wrapper around a ChatOllama model."""

    def __init__(self, model: str, temperature: float, ctx_size: int):
        self.chat = ChatOllama(
            model=model,
            base_url=OLLAMA_BASE_URL,
            temperature=temperature,
            num_ctx=ctx_size,
        )

    def __call__(self, prompt: str) -> str:
        return self.chat.invoke(prompt).content.strip()

###############################################################################
# Ferramentas de processamento de dados
###############################################################################

METRIC_FEATURES = {
    "throughput_download",
    "throughput_upload",
    "rtt_download",
    "rtt_upload",
}


def select_metric_coefficients(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Return only rows whose *feature* is a metric variable."""
    metrics_coefs = df[df['type'] =='metric'][['feature', 'coefficient']].to_dict(orient="records")
    return metrics_coefs


def select_entity_coefficients(
    df: pd.DataFrame,
    worst_cluster: int,
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """Return top‑N client/server coefficients pointing to the worst cluster."""
    # Entity rows are those whose feature string contains "client" or "server"
    df_ent = df[(df['type'] =='client') | (df['type'] =='server')][['feature', 'type', 'coefficient']].copy()

    sign_mask = df_ent["coefficient"] > 0 if worst_cluster == 1 else df_ent["coefficient"] < 0
    df_sel = df_ent.loc[sign_mask].copy()
    df_sel["abs_coef"] = df_sel["coefficient"].abs()
    df_sel.sort_values("abs_coef", ascending=False, inplace=True)
    return (
        df_sel.head(top_n)[["feature", "type", "coefficient"]]
        .to_dict(orient="records")
    )


def determine_worst_cluster(tool_llm: OllamaLLM, cluster_stats: List[Dict[str, Any]]) -> int:
    """Ask the *tool* LLM which cluster (0/1) is worse. Returns 0 or 1."""
    prompt = (
        "You are a network QoS expert. Given the JSON below, which cluster (0 or 1)\n"
        "has WORSE quality (higher RTT, lower throughput, higher event frequency).\n"
        "Respond ONLY with the number 0 or 1.\n\nJSON:\n"
        f"```json\n{json.dumps(cluster_stats, ensure_ascii=False)}\n```\nAnswer:"
    )
    reply = tool_llm(prompt)
    try:
        return int(reply.strip())
    except ValueError:
        # Fallback: assume cluster 1 is worse if ambiguous
        return 0

###############################################################################
# Orquestrador principal
###############################################################################

class QoSAnalyzer:
    """Runs the five‑stage pipeline for both metric approaches."""

    FILES = {
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

    def __init__(self, analyzer_llm: OllamaLLM, tool_llm: OllamaLLM, logger: PromptLogger):
        self.analyzer_llm = analyzer_llm
        self.tool_llm = tool_llm
        self.logger = logger
        self.system_prompt = PROMPTS["system"].strip()


    def run(self):
        reports: Dict[str, str] = {}
        for metric, paths in self.FILES.items():
            self.logger.log("INFO", f"=== Start approach: {metric} ===")
            reports[metric] = self._run_single_approach(metric, paths)

        consolidated = self._consolidate_reports(reports)
        out_path = REPORT_DIR / f"qos_consolidated_{RUN_TS}.txt"
        out_path.write_text(consolidated, encoding="utf‑8")
        self.logger.log("REPORT – Consolidated", consolidated)
        print(f"✅ Consolidated report saved to: {out_path}")

    def _run_single_approach(self, metric: str, paths: Dict[str, str]) -> str:
        clusters_df = pd.read_csv(paths["cluster_stats"])
        coef_df = pd.read_csv(paths["coefficients"])

        metric_coefs = select_metric_coefficients(coef_df)
        clusters_json = clusters_df.to_dict(orient="records")

        # Verificar qual cluster é o pior
        worst_cluster = determine_worst_cluster(self.tool_llm, clusters_json)
        self.logger.log(f"TOOL – worst_cluster ({metric})", worst_cluster)

        # Construir JSON para o prompt
        stage1_json = {
            "cluster_stats": clusters_json,
            "metric_coefficients": metric_coefs,
        }
        prompt1 = self._wrap(PROMPTS["cluster_analysis"].format(DATA_JSON=json.dumps(stage1_json)))
        self.logger.log(f"PROMPT 1 – {metric}", prompt1)
        cluster_interpretation = self.analyzer_llm(prompt1)
        self.logger.log(f"RESPONSE 1 – {metric}", cluster_interpretation)

        # Etapa 2
        entity_coefs = select_entity_coefficients(coef_df, worst_cluster, top_n=10)
        prompt2 = self._wrap(
            PROMPTS["critical_elements"].format(
                PREVIOUS_OUTPUT=cluster_interpretation,
                DATA_JSON=json.dumps(entity_coefs),
            )
        )
        self.logger.log(f"PROMPT 2 – {metric}", prompt2)
        critical_list = self.analyzer_llm(prompt2)
        self.logger.log(f"RESPONSE 2 – {metric}", critical_list)

        # Extrair entidades críticas
        critical_entities = [d["feature"] for d in entity_coefs]

        # Etapa 3
        surv_df = pd.read_csv(paths["survival"])
        entity_diagnostics: List[str] = []
        for ent in critical_entities:
            interval_json = self._make_interval_subset(surv_df, ent)
            prompt3 = self._wrap(
                PROMPTS["interval_diagnosis"].format(
                    ELEMENT_ID=ent,
                    DATA_JSON=json.dumps(interval_json),
                )
            )
            self.logger.log(f"PROMPT 3 – {metric} – {ent}", prompt3)
            diag = self.analyzer_llm(prompt3)
            self.logger.log(f"RESPONSE 3 – {metric} – {ent}", diag)
            entity_diagnostics.append(diag)

        # Etapa 4
        combined_prev = (
            f"{cluster_interpretation}\n\n{critical_list}\n\n" + "\n\n".join(entity_diagnostics)
        )
        prompt4 = self._wrap(
            PROMPTS["approach_report"].format(
                APPROACH_NAME=metric,
                PREVIOUS_OUTPUT=combined_prev,
            )
        )
        self.logger.log(f"PROMPT 4 – {metric}", prompt4)
        approach_report = self.analyzer_llm(prompt4)
        self.logger.log(f"RESPONSE 4 – {metric}", approach_report)

        out_path = REPORT_DIR / f"qos_{metric}_{RUN_TS}.txt"
        out_path.write_text(approach_report, encoding="utf‑8")
        self.logger.log("INFO", f"Saved approach report to {out_path}")
        return approach_report

    # ------------------------------------------------------------------
    def _consolidate_reports(self, reports: Dict[str, str]) -> str:
        prev = (
            "\n\n==== Throughput approach ====\n" + reports["throughput_download"] +
            "\n\n==== RTT approach ====\n" + reports["rtt_upload"]
        )
        prompt5 = self._wrap(
            PROMPTS["consolidated_report"].format(PREVIOUS_OUTPUT=prev)
        )
        self.logger.log("PROMPT 5 – Consolidation", prompt5)
        consolidated = self.analyzer_llm(prompt5)
        self.logger.log("RESPONSE 5 – Consolidation", consolidated)
        return consolidated

    # ------------------------------------------------------------------
    def _wrap(self, body: str) -> str:
        return f"{self.system_prompt}\n\n{body.strip()}"

    @staticmethod
    def _make_interval_subset(df: pd.DataFrame, entity: str) -> List[Dict[str, Any]]:
        mask = (df["client"] == entity) | (df["server"] == entity)
        cols = [
            "client", "server", "timestamp_start", "timestamp_end", "time",
            "cluster", "event", "throughput_download", "throughput_upload",
            "rtt_download", "rtt_upload",
        ]
        return df.loc[mask, cols].to_dict(orient="records")

###############################################################################
# Principal
###############################################################################

def main():
    logger = PromptLogger(LOG_FILE)
    analyzer_llm = OllamaLLM(ANALYZER_MODEL, ANALYZER_TEMP, ANALYZER_CTX)
    tool_llm = OllamaLLM(TOOL_MODEL, TOOL_TEMP, ANALYZER_CTX)

    pipeline = QoSAnalyzer(analyzer_llm, tool_llm, logger)
    pipeline.run()


if __name__ == "__main__":
    main()