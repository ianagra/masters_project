from __future__ import annotations
from dotenv import load_dotenv
import os, json, re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from langchain_ollama import ChatOllama
from prompt_templates import PROMPTS

###############################################################################
# Configurações
###############################################################################
load_dotenv(".env")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

ANALYZER_MODEL = "phi4-reasoning:plus"
ANALYZER_TEMP  = 0.3
MODEL_NAME     = ANALYZER_MODEL.replace(":", "-")

TOOL_MODEL = "llama3.2:1b"
TOOL_TEMP  = 0.0

RUN_TS     = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR    = Path("logs");    LOG_DIR.mkdir(exist_ok=True)
REPORT_DIR = Path("reports"); REPORT_DIR.mkdir(exist_ok=True)
LOG_FILE   = LOG_DIR / f"{RUN_TS}_{MODEL_NAME}.txt"

###############################################################################
# Utilidades – remoção opcional de blocos de raciocínio
###############################################################################
_THINK_REGEX = re.compile(
    r"(?:<think>.*?</think>|Thinking\.\.\..*?\.\.\.done thinking\.)",
    flags=re.DOTALL | re.IGNORECASE,
)

def strip_think_blocks(text: str) -> str:
    """Remove blocos de raciocínio interno (<think>…</think> ou Thinking…done)."""
    cleaned = _THINK_REGEX.sub("", text)
    cleaned = re.sub(r"\n\s*\n", "\n\n", cleaned).strip()
    return cleaned

###############################################################################
# Logger – grava tudo em um único arquivo
###############################################################################
class PromptLogger:
    SEPARATOR = "=" * 100

    def __init__(self, path: Path, system_prompt: str):
        self.path = path
        header = (
            f"# QoS-LLM unified log – {datetime.now().isoformat(timespec='seconds')}\n"
            f"{self.SEPARATOR}\nSYSTEM PROMPT\n{self.SEPARATOR}\n{system_prompt.strip()}\n"
        )
        path.write_text(header, encoding="utf-8")

    def _write(self, text: str):
        with self.path.open("a", encoding="utf-8") as f:
            f.write(text)

    def log(self, stage: str, metric: str, kind: str, content: str | Dict | List):
        body = json.dumps(content, indent=2, ensure_ascii=False) if isinstance(content, (dict, list)) else str(content)
        header = f"\n\n{self.SEPARATOR}\n[{metric.upper()}] {stage} – {kind}\n{self.SEPARATOR}\n"
        self._write(header + body + "\n")

###############################################################################
# Wrapper LLM
###############################################################################
class OllamaLLM:
    def __init__(self, model: str, temperature: float):
        self.chat = ChatOllama(model=model, base_url=OLLAMA_BASE_URL, temperature=temperature)

    def __call__(self, prompt: str) -> str:
        return self.chat.invoke(prompt).content.strip()

###############################################################################
# Seletores de coeficientes
###############################################################################
def select_entity_coefficients(df: pd.DataFrame, worst_cluster: int) -> List[Dict[str, Any]]:
    df_ent = df[df["type"].isin({"client", "server"})][["feature", "type", "coefficient"]].copy()
    sign_mask = df_ent["coefficient"] > 0 if worst_cluster == 1 else df_ent["coefficient"] < 0
    df_sel = df_ent.loc[sign_mask].copy()
    df_sel["abs_coef"] = df_sel["coefficient"].abs()
    thr = df_sel["abs_coef"].mean()
    df_fin = df_sel[df_sel["abs_coef"] > thr].sort_values("abs_coef", ascending=False).copy()
    return df_fin.head(10)[["feature", "type", "coefficient"]].to_dict("records")

###############################################################################
# Cluster pior – agora também registra PROMPT / RESPONSE / RESULT do tool-LLM
###############################################################################
def determine_worst_cluster(
    tool_llm: "OllamaLLM",
    stats: List[Dict[str, Any]],
    logger: PromptLogger,
    metric: str,
) -> int:
    prompt = (
        "You are a network QoS expert. Given the JSON below, which cluster (0 or 1) "
        "has WORSE quality (higher RTT, lower throughput, higher event frequency). "
        "Respond ONLY with the number 0 or 1.\n\nJSON:\n"
        f"```json\n{json.dumps(stats, ensure_ascii=False)}\n```\nAnswer:"
    )

    # ---- logging da chamada (function-call) ----
    logger.log("STAGE 0", metric, "TOOL PROMPT", prompt)

    raw_response = tool_llm(prompt)

    logger.log("STAGE 0", metric, "TOOL RESPONSE", raw_response)

    worst = int(strip_think_blocks(raw_response))

    logger.log("STAGE 0", metric, "TOOL RESULT", f"worst_cluster = {worst}")

    return worst

###############################################################################
# Pipeline
###############################################################################
class QoSAnalyzer:
    FILES = {
        "throughput_download": {
            "cluster_stats": "output_files/clusters_stats_throughput_download.csv",
            "coefficients":   "output_files/coefficients_throughput_download.csv",
            "survival":       "output_files/dataset_survival_throughput_download.csv",
        },
        "rtt_upload": {
            "cluster_stats": "output_files/clusters_stats_rtt_upload.csv",
            "coefficients":   "output_files/coefficients_rtt_upload.csv",
            "survival":       "output_files/dataset_survival_rtt_upload.csv",
        },
    }

    def __init__(self, analyzer_llm: OllamaLLM, tool_llm: OllamaLLM, log: PromptLogger):
        self.analyzer_llm, self.tool_llm, self.log = analyzer_llm, tool_llm, log
        self.system_prompt = PROMPTS["system"].strip()

    def _wrap(self, body: str) -> str:
        """Concatena o prompt de sistema ao corpo antes de enviar ao modelo."""
        return f"{self.system_prompt}\n\n{body.strip()}"

    @staticmethod
    def _subset(df: pd.DataFrame, ent: str) -> List[Dict]:
        mask = (df.client == ent) | (df.site == ent)
        cols = ["client","site","timestamp_start","timestamp_end","time","cluster","event",
                "throughput_download_mean","throughput_upload_mean","rtt_download_mean",
                "rtt_upload_mean","throughput_download_std","throughput_upload_std",
                "rtt_download_std","rtt_upload_std"]
        return df.loc[mask, cols].to_dict("records")

    # ---------------------------- etapas -------------------------------- #
    def _run_single(self, metric: str, paths: Dict[str, str]) -> str:
        stage = lambda n: f"STAGE {n}"

        clusters_df = pd.read_csv(paths["cluster_stats"])
        coef_df     = pd.read_csv(paths["coefficients"])
        clusters_json = clusters_df.to_dict("records")

        # Etapa 0 – usa tool-LLM e agora loga todo o ciclo
        worst = determine_worst_cluster(self.tool_llm, clusters_json, self.log, metric)

        # Etapa 1 – Cluster Analysis
        p1_body = PROMPTS["cluster_analysis"].format(
            DATA_JSON=json.dumps({"cluster_stats": clusters_json}, indent=2, ensure_ascii=False))
        r1_raw = self.analyzer_llm(self._wrap(p1_body))
        self.log.log(stage("1"), metric, "PROMPT", p1_body)
        self.log.log(stage("1"), metric, "RESPONSE", r1_raw)
        r1 = strip_think_blocks(r1_raw)

        # Etapa 2 – Critical Elements
        ent_coefs = select_entity_coefficients(coef_df, worst)
        p2_body = PROMPTS["critical_elements"].format(
            PREVIOUS_OUTPUT=r1,
            DATA_JSON=json.dumps(ent_coefs, indent=2, ensure_ascii=False))
        r2_raw = self.analyzer_llm(self._wrap(p2_body))
        self.log.log(stage("2"), metric, "PROMPT", p2_body)
        self.log.log(stage("2"), metric, "RESPONSE", r2_raw)
        r2 = strip_think_blocks(r2_raw)

        # Etapa 3 – Interval Diagnosis
        surv_df = pd.read_csv(paths["survival"])
        diags = []
        for ent in [d["feature"] for d in ent_coefs]:
            p3_body = PROMPTS["interval_diagnosis"].format(
                ELEMENT_ID=ent,
                DATA_JSON=json.dumps(self._subset(surv_df, ent)))
            r3_raw = self.analyzer_llm(self._wrap(p3_body))
            self.log.log(stage("3"), metric, "PROMPT", p3_body)
            self.log.log(stage("3"), metric, "RESPONSE", r3_raw)
            diags.append(strip_think_blocks(r3_raw))

        # Etapa 4 – Approach Report
        prev = f"{r1}\n\n{r2}\n\n" + "\n\n".join(diags)
        p4_body = PROMPTS["approach_report"].format(
            APPROACH_NAME=metric, PREVIOUS_OUTPUT=prev)
        r4_raw = self.analyzer_llm(self._wrap(p4_body))
        self.log.log(stage("4"), metric, "PROMPT", p4_body)
        self.log.log(stage("4"), metric, "RESPONSE", r4_raw)
        report = strip_think_blocks(r4_raw)

        out = REPORT_DIR / f"{RUN_TS}_{MODEL_NAME}_{metric}.txt"
        out.write_text(report, encoding="utf-8")
        return report

    # Consolidação
    def _consolidate(self, rep: Dict[str, str]) -> str:
        prev = ("\n\n==== Throughput approach ====\n" + rep["throughput_download"] +
                "\n\n==== RTT approach ====\n"        + rep["rtt_upload"])
        p5_body = PROMPTS["consolidated_report"].format(PREVIOUS_OUTPUT=prev)
        raw = self.analyzer_llm(self._wrap(p5_body))
        self.log.log("STAGE 5", "consolidated", "PROMPT", p5_body)
        self.log.log("STAGE 5", "consolidated", "RESPONSE", raw)
        return strip_think_blocks(raw)

    # Pipeline completo
    def run(self):
        rep = {m: self._run_single(m, p) for m, p in self.FILES.items()}
        final = self._consolidate(rep)
        dest  = REPORT_DIR / f"{RUN_TS}_{MODEL_NAME}_consolidated.txt"
        dest.write_text(final, encoding="utf-8")
        print(f"✅ Consolidated report saved to: {dest}")

###############################################################################
# Execução
###############################################################################
def main():
    system_prompt = PROMPTS["system"].strip()
    log = PromptLogger(LOG_FILE, system_prompt)
    a_llm = OllamaLLM(ANALYZER_MODEL, ANALYZER_TEMP)
    t_llm = OllamaLLM(TOOL_MODEL, TOOL_TEMP)
    QoSAnalyzer(a_llm, t_llm, log).run()

if __name__ == "__main__":
    main()