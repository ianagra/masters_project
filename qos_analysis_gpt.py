from __future__ import annotations
from dotenv import load_dotenv
import os, json, re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from langchain_openai import ChatOpenAI
from prompt_templates import PROMPTS

###############################################################################
# Configurações
###############################################################################
load_dotenv(".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MODEL = "gpt-5"
TEMP  = 0.3

RUN_TS     = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR    = Path("logs");    LOG_DIR.mkdir(exist_ok=True)
REPORT_DIR = Path("reports"); REPORT_DIR.mkdir(exist_ok=True)
LOG_FILE   = LOG_DIR / f"{RUN_TS}_{MODEL}.txt"

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
# Wrapper LLM (OpenAI)
###############################################################################
class OpenAILLM:
    def __init__(self, model: str):
        self.chat = ChatOpenAI(
            model=model,
            openai_api_key=OPENAI_API_KEY
        )

    def __call__(self, prompt: str) -> str:
        return self.chat.invoke(prompt).content.strip()

###############################################################################
# Seletores de coeficientes e métricas
###############################################################################
def select_client_coefficients(df: pd.DataFrame, worst_cluster: int) -> List[Dict[str, Any]]:
    df_ent = df[df["type"] == "client"][["feature", "type", "coefficient"]].copy()
    sign_mask = df_ent["coefficient"] > 0 if worst_cluster == 1 else df_ent["coefficient"] < 0
    df_sel = df_ent.loc[sign_mask].copy()
    df_sel["abs_coef"] = df_sel["coefficient"].abs()
    #thr = df_sel["abs_coef"].quantile(0.75)
    thr = df_sel["abs_coef"].median()
    df_fin = df_sel[df_sel["abs_coef"] > thr].sort_values("abs_coef", ascending=False).copy()
    return df_fin[["feature", "type", "coefficient"]].to_dict("records")

def select_server_coefficients(df: pd.DataFrame, worst_cluster: int) -> List[Dict[str, Any]]:
    df_ent = df[df["type"] == "server"][["feature", "type", "coefficient"]].copy()
    sign_mask = df_ent["coefficient"] > 0 if worst_cluster == 1 else df_ent["coefficient"] < 0
    df_sel = df_ent.loc[sign_mask].copy()
    df_sel["abs_coef"] = df_sel["coefficient"].abs()
    #thr = df_sel["abs_coef"].quantile(0.75)
    thr = df_sel["abs_coef"].median()
    df_fin = df_sel[df_sel["abs_coef"] > thr].sort_values("abs_coef", ascending=False).copy()
    return df_fin[["feature", "type", "coefficient"]].to_dict("records")

def select_metric_coefficients(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Filtra e seleciona os coeficientes das métricas para análise."""
    df_met = df[df["type"] == "metric"].copy()
    df_met["abs_coef"] = df_met["coefficient"].abs()
    #thr = df_sel["abs_coef"].quantile(0.75)
    thr = df_met["abs_coef"].median()
    df_fin = df_met[df_met["abs_coef"] > thr].sort_values("abs_coef", ascending=False).copy()
    return df_fin[["feature", "coefficient"]].to_dict("records")

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

    def __init__(self, analyzer_llm: OpenAILLM, log: PromptLogger):
        self.analyzer_llm, self.log = analyzer_llm, log
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

        # ===================== ETAPA 1 =====================
        # 1A) Cluster Analysis
        p1_body = PROMPTS["cluster_analysis"].format(
            DATA_JSON=json.dumps({"cluster_stats": clusters_json}, indent=2, ensure_ascii=False))
        self.log.log(stage("1A"), metric, "PROMPT", p1_body)
        r1_raw = self.analyzer_llm(self._wrap(p1_body))
        self.log.log(stage("1A"), metric, "RESPONSE", r1_raw)
        r1 = strip_think_blocks(r1_raw)

        # 1B) Extração do cluster pior usando a PRÓPRIA resposta da 1A
        p1b_body = PROMPTS["worst_cluster_from_profile"].format(PREVIOUS_OUTPUT=r1)
        self.log.log(stage("1B"), metric, "PROMPT", p1b_body)
        r1b_raw = self.analyzer_llm(self._wrap(p1b_body))
        self.log.log(stage("1B"), metric, "RESPONSE", r1b_raw)
        r1b = strip_think_blocks(r1b_raw)

        # Sanitização para inteiro 0/1
        m = re.search(r"[01]", r1b)
        if not m:
            # fallback conservador: tenta inferir por palavras-chave se o modelo não seguiu estritamente
            # Preferência para "Cluster 1" se mencionado como high-risk; senão, 0.
            worst = 1 if re.search(r"cluster\s*1", r1, flags=re.I) and re.search(r"high[- ]?risk", r1, flags=re.I) else 0
        else:
            worst = int(m.group(0))

        self.log.log(stage("1B"), metric, "RESULT", f"worst_cluster = {worst}")

        # ===================== ETAPA 2 =====================
        metric_coefs = select_metric_coefficients(coef_df)
        p2_body = PROMPTS["critical_metrics"].format(
            WORST_CLUSTER_ID=worst,
            PREVIOUS_OUTPUT=r1,
            DATA_JSON=json.dumps(metric_coefs, indent=2, ensure_ascii=False))
        self.log.log(stage("2"), metric, "PROMPT", p2_body)
        r2_raw = self.analyzer_llm(self._wrap(p2_body))
        self.log.log(stage("2"), metric, "RESPONSE", r2_raw)
        r2 = strip_think_blocks(r2_raw)

        # ===================== ETAPA 3 =====================
        client_coefs = select_client_coefficients(coef_df, worst)
        p3_body = PROMPTS["critical_clients"].format(
            WORST_CLUSTER_ID=worst,
            PREVIOUS_OUTPUT=r1,
            DATA_JSON=json.dumps(client_coefs, indent=2, ensure_ascii=False))
        self.log.log(stage("3"), metric, "PROMPT", p3_body)
        r3_raw = self.analyzer_llm(self._wrap(p3_body))
        self.log.log(stage("3"), metric, "RESPONSE", r3_raw)
        r3 = strip_think_blocks(r3_raw)

        # ===================== ETAPA 4 =====================
        surv_df = pd.read_csv(paths["survival"])
        diags_clients = []
        for client in [d["feature"] for d in client_coefs]:
            p4_body = PROMPTS["interval_diagnosis_client"].format(
                CLIENT_ID=client,
                CRITICAL_METRICS_ANALYSIS=r2,
                DATA_JSON=json.dumps(self._subset(surv_df, client)))
            self.log.log(stage("4"), metric, "PROMPT", p4_body)
            r4_raw = self.analyzer_llm(self._wrap(p4_body))
            self.log.log(stage("4"), metric, "RESPONSE", r4_raw)
            diags_clients.append(strip_think_blocks(r4_raw))

        # ===================== ETAPA 5 =====================
        server_coefs = select_server_coefficients(coef_df, worst)
        p5_body = PROMPTS["critical_servers"].format(
            WORST_CLUSTER_ID=worst,
            PREVIOUS_OUTPUT=r1,
            DATA_JSON=json.dumps(server_coefs, indent=2, ensure_ascii=False))
        self.log.log(stage("5"), metric, "PROMPT", p5_body)
        r5_raw = self.analyzer_llm(self._wrap(p5_body))
        self.log.log(stage("5"), metric, "RESPONSE", r5_raw)
        r5 = strip_think_blocks(r5_raw)

        # ===================== ETAPA 6 =====================
        diags_servers = []
        for server in [d["feature"] for d in server_coefs]:
            p6_body = PROMPTS["interval_diagnosis_server"].format(
                SERVER_ID=server,
                CRITICAL_METRICS_ANALYSIS=r2,
                DATA_JSON=json.dumps(self._subset(surv_df, server)))
            self.log.log(stage("6"), metric, "PROMPT", p6_body)
            r6_raw = self.analyzer_llm(self._wrap(p6_body))
            self.log.log(stage("6"), metric, "RESPONSE", r6_raw)
            diags_servers.append(strip_think_blocks(r6_raw))

        # ===================== ETAPA 7 =====================
        prev = f"{r1}\n\n{r2}\n\n{r3}\n\n{r5}\n\n" + "\n\n".join(diags_clients) + "\n\n" + "\n\n".join(diags_servers)
        p7_body = PROMPTS["approach_report"].format(
            APPROACH_NAME=metric, PREVIOUS_OUTPUT=prev)
        self.log.log(stage("7"), metric, "PROMPT", p7_body)
        r7_raw = self.analyzer_llm(self._wrap(p7_body))
        self.log.log(stage("7"), metric, "RESPONSE", r7_raw)
        report = strip_think_blocks(r7_raw)

        out = REPORT_DIR / f"{RUN_TS}_{MODEL}_{metric}.txt"
        out.write_text(report, encoding="utf-8")
        return report

    # Etapa 8 - Consolidação
    def _consolidate(self, rep: Dict[str, str]) -> str:
        prev = ("\n\n==== Throughput approach ====\n" + rep["throughput_download"] +
                "\n\n==== RTT approach ====\n"        + rep["rtt_upload"])
        p8_body = PROMPTS["consolidated_report"].format(PREVIOUS_OUTPUT=prev)
        self.log.log("STAGE 8", "consolidated", "PROMPT", p8_body)
        raw = self.analyzer_llm(self._wrap(p8_body))
        self.log.log("STAGE 8", "consolidated", "RESPONSE", raw)
        return strip_think_blocks(raw)

    # Pipeline completo
    def run(self):
        rep = {m: self._run_single(m, p) for m, p in self.FILES.items()}
        final = self._consolidate(rep)
        dest  = REPORT_DIR / f"{RUN_TS}_{MODEL}_consolidated.txt"
        dest.write_text(final, encoding="utf-8")
        print(f"✅ Consolidated report saved to: {dest}")

###############################################################################
# Execução
###############################################################################
def main():
    system_prompt = PROMPTS["system"].strip()
    log = PromptLogger(LOG_FILE, system_prompt)
    a_llm = OpenAILLM(MODEL)
    QoSAnalyzer(a_llm, log).run()

if __name__ == "__main__":
    main()