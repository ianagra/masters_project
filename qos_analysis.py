import pandas as pd
import json
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# Log file for saving prompts and responses
LOG_FILE = "qos_analysis_log.txt"


def load_csv(metric: str):
    """
    Load survival, coefficients, and cluster stats CSVs for the given metric.
    """
    df_surv = pd.read_csv(f'dataset_survival_{metric}.csv')
    df_coef = pd.read_csv(f'coefficients_odds_ratio_{metric}.csv')
    df_stats = pd.read_csv(f'clusters_stats_{metric}.csv')
    return df_surv, df_coef, df_stats


def jsonify_df(df: pd.DataFrame):
    """
    Convert a DataFrame to a JSON-serializable Python object.
    """
    return json.loads(df.to_json(orient='records'))


def run_step(model, prompt: str, step_name: str, log_file: str):
    """
    Generic runner: sends prompt, gets response, and logs both.
    """
    # Invoke the model
    response = model.invoke([HumanMessage(content=prompt)]).content

    # Append prompt and response to log
    with open(log_file, "a") as f:
        f.write(f"===== {step_name} =====\n")
        f.write("PROMPT:\n")
        f.write(prompt + "\n\n")
        f.write("RESPONSE:\n")
        f.write(response + "\n\n")

    return response


def main():
    # Initialize/clear the log file
    open(LOG_FILE, "w").close()

    # Initialize the LLM using deepseek-r1-14b
    model = ChatOllama(
        model="deepseek-r1:14b",
        base_url="http://localhost:10000",
        temperature=0,
        num_ctx=8192
    )

    metrics = ["throughput_download", "rtt_upload"]
    approach_reports = {}

    for metric in metrics:
        # Load data
        df_surv, df_coef, df_stats = load_csv(metric)

        # Prepare JSON context
        cluster_stats_json = jsonify_df(df_stats)
        coef_json = jsonify_df(df_coef)
        surv_json = jsonify_df(df_surv)

        # Step 1: Cluster Analysis
        prompt1 = (
            "Step 1: Cluster Analysis - Given the following cluster statistics in JSON format, "
            "generate a detailed descriptive summary of each performance cluster in clear, operational language for network operators.\n\n"
            + json.dumps(cluster_stats_json, indent=2)
        )
        desc = run_step(model, prompt1, "Step 1: Cluster Analysis", LOG_FILE)

        # Step 2: Identify Critical Elements
        prompt2 = (
            "Step 2: Identify Critical Elements - Using the following odds ratio data in JSON format and the cluster descriptions, "
            "list network elements (clients and servers) that are critical (high odds ratios for unstable clusters) and explain why each is critical.\n\n"
            + "Odds Ratio Data:\n" + json.dumps(coef_json, indent=2) + "\n\n"
            + "Cluster Descriptions:\n" + desc
        )
        crit = run_step(model, prompt2, "Step 2: Identify Critical Elements", LOG_FILE)

        # Step 3: Interval Analysis
        prompt3 = (
            "Step 3: Interval Analysis - For each critical element listed, examine its historical intervals from the survival data in JSON format. "
            "Analyze temporal evolution, interactions with other elements, and performance metrics. Provide a detailed diagnostic for each.\n\n"
            + "Survival Data:\n" + json.dumps(surv_json, indent=2) + "\n\n"
            + "Critical Elements:\n" + crit
        )
        diag = run_step(model, prompt3, "Step 3: Interval Analysis", LOG_FILE)

        # Step 4: Generate Approach Report
        prompt4 = (
            f"Step 4: Generate Approach Report for metric {metric} - Using outputs from Steps 1, 2, and 3, create a detailed QoS report for network operators.\n\n"
            + "Cluster Analysis:\n" + desc + "\n\n"
            + "Critical Elements Analysis:\n" + crit + "\n\n"
            + "Interval Diagnostics:\n" + diag
        )
        rpt = run_step(model, prompt4, f"Step 4: Approach Report ({metric})", LOG_FILE)
        approach_reports[metric] = rpt

    # Final Step: Consolidated QoS Report
    final_prompt = (
        "Final Step: Consolidated QoS Report - You have individual reports for different metrics. Compare and contrast these approaches, "
        "highlight complementarities and peculiarities, and provide practical recommendations for network operators.\n\n"
        + "\n\n".join([f"Report for {m}:\n{r}" for m, r in approach_reports.items()])
    )
    final_report = run_step(model, final_prompt, "Final Step: Consolidated QoS Report", LOG_FILE)

    # Output final report
    print("=== Consolidated QoS Report ===\n")
    print(final_report)


if __name__ == "__main__":
    main()