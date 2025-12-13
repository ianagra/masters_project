PROMPTS = {
    # ------------------------------------------------------------------
    # Global system prompt (fed once at model initialisation)
    # ------------------------------------------------------------------
    "system":
        """
# Persona
You are a senior network performance analyst at a major Brazilian ISP. Your task is to diagnose Quality of Service (QoS) issues by interpreting measurement data from network probes.

# Experiment Context
You will analyze data from two parallel approaches, each defined by a different reference metric for detecting instability (statistical changes).

## Data Collection
- Probes: 16 Raspberry Pi devices ran Network Diagnostic Tool (NDT) tests every 30 minutes.
- Raw Metrics: Throughput (download/upload) and RTT (download/upload).
- Targets: RNP (Brazilian National Research and Education Network) servers.

## Methodological Pipeline
1. Change-Point Detection: For each client-server pair, statistical change-points were detected for the reference metric, slicing the timeline into "intervals" of homogeneous behavior.
2. Multivariate Interval Characterization: While each interval is defined by a single reference metric (either throughput or RTT), every interval was characterized by the local mean and standard deviation of ALL FOUR metrics.
3. Clustering and Logistic Regression:
    - A clustering algorithm grouped all intervals into exactly two clusters: Cluster 0 and Cluster 1.
    - The clustering algorithm uses ALL interval's features to define the clusters: time, event occurrence, client/server ID, and the local mean/std of all the four metrics.
    - The model uses multinomial logistic regression with Cluster 0 as the baseline to define the probability of an interval belonging to one cluster or the other.
    - CRUCIAL: The logistic regression coefficients you will see are for the log-odds of an interval belonging to Cluster 1. A positive coefficient for a variable increases the odds of being in Cluster 1; a negative coefficient increases the odds of being in Cluster 0.

# Your Eight-Stage Mission
1. Cluster Analysis: Interpret aggregated cluster statistics to define a risk profile for each cluster.
2. Identification of the most influential metrics: use the logistic regression coefficients to identify the metrics that have the greatest impact on the definition of the clusters.
3. Critical Clients Identification: Use the logistic regression coefficients to identify clients most strongly associated with the high risk cluster.
4. Critical Clients' Interval Diagnosis: Analyze the temporal performance evolution of each of these critical clients.
5. Critical Servers Identification: Use the logistic regression coefficients to identify servers most strongly associated with the high risk cluster.
6. Critical Servers' Interval Diagnosis: Analyze the temporal performance evolution of each of these critical servers.
7. Technical Report per Approach: Consolidate findings for each approach (throughput and RTT) into a technical report.
8. Consolidated Operational Report: Synthesize both technical reports into a single, actionable document for network operations teams.

# Core Directives
- Data Grounding: Base ALL conclusions strictly on the data provided in each step. Do not extrapolate or assume.
- Output Language: All technical analyses (Stages 1-7) must be in expert-level English. The final Consolidated Operational Report (Stage 8) MUST be written in clear, accessible Brazilian Portuguese (PT-BR), keeping technical terms like "RTT" and "throughput" in English.
- Formatting: Use Markdown `##` titles for structure. DO NOT write tables.
        """,

# ------------------------------------------------------------------
# Stage-1A prompt – cluster analysis
# ------------------------------------------------------------------
    "cluster_analysis":
        """
## Context
Analyze the following JSON, which contains aggregated statistics for two network performance clusters. The data includes metrics for throughput, RTT, interval duration, and survival probability (the likelihood of an interval remaining stable over time).

```json
{DATA_JSON}
````

## Task

For each cluster (Cluster 0 and Cluster 1), provide the following analysis:

1. Risk Profile Assignment: Label the cluster as either "High-Risk Profile (Unstable)" or "Low-Risk Profile (Stable)".
2. Justification: Provide a concise rationale for your assignment based on three key areas:
      - Connection Stability: Analyze `time_days_mean`, `event_frequency`, and `survival_probability`. Shorter durations and lower survival probabilities indicate instability.
      - Throughput Performance: Evaluate the mean values for `throughput_download` and `throughput_upload`.
      - Latency Performance: Evaluate the mean values for `rtt_download` and `rtt_upload`.

Structure your response under a main Markdown heading "## Cluster Profile Analysis", with a separate subsection for each cluster.
""",

# ------------------------------------------------------------------
# Stage-1B prompt – worst cluster identification
# ------------------------------------------------------------------
    "worst_cluster_from_profile":
        """
# Context
You are a network QoS expert. You are given the "Cluster Profile Analysis" section, which contains the risk profiles of two clusters (0 and 1).

# Task
Determine which cluster (0 or 1) is the WORST (i.e., the one with high-risk profile).

# Tie-braking rules
Tie-breaking rules if both appear high-risk or the label is not explicit:
1. Choose the cluster with LOWER survival probability and/or HIGHER event frequency. 
2. If still tied, choose the one with WORSE latency (higher RTT) or WORSE throughput (lower). 
3. If still tied, return None.

---

{PREVIOUS_OUTPUT}

---

Return ONLY the digit 0 or 1. Do not include any other text, punctuation, or code fences.
""",

# --------------------------------------------------------------------------
# Stage-2 prompt – critical metrics detection
# --------------------------------------------------------------------------
    "critical_metrics":
    """
## Context
You will now explain why the selected performance metrics are considered the primary drivers of network instability.  
These metrics were pre-selected based on their logistic regression coefficients: only those whose absolute values are above the 50th percentile were retained.  
This ensures that only the most impactful metrics are included, according to their statistical weight in the model.

## Inputs
1. Previous Cluster Profile Analysis (Worst cluster is Cluster {WORST_CLUSTER_ID}):

    {PREVIOUS_OUTPUT}

2. Logistic Regression Coefficients for Metrics (filtered to include only the most impactful ones):

```json
{DATA_JSON}
```

## Model Reminder
The logistic regression coefficients are for the log-odds of an interval belonging to Cluster 1. Cluster 0 is the baseline.
A negative coefficient for a variable increases the odds of it belonging to Cluster 0.

## Task
For each metric in the provided list:

1. Explain why this metric is important for determining connection stability.
2. Interpret the sign of the coefficient in relation to the high-risk cluster (Cluster {WORST_CLUSTER_ID}).

   - Example: "A large negative coefficient for 'rtt_download_mean' means that a higher average download RTT strongly increases the likelihood of an interval belonging to the high-risk Cluster 0."

Format the output under a heading "## Critical Metrics Explanation".
""",

# ------------------------------------------------------------------
# Stage-3 prompt – critical clients detection
# ------------------------------------------------------------------
    "critical_clients":
    """
## Inputs
1. Previous Cluster Profile Analysis (worst cluster is {WORST_CLUSTER_ID}):

    {PREVIOUS_OUTPUT}

2. Logistic Regression Coefficients for Clients (pre-filtered):
   - The JSON includes only clients whose absolute coefficients are above the 50th percentile within the client group, and whose coefficient sign points to the high-risk cluster {WORST_CLUSTER_ID}.

    ```json
    {DATA_JSON}
    ```

## Model Reminder
- Coefficients are for the log-odds of belonging to Cluster 1 (Cluster 0 is the baseline).
- A negative coefficient increases the odds of Cluster 0; a positive coefficient increases the odds of Cluster 1.
- Since the data is pre-filtered, all listed clients already point to the high-risk cluster {WORST_CLUSTER_ID} by sign.

## Task
For each client in the JSON (do not add clients beyond those provided):

1. Provide the client's name (e.g., `client07`).
2. Provide the exact coefficient value (with sign).
3. In one sentence, explain why this value indicates association with the high-risk cluster {WORST_CLUSTER_ID}, referencing the sign logic above.
4. Order the list by descending absolute coefficient magnitude.

Format the output as a Markdown list under the heading:
"## Critical Clients"
""",

# ------------------------------------------------------------------
# Stage-4 prompt – clients intervals deep dive
# ------------------------------------------------------------------
    "interval_diagnosis_client":
    """
## Inputs
1. Critical Metrics Analysis:

    {CRITICAL_METRICS_ANALYSIS}

2. Critical Client Under Review: `{CLIENT_ID}`

3. Interval Records for this client (JSON): Each record contains timestamps, cluster assignment, event flag, and local performance metrics for a specific time window.

    ```json
    {DATA_JSON}
    ```

## Task

Provide a concise diagnostic for `{CLIENT_ID}`:

1. Summarize Temporal Behavior: Describe the client's performance evolution. Was it consistently unstable, or were there specific periods of degradation? Compare the proportion of intervals in each cluster.
2. Identify Key Interactions: Note any partner servers that consistently appear in its high-risk intervals.
3. Analyse which metrics are most strongly correlated with the client's instability.
4. Formulate an Actionable Insight: Conclude with a specific, testable hypothesis for the operations team (e.g., "The consistent high RTT during interactions with `rnp_rj` suggests a potential routing issue to be investigated via `traceroute`.").

Produce a subsection titled "Diagnosis of client {CLIENT_ID}".
""",

# ------------------------------------------------------------------
# Stage-5 prompt – critical servers detection
# ------------------------------------------------------------------
    "critical_servers":
    """
## Inputs
1. Previous Cluster Profile Analysis (worst cluster is {WORST_CLUSTER_ID}):

    {PREVIOUS_OUTPUT}

2. Logistic Regression Coefficients for Servers (pre-filtered):
   - The JSON includes only servers whose absolute coefficients are above the 50th percentile within the server group, and whose coefficient sign points to the high-risk cluster {WORST_CLUSTER_ID}.

    ```json
    {DATA_JSON}
    ```

## Model Reminder
- Coefficients are for the log-odds of belonging to Cluster 1 (Cluster 0 is the baseline).
- A negative coefficient increases the odds of Cluster 0; a positive coefficient increases the odds of Cluster 1.
- Since the data is pre-filtered, all listed servers already point to the high-risk cluster {WORST_CLUSTER_ID} by sign.

## Task
For each server in the JSON (do not add servers beyond those provided):

1. Provide the server's name (e.g., `gru03`).
2. Provide the exact coefficient value (with sign).
3. In one sentence, explain why this value indicates association with the high-risk cluster {WORST_CLUSTER_ID}, referencing the sign logic above.
4. Order the list by descending absolute coefficient magnitude.

Format the output as a Markdown list under the heading:
"## Critical Servers"
""",

# ------------------------------------------------------------------
# Stage-6 prompt – servers intervals deep dive
# ------------------------------------------------------------------
    "interval_diagnosis_server":
    """
## Inputs
1. Critical Metrics Analysis:

    {CRITICAL_METRICS_ANALYSIS}

2. Critical Server Under Review: `{SERVER_ID}`

3. Interval Records for this server (JSON): Each record contains timestamps, cluster assignment, event flag, and local performance metrics for a specific time window.

    ```json
    {DATA_JSON}
    ```

## Task

Provide a concise diagnostic for `{SERVER_ID}`:

1. Summarize Temporal Behavior: Describe the server's performance evolution. Was it consistently unstable, or were there specific periods of degradation? Compare the proportion of intervals in each cluster.
2. Identify Key Interactions: Note any partner clients that consistently appear in its high-risk intervals.
3. Analyse which metrics are most strongly correlated with the client's instability.
4. Formulate an Actionable Insight: Conclude with a specific, testable hypothesis for the operations team (e.g., "The consistent high RTT during interactions with `client03` suggests a potential routing issue to be investigated via `traceroute`.").

Produce a subsection titled "Diagnosis of server {SERVER_ID}".
""",

# ------------------------------------------------------------------
# Stage-7 prompt – approach report
# ------------------------------------------------------------------
    "approach_report":
    """
## Inputs

Consolidate the following analyses in order:

1. Cluster Profile Analysis.
2. Most impactful Metrics.
3. Critical Clients.
4. Critical Servers.
5. Individual Diagnoses for each critical client.
6. Individual Diagnoses for each critical server.

---

{PREVIOUS_OUTPUT}

---

## Task

Draft a concise Technical Report titled "QoS Assessment Report: {APPROACH_NAME} Approach". Structure it as follows:

1. Executive Summary: A brief paragraph summarizing the main findings.

2. Detailed Findings: Integrate the provided analyses into a coherent narrative. Include:
    - A list of the most critical clients.
    - A list of the most critical servers.
    - A list of the metrics with greatest impact on stability, explaining their influence.

3. Recommended Actions: A bulleted list of technical next steps based on your findings.
    """,

# ------------------------------------------------------------------
# Stage-8 prompt – consolidated operator report (Output in PT-BR)
# ------------------------------------------------------------------

    "consolidated_report":
    """

## Inputs

You are provided with two separate technical reports: one based on DOWNLOAD THROUGHPUT instability and one on UPLOAD RTT instability.

---

## {PREVIOUS_OUTPUT}

---

## Final Task

Your final task is to write a single, consolidated Operational Report for the network operations and field teams. This report MUST be written in clear, direct Brazilian Portuguese (PT-BR).

Structure the report as follows:

### 1. Executive Summary: Capacity vs. Responsiveness

Explain in simple terms why we analyze the network from these two perspectives (one measures “delivery capacity” and the other the “responsiveness of the connection”) and what the main finding was for each.

### 2. Unified Diagnosis: Where Are the Critical Points?

- List the clients with critical performance, indicating whether they were identified in both approaches or only in one.
- List the servers with critical performance, indicating whether they were identified in both approaches or only in one.
- Summarize the main metrics that have the greatest impact on service quality, clearly explaining their role and relevance to the diagnosis.
- Highlight consistent bottlenecks (critical elements present in both analyses) as top priority.
- Also mention issues that are exclusive to one approach and what this might indicate.

### 3. Prioritized Action Plan

Provide a list of practical and sequential actions for the technical team. Organize the actions by priority level (High, Medium, Low).

- Example of High Priority Action:

  - Asset: Server `gru03` (critical in both analyses).
  - Action 1 (NOC): Immediately check the server’s monitoring dashboards for hardware alerts (CPU, memory) and link saturation.
  - Action 2 (Field Team): Run a `traceroute` from a problematic client (e.g., `client12`) to `gru03` to map the path and identify hops with high latency.
  - Action 3 (Engineering): Escalate to the Network Engineering team to review routing policies and the peering agreement with RNP at this location.

Be objective and focus on verifiable actions. Avoid statistical or machine learning jargon.
"""
}