from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

###############################################################################
# Configurações
###############################################################################
load_dotenv(".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MODEL = "gpt-5"
TEMP  = 0.3

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

# Enviar prompt para o modelo
def query_model(prompt: str) -> str:
    llm = OpenAILLM(model=MODEL)
    response = llm(prompt)
    return response

prompt = """
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
    - The analysis revealed two distinct operational profiles: an "unstable" cluster (Cluster 0), characterized by frequent statistical changes and degraded performance metrics such as higher latency and lower throughput, and a "stable" cluster (Cluster 1), defined by extended periods of consistent performance and superior quality of service.
    - The clustering algorithm uses ALL interval's features to define the clusters: time, event occurrence, client/server ID, and the local mean/std of all the four metrics.
    - The model uses multinomial logistic regression with Cluster 0 as the baseline to define the probability of an interval belonging to one cluster or the other.
    - CRUCIAL: The logistic regression coefficients you will see are for the log-odds of an interval belonging to Cluster 1. A positive coefficient for a variable increases the odds of being in Cluster 1; a negative coefficient increases the odds of being in Cluster 0.

# Logistic Regression Coefficients

1. Approach 1 (based on changepoints in download throughput):

```json
[
  {
    "feature":"client01",
    "coefficient":0.8325555256
  },
  {
    "feature":"client02",
    "coefficient":-1.9520590967
  },
  {
    "feature":"client03",
    "coefficient":0.47349628
  },
  {
    "feature":"client04",
    "coefficient":0.123274743
  },
  {
    "feature":"client05",
    "coefficient":1.3414314327
  },
  {
    "feature":"client06",
    "coefficient":0.3746463827
  },
  {
    "feature":"client07",
    "coefficient":-1.7344425085
  },
  {
    "feature":"client08",
    "coefficient":-0.4008017327
  },
  {
    "feature":"client09",
    "coefficient":-0.3169876699
  },
  {
    "feature":"client10",
    "coefficient":-0.2823155657
  },
  {
    "feature":"client11",
    "coefficient":0.6814997914
  },
  {
    "feature":"client12",
    "coefficient":0.48601729
  },
  {
    "feature":"client13",
    "coefficient":-0.6060202457
  },
  {
    "feature":"client14",
    "coefficient":0.5455196667
  },
  {
    "feature":"client15",
    "coefficient":0.4047569824
  },
  {
    "feature":"client16",
    "coefficient":0.0461249131
  },
  {
    "feature":"gru03",
    "coefficient":-0.4635384459
  },
  {
    "feature":"gru05",
    "coefficient":1.8157559834
  },
  {
    "feature":"gru06",
    "coefficient":1.2646868813
  },
  {
    "feature":"rnp_rj",
    "coefficient":-1.3675436001
  },
  {
    "feature":"rnp_sp",
    "coefficient":-1.2326646303
  },
  {
    "feature":"throughput_download_mean",
    "coefficient":0.1128004894
  },
  {
    "feature":"throughput_upload_mean",
    "coefficient":0.4125980697
  },
  {
    "feature":"rtt_download_mean",
    "coefficient":-1.4540723482
  },
  {
    "feature":"rtt_upload_mean",
    "coefficient":-0.6237261322
  },
  {
    "feature":"throughput_download_std",
    "coefficient":-0.5446198542
  },
  {
    "feature":"throughput_upload_std",
    "coefficient":0.4792671179
  },
  {
    "feature":"rtt_download_std",
    "coefficient":-0.9938423881
  },
  {
    "feature":"rtt_upload_std",
    "coefficient":0.8974343785
  }
]
```

2. Approach 2 (based on changepoints in upload RTT):

```json
[
  {
    "feature":"client01",
    "coefficient":0.0250031033
  },
  {
    "feature":"client02",
    "coefficient":-0.6515054253
  },
  {
    "feature":"client03",
    "coefficient":0.6016035555
  },
  {
    "feature":"client04",
    "coefficient":-0.2014864509
  },
  {
    "feature":"client05",
    "coefficient":0.8722492779
  },
  {
    "feature":"client06",
    "coefficient":0.4000450999
  },
  {
    "feature":"client07",
    "coefficient":-0.6036415681
  },
  {
    "feature":"client08",
    "coefficient":-0.4561600495
  },
  {
    "feature":"client09",
    "coefficient":-0.9922840081
  },
  {
    "feature":"client10",
    "coefficient":-0.0108681562
  },
  {
    "feature":"client11",
    "coefficient":0.0927346616
  },
  {
    "feature":"client12",
    "coefficient":-0.7821428635
  },
  {
    "feature":"client13",
    "coefficient":0.5795422958
  },
  {
    "feature":"client14",
    "coefficient":0.1137166309
  },
  {
    "feature":"client15",
    "coefficient":0.0987833045
  },
  {
    "feature":"client16",
    "coefficient":0.9177596356
  },
  {
    "feature":"gru03",
    "coefficient":0.1500247571
  },
  {
    "feature":"gru05",
    "coefficient":0.5907924468
  },
  {
    "feature":"gru06",
    "coefficient":0.5872500008
  },
  {
    "feature":"rnp_rj",
    "coefficient":-0.903056362
  },
  {
    "feature":"rnp_sp",
    "coefficient":-0.4216617993
  },
  {
    "feature":"throughput_download_mean",
    "coefficient":1.3197135225
  },
  {
    "feature":"throughput_upload_mean",
    "coefficient":0.6422444231
  },
  {
    "feature":"rtt_download_mean",
    "coefficient":-0.6624600468
  },
  {
    "feature":"rtt_upload_mean",
    "coefficient":0.2166369645
  },
  {
    "feature":"throughput_download_std",
    "coefficient":-0.5893792859
  },
  {
    "feature":"throughput_upload_std",
    "coefficient":0.828071721
  },
  {
    "feature":"rtt_download_std",
    "coefficient":-1.9935349937
  },
  {
    "feature":"rtt_upload_std",
    "coefficient":-0.1242569566
  }
]
```

# Your Mission
1. Identification of the most influential metrics: use the logistic regression coefficients to identify the metrics that have the greatest impact on the definition of the clusters.
2. Critical Clients Identification: Use the logistic regression coefficients to identify clients most strongly associated with Cluster 0.
3. Critical Servers Identification: Use the logistic regression coefficients to identify servers most strongly associated with Cluster 0.
4. Operational Report: Synthesize findings for both approaches (throughput and RTT) into a technical report. into a single, actionable document for network operations teams.

# Core Directives
- Data Grounding: Base ALL conclusions strictly on the data provided. Do not extrapolate or assume.
- Output Language: English.
- Formatting: Use Markdown `##` titles for structure. DO NOT write tables.
"""

response = query_model(prompt)  # Teste de conexão com o modelo
print(response)
