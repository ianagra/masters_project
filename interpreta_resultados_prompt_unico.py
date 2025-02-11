from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import pandas as pd
import numpy as np
import json
from datetime import datetime

def numpy_to_python(obj):
    """
    Convert numpy types to native Python types for JSON serialization
    """
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
        np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj

def calculate_odds_ratios(df):
    """
    Calculate odds ratios from coefficients
    """
    df['odds_ratio'] = np.exp(df['coefficient'])
    return df

def preprocess_data(survival_data, coefficients_data):
    """
    Preprocesses the data to create more structured and informative inputs for the LLM.
    """
    # Calculate odds ratios
    coefficients_data = calculate_odds_ratios(coefficients_data)
    
    # Criar dataframe com dados das features para cada cluster
    clusters_data = survival_data.groupby('cluster').agg({
        'time': 'mean',
        'event': 'sum',
        'throughput_download': 'mean',
        'throughput_download_std': 'mean',
        'throughput_upload': 'mean', 
        'throughput_upload_std': 'mean',
        'rtt_download': 'mean',
        'rtt_download_std': 'mean',
        'rtt_upload': 'mean',
        'rtt_upload_std': 'mean'
    }).reset_index()

    # Renomear colunas de features não terminadas em _std para incluir _mean
    clusters_data.columns = [col + '_mean' if not (col.endswith('_std') or col == 'event' or col == 'cluster') else col for col in clusters_data.columns]
        
    return clusters_data, coefficients_data

def create_enhanced_prompt(survival_data, coefficients_data):
    """
    Creates a structured prompt for the LLM without biasing cluster interpretation.
    """
    system_context = """
You are a machine learning and computer networks specialist.
You are analyzing network performance data collected from an ISP network.

The data was processed as follows:

1. Change points were detected in download throughput time series for each client-server pair.
2. Intervals between changes were analyzed using survival analysis.
3. Intervals were clustered into 2 groups based on:
- Interval duration, in days;
- Associated metrics (throughput, RTT);
- Client and server IDs; and
- Event occurrence (1) or censored data (0).
4. Logistic regression was used to determine feature importance for cluster membership.

Keep responses direct and actionable. Focus on identifying specific elements needing intervention.
Base your analysis on the data provided.
Present your analysis in clear language suitable for network operators.
    """
    
    analysis_template = """
Based on the following cluster statistics:

{cluster_stats}

Provide a comparative analysis of the clusters focusing on:

1. Performance metrics (throughput and RTT).
2. Stability (duration in days between changes).
3. Event frequency (changes in performance).

In the end, which cluster represents better network performance and why?


Now, here are the logistic regression coefficients and odds ratios related to Cluster 1:
{feature_categories}

Based on your cluster analysis and the latter data, please provide:

1. ALL clients with poor performance.
2. ALL servers with poor performance.
3. The top 3 metrics that most strongly indicate poor performance.
4. Practical and specific recommendations for network service improvement.
        """
    
    formatted_analysis_template = analysis_template.format(
        cluster_stats=json.dumps(survival_data.to_dict(orient="records"), indent=2),
        feature_categories=json.dumps(coefficients_data.to_dict(orient="records"), indent=2)
    )
    
    return system_context, formatted_analysis_template

def save_conversation(model_name, system_prompt, user_prompt, assistant_response):
    """
    Saves the complete conversation to a file with timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"analysis_{model_name}_{timestamp}.txt"
    
    conversation = f"""System Prompt:
{'-' * 80}
{system_prompt}

User Prompt:
{'-' * 80}
{user_prompt}

Assistant Response:
{'-' * 80}
{assistant_response}
"""
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(conversation)
    
    return filename

def main():
    # Load data
    survival_data = pd.read_csv("dataset_survival_thr_d.csv")
    coefficients_data = pd.read_csv("coefficients_thr_d.csv")
    
    # Process data
    survival_data, coefficients_data = preprocess_data(survival_data, coefficients_data)
    system_context, analysis_prompt = create_enhanced_prompt(survival_data, coefficients_data)
    
    # Initialize model
    model = ChatOllama(
        model='llama3.2:latest',
        base_url='http://10.246.47.169:10000',
        temperature=0,
        #top_p=0.3,
    )
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_context),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessage(content=analysis_prompt)
    ])
    
    # Create chain with RunnablePassthrough
    chain = (
        RunnablePassthrough.assign(
            survival_data=lambda _: survival_data,
            coefficients_data=lambda _: coefficients_data,
            chat_history=lambda _: []
        )
        | prompt
        | model
        | StrOutputParser()
    )
    
    try:
        # Get the analysis result
        analysis = chain.invoke({})
        
        # Save the complete conversation
        saved_file = save_conversation(
            model_name="llama3.2",
            system_prompt=system_context,
            user_prompt=analysis_prompt,
            assistant_response=analysis
        )
        
        print(f"Conversation saved to: {saved_file}")
            
    except Exception as e:
        print(f"Error in analysis: {str(e)}")

if __name__ == "__main__":
    main()