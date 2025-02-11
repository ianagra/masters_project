from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
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
    
    # Create dataframe with features for each cluster
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

    # Rename non-std feature columns to include _mean
    clusters_data.columns = [col + '_mean' if not (col.endswith('_std') or col == 'event' or col == 'cluster') else col for col in clusters_data.columns]
        
    return clusters_data, coefficients_data

def get_system_context():
    """
    Returns the system context for the chat
    """
    return """
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

def get_messages_chain(system_message, messages):
    """
    Creates a list of messages including system message and chat history
    """
    all_messages = [SystemMessage(content=system_message)]
    all_messages.extend(messages)
    return all_messages

def create_chat_chain(model):
    """
    Creates the chat chain with context management
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", get_system_context()),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    chain = prompt | model | StrOutputParser()
    
    return chain

def save_conversation(chat_history, filename=None, model_name=None):
    """
    Saves the conversation history to a file
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{model_name}_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("Network Performance Analysis Conversation\n")
        f.write("=" * 50 + "\n\n")
        
        for message in chat_history:
            if isinstance(message, HumanMessage):
                f.write("Human:\n")
            elif isinstance(message, AIMessage):
                f.write("\nAssistant:\n")
            elif isinstance(message, SystemMessage):
                f.write("System:\n")
                
            f.write(message.content + "\n\n")
            f.write("-" * 50 + "\n\n")
    
    return filename

def format_cluster_stats(survival_data):
    """
    Format cluster statistics in a more readable way
    """
    # Round numerical values to 2 decimal places
    formatted_data = survival_data.round(2)
    
    # Convert to dictionary with proper formatting
    return json.dumps(formatted_data.to_dict(orient="records"), 
                     indent=2, 
                     default=numpy_to_python)

def format_coefficients(coefficients_data):
    """
    Format coefficients data in a more readable way
    """
    # Round numerical values to 4 decimal places for coefficients
    formatted_data = coefficients_data.round(4)
    
    # Sort by absolute coefficient value to show most important features first
    formatted_data = formatted_data.reindex(
        formatted_data.coefficient.abs().sort_values(ascending=False).index
    )
    
    # Convert to dictionary with proper formatting
    return json.dumps(formatted_data.to_dict(orient="records"), 
                     indent=2, 
                     default=numpy_to_python)

def main():
    # Load data
    survival_data = pd.read_csv("dataset_survival_thr_d.csv")
    coefficients_data = pd.read_csv("coefficients_thr_d.csv")
    
    # Process data
    clusters_data, coefficients_data = preprocess_data(survival_data, coefficients_data)
    
    # Initialize model
    model = ChatOllama(
        model='qwq',
        base_url='http://10.246.47.169:10000',
        temperature=0,
    )
    
    # Create chat chain
    chain = create_chat_chain(model)
    
    # Initialize chat history
    chat_history = []
    
    try:
        # First prompt - Cluster Analysis
        first_prompt = """
Based on the following cluster statistics (mean values for each metric):

{cluster_stats}

Please provide a comparative analysis of the clusters focusing on:

1. Performance metrics:
   - Download throughput (mean and standard deviation)
   - Upload throughput (mean and standard deviation)
   - Download RTT (mean and standard deviation)
   - Upload RTT (mean and standard deviation)
2. Stability (time_mean represents average duration in days between changes)
3. Event frequency (event represents the number of performance changes)

In your conclusion, specify which cluster represents better network performance and explain why.
        """
        
        formatted_first_prompt = first_prompt.format(
            cluster_stats=format_cluster_stats(clusters_data)
        )
        
        print("\nSending first prompt for cluster analysis...")
        
        # Get first response with empty chat history
        response = chain.invoke({
            "input": formatted_first_prompt,
            "chat_history": chat_history
        })
        print("\nAI Response to Cluster Analysis:")
        print(response)
        
        # Add the prompt and response to chat history
        chat_history.extend([
            HumanMessage(content=formatted_first_prompt),
            AIMessage(content=response)
        ])
        
        # Second prompt - Performance Analysis
        second_prompt = """
Based on the logistic regression analysis below, which shows coefficients and odds ratios for predicting membership in Cluster 1:

{feature_categories}

Using this information and your previous cluster analysis, please provide:

1. Client identification:
   - List any client IDs associated with poor performance
   - Explain how the coefficients indicate their impact

2. Server identification:
   - List any server IDs associated with poor performance
   - Explain how the coefficients indicate their impact

3. Critical metrics:
   - Identify the top 3 metrics most strongly indicating poor performance
   - Explain their impact using the odds ratios

4. Recommendations:
   - Provide specific, actionable steps for improving network service
   - Prioritize recommendations based on the coefficient magnitudes

Note: A positive coefficient indicates higher likelihood of being in Cluster 1.
        """
        
        formatted_second_prompt = second_prompt.format(
            feature_categories=format_coefficients(coefficients_data)
        )
        
        print("\nSending second prompt for performance analysis...")
        
        # Get second response with updated chat history
        response = chain.invoke({
            "input": formatted_second_prompt,
            "chat_history": chat_history
        })
        print("\nAI Response to Performance Analysis:")
        print(response)
        
        # Add the prompt and response to chat history
        chat_history.extend([
            HumanMessage(content=formatted_second_prompt),
            AIMessage(content=response)
        ])
        
        # Save conversation with system message
        full_history = [SystemMessage(content=get_system_context())] + chat_history
        saved_file = save_conversation(full_history, model_name='qwq')
        print(f"\nConversation saved to: {saved_file}")
                
    except Exception as e:
        print(f"Error in analysis: {str(e)}")

if __name__ == "__main__":
    main()