from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Optional
import datetime
import os

# Carregar chave de API da OpenAI
try:
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        # Carregar chave do arquivo .env
        try:
            from dotenv import load_dotenv
            load_dotenv()
            openai_api_key = os.environ.get("OPENAI_API_KEY")
        except ImportError:
            print("Warning: python-dotenv not installed, trying to use environment variables directly")
    
    if not openai_api_key:
        print("Warning: OPENAI_API_KEY not found. Evaluation functionality will be disabled.")
        evaluator_available = False
    else:
        evaluator_available = True
except Exception as e:
    print(f"Error setting up OpenAI API key: {e}")
    evaluator_available = False

# Criar diretório para os logs caso não exista
os.makedirs("chat_logs", exist_ok=True)

# Classe que cria o log da conversa
class ConversationLogger:
    def __init__(self, filename_prefix="qa_log"):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"chat_logs/{filename_prefix}_{timestamp}.txt"
        # Clear previous log file
        with open(self.filename, 'w', encoding='utf-8') as f:
            f.write("Complete Conversation Log\n")
        self.conversation = []

    def log_entry(self, entry_type: str, model: str, content: Any) -> bool:
        """Write an entry to the log file"""
        try:
            with open(self.filename, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*50}\n")
                if model:
                    f.write(f"{model} - {entry_type}:\n")
                else:
                    f.write(f"{entry_type}:\n")
                f.write(f"{'='*50}\n")
                
                if isinstance(content, dict):
                    f.write(f"{json.dumps(content, indent=2)}\n")
                else:
                    f.write(f"{str(content)}\n")
                
                f.flush()
            self.conversation.append((entry_type, model, content))
            return True
        except Exception as e:
            print(f"Error writing to log file: {str(e)}")
            return False

    def log_system_prompt(self, model: str, system_prompt):
        """Log a system prompt for a model"""
        return self.log_entry(f"SYSTEM PROMPT", model, system_prompt)
    
    def log_user_prompt(self, prompt):
        """Log the user prompt"""
        return self.log_entry("USER PROMPT", "", prompt)

    def log_tool_call(self, tool_name, parameters, result):
        """Log a tool call and its result"""
        content = {
            "tool": tool_name,
            "parameters": parameters,
            "result": result
        }
        return self.log_entry("TOOL CALL", "", content)

    def log_response(self, model: str, response):
        """Log the model's response"""
        return self.log_entry("LLM RESPONSE", model, response)
        
    def log_analyst_prompt(self, prompt):
        """Log the prompt sent to the analyst model"""
        return self.log_entry("LLM PROMPT", "DeepSeek-R1-14B - Analysis", prompt)

try:
    surv = pd.read_csv('dataset_survival_rtt_u.csv')
    coef = pd.read_csv('coefficients_rtt_u.csv')
    coef['odds_ratio'] = np.exp(coef['coefficient'])
    datasets_loaded = True
except Exception as e:
    print(f"Error loading datasets: {e}")
    surv = pd.DataFrame()
    coef = pd.DataFrame()
    datasets_loaded = False

# Ferramentas de análise dos dados
@tool
def analyze_clusters() -> Dict:
    """
    Performs comparative analysis of clusters to identify performance profiles.
    Use this tool when you need to understand the general characteristics and differences between performance clusters.
    """
    if not datasets_loaded:
        return {"error": "Datasets not loaded"}
    
    clusters_analysis = {}
    
    for cluster in [0, 1]:
        cluster_data = surv[surv['cluster'] == cluster]
        
        metrics = {
            'throughput_download': {
                'mean': float(cluster_data['throughput_download'].mean()),
                'std': float(cluster_data['throughput_download_std'].mean())
            },
            'throughput_upload': {
                'mean': float(cluster_data['throughput_upload'].mean()),
                'std': float(cluster_data['throughput_upload_std'].mean())
            },
            'rtt_download': {
                'mean': float(cluster_data['rtt_download'].mean()),
                'std': float(cluster_data['rtt_download_std'].mean())
            },
            'rtt_upload': {
                'mean': float(cluster_data['rtt_upload'].mean()),
                'std': float(cluster_data['rtt_upload_std'].mean())
            },
            'time_intervals_days': {
                'mean': float(cluster_data['time'].mean()),
                'median': float(cluster_data['time'].median()),
                'std': float(cluster_data['time'].std()),
                'min': float(cluster_data['time'].min()),
                'max': float(cluster_data['time'].max())
            },
            'sample_size': len(cluster_data)
        }
        
        clusters_analysis[f'cluster_{cluster}'] = metrics
    
    return clusters_analysis

@tool
def analyze_all_clients(self) -> Dict:
    """
    Analyzes and compares the performance of all clients.
    Returns comparative statistics and rankings.
    """
    if self.df_surv.empty:
        return {"error": "Dataset não carregado"}

    # Agregar métricas por cliente
    client_stats = self.df_surv.groupby('client').agg({
        'throughput_download': 'mean',
        'throughput_upload': 'mean',
        'rtt_download': 'mean',
        'rtt_upload': 'mean',
        'event': 'sum',
        'cluster': lambda x: (x == 1).mean(),
        'time': ['count', 'mean'],
        'site': 'nunique'
    })

    # Tratando a estrutura multinível
    time_count = client_stats['time']['count']
    time_mean = client_stats['time']['mean']
    
    # Removendo a estrutura multinível das outras colunas
    client_stats = client_stats.droplevel(level=1, axis=1) if isinstance(client_stats.columns, pd.MultiIndex) else client_stats
    
    # Readicionando as colunas de tempo
    client_stats['time_count'] = time_count
    client_stats['time_mean'] = time_mean
    
    client_stats = client_stats.reset_index()
    
    # Calcular percentis para cada métrica
    metrics_config = {
        'throughput_download': {'ascending': True},
        'throughput_upload': {'ascending': True},
        'rtt_download': {'ascending': False},
        'rtt_upload': {'ascending': False},
        'cluster': {'ascending': True}
    }
    
    analysis = {}
    for _, row in client_stats.iterrows():
        client = row['client']
        client_coef = self.df_coef[self.df_coef['feature'] == client]
        
        client_analysis = {
            'throughput_download': {
                'value': float(row['throughput_download']),
                'percentile_rank': float(client_stats['throughput_download'].rank(pct=True)[client_stats['client'] == client].iloc[0] * 100)
            },
            'throughput_upload': {
                'value': float(row['throughput_upload']),
                'percentile_rank': float(client_stats['throughput_upload'].rank(pct=True)[client_stats['client'] == client].iloc[0] * 100)
            },
            'rtt_download': {
                'value': float(row['rtt_download']),
                'percentile_rank': float(client_stats['rtt_download'].rank(pct=True, ascending=False)[client_stats['client'] == client].iloc[0] * 100)
            },
            'rtt_upload': {
                'value': float(row['rtt_upload']),
                'percentile_rank': float(client_stats['rtt_upload'].rank(pct=True, ascending=False)[client_stats['client'] == client].iloc[0] * 100)
            },
            'cluster_1_ratio': {
                'value': float(row['cluster']),
                'percentile_rank': float(client_stats['cluster'].rank(pct=True)[client_stats['client'] == client].iloc[0] * 100)
            },
            'intervals': int(row['time_count']),
            'avg_interval_length_days': int(row['time_mean']),
            'events': int(row['event']),
            'cluster_1_odds_ratio': float(client_coef['odds_ratio'].iloc[0]) if not client_coef.empty else 0.0,
            'unique_servers': int(row['site'])
        }
        analysis[client] = client_analysis

    # Estatísticas gerais para comparação
    overall_stats = {
        'avg_throughput_download': float(client_stats['throughput_download'].mean()),
        'median_throughput_download': float(client_stats['throughput_download'].median()),
        'std_throughput_download': float(client_stats['throughput_download'].std()),
        'avg_throughput_upload': float(client_stats['throughput_upload'].mean()),
        'median_throughput_upload': float(client_stats['throughput_upload'].median()),
        'std_throughput_upload': float(client_stats['throughput_upload'].std()),
        'avg_rtt_download': float(client_stats['rtt_download'].mean()),
        'median_rtt_download': float(client_stats['rtt_download'].median()),
        'std_rtt_download': float(client_stats['rtt_download'].std()),
        'avg_rtt_upload': float(client_stats['rtt_upload'].mean()),
        'median_rtt_upload': float(client_stats['rtt_upload'].median()),
        'std_rtt_upload': float(client_stats['rtt_upload'].std())
    }

    return {
        'clients': analysis,
        'overall_stats': overall_stats,
        'total_clients': len(client_stats)
    }

@tool
def analyze_all_servers(self) -> Dict:
    """
    Analyzes and compares the performance of all servers.
    Returns comparative statistics and rankings.
    """
    if self.df_surv.empty:
        return {"error": "Dataset não carregado"}

    # Agregar métricas por servidor
    server_stats = self.df_surv.groupby('site').agg({
        'throughput_download': 'mean',
        'throughput_upload': 'mean',
        'rtt_download': 'mean',
        'rtt_upload': 'mean',
        'event': 'sum',
        'cluster': lambda x: (x == 1).mean(),
        'time': ['count', 'mean'],
        'client': 'nunique'
    })

    # Tratando a estrutura multinível
    time_count = server_stats['time']['count']
    time_mean = server_stats['time']['mean']
    
    # Removendo a estrutura multinível das outras colunas
    server_stats = server_stats.droplevel(level=1, axis=1) if isinstance(server_stats.columns, pd.MultiIndex) else server_stats
    
    # Readicionando as colunas de tempo
    server_stats['time_count'] = time_count
    server_stats['time_mean'] = time_mean
    
    server_stats = server_stats.reset_index()
    
    analysis = {}
    for _, row in server_stats.iterrows():
        server = row['site']
        server_coef = self.df_coef[self.df_coef['feature'] == server]
        
        server_analysis = {
            'throughput_download': {
                'value': float(row['throughput_download']),
                'percentile_rank': float(server_stats['throughput_download'].rank(pct=True)[server_stats['site'] == server].iloc[0] * 100)
            },
            'throughput_upload': {
                'value': float(row['throughput_upload']),
                'percentile_rank': float(server_stats['throughput_upload'].rank(pct=True)[server_stats['site'] == server].iloc[0] * 100)
            },
            'rtt_download': {
                'value': float(row['rtt_download']),
                'percentile_rank': float(server_stats['rtt_download'].rank(pct=True, ascending=False)[server_stats['site'] == server].iloc[0] * 100)
            },
            'rtt_upload': {
                'value': float(row['rtt_upload']),
                'percentile_rank': float(server_stats['rtt_upload'].rank(pct=True, ascending=False)[server_stats['site'] == server].iloc[0] * 100)
            },
            'cluster_1_ratio': {
                'value': float(row['cluster']),
                'percentile_rank': float(server_stats['cluster'].rank(pct=True)[server_stats['site'] == server].iloc[0] * 100)
            },
            'intervals': int(row['time_count']),
            'avg_interval_length_days': int(row['time_mean']),
            'events': int(row['event']),
            'cluster_1_odds_ratio': float(server_coef['odds_ratio'].iloc[0]) if not server_coef.empty else 0.0,
            'unique_clients': int(row['client'])
        }
        analysis[server] = server_analysis

    # Estatísticas gerais para comparação
    overall_stats = {
        'avg_throughput_download': float(server_stats['throughput_download'].mean()),
        'median_throughput_download': float(server_stats['throughput_download'].median()),
        'std_throughput_download': float(server_stats['throughput_download'].std()),
        'avg_throughput_upload': float(server_stats['throughput_upload'].mean()),
        'median_throughput_upload': float(server_stats['throughput_upload'].median()),
        'std_throughput_upload': float(server_stats['throughput_upload'].std()),
        'avg_rtt_download': float(server_stats['rtt_download'].mean()),
        'median_rtt_download': float(server_stats['rtt_download'].median()),
        'std_rtt_download': float(server_stats['rtt_download'].std()),
        'avg_rtt_upload': float(server_stats['rtt_upload'].mean()),
        'median_rtt_upload': float(server_stats['rtt_upload'].median()),
        'std_rtt_upload': float(server_stats['rtt_upload'].std())
    }

    return {
        'servers': analysis,
        'overall_stats': overall_stats,
        'total_servers': len(server_stats)
    }

@tool
def analyze_client(self, client_id: str) -> Dict:
    """
    Analyzes the detailed performance of a specific client.
    Returns comparative statistics and change points.
    """
    client_data = self.df_surv[self.df_surv['client'] == client_id]
    client_coef = self.df_coef[self.df_coef['feature'] == client_id]
        
    if len(client_data) == 0:
        return {"error": f"No data found for client {client_id}"}
            
    # Análise geral e métricas
    performance = {
        'general_stats': {
            'total_intervals': len(client_data),
            'cluster_distribution': {
                'cluster_0': float((client_data['cluster'] == 0).mean()),
                'cluster_1': float((client_data['cluster'] == 1).mean())
            },
            'avg_interval_length_days': float(client_data['time'].mean()),
            'total_events': int(client_data['event'].sum()),
            'odds_ratio_for_cluster_1': float(client_coef['odds_ratio'].iloc[0]) if not client_coef.empty else None
        },
        'metrics': {
            'throughput_download': {
                'mean': float(client_data['throughput_download'].mean()),
                'std': float(client_data['throughput_download_std'].mean())
            },
            'throughput_upload': {
                'mean': float(client_data['throughput_upload'].mean()),
                'std': float(client_data['throughput_upload_std'].mean())
            },
            'rtt_download': {
                'mean': float(client_data['rtt_download'].mean()),
                'std': float(client_data['rtt_download_std'].mean())
            },
            'rtt_upload': {
                'mean': float(client_data['rtt_upload'].mean()),
                'std': float(client_data['rtt_upload_std'].mean())
            }
        }
    }
    
    # Processamento dos pontos de mudança
    # Filtrar apenas os dados do cliente e ordenar
    df_cliente = client_data.sort_values(by=['site', 'timestamp_start']).copy()
    
    # Identificar os pontos de mudança (event = 1)
    pontos_mudanca = df_cliente[df_cliente['event'] == 1].copy()
    change_points = []
    
    # Calcular as diferenças de métricas para cada ponto de mudança
    for idx, mudanca in pontos_mudanca.iterrows():
        # Encontrar o próximo intervalo para o mesmo par cliente-servidor
        proximo_intervalo = df_cliente[
            (df_cliente['site'] == mudanca['site']) & 
            (pd.to_datetime(df_cliente['timestamp_start']) > pd.to_datetime(mudanca['timestamp_end']))
        ].sort_values('timestamp_start').head(1)
        
        # Se não houver próximo intervalo, pular o ponto de mudança
        if proximo_intervalo.empty:
            continue
            
        proximo = proximo_intervalo.iloc[0]
        
        # Calcular as diferenças de métricas
        resultado = {
            'timestamp': str(mudanca['timestamp_end']),
            'server': mudanca['site'],
            'interval_length': float(mudanca['time']),
            'cluster_before_changepoint': int(mudanca['cluster']),
            'cluster_after_changepoint': int(proximo['cluster']),
            'throughput_download_difference': float(proximo['throughput_download'] - mudanca['throughput_download']),
            'throughput_upload_difference': float(proximo['throughput_upload'] - mudanca['throughput_upload']),
            'rtt_download_difference': float(proximo['rtt_download'] - mudanca['rtt_download']),
            'rtt_upload_difference': float(proximo['rtt_upload'] - mudanca['rtt_upload'])
        }
        
        change_points.append(resultado)
    
    # Adicionar os pontos de mudança ao resultado
    performance['change_points'] = change_points
    
    return performance

@tool
def analyze_server(self, server_id: str) -> Dict:
    """
    Analyzes the detailed performance of a specific server.
    Returns comparative statistics and change points.
    """
    server_data = self.df_surv[self.df_surv['site'] == server_id]
    server_coef = self.df_coef[self.df_coef['feature'] == server_id]
    
    if len(server_data) == 0:
        return {"error": f"No data found for server {server_id}"}
        
    performance = {
        'general_stats': {
            'total_intervals': len(server_data),
            'cluster_distribution': {
                'cluster_0': float((server_data['cluster'] == 0).mean()),
                'cluster_1': float((server_data['cluster'] == 1).mean())
            },
            'avg_interval_length_days': float(server_data['time'].mean()),
            'total_events': int(server_data['event'].sum()),
            'cluster_1_odds_ratio': float(server_coef['odds_ratio'].iloc[0])
        },
        'metrics': {
            'throughput_download': {
                'mean': float(server_data['throughput_download'].mean()),
                'std': float(server_data['throughput_download_std'].mean())
            },
            'rtt_download': {
                'mean': float(server_data['rtt_download'].mean()),
                'std': float(server_data['rtt_download_std'].mean())
            }
        }
    }

    # Processamento dos pontos de mudança
    # Filtrar apenas os dados do servidor e ordenar
    df_servidor = server_data.sort_values(by=['client', 'timestamp_start']).copy()

    # Identificar os pontos de mudança
    pontos_mudanca = df_servidor[df_servidor['event'] == 1].copy()
    change_points = []

    # Calcular as diferenças de métricas para cada ponto de mudança
    for idx, mudanca in pontos_mudanca.iterrows():
        # Encontrar o próximo intervalo para o mesmo par cliente-servidor
        proximo_intervalo = df_servidor[
            (df_servidor['client'] == mudanca['client']) & 
            (pd.to_datetime(df_servidor['timestamp_start']) > pd.to_datetime(mudanca['timestamp_end']))
        ].sort_values('timestamp_start').head(1)

        # Se não houver próximo intervalo, pular o ponto de mudança
        if proximo_intervalo.empty:
            continue

        proximo = proximo_intervalo.iloc[0]

        # Calcular as diferenças de métricas
        resultado = {
            'timestamp': str(mudanca['timestamp_end']),
            'client': mudanca['client'],
            'interval_length': float(mudanca['time']),
            'cluster_before_changepoint': int(mudanca['cluster']),
            'cluster_after_changepoint': int(proximo['cluster']),
            'throughput_download_difference': float(proximo['throughput_download'] - mudanca['throughput_download']),
            'rtt_download_difference': float(proximo['rtt_download'] - mudanca['rtt_download'])
        }

        change_points.append(resultado)

    # Adicionar os pontos de mudança ao resultado
    performance['change_points'] = change_points
    
    return performance

@tool
def analyze_pair(self, client_id: str, server_id: str) -> Dict:
    """
    Analyzes the detailed performance of a specific client-server pair.
    Returns statistics and change points.
    """
    connection_data = self.df_surv[
        (self.df_surv['client'] == client_id) & 
        (self.df_surv['site'] == server_id)
    ]
    client_coef = self.df_coef[self.df_coef['feature'] == client_id]
    server_coef = self.df_coef[self.df_coef['feature'] == server_id]

    if len(connection_data) == 0:
        return {"error": f"No connection data found between client {client_id} and server {server_id}"}
    
    connection_data['timestamp'] = pd.to_datetime(connection_data['timestamp_start'])
    
    analysis = {
        'pair_info': {
            'total_intervals': len(connection_data),
            'cluster_distribution': {
                'cluster_0': float((connection_data['cluster'] == 0).mean()),
                'cluster_1': float((connection_data['cluster'] == 1).mean())
            },
            'avg_interval_length_days': float(connection_data['time'].mean()),
            'total_events': int(connection_data['event'].sum()),
            # Aqui está a correção - acessando o valor da coluna 'odds_ratio' de forma segura
            'client_odds_ratio_cluster_1': float(client_coef['odds_ratio'].iloc[0]) if not client_coef.empty else 0.0,
            'server_odds_ratio_cluster_1': float(server_coef['odds_ratio'].iloc[0]) if not server_coef.empty else 0.0
        },
        'timespan': {
            'start': connection_data['timestamp'].min().isoformat(),
            'end': connection_data['timestamp'].max().isoformat(),
            'total_days': (connection_data['timestamp'].max() - connection_data['timestamp'].min()).days
        },
        'overall_metrics': {
            'throughput_download': {
                'mean': float(connection_data['throughput_download'].mean()),
                'std': float(connection_data['throughput_download_std'].mean())
            },
            'throughput_upload': {
                'mean': float(connection_data['throughput_upload'].mean()),
                'std': float(connection_data['throughput_upload_std'].mean())
            },
            'rtt_download': {
                'mean': float(connection_data['rtt_download'].mean()),
                'std': float(connection_data['rtt_download_std'].mean())
            },
            'rtt_upload': {
                'mean': float(connection_data['rtt_upload'].mean()),
                'std': float(connection_data['rtt_upload_std'].mean())
            }
        }
    }
    
    # Processamento dos pontos de mudança
    # Filtrar apenas os dados do servidor e ordenar
    df_pair = connection_data.sort_values(by=['timestamp_start']).copy()

    # Identificar os pontos de mudança
    pontos_mudanca = df_pair[df_pair['event'] == 1].copy()
    change_points = []

    # Calcular as diferenças de métricas para cada ponto de mudança
    for idx, mudanca in pontos_mudanca.iterrows():
        # Encontrar o próximo intervalo para o mesmo par cliente-servidor
        proximo_intervalo = df_pair[
            (df_pair['client'] == mudanca['client']) & (df_pair['site'] == mudanca['site']) & (pd.to_datetime(df_pair['timestamp_start']) > pd.to_datetime(mudanca['timestamp_end']))
        ].sort_values('timestamp_start').head(1)

        # Se não houver próximo intervalo, pular o ponto de mudança
        if proximo_intervalo.empty:
            continue

        proximo = proximo_intervalo.iloc[0]

        # Calcular as diferenças de métricas
        resultado = {
            'timestamp': str(mudanca['timestamp_end']),
            'client': mudanca['client'],
            'interval_length': float(mudanca['time']),
            'cluster_before_changepoint': int(mudanca['cluster']),
            'cluster_after_changepoint': int(proximo['cluster']),
            'throughput_download_difference': float(proximo['throughput_download'] - mudanca['throughput_download']),
            'rtt_download_difference': float(proximo['rtt_download'] - mudanca['rtt_download'])
        }

        change_points.append(resultado)

    # Adicionar os pontos de mudança ao resultado
    analysis['change_points'] = change_points

    return analysis

# Dicionário para mapear os nomes das ferramentas para suas funções
tool_map = {
    "analyze_clusters": analyze_clusters,
    "analyze_all_clients": analyze_all_clients,
    "analyze_all_servers": analyze_all_servers,
    "analyze_client": analyze_client,
    "analyze_server": analyze_server,
    "analyze_pair": analyze_pair
}

# Definir todas as ferramentas disponíveis
available_tools = [
    analyze_clusters,
    analyze_all_clients,
    analyze_all_servers,
    analyze_client,
    analyze_server,
    analyze_pair
]

def _extract_analysis_content(response: str) -> str:
    """
    Extract only the analysis part after the </think> tag.
    If the tag is not found, return the complete response.
    """
    try:
        if '</think>' in response:
            analysis = response.split('</think>')[-1].strip()
            return analysis
        return response.strip()
    except Exception as e:
        print(f"Error extracting analysis content: {e}")
        return response

def process_question(query: str, logger: ConversationLogger) -> str:
    """
    Process a user question through the network analysis pipeline.
    
    Args:
        query: The user's question
        logger: Logger instance to record the conversation
    
    Returns:
        The final analysis response
    """
    # Salvar o prompt do usuário no log
    logger.log_user_prompt(query)
    
    # Instanciar modelos
    tool_selector_model = ChatOllama(
        model='llama3.2',
        base_url='http://10.246.47.169:10000',
        temperature=0
    )
    
    analyst_model = ChatOllama(
        model='deepseek-r1:14b',
        base_url='http://10.246.47.169:10000',
        temperature=0,
        num_ctx=8192
    )

    if evaluator_available:
        evaluator_model = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=openai_api_key
        )
    
    # Bind tools to selector model for tool selection
    tool_selector_model_with_tools = tool_selector_model.bind_tools(available_tools)
    
    # Create system messages
    tool_selector_model_system_content = """You are a computer networks specialist.
Your task is to select the most appropriate tool for analyzing network data based on the user's question.

Available tools:

1. analyze_clusters()
   - Use when analyzing general patterns and differences between performance clusters
   - Provides comparative statistics between clusters
                                 
2. analyze_all_clients()
   - Use when comparing the performance of all clients
   - Provides detailed metrics and rankings for all clients
   - Note: This tool takes no parameters, use empty parameters object {}

3. analyze_all_servers()
   - Use when comparing the performance of all servers
   - Provides detailed metrics and rankings for all servers
   - Note: This tool takes no parameters, use empty parameters object {}

4. analyze_client(client_id: str)
   - Use when analyzing overall performance of a specific client
   - Provides detailed metrics and temporal evolution for that client

5. analyze_server(server_id: str)
   - Use when analyzing overall performance of a specific server
   - Provides detailed metrics and temporal evolution for that server

6. analyze_pair(client_id: str, server_id: str)
   - Use when analyzing overall performance of a specific client-server pair
   - Provides detailed metrics and cluster analysis for that connection

Select the most appropriate tool based on the user's question."""
    
    tool_selector_model_system = SystemMessage(content=tool_selector_model_system_content)
    
    # Log the system prompt for the tool selector
    logger.log_system_prompt("LLAMA 3.2 - Tool Selection", tool_selector_model_system_content)
    
    analyst_model_system_content = """You are a computer networks specialist analyzing network performance data collected from an ISP network.

The data was processed as follows:

1. Change points were detected in RTT time series for each client-server pair.
2. Intervals between changes were analyzed using survival analysis.
3. Intervals were clustered into 2 groups based on:
- Interval duration, in days;
- Associated metrics (throughput, RTT);
- Client and server IDs; and
- Event occurrence (1) or censored data (0).
4. Logistic regression was used to determine feature importance for cluster membership.

Keep responses direct and actionable. Focus on identifying specific elements needing intervention.
Always base your analysis on the data provided.

When comparing clusters:
- Explain the characteristics of each cluster
- Highlight the key differences between clusters
- Indicate which represents better performance

Respond in clear language suitable for network operators."""
    
    analyst_model_system = SystemMessage(content=analyst_model_system_content)
    
    # Log the system prompt for the analyst model
    logger.log_system_prompt("DeepSeek-R1-14B - Analysis", analyst_model_system_content)
    
    evaluator_model_system_content = """You are an evaluator assessing the quality and accuracy of network performance analysis responses.
Your task is to evaluate responses based on:

1. Technical Accuracy
 - Are the interpretations of metrics correct?
 - Are the conclusions supported by the data?
 - Are there any technical errors or misunderstandings?

2. Completeness
 - Does the response address all aspects of the question?
 - Are important metrics or patterns discussed?
 - Is sufficient context provided?

3. Actionability
 - Are the insights practical and useful?
 - Are recommendations specific and implementable?
 - Is the importance of findings clearly explained?

4. Clarity
 - Is the response clear and well-structured?
 - Is technical language used appropriately?
 - Would network operators understand the response?

Provide a concise evaluation highlighting strengths and any areas for improvement.
Focus on substantial issues rather than minor details.
If you identify errors, explain why they are incorrect and what the correct interpretation should be."""
    
    evaluator_model_system = SystemMessage(content=evaluator_model_system_content)
    
    # Log the system prompt for the evaluator model
    if evaluator_available:
        logger.log_system_prompt("GPT-4o - Evaluation", evaluator_model_system_content)
    
    # Criar a mensagem do usuário
    user_message = HumanMessage(content=query)
    
    # Seleção de ferramentas
    tool_selector_model_messages = [tool_selector_model_system, user_message]
    try:
        # Obter chamadas de função com o modelo seletor
        tool_calls_response = tool_selector_model_with_tools.invoke(tool_selector_model_messages)
        logger.log_response("LLAMA 3.2 - Tool Selection", tool_calls_response.content)
        
        # Extrair a chamada de função
        tool_calls = tool_calls_response.tool_calls
        if not tool_calls:
            return "No appropriate tool was selected to answer your question."
        
        # Executar as funções e obter os resultados
        tool_results = []
        for call in tool_calls:
            tool_name = call.get('name')
            tool_args = call.get('args', {})
            
            # Registrar a chamada de função
            logger.log_tool_call(tool_name, tool_args, "Executing...")
            
            # Executar a função de forma segura
            if tool_name in tool_map:
                try:
                    result = tool_map[tool_name].invoke(tool_args)
                    tool_results.append(AIMessage(content=json.dumps(result, indent=2)))
                    logger.log_tool_call(tool_name, tool_args, result)
                except Exception as e:
                    error_msg = f"Error executing tool {tool_name}: {str(e)}"
                    tool_results.append(AIMessage(content=error_msg))
                    logger.log_tool_call(tool_name, tool_args, {"error": error_msg})
            else:
                error_msg = f"Unknown tool: {tool_name}"
                tool_results.append(AIMessage(content=error_msg))
                logger.log_tool_call(tool_name, tool_args, {"error": error_msg})

        # Criar prompt para o modelo analista com os resultados das ferramentas
        analyst_model_prompt = f"""Based on this network performance data:

{tool_results[0].content}

{query}

Answer in clear language, suitable for network operators.
If you have already analyzed the clusters, keep in mind which cluster represents better performance.
Always compare the metrics with the time between changes, the interval length, and the Odds Ratio related to Cluster 1.
"""
        
        # Log the prompt sent to the analyst model
        logger.log_analyst_prompt(analyst_model_prompt)
        
        # Análise com o modelo analista
        analyst_model_messages = [
            analyst_model_system,
            user_message,
            *tool_results
        ]
        
        # Obter a análise final
        analyst_model_response = analyst_model.invoke(analyst_model_messages)
        logger.log_response("DeepSeek-R1-14B - Analysis", analyst_model_response.content)
        
        # Extrair apenas a parte da resposta após </think>
        clean_analysis = _extract_analysis_content(analyst_model_response.content)
        
        # Avaliação com o modelo avaliador
        if evaluator_available:
            # Prepare the tool results for evaluation
            tool_data_summary = ""
            for i, tool_result in enumerate(tool_results):
                if hasattr(tool_result, 'content') and tool_result.content:
                    # Truncate long content for readability
                    content_preview = tool_result.content[:1000] + "... [truncated]" if len(tool_result.content) > 1000 else tool_result.content
                    tool_data_summary += f"Tool result {i+1}:\n{content_preview}\n\n"
    
            # Create evaluation messages
            evaluator_prompt = f"""Prompt sent to analyst model:
{analyst_model_prompt}

Data provided to analyst model:
{tool_data_summary}

Analyst model's response:
{clean_analysis}

Please evaluate this response."""

            evaluator_model_messages = [
                evaluator_model_system,
                HumanMessage(content=evaluator_prompt)
            ]
            
            # Obter avaliação da análise
            evaluator_model_response = evaluator_model.invoke(evaluator_model_messages)
            logger.log_response("GPT-4o - Evaluation", evaluator_model_response.content)
            
            return f"""Evaluation:
{evaluator_model_response.content}"""
        else:
            # Return just the analysis if evaluator is not available
            return clean_analysis
        
    except Exception as e:
        error_message = f"An error occurred during processing: {str(e)}"
        logger.log_entry("ERROR", "", error_message)
        return error_message

def main():
    """Main function to run the interactive CLI."""
    logger = ConversationLogger()
    
    if not datasets_loaded:
        print("Warning: Datasets could not be loaded. Tool functionality will be limited.")
    
    print("Network Performance Analysis Agent")
    print("Enter your questions about network performance (or 'quit' to exit)")
    print("\nExample questions:")
    print("- What are the general characteristics of the performance clusters?")
    print("- Compare the performance clusters")
    
    while True:
        query = input("\nYour question: ")
        if query.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        print("\nAnalyzing...")
        response = process_question(query, logger)
        print("\nAnalysis:")
        print(response)

if __name__ == "__main__":
    main()