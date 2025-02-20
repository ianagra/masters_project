from langchain_core.tools import tool, StructuredTool
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import pandas as pd
import numpy as np
import json
from typing import Dict
import re
import datetime

class NetworkTools:
    def __init__(self):
        """Inicializa a classe NetworkTools carregando o dataset de sobrevivência e os coeficientes da regressão logística"""
        try:
            self.df_surv = pd.read_csv('dataset_survival_rtt_u.csv')
        except Exception as e:
            print(f"Error loading CSV: {e}")
            self.df_surv = pd.DataFrame()
        try:
            self.df_coef = pd.read_csv('coefficients_rtt_u.csv')
            self.df_coef['odds_ratio'] = np.exp(self.df_coef['coefficient'])
        except Exception as e:
            print(f"Error loading CSV: {e}")
            self.df_coef = pd.DataFrame()

    def analyze_clusters(self) -> Dict:
        """
        Realiza análise comparativa dos clusters para identificar perfis de desempenho.
        """
        clusters_analysis = {}
        
        for cluster in [0, 1]:
            cluster_data = self.df_surv[self.df_surv['cluster'] == cluster]
            
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

    # Método para análise de desempenho de todos os clientes
    def analyze_all_clients(self) -> Dict:
        """
        Analisa e compara o desempenho de todos os clientes.
        Retorna estatísticas comparativas e rankings.
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

    # Método para análise de desempenho de todos os servidores
    def analyze_all_servers(self) -> Dict:
        """
        Analisa e compara o desempenho de todos os servidores.
        Retorna estatísticas comparativas e rankings.
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

    # Método para análise de desempenho de um cliente específico
    def analyze_client(self, client_id: str) -> Dict:
        """
        Analisa o desempenho detalhado de um cliente específico, incluindo os pontos de mudança.
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

    # Método para análise de desempenho de um servidor específico
    def analyze_server(self, server_id: str) -> Dict:
        """
        Analisa o desempenho detalhado de um servidor específico.
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

    # Método para análise de um par cliente-servidor específico
    def analyze_pair(self, client_id: str, server_id: str) -> Dict:
        """
        Analisa o desempenho detalhado de um par cliente-servidor específico.
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
            'general_info': {
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
            },
            'cluster_analysis': {}
        }
        
        return analysis

class NetworkAgent:
    def __init__(self, llama_base_url: str = 'http://10.246.47.169:10000', phi_base_url: str = 'http://10.246.47.169:10000'):
        self.tools = NetworkTools()
        self.logger = ConversationLogger()
        
        # Initialize the tool definitions
        self.analyze_clusters_tool = StructuredTool.from_function(
            func=self.tools.analyze_clusters,
            name="analyze_clusters",
            description="Analyzes general patterns and performance differences between clusters"
        )
        
        self.analyze_all_clients_tool = StructuredTool.from_function(
            func=self.tools.analyze_all_clients,
            name="analyze_all_clients",
            description="Compare the performance of all clients with each other. Use to rank the clients by performance"
        )

        self.analyze_all_servers_tool = StructuredTool.from_function(
            func=self.tools.analyze_all_servers,
            name="analyze_all_servers",
            description="Compare the performance of all servers with each other. Use to rank the servers by performance"
        )

        self.analyze_client_tool = StructuredTool.from_function(
            func=self.tools.analyze_client,
            name="analyze_client",
            description="Analyzes the performance of a specific client"
        )
        
        self.analyze_server_tool = StructuredTool.from_function(
            func=self.tools.analyze_server,
            name="analyze_server",
            description="Analyzes the performance of a specific server"
        )
        
        self.analyze_pair_tool = StructuredTool.from_function(
            func=self.tools.analyze_pair,
            name="analyze_pair",
            description="Analyzes the performance of a specific client-server pair"
        )
        
        self.tool_names = {
            'analyze_clusters': self.analyze_clusters_tool,
            'analyze_all_clients': self.analyze_all_clients_tool,
            'analyze_all_servers': self.analyze_all_servers_tool,
            'analyze_client': self.analyze_client_tool,
            'analyze_server': self.analyze_server_tool,
            'analyze_pair': self.analyze_pair_tool
        }
        
        # Initialize models
        self.tool_selector_model = ChatOllama(
            model="llama3.2",
            base_url=llama_base_url,
            temperature=0,
        )
        
        self.analysis_model = ChatOllama(
            model="phi4",
            base_url=phi_base_url,
            temperature=0,
        )
        
        self.tool_selector_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_tool_selection_context()),
            ("human", "{input}\n\nRemember to respond ONLY with a JSON object and nothing else."),
        ])
        
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_analysis_context()),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        self.chat_history = []
        
        # Log the system contexts
        self.logger.log_system_prompt("LLAMA 3.2 - Tool Selection", self._get_tool_selection_context())
        self.logger.log_system_prompt("PHI 4 - Analysis", self._get_analysis_context())

    def _get_tool_selection_context(self) -> str:
        return """You are a computer networks specialist.
Your task is to select the most appropriate tool for analyzing network data based on the user's question.

Available tools:

1. analyze_clusters()
   - Use when analyzing general patterns and differences between performance clusters
   - Provides comparative statistics between clusters
   - Note: This tool takes no parameters, use empty parameters object {{}}

2. analyze_all_clients()
   - Use when comparing the performance of all clients
   - Provides detailed metrics and rankings for all clients
   - Note: This tool takes no parameters, use empty parameters object {{}}

3. analyze_all_servers()
   - Use when comparing the performance of all servers
   - Provides detailed metrics and rankings for all servers
   - Note: This tool takes no parameters, use empty parameters object {{}}

4. analyze_client(client_id: str)
   - Use when analyzing overall performance of a specific client
   - Provides detailed metrics and temporal evolution for that client

5. analyze_server(server_id: str)
   - Use when analyzing overall performance of a specific server
   - Provides detailed metrics and temporal evolution for that server

6. analyze_pair(client_id: str, server_id: str)
   - Use when analyzing overall performance of a specific client-server pair
   - Provides detailed metrics and cluster analysis for that connection

Your response must be ONLY a JSON object with no additional text, containing:
{{
    "tool": "tool_name",
    "parameters": {{}}
}}

For example, for analyzing clusters:
{{
    "tool": "analyze_clusters",
    "parameters": {{}}
}}

Select the most appropriate tool based on the user's question. Do not include any explanation or additional text."""

    def _get_analysis_context(self) -> str:
        return """You are a computer networks specialist analyzing network performance data collected from an ISP network.

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
Always base your analysis on the data provided.

When analyzing temporal evolution:
- Look for trends and patterns in the data
- Identify periods of degraded performance
- Note any improvements or deteriorations over time

When analyzing clients or servers:
- Always consider the number of events, the time between changes and the odds ratios for cluster membership
- Highlight the key differences between clients or servers
- Provide clear, actionable insights and specific recommendations based on the data

When comparing clusters:
- Explain the characteristics of each cluster
- Highlight the key differences between clusters
- Indicate which represents better performance

Respond in clear language suitable for network operators."""

    def _extract_tool_call(self, response: str) -> tuple:
        """
        Extrai a ferramenta e parâmetros da resposta do modelo
        """
        try:
            # Clean the response to extract only the JSON part
            json_str = response.strip()
            # If response contains multiple lines, try to find the JSON block
            if '\n' in json_str:
                # Look for content between { and }
                match = re.search(r'\{.*\}', json_str, re.DOTALL)
                if match:
                    json_str = match.group(0)
            
            tool_call = json.loads(json_str)
            return tool_call["tool"], tool_call["parameters"]
        except Exception as e:
            print(f"Error parsing tool call: {e}")
            print(f"Response content: {response}")  # Added for debugging
            return None, None

    def process_question(self, question: str) -> str:
        """
        Processa uma pergunta do usuário e retorna a análise
        """
        try:
            # 1. Log user question
            self.logger.log_user_prompt(question)
            
            # 2. Select tool using LLAMA 3.2
            tool_selection = self.tool_selector_prompt.invoke({
                "input": question
            }).to_messages()
            
            tool_response = self.tool_selector_model.invoke(tool_selection)
            self.logger.log_llm_response("LLAMA 3.2 - Tool Selection", tool_response.content)
            
            tool_name, parameters = self._extract_tool_call(tool_response.content)
            
            if not tool_name or tool_name not in self.tool_names:
                return f"I couldn't determine how to analyze this question. Please try rephrasing it.\nDebug info - Response: {tool_response.content}"
            
            # 3. Execute and log tool call
            tool = self.tool_names[tool_name]
            if tool_name in ['analyze_clusters', 'analyze_all_clients', 'analyze_all_servers']:
                data = tool.invoke(input="")
            elif tool_name == 'analyze_pair':
                data = tool.invoke(input={
                    'client_id': parameters['client_id'],
                    'server_id': parameters['server_id']
                })
            else:
                data = tool.invoke(input=next(iter(parameters.values())))
            
            self.logger.log_tool_call(tool_name, parameters, data)
            
            if not data:
                return "There was an error analyzing the data. Please try again."
            
            # 4. Prepare and log analysis using PHI 4
            analysis_prompt = f"""Based on this network performance data:

{json.dumps(data, indent=2)}

{question}

Answer in clear language, suitable for network operators.
If you have already analyzed the clusters, keep in mind which cluster represents better performance.
Always compare the metrics with the time between changes, the interval length, and the Odds Ratio related to Cluster 1.
"""
            
            self.logger.log_llm_prompt("PHI 4 - Analysis", analysis_prompt)
            
            response = self.analysis_prompt.invoke({
                "input": analysis_prompt,
                "chat_history": self.chat_history
            }).to_messages()
            
            analysis = self.analysis_model.invoke(response)
            self.logger.log_llm_response("PHI 4 - Analysis", analysis.content)
            
            # 5. Update history
            self.chat_history.extend([
                HumanMessage(content=analysis_prompt),
                AIMessage(content=analysis.content)
            ])
            
            return analysis.content
            
        except Exception as e:
            return f"An error occurred: {str(e)}"
        
class ConversationLogger:
    def __init__(self, filename_prefix="qa_log"):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"chat_logs/{filename_prefix}_llama32_phi4_{timestamp}.txt"
        # Clear previous log file
        with open(self.filename, 'w', encoding='utf-8') as f:
            f.write("New Conversation Log\n")
        self.conversation = []

    def _write_to_file(self, entry_type: str, model: str, content) -> bool:
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
                    for key, value in content.items():
                        f.write(f"\n{key}:\n")
                        try:
                            if isinstance(value, (dict, list)):
                                f.write(f"{json.dumps(value, indent=2)}\n")
                            else:
                                f.write(f"{str(value)}\n")
                        except Exception as e:
                            f.write(f"[Error serializing value: {str(e)}]\n")
                else:
                    f.write(f"{str(content)}\n")
                
                f.flush()
            return True
        except Exception as e:
            print(f"Error writing to log file: {str(e)}")
            return False

    def log_system_prompt(self, model: str, prompt):
        """Log the system prompt"""
        success = self._write_to_file("SYSTEM PROMPT", model, prompt)
        if success:
            self.conversation.append(("SYSTEM PROMPT", model, prompt))

    def log_user_prompt(self, prompt):
        """Log the user prompt"""
        success = self._write_to_file("USER PROMPT", "", prompt)
        if success:
            self.conversation.append(("USER PROMPT", "", prompt))

    def log_tool_call(self, tool_name, parameters, result):
        """Log a tool call and its result"""
        content = {
            "tool": tool_name,
            "parameters": parameters,
            "result": result
        }
        success = self._write_to_file("TOOL CALL", "", content)
        if success:
            self.conversation.append(("TOOL CALL", "", content))

    def log_llm_prompt(self, model: str, prompt):
        """Log the prompt sent to the model"""
        success = self._write_to_file("LLM PROMPT", model, prompt)
        if success:
            self.conversation.append(("LLM PROMPT", model, prompt))

    def log_llm_response(self, model: str, response):
        """Log the model's response"""
        success = self._write_to_file("LLM RESPONSE", model, response)
        if success:
            self.conversation.append(("LLM RESPONSE", model, response))

    def save(self):
        """Verify and rewrite the entire log to ensure completeness"""
        try:
            if not self.conversation:
                print("No conversation data to save")
                return
            
            with open(self.filename, 'w', encoding='utf-8') as f:
                f.write("Complete Conversation Log\n")
                for entry_type, model, content in self.conversation:
                    f.write(f"\n{'='*50}\n")
                    if model:
                        f.write(f"{model} - {entry_type}:\n")
                    else:
                        f.write(f"{entry_type}:\n")
                    f.write(f"{'='*50}\n")
                    
                    if isinstance(content, dict):
                        for key, value in content.items():
                            f.write(f"\n{key}:\n")
                            try:
                                if isinstance(value, (dict, list)):
                                    f.write(f"{json.dumps(value, indent=2)}\n")
                                else:
                                    f.write(f"{str(value)}\n")
                            except Exception as e:
                                f.write(f"[Error serializing value: {str(e)}]\n")
                    else:
                        f.write(f"{str(content)}\n")
            
            print(f"Successfully saved complete log to {self.filename}")
            
        except Exception as e:
            print(f"Error saving complete log: {str(e)}")

def main():
    # Inicializar agente
    agent = NetworkAgent()
    
    print("Network Performance Analysis Agent")
    print("Enter your questions about network performance (or 'quit' to exit)")
    print("\nExample questions:")
    print("- What are the general characteristics of the performance clusters?")
    print("- How is client01 performing?")
    print("- What are the issues with server gru03?")
    print("- Analyze the connection between client01 and gru03")
    print("- Compare the performance clusters")
    
    while True:
        question = input("\nYour question: ")
        if question.lower() == 'quit':
            agent.logger.save()
            break
            
        print("\nAnalyzing...")
        response = agent.process_question(question)
        print("\nAnalysis:")
        print(response)

if __name__ == "__main__":
    main()