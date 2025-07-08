from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
import pandas as pd
import numpy as np
import json
import datetime
import os

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

class NetworkDataHandler:
    """
    Classe para carregar e preparar os dados da rede para as diferentes abordagens de análise.
    """
    def __init__(self, approach: str):
        """
        Inicializa o manipulador de dados para uma abordagem específica.

        Args:
            approach (str): A abordagem de análise, 'throughput_download' ou 'rtt_upload'.
        """
        self.approach = approach
        self.df_intervals = None
        self.df_coef = None
        self.df_cluster_stats = None

        if self.approach == 'throughput_download':
            interval_file = 'dataset_survival_throughput_download.csv'
            coef_file = 'coefficients_odds_ratio_throughput_download.csv'
            stats_file = 'clusters_stats_throughput_download.csv'
        elif self.approach == 'rtt_upload':
            # Assumindo um padrão de nomenclatura simétrico para a abordagem de RTT
            interval_file = 'dataset_survival_rtt_upload.csv'
            coef_file = 'coefficients_odds_ratio_rtt_upload.csv'
            stats_file = 'clusters_stats_rtt_upload.csv'
        else:
            raise ValueError("A abordagem deve ser 'throughput_download' ou 'rtt_upload'")

        print(f"\nINFO: Carregando dados para a abordagem '{self.approach}':")
        print(f" - {interval_file}\n - {coef_file}\n - {stats_file}")

        try:
            self.df_intervals = pd.read_csv(interval_file)
            self.df_coef = pd.read_csv(coef_file)
            self.df_cluster_stats = pd.read_csv(stats_file)
        except FileNotFoundError as e:
            print(f"ERRO: Arquivo não encontrado para a abordagem '{self.approach}'. Certifique-se de que os arquivos existam. Detalhes: {e}")
            # Inicializa dataframes vazios para evitar falhas posteriores
            self.df_intervals = pd.DataFrame()
            self.df_coef = pd.DataFrame()
            self.df_cluster_stats = pd.DataFrame()
        except Exception as e:
            print(f"ERRO: Ocorreu um erro ao carregar os dados para a abordagem '{self.approach}': {e}")
            self.df_intervals = pd.DataFrame()
            self.df_coef = pd.DataFrame()
            self.df_cluster_stats = pd.DataFrame()

    def get_cluster_descriptive_stats(self) -> Dict:
        """
        Carrega e formata as estatísticas descritivas dos clusters a partir do arquivo CSV.
        """
        if self.df_cluster_stats.empty:
            return {"erro": "O dataset de estatísticas dos clusters não foi carregado."}
        
        # Converte o dataframe de estatísticas para o formato de dicionário esperado
        stats_dict = {}
        for _, row in self.df_cluster_stats.iterrows():
            cluster_id = f"cluster_{int(row['cluster'])}"
            stats_dict[cluster_id] = {
                'total_intervals': int(row['total_intervals']),
                'event_frequency': float(row['event_frequency']),
                'interval_duration_days': {
                    'mean': float(row['duration_mean']),
                    'median': float(row['duration_median']),
                    'std': float(row['duration_std'])
                },
                'throughput_download_mbps': {
                    'mean': float(row['throughput_download_mean']),
                    'median': float(row['throughput_download_median']),
                    'std': float(row['throughput_download_std'])
                },
                'throughput_upload_mbps': {
                    'mean': float(row['throughput_upload_mean']),
                    'median': float(row['throughput_upload_median']),
                    'std': float(row['throughput_upload_std'])
                },
                'rtt_download_ms': {
                    'mean': float(row['rtt_download_mean']),
                    'median': float(row['rtt_download_median']),
                    'std': float(row['rtt_download_std'])
                },
                'rtt_upload_ms': {
                    'mean': float(row['rtt_upload_mean']),
                    'median': float(row['rtt_upload_median']),
                    'std': float(row['rtt_upload_std'])
                }
            }
        return stats_dict


    def get_coefficients(self) -> Dict:
        """Retorna os coeficientes da regressão logística."""
        if self.df_coef.empty:
            return {"erro": "O dataset de coeficientes não foi carregado."}
        return self.df_coef.to_dict(orient='records')

    def get_intervals_data(self) -> Dict:
        """Retorna o conjunto de dados completo de intervalos."""
        if self.df_intervals.empty:
            return {"erro": "O dataset de intervalos não foi carregado."}
        # Retornando uma amostra para brevidade nos logs, mas o modelo receberia os dados completos.
        return self.df_intervals.head(20).to_dict(orient='records')


class ConversationLogger:
    """Registra o fluxo da conversa para análise e depuração."""
    def __init__(self, filename_prefix="analysis_log"):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"chat_logs/{filename_prefix}_{timestamp}.txt"
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        with open(self.filename, 'w', encoding='utf-8') as f:
            f.write(f"Novo Log de Análise - {timestamp}\n")

    def log_step(self, approach: str, step_name: str, prompt: str, response: str):
        """Registra um passo no processo de análise."""
        try:
            with open(self.filename, 'a', encoding='utf-8') as f:
                f.write("\n" + "="*80 + "\n")
                f.write(f"ABORDAGEM: {approach} | ETAPA: {step_name}\n")
                f.write("="*80 + "\n\n")
                f.write("--- PROMPT ENVIADO AO LLM ---\n")
                f.write(prompt + "\n\n")
                f.write("--- RESPOSTA DO LLM ---\n")
                f.write(response + "\n\n")
                f.flush()
        except Exception as e:
            print(f"Erro ao escrever no arquivo de log: {e}")

    def log_final_comparison(self, prompt: str, response: str):
        """Registra a análise comparativa final."""
        self.log_step("Comparativa", "Comparação Final", prompt, response)


class NetworkAnalysisAgent:
    """
    Um agente que realiza uma análise de múltiplos passos sobre dados de QoS de rede usando um fluxo de LLM conversacional.
    """
    def __init__(self, analyst_base_url: str = 'http://10.246.47.169:10000'):
        """
        Inicializa o agente com um único LLM para todos os passos analíticos.
        """
        self.logger = ConversationLogger()
        
        # Inicializa o modelo único para análise
        self.analyst_model = ChatOllama(
            model="deepseek-r1:14b",
            base_url=analyst_base_url,
            temperature=0,
            num_ctx=8192
        )
        
        # Define a estrutura do prompt para o fluxo conversacional
        self.analysis_prompt_template = ChatPromptTemplate.from_messages([
            ("system", self._get_analysis_context()),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        self.analyst_chain = self.analysis_prompt_template | self.analyst_model

    def _get_analysis_context(self) -> str:
        """Prompt de sistema que define a persona e a tarefa para o LLM."""
        return """Você é um especialista sênior em redes de computadores, um perito em analisar dados de Qualidade de Serviço (QoS). Sua tarefa é interpretar dados estatísticos de uma rede de provedor de internet e fornecer insights claros e acionáveis para os operadores de rede.

A análise é baseada na seguinte metodologia:
1. Pontos de mudança foram detectados em séries temporais de métricas-chave (como throughput de download ou RTT de upload).
2. Os intervalos entre esses pontos de mudança foram agrupados em dois clusters (Cluster 0 e Cluster 1) usando análise de sobrevivência.
3. Regressão logística foi usada para identificar quais elementos da rede (clientes, servidores) e métricas estão mais associados a cada cluster.

Sua análise procederá em etapas. Considere cuidadosamente os dados fornecidos em cada etapa e o histórico da conversa para construir uma compreensão abrangente. Seja sempre direto, orientado por dados e foque em fornecer recomendações práticas.
"""
    
    def _run_analysis_for_approach(self, approach: str) -> str:
        """
        Executa a análise completa de 4 etapas para uma determinada abordagem.
        
        Args:
            approach (str): A abordagem a ser analisada ('throughput_download' ou 'rtt_upload').

        Returns:
            str: O relatório final e sintetizado para a abordagem.
        """
        print(f"\n--- Iniciando Análise para a Abordagem: {approach.replace('_', ' ').title()} ---")
        data_handler = NetworkDataHandler(approach)
        chat_history = []
        
        # Verifica se os dados foram carregados corretamente
        if data_handler.df_intervals.empty or data_handler.df_coef.empty or data_handler.df_cluster_stats.empty:
            error_message = f"Não foi possível prosseguir com a análise para '{approach}' devido a um erro no carregamento de dados."
            print(error_message)
            return error_message


        # === ETAPA 1: Interpretar Clusters ===
        print("Etapa 1: Interpretando os Clusters...")
        cluster_stats = data_handler.get_cluster_descriptive_stats()
        step1_prompt = f"""
        Vamos começar a análise para a abordagem '{approach}'.
        Aqui estão as estatísticas descritivas para os dois clusters de desempenho identificados.
        
        Dados:
        {json.dumps(cluster_stats, indent=2)}

        Com base nestes dados, por favor:
        1. Crie um perfil descritivo para o Cluster 0 e o Cluster 1.
        2. Declare claramente qual cluster representa o melhor desempenho e justifique sua conclusão comparando suas métricas-chave (duração do intervalo, throughput, RTT, frequência de eventos).
        Este perfil será a base para nossa análise subsequente.
        """
        step1_response_msg = self.analyst_chain.invoke({
            "chat_history": chat_history,
            "input": step1_prompt
        })
        step1_response = step1_response_msg.content
        self.logger.log_step(approach, "1_Interpretacao_Cluster", step1_prompt, step1_response)
        chat_history.extend([HumanMessage(content=step1_prompt), AIMessage(content=step1_response)])

        # === ETAPA 2: Avaliar Risco dos Elementos ===
        print("Etapa 2: Avaliando o Risco dos Elementos...")
        coefficients = data_handler.get_coefficients()
        step2_prompt = f"""
        Excelente. Estabelecemos os perfis de desempenho para nossos clusters.
        Agora, vamos identificar quais elementos da rede têm o maior risco de apresentar baixo desempenho. Aqui estão os coeficientes da regressão logística. Eles mostram a influência de cada elemento na probabilidade de pertencer ao Cluster 1 (o cluster de melhor desempenho). Uma odds_ratio < 1 indica um risco maior de estar no Cluster 0, de pior desempenho.

        Dados:
        {json.dumps(coefficients, indent=2)}

        Com base nesses dados e em nossos perfis de cluster estabelecidos, por favor:
        1. Avalie o perfil de risco para cada cliente e servidor.
        2. Crie uma pontuação de risco ponderada simples para cada elemento. Uma pontuação básica pode ser `(1 - odds_ratio)`. Pontuações mais altas significam maior risco.
        3. Liste os 5 elementos de maior risco (clientes ou servidores combinados).
        """
        step2_response_msg = self.analyst_chain.invoke({
            "chat_history": chat_history,
            "input": step2_prompt
        })
        step2_response = step2_response_msg.content
        self.logger.log_step(approach, "2_Avaliacao_Risco_Elemento", step2_prompt, step2_response)
        chat_history.extend([HumanMessage(content=step2_prompt), AIMessage(content=step2_response)])

        # === ETAPA 3: Analisar Elementos Críticos ===
        print("Etapa 3: Analisando os Elementos Críticos...")
        intervals_data = data_handler.get_intervals_data()
        step3_prompt = f"""
        Ótimo, temos uma lista de observação de elementos de alto risco.
        Agora, vamos realizar uma análise mais profunda de seu comportamento. Você receberá uma amostra dos dados brutos de intervalo para todas as conexões.

        Amostra de Dados de Intervalo:
        {json.dumps(intervals_data, indent=2)}

        Focando nos elementos de alto risco identificados na etapa anterior:
        1. Analise sua evolução temporal. Eles mostram consistentemente um desempenho ruim, ou é intermitente?
        2. Examine os outros elementos (clientes/servidores) com os quais eles se conectam. Existem padrões?
        3. Com base no contexto completo (perfis de cluster, pontuações de risco, dados de intervalo), proponha possíveis causas para seu status de alto risco.
        """
        step3_response_msg = self.analyst_chain.invoke({
            "chat_history": chat_history,
            "input": step3_prompt
        })
        step3_response = step3_response_msg.content
        self.logger.log_step(approach, "3_Analise_Elemento_Critico", step3_prompt, step3_response)
        chat_history.extend([HumanMessage(content=step3_prompt), AIMessage(content=step3_response)])

        # === ETAPA 4: Sintetizar Relatório Final ===
        print("Etapa 4: Sintetizando o Relatório Final...")
        step4_prompt = f"""
        Isto conclui nossa análise detalhada para a abordagem '{approach}'.
        Agora, por favor, sintetize todas as informações de toda a nossa conversa (Perfis de Cluster, Avaliação de Risco e Análise de Elementos Críticos) em um único relatório abrangente.

        O relatório deve ser estruturado para operadores de rede e deve incluir:
        1. Um resumo executivo da saúde da rede a partir desta perspectiva.
        2. Uma descrição clara dos perfis de desempenho identificados (clusters).
        3. Uma lista priorizada dos elementos críticos e de alto risco.
        4. Um resumo das causas prováveis para seu baixo desempenho.
        5. Um conjunto de recomendações claras e acionáveis para a equipe de operações de rede mitigar esses riscos e melhorar a QoS.
        """
        step4_response_msg = self.analyst_chain.invoke({
            "chat_history": chat_history,
            "input": step4_prompt
        })
        final_report = step4_response_msg.content
        self.logger.log_step(approach, "4_Relatorio_Final", step4_prompt, final_report)
        print(f"--- Análise para {approach.replace('_', ' ').title()} Concluída ---")
        
        return final_report

    def run_full_analysis(self):
        """
        Orquestra todo o processo de análise, executando ambas as abordagens e, em seguida, comparando-as.
        """
        try:
            # Executa a análise para a primeira abordagem
            report_throughput = self._run_analysis_for_approach('throughput_download')

            # Executa a análise para a segunda abordagem
            report_rtt = self._run_analysis_for_approach('rtt_upload')

            # === ETAPA FINAL: Comparar Abordagens ===
            # Verifica se os relatórios foram gerados com sucesso antes de comparar
            if "erro" in report_throughput.lower() or "erro" in report_rtt.lower():
                print("\nAVISO: A análise comparativa não pode ser executada porque uma ou mais das análises de abordagem falharam.")
                return

            print("\n--- Iniciando Etapa Final: Comparando Abordagens ---")
            comparison_prompt = f"""
            Concluímos duas análises separadas e aprofundadas da mesma rede.
            -   **Análise 1 (Baseada em Throughput):** Focou em intervalos definidos por mudanças no 'throughput de download'.
            -   **Análise 2 (Baseada em RTT):** Focou em intervalos definidos por mudanças no 'RTT de upload'.

            Aqui estão os relatórios finais de cada análise:

            --- RELATÓRIO 1: ANÁLISE BASEADA EM THROUGHPUT ---
            {report_throughput}
            -------------------------------------------------

            --- RELATÓRIO 2: ANÁLISE BASEADA EM RTT ---
            {report_rtt}
            -----------------------------------------

            Como analista de dados sênior, por favor, escreva agora uma meta-análise final comparando essas duas visões. Sua comparação deve explicar:
            1.  **Pontos em Comum:** Quais descobertas chave (ex: elementos problemáticos como `gru03`) são consistentes em ambas as análises?
            2.  **Insights Únicos:** O que a análise de throughput revelou que a análise de RTT não revelou, e vice-versa?
            3.  **Complementaridade:** Explique como essas duas métricas (throughput e latência) fornecem uma visão complementar, e não contraditória, da saúde da rede. Throughput mede capacidade, enquanto RTT mede responsividade.
            4.  **Estratégia Integrada:** Como um operador de rede deve combinar essas duas perspectivas para uma compreensão holística e uma estratégia de gerenciamento de rede mais eficaz? Forneça conselhos práticos e acionáveis.
            """
            
            final_comparison_msg = self.analyst_chain.invoke({
                "chat_history": [], # Começa do zero para a meta-análise final
                "input": comparison_prompt
            })
            final_comparison = final_comparison_msg.content
            self.logger.log_final_comparison(comparison_prompt, final_comparison)

            print("\n" + "="*40)
            print("   ANÁLISE COMPLETA DE QOS DA REDE   ")
            print("="*40 + "\n")
            print("--- Análise Baseada em Throughput de Download ---")
            print(report_throughput)
            print("\n--- Análise Baseada em RTT de Upload ---")
            print(report_rtt)
            print("\n--- Análise Comparativa ---")
            print(final_comparison)
            print("\n" + "="*40)
            print(f"Log completo da análise salvo em: {self.logger.filename}")

        except Exception as e:
            print(f"Ocorreu um erro inesperado durante a análise: {e}")
            import traceback
            traceback.print_exc()

def main():
    """
    Função principal para inicializar e executar o agente de análise de rede.
    """
    print("Inicializando o Agente de Análise de QoS da Rede...")
    print("Este script realizará automaticamente uma análise de múltiplos passos.")
    
    try:
        agent = NetworkAnalysisAgent()
        agent.run_full_analysis()
    except Exception as e:
        print(f"Falha ao inicializar ou executar o agente. Erro: {e}")

if __name__ == "__main__":
    main()