from typing import Optional, Dict, List
import pandas as pd
import numpy as np
import os
from pathlib import Path

def decode_one_hot(df, prefix):
    """
    Decodifica um conjunto de colunas one-hot encoded em uma única coluna com as labels originais.

    Parâmetros:
        df (pd.DataFrame): DataFrame com as colunas one-hot encoded.
        prefix (str): Prefixo das colunas one-hot encoded.

    Retorna:
        pd.DataFrame: DataFrame com as colunas decodificadas.
    """
    cols = [col for col in df.columns if col.startswith(prefix)]
    decoded_column = df[cols].idxmax(axis=1).str.replace(prefix, '', regex=False)
    df = df.drop(columns=cols).assign(**{prefix.replace('_',''): decoded_column})
    return df


def assign_labels(survival_df, data, timestamp_col='timestamp', survival_functions=None, multivariate=False):
    """
    Associa os rótulos de clusters gerados pelo SurvMixClust às séries temporais originais.

    Parâmetros:
    ----------
    survival_df : pd.DataFrame
        Dataset de sobrevivência com os rótulos dos clusters adicionados.
    series_dir : str
        Diretório contendo as séries temporais originais em formato Parquet.
    output_dir : str
        Diretório onde as séries temporais atualizadas serão salvas.
    timestamp_col : str, default='timestamp'
        Nome da coluna de timestamp nas séries temporais.
    survival_functions : pd.DataFrame, default=None
        Funções de sobrevivência para cada intervalo de tempo.
    multivariate : bool, default=False
        Se True, considera os changepoints detectados com a abordagem multivariada do VWCD.

    Retorna:
    -------
    None
        Salva as séries temporais atualizadas com os rótulos dos clusters no diretório `output_dir`.
    """
    if multivariate:
        series_dir = f'datasets/ts_{data}_cp_mv/'
        output_dir = f'datasets/ts_{data}_results_mv/'
    else:
        series_dir = f'datasets/ts_{data}_cp/'
        output_dir = f'datasets/ts_{data}_results/'

    os.makedirs(output_dir, exist_ok=True)

    # Criar colunas para as médias e desvios padrão locais e a taxa de outliers
    for feature in ['rtt_download', 'rtt_upload', 'throughput_download', 'throughput_upload']:
        #survival_df[f'{feature}_local_mean'] = -1
        #survival_df[f'{feature}_local_std'] = -1
        survival_df[f'{feature}_outliers_rate'] = -1

    for file in os.listdir(series_dir):
        if file.endswith(".parquet"):
            series_df = pd.read_parquet(os.path.join(series_dir, file))
            client, site = file.split('.')[0].split('_', 1)

            # Analisar os intervalos do dataset de sobrevivência correspondentes ao par cliente-site
            if f'client_{client}' in survival_df.columns and f'site_{site}' in survival_df.columns:
                intervals = survival_df[(survival_df[f'client_{client}'] == 1) & 
                                        (survival_df[f'site_{site}'] == 1)]
            else:
                print(f"Colunas 'client_{client}' ou 'site_{site}' não encontradas. Pulando {file}.")
                continue

            series_df[timestamp_col] = pd.to_datetime(series_df[timestamp_col])
            series_df['cluster'] = -1
            series_df['cluster_probability'] = -1
            series_df['survival_probability'] = -1

            for idx, row in intervals.iterrows():
                start_time = pd.to_datetime(row['timestamp_start'])
                end_time = pd.to_datetime(row['timestamp_end'])
                cluster_label = row['cluster']
                cluster_prob = row.get(f'probability_cluster_{cluster_label}', -1)

                mask = (series_df[timestamp_col] >= start_time) & (series_df[timestamp_col] <= end_time)
                series_df.loc[mask, 'cluster'] = cluster_label
                series_df.loc[mask, 'cluster_probability'] = cluster_prob

                for _, row_series in series_df[mask].iterrows():
                    # Interpolando a função de sobrevivência
                    time = row_series[timestamp_col] - start_time
                    time = time.total_seconds() / (60 * 60 * 24)  # Converter para dias
                    
                    # Interpolação linear para encontrar a probabilidade de sobrevivência para o tempo 'time'.
                    surv_prob = np.interp(
                        time, 
                        survival_functions.index.to_numpy().flatten(),
                        survival_functions[idx].values.flatten()
                    )

                    # Atribuir a probabilidade de sobrevivência à coluna 'survival' da entrada correspondente
                    series_df.loc[series_df[timestamp_col] == row_series[timestamp_col], 'survival_probability'] = surv_prob
                    
                # Média e desvio padrão local das métricas
                for feature in ['rtt_download', 'rtt_upload', 'throughput_download', 'throughput_upload']:
                    local_mean = series_df.loc[mask, feature].mean()
                    local_std = series_df.loc[mask, feature].std()

                    if pd.notnull(local_std) and local_std != 0:
                        z_scores = (series_df.loc[mask, feature] - local_mean) / local_std
                        outliers_rate = (z_scores > 3).sum() / len(z_scores)
                    else:
                        outliers_rate = 0

                    #series_df.loc[mask, f'{feature}_local_mean'] = local_mean
                    #series_df.loc[mask, f'{feature}_local_std'] = local_std
                    series_df.loc[mask, f'{feature}_outliers_rate'] = outliers_rate

                    # Atribuir a média e desvio padrão locais e a taxa de outliers à entrada correspondente no surival_df
                    #survival_df.loc[idx, f'{feature}_local_mean'] = local_mean
                    #survival_df.loc[idx, f'{feature}_local_std'] = local_std
                    survival_df.loc[idx, f'{feature}_outliers_rate'] = outliers_rate

            series_df.to_parquet(os.path.join(output_dir, file), index=False)

    print(f"Clusters associados e séries temporais salvas em: {output_dir}")

    return survival_df


def calculate_cluster_proportions(df_surv):
    """
    Calcula a proporção de tempo em cada cluster separadamente para clientes e servidores.
    
    Parameters:
    df_surv (pd.DataFrame): DataFrame com colunas 'time', 'cluster' e colunas one-hot para clientes e servidores
    
    Returns:
    tuple: (client_proportions, site_proportions) - DataFrames com as proporções de tempo por cluster
    """
    # Identificar colunas de clientes e servidores
    client_cols = [col for col in df_surv.columns if col.startswith('client_')]
    site_cols = [col for col in df_surv.columns if col.startswith('site_')]
    
    # Função auxiliar para calcular proporções
    def calculate_entity_proportions(entity_cols, entity_prefix):
        results = []
        
        for entity_col in entity_cols:
            # Filtrar registros para esta entidade
            subset = df_surv[df_surv[entity_col] == 1]
            
            if len(subset) > 0:
                # Calcular tempo total para esta entidade
                total_time = subset['time'].sum()
                
                # Calcular tempo por cluster
                cluster_times = subset.groupby('cluster')['time'].sum()
                
                # Calcular proporções
                proportions = cluster_times / total_time
                
                # Criar registro para esta entidade
                for cluster, proportion in proportions.items():
                    results.append({
                        'entity': entity_col.replace(entity_prefix, ''),
                        'cluster': cluster,
                        'proportion': proportion,
                        'total_time': total_time
                    })
        
        # Criar DataFrame com os resultados
        result_df = pd.DataFrame(results)
        
        if len(result_df) > 0:
            # Pivotear a tabela para ter clusters como colunas
            pivot_df = result_df.pivot_table(
                index='entity',
                columns='cluster',
                values='proportion',
                fill_value=0
            )
            
            # Adicionar coluna de tempo total
            total_time_df = result_df.groupby('entity')['total_time'].first()
            pivot_df['total_time'] = total_time_df
            
            # Renomear colunas de cluster para melhor clareza
            pivot_df.columns = [f'cluster_{c}' if isinstance(c, (int, float)) else c 
                              for c in pivot_df.columns]
            
            return pivot_df
        
        return pd.DataFrame()
    
    # Calcular proporções para clientes e servidores
    client_proportions = calculate_entity_proportions(client_cols, 'client_')
    site_proportions = calculate_entity_proportions(site_cols, 'site_')
    
    return client_proportions, site_proportions


def compute_clusters_stats(df_surv: pd.DataFrame, model):
    """
    Cria um dataframe com as estatísticas descritivas de cada cluster.
    
    Args:
        df_surv (pd.DataFrame): DataFrame com os intervalos. Deve conter as 
            colunas 'timestamp_start', 'timestamp_end', 'time', 'event', 'cluster', 'client', 'site'.
    
    Returns:
        pd.DataFrame: DataFrame com estatísticas descritivas para cada cluster.
    """
    # Validação e Preparação
    required_cols = ['timestamp_start', 'timestamp_end', 'time', 'event', 'cluster', 'client', 'site']
    for col in required_cols:
        if col not in df_surv.columns:
            raise ValueError(f"A coluna '{col}' está faltando no DataFrame de sobrevivência (df_surv).")
    
    # Lista para armazenar estatísticas de cada cluster
    cluster_stats_list = []
    
    # Iterar sobre cada cluster único
    unique_clusters = df_surv['cluster'].unique()
    print(f"Processando {len(unique_clusters)} clusters...")
    
    for cluster_id in unique_clusters:
        print(f"\nProcessando cluster {cluster_id}...")
        
        # Filtrar dados do cluster atual
        cluster_data = df_surv[df_surv['cluster'] == cluster_id]
        
        # Calcular estatísticas de tempo e eventos
        time_stats = {
            'cluster': cluster_id,
            'interval_count': len(cluster_data),
            'time_mean': cluster_data['time'].mean(),
            'time_median': cluster_data['time'].median(),
            'time_std': cluster_data['time'].std(),
            'event_frequency': cluster_data['event'].sum() / len(cluster_data),
            'throughput_download_mean': cluster_data['throughput_download_mean'].mean(),
            'throughput_download_median': cluster_data['throughput_download_mean'].median(),
            'throughput_download_std': cluster_data['throughput_download_mean'].std(),
            'throughput_uplosad_mean': cluster_data['throughput_upload_mean'].mean(),
            'throughput_upload_median': cluster_data['throughput_upload_mean'].median(),
            'throughput_upload_std': cluster_data['throughput_upload_mean'].std(),
            'rtt_download_mean': cluster_data['rtt_download_mean'].mean(),
            'rtt_download_median': cluster_data['rtt_download_mean'].median(),
            'rtt_download_std': cluster_data['rtt_download_mean'].std(),
            'rtt_upload_mean': cluster_data['rtt_upload_mean'].mean(),
            'rtt_upload_median': cluster_data['rtt_upload_mean'].median(),
            'rtt_upload_std': cluster_data['rtt_upload_mean'].std(),
        }
        
        # Adicionar função de sobrevivência
        kmf = model.kmfs[cluster_id]['kmf']
        surv_fcn_df = kmf.survival_function_.reset_index()
        surv_fcn_df.columns = ['time', 'survival_probability']
        
        # Definir tempos fixos para interpolação
        fixed_times = [1, 7, 15, 30, 60, 90]
        
        # Interpolar valores da função de sobrevivência nos tempos fixos
        interpolated_probs = np.interp(
            fixed_times, 
            surv_fcn_df['time'].values, 
            surv_fcn_df['survival_probability'].values
        )
        
        # Criar lista de pontos [tempo, probabilidade] para os tempos fixos
        survival_function = []
        for i in range(len(fixed_times)):
            point = {
                'time_days': fixed_times[i],
                'survival_probability': round(interpolated_probs[i], 3)
            }
            survival_function.append(point)
        
        time_stats['survival_function'] = survival_function
        
        cluster_stats_list.append(time_stats)
    
    # Criar DataFrame final
    result_df = pd.DataFrame(cluster_stats_list)
        
    return result_df

def compute_clusters_stats_ts(df_surv: pd.DataFrame, model):
    """
    Cria um dataframe com as estatísticas descritivas de cada cluster.
    
    Args:
        df_surv (pd.DataFrame): DataFrame com os intervalos. Deve conter as 
            colunas 'timestamp_start', 'timestamp_end', 'time', 'event', 'cluster', 'client', 'site'.
    
    Returns:
        pd.DataFrame: DataFrame com estatísticas descritivas para cada cluster.
    """
    # Validação e Preparação
    required_cols = ['timestamp_start', 'timestamp_end', 'time', 'event', 'cluster', 'client', 'site']
    for col in required_cols:
        if col not in df_surv.columns:
            raise ValueError(f"A coluna '{col}' está faltando no DataFrame de sobrevivência (df_surv).")
    
    metrics = ['throughput_download', 'throughput_upload', 'rtt_download', 'rtt_upload']
    time_series_path = 'datasets/ts_ndt_cp'
    
    # Cache para arquivos lidos
    read_files_cache = {}
    
    # Lista para armazenar estatísticas de cada cluster
    cluster_stats_list = []
    
    # Iterar sobre cada cluster único
    unique_clusters = df_surv['cluster'].unique()
    print(f"Processando {len(unique_clusters)} clusters...")
    
    for cluster_id in unique_clusters:
        print(f"\nProcessando cluster {cluster_id}...")
        
        # Filtrar dados do cluster atual
        cluster_data = df_surv[df_surv['cluster'] == cluster_id]
        
        # Calcular estatísticas de tempo e eventos
        time_stats = {
            'cluster': cluster_id,
            'interval_count': len(cluster_data),
            'time_mean': cluster_data['time'].mean(),
            'time_median': cluster_data['time'].median(),
            'time_std': cluster_data['time'].std(),
            'event_frequency': cluster_data['event'].sum() / len(cluster_data)
        }
        
        # Coletar séries temporais para este cluster
        cluster_time_series_list = []
        
        for _, interval in cluster_data.iterrows():
            client = interval['client']
            server = interval['site']
            
            file_name = f"{client}_{server}.parquet"
            file_path = os.path.join(time_series_path, file_name)
            
            try:
                # Verificar cache
                if file_path not in read_files_cache:
                    df_ts_raw = pd.read_parquet(file_path)
                    df_ts_raw['timestamp'] = pd.to_datetime(df_ts_raw['timestamp'])
                    read_files_cache[file_path] = df_ts_raw
                else:
                    df_ts_raw = read_files_cache[file_path]
                
                # Filtrar período do intervalo
                mask = ((df_ts_raw['timestamp'] >= interval['timestamp_start']) & 
                       (df_ts_raw['timestamp'] <= interval['timestamp_end']))
                filtered_ts = df_ts_raw.loc[mask, metrics].copy()
                
                if not filtered_ts.empty:
                    cluster_time_series_list.append(filtered_ts)
                    
            except FileNotFoundError:
                print(f"  AVISO: Arquivo não encontrado: {file_name}")
                continue
        
        # Calcular estatísticas das métricas se houver dados
        if cluster_time_series_list:
            # Concatenar todas as séries temporais do cluster
            cluster_ts_combined = pd.concat(cluster_time_series_list, ignore_index=True)
            
            # Calcular estatísticas para cada métrica
            for metric in metrics:
                metric_data = cluster_ts_combined[metric]
                time_stats[f'{metric}_mean'] = metric_data.mean()
                time_stats[f'{metric}_median'] = metric_data.median()
                time_stats[f'{metric}_std'] = metric_data.std()

        else:
            # Se não houver dados de séries temporais, preencher com NaN
            for metric in metrics:
                time_stats[f'{metric}_mean'] = np.nan
                time_stats[f'{metric}_median'] = np.nan
                time_stats[f'{metric}_std'] = np.nan
        
        # Adicionar função de sobrevivência
        kmf = model.kmfs[cluster_id]['kmf']
        surv_fcn_df = kmf.survival_function_.reset_index()
        surv_fcn_df.columns = ['time', 'survival_probability']
        
        # Definir tempos fixos para interpolação
        fixed_times = [1, 7, 15, 30, 60, 90]
        
        # Interpolar valores da função de sobrevivência nos tempos fixos
        interpolated_probs = np.interp(
            fixed_times, 
            surv_fcn_df['time'].values, 
            surv_fcn_df['survival_probability'].values
        )
        
        # Criar lista de pontos [tempo, probabilidade] para os tempos fixos
        survival_function = []
        for i in range(len(fixed_times)):
            point = {
                'time_days': fixed_times[i],
                'survival_probability': round(interpolated_probs[i], 3)
            }
            survival_function.append(point)
        
        time_stats['survival_function'] = survival_function
        
        cluster_stats_list.append(time_stats)
    
    # Criar DataFrame final
    result_df = pd.DataFrame(cluster_stats_list)
    
    print(f"\nProcessamento concluído. Estatísticas calculadas para {len(result_df)} clusters.")
    
    return result_df