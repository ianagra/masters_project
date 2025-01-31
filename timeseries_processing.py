import numpy as np
import pandas as pd
import os
from VWCD import vwcd, vwcd_mv

def map_mac_to_client(df, mac_column='ClientMac'):
    """
    Mapeia endereços MAC para identificadores no formato 'clientXX'.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contendo a coluna com endereços MAC
    mac_column : str, optional
        Nome da coluna contendo os endereços MAC (default: 'ClientMac')
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame com a coluna original e nova coluna mapeada
    dict
        Dicionário com o mapeamento realizado
    """
    
    # Validação da entrada
    if mac_column not in df.columns:
        raise ValueError(f"Coluna {mac_column} não encontrada no DataFrame")
    
    # Mapeamento inicial específico
    initial_mapping = {
        'e45f01963c21': 'client01',
        'e45f01359a20': 'client02',
        'dca6326b9ca8': 'client03',
        'dca6326b9c99': 'client04'
    }
    
    # Pegar todos os valores únicos da coluna
    unique_macs = df[mac_column].unique()
    
    # Criar mapeamento completo
    mac_mapping = initial_mapping.copy()
    current_index = 5
    
    for mac in unique_macs:
        if mac not in mac_mapping:
            mac_mapping[mac] = f'client{current_index:02d}'
            current_index += 1
    
    # Criar nova coluna com o mapeamento
    df_mapped = df.copy()
    df_mapped['ClientId'] = df[mac_column].map(mac_mapping)
    
    return df_mapped, mac_mapping


def export_time_series(data):
    """
    Exporta séries temporais das variáveis de download e upload para cada par cliente-site.

    Para cada cliente e site:
    - Cria um DataFrame contendo os dados de download (DownRTTMean, Download) e de upload (UpRTTMean, Upload) para cada par cliente-site.
    - Cada DataFrame é salvos em arquivo .parquet.
    - Gera metadados consolidados contendo informações resumidas sobre as séries temporais.

    Parâmetros:
    ----------
    data : str
        Identificador do conjunto de dados (usado para carregar e nomear os arquivos correspondentes).

    Saída:
    ------
    pd.DataFrame
        Um DataFrame contendo os metadados das séries temporais, incluindo:
        - client (str): Identificação do cliente.
        - site (str): Identificação do site.
        - inicio (datetime): Timestamp da primeira medição de download.
        - fim (datetime): Timestamp da última medição de download.
        - num_med (int): Número de medições.
        - mean_time (float): Intervalo médio entre medições, em horas.
        - file_prefix (str): Prefixo usado nos nomes dos arquivos gerados.

    Diretórios criados:
    -------------------
    datasets/ts_<data>/

    Arquivos gerados:
    -----------------
    - Séries temporais para cada par cliente-site: <client>_<site>.parquet
    - Metadados: ts_metadata_<data>.parquet
    """
    # Importação dos dados
    df = pd.read_parquet(f'datasets/dados_{data}.parquet')

    # Filtros
    clients = df['ClientId'].unique()
    sites = df['Site'].unique()

    # Diretórios de saída
    output_dir = f'datasets/ts_{data}/'
    os.makedirs(output_dir, exist_ok=True)

    med = []
    for c in clients:
        for s in sites:
            # Filtros por cliente e site para download e upload
            df_pair = df[(df.ClientId == c) & (df.Site == s)]
                    
            if len(df_pair) >= 100:
                # Criar DataFrame para download
                df_ts = pd.DataFrame({
                    'timestamp': df_pair['DataHora'].values,
                    'rtt_download': df_pair['DownRTTMean'].values,
                    'throughput_download': df_pair['Download'].values,
                    'rtt_upload': df_pair['UpRTTMean'].values,
                    'throughput_upload': df_pair['Upload'].values
                })

                # Ordenar por timestamp
                df_ts.sort_values(by='timestamp', inplace=True)

                # Salvar em arquivo
                output_file = f"{output_dir}/{c}_{s}.parquet"
                df_ts.to_parquet(output_file, index=False)

                # Coletar metadados
                inicio = df_pair['DataHora'].iloc[0]
                fim = df_pair['DataHora'].iloc[-1]
                num_med = len(df_pair)
                mean_time = np.round(df_pair['DataHora'].diff().mean().seconds / 3600, 1)
                file_prefix = f"{c}_{s}"
                
                quant = {
                    "client": c, 
                    "site": s, 
                    "inicio": inicio,
                    "fim": fim,
                    "num_med": num_med,
                    "mean_time": mean_time,
                    "file_prefix": file_prefix
                }
                med.append(quant)

    # Conjunto de metadados
    df_series = pd.DataFrame(med)
    df_series.to_parquet(f'datasets/ts_metadata_{data}.parquet', index=False)

    return df_series


def detect_changepoints(data, wv, ab, p_thr, vote_p_thr, vote_n_thr, y0, yw, aggreg):
    """
    Detecta pontos de mudança (changepoints) em todas as colunas numéricas de séries temporais.

    Para cada arquivo Parquet gerado pela função `export_time_series`, a função:
    - Detecta changepoints em todas as colunas numéricas (exceto 'timestamp').
    - Adiciona colunas binárias indicando os changepoints para cada variável.
    - Adiciona colunas com o número de votos para cada ponto da série temporal.
    - Adiciona colunas com a média e desvio padrão local para cada ponto.

    Parâmetros:
    ----------
    data : str
        Identificador do conjunto de dados (usado para localizar os arquivos de séries temporais).
    wv : int
        Tamanho da janela deslizante de votação.
    ab : float
        Hiperparâmetros alfa e beta da distribuição beta-binomial.
    p_thr : float
        Limiar de probabilidade para o voto de uma janela ser registrado.
    vote_p_thr : float
        Limiar de probabilidade para definir um ponto de mudança após a agregação dos votos.
    vote_n_thr : float
        Fração mínima da janela que precisa votar para definir um ponto de mudança.
    y0 : float
        Probabilidade a priori da função logística (início da janela).
    yw : float
        Probabilidade a priori da função logística (tamanho da janela).
    aggreg : str
        Função de agregação para os votos ('posterior' ou 'mean').

    Retorna:
    -------
    None
        Cria novos arquivos Parquet com colunas binárias indicando os changepoints,
        número de votos, médias e desvios padrão locais para cada variável.
    """
    # Diretório de entrada e saída
    input_dir = f'datasets/ts_{data}/'
    output_dir = f'datasets/ts_{data}_cp/'
    os.makedirs(output_dir, exist_ok=True)

    # Processar cada arquivo Parquet de séries temporais
    for file in os.listdir(input_dir):
        if file.endswith(".parquet"):
            # Carregar a série temporal
            df = pd.read_parquet(os.path.join(input_dir, file))

            # Iterar pelas colunas numéricas, exceto 'timestamp'
            for column in df.select_dtypes(include=[np.number]).columns:
                y = df[column].dropna().values

                # Parâmetros do algoritmo VWCD
                kargs = {
                    'X': y, 'w': wv, 'w0': wv, 'ab': ab,
                    'p_thr': p_thr, 'vote_p_thr': vote_p_thr,
                    'vote_n_thr': vote_n_thr, 'y0': y0, 'yw': yw, 'aggreg': aggreg
                }

                # Executar a detecção de changepoints com VWCD
                CP, M0, S0, _, vote_counts, vote_probs, agg_probs = vwcd(**kargs)

                # Criar uma coluna binária indicando changepoints
                changepoints = np.zeros(len(y), dtype=int)
                changepoints[CP] = 1

                # Inicializar arrays para médias e desvios padrão locais
                local_means = np.zeros(len(y))
                local_stds = np.zeros(len(y))

                # Preencher médias e desvios padrão locais
                if len(CP) > 0:
                    # Primeiro segmento (do início até o primeiro CP)
                    local_means[:CP[0]] = M0[0]
                    local_stds[:CP[0]] = S0[0]

                    # Segmentos intermediários
                    for i in range(len(CP)-1):
                        local_means[CP[i]:CP[i+1]] = M0[i+1]
                        local_stds[CP[i]:CP[i+1]] = S0[i+1]

                    # Último segmento (do último CP até o fim)
                    local_means[CP[-1]:] = M0[-1]
                    local_stds[CP[-1]:] = S0[-1]
                else:
                    # Se não houver CPs, usar a média e desvio padrão de toda a série
                    local_means[:] = y.mean()
                    local_stds[:] = y.std(ddof=1)

                # Adicionar todas as colunas ao DataFrame
                # Changepoints
                changepoint_column = f'{column}_cp'
                df[changepoint_column] = 0
                df.loc[df[column].dropna().index, changepoint_column] = changepoints

                # Votos
                votes_column = f'{column}_votes'
                df[votes_column] = 0
                df.loc[df[column].dropna().index, votes_column] = vote_counts

                # Probabilidades individuais dos votos
                probs_column = f'{column}_vote_probs'
                df[probs_column] = 0
                df.loc[df[column].dropna().index, probs_column] = vote_probs

                # Probabilidades agregadas
                agg_probs_column = f'{column}_agg_probs'
                df[agg_probs_column] = 0
                df.loc[df[column].dropna().index, agg_probs_column] = agg_probs

                # Médias locais
                means_column = f'{column}_local_mean'
                df[means_column] = 0
                df.loc[df[column].dropna().index, means_column] = local_means

                # Desvios padrão locais
                stds_column = f'{column}_local_std'
                df[stds_column] = 0
                df.loc[df[column].dropna().index, stds_column] = local_stds

            # Salvar o DataFrame com todas as novas colunas
            output_file = os.path.join(output_dir, file)
            df.to_parquet(output_file, index=False)

    print(f"Changepoints, contagem de votos e estatísticas locais detectados e salvos em: {output_dir}")


def detect_changepoints_mv(data, wv, ab, p_thr, vote_p_thr, vote_n_thr, y0, yw, aggreg):
    """
    Detecta pontos de mudança em séries temporais multivariadas.
    
    Salva os dados originais, junto com:
    - Pontos de mudança (changepoints)
    - Número de votos
    - Probabilidades dos votos
    - Probabilidades agregadas
    - Médias locais (na escala original)
    - Desvios padrão locais (na escala original)
    """
    input_dir = f'datasets/ts_{data}/'
    output_dir = f'datasets/ts_{data}_cp_mv/'
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith(".parquet"):
            df = pd.read_parquet(os.path.join(input_dir, file))
            
            # Selecionar colunas numéricas
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            # Preparar dados multivariados
            X = df[numeric_cols].values
            
            # Executar VWCD multivariado
            kargs = {
                'X': X, 'w': wv, 'w0': wv, 'ab': ab,
                'p_thr': p_thr, 'vote_p_thr': vote_p_thr,
                'vote_n_thr': vote_n_thr, 'y0': y0, 'yw': yw, 'aggreg': aggreg
            }
            
            CP, M0, S0, _, vote_counts, vote_probs, agg_probs = vwcd_mv(**kargs)
            
            # Criar colunas de changepoints e votos
            changepoints = np.zeros(len(df), dtype=int)
            changepoints[CP] = 1
            
            # Adicionar colunas globais ao DataFrame
            df['cp'] = changepoints
            df['votes'] = vote_counts
            df['vote_probs'] = vote_probs
            df['agg_probs'] = agg_probs
            
            # Inicializar arrays para médias e desvios padrão locais
            local_means = np.zeros((len(df), len(numeric_cols)))
            local_stds = np.zeros((len(df), len(numeric_cols)))
            
            # Preencher médias e desvios padrão locais (na escala original)
            if len(CP) > 0:
                # Primeiro segmento (do início até o primeiro CP)
                local_means[:CP[0]] = M0[0]
                local_stds[:CP[0]] = np.sqrt(np.diag(S0[0]))
                
                # Segmentos intermediários
                for i in range(len(CP)-1):
                    local_means[CP[i]:CP[i+1]] = M0[i+1]
                    local_stds[CP[i]:CP[i+1]] = np.sqrt(np.diag(S0[i+1]))
                
                # Último segmento (do último CP até o fim)
                local_means[CP[-1]:] = M0[-1]
                local_stds[CP[-1]:] = np.sqrt(np.diag(S0[-1]))
            else:
                # Se não houver CPs, usar a média e desvio padrão de toda a série
                local_means[:] = X.mean(axis=0)
                local_stds[:] = X.std(axis=0, ddof=1)
            
            # Adicionar colunas de estatísticas locais para cada variável
            for i, col in enumerate(numeric_cols):
                # Médias locais
                means_column = f'{col}_local_mean'
                df[means_column] = local_means[:, i]
                
                # Desvios padrão locais
                stds_column = f'{col}_local_std'
                df[stds_column] = local_stds[:, i]
            
            # Salvar resultados
            output_file = os.path.join(output_dir, file)
            df.to_parquet(output_file, index=False)
            
    print(f"Changepoints, votos e estatísticas locais multivariados detectados e salvos em: {output_dir}")


def create_survival_dataset(data, feature, max_gap_days=3, multivariate=False):
    """
    Cria um dataset de sobrevivência baseado em séries temporais e pontos de mudança de uma variável específica.

    Parâmetros:
    ----------
    data : str
        Identificador do conjunto de dados (usado para localizar os arquivos de séries temporais).
    feature : str
        Nome da variável para a qual os changepoints serão considerados.
    max_gap_days : int
        Intervalo máximo de dias permitido entre medições consecutivas antes de considerar um intervalo censurado.
    multivariate : bool
        Se True, considera considera os changepoints detectados com a abordagem multivariada do VWCD.

    Retorna:
    -------
    pd.DataFrame
        Dataset de sobrevivência com as seguintes colunas:
        - 'client', 'site': Identificação do cliente e site.
        - 'time': Duração do intervalo em dias.
        - 'timestamp_start', 'timestamp_end': Timestamps de início e fim do intervalo.
        - Variáveis originais: Valores no início do intervalo.
        - 'event': 1 se o intervalo termina em um changepoint, 0 se for censurado.
    """
    survival_data = []
    if multivariate:
        cp_dir = f'datasets/ts_{data}_cp_mv/'
    else:
        cp_dir = f'datasets/ts_{data}_cp/'

    for file in os.listdir(cp_dir):
        if file.endswith(".parquet"):
            df = pd.read_parquet(os.path.join(cp_dir, file))
            client, site = file.split('.')[0].split('_', 1)

            # Ordenar pelo timestamp
            df.sort_values(by='timestamp', inplace=True)

            # Identificar changepoints
            if multivariate:
                changepoint_column = 'cp'
            else:
                changepoint_column = f'{feature}_cp'

            if changepoint_column not in df.columns:
                print(f"Changepoint column '{changepoint_column}' not found in {file}. Skipping.")
                continue
            changepoint_indices = df.index[df[changepoint_column] == 1].tolist()

            # Calcular gaps de tempo
            df['time_diff'] = df['timestamp'].diff().dt.total_seconds() / (60 * 60 * 24)
            large_gaps = df.index[df['time_diff'] > max_gap_days].tolist()
            split_indices = [0]

            # Adicionar os índices antes e depois dos gaps como split indices
            for gap in large_gaps:
                if gap - 1 >= 0:
                    split_indices.append(gap - 1)  # Antes do gap
                split_indices.append(gap)  # Depois do gap
            split_indices.append(len(df) - 1)

            split_indices = sorted(set(split_indices))  # Garantir que os índices são únicos e ordenados

            # Iterar entre as subdivisões
            for i, (start, end) in enumerate(zip(split_indices[:-1], split_indices[1:])):
                sub_df = df.iloc[start:end + 1]  # Incluir o índice final
                if len(sub_df) < 2:
                    continue

                # Identificar changepoints nesta subsequência
                sub_changepoints = [cp for cp in changepoint_indices if start <= cp <= end]
                all_points = [start] + sub_changepoints + [end]

                # Iterar sobre os intervalos entre todos os pontos
                for j in range(len(all_points) - 1):
                    start_idx = all_points[j]
                    end_idx = all_points[j + 1]
                    if start_idx >= len(df) or end_idx >= len(df) or start_idx >= end_idx:
                        continue

                    start_time = df['timestamp'].iloc[start_idx]
                    end_time = df['timestamp'].iloc[end_idx]
                    duration = (end_time - start_time).total_seconds() / (60 * 60 * 24)

                    initial_values = df.iloc[start_idx].to_dict()
                    
                    # Buscar o primeiro valor válido para cada variável
                    throughput_download_local_mean = initial_values.get('throughput_download_local_mean', np.nan)
                    if pd.isna(throughput_download_local_mean):
                        throughput_download_local_mean = df['throughput_download_local_mean'].iloc[start_idx:end_idx+1].dropna().values[0] if not df['throughput_download_local_mean'].iloc[start_idx:end_idx+1].dropna().empty else np.nan

                    throughput_upload_local_mean = initial_values.get('throughput_upload_local_mean', np.nan)
                    if pd.isna(throughput_upload_local_mean):
                        throughput_upload_local_mean = df['throughput_upload_local_mean'].iloc[startthroughput_u_idx:end_idx+1].dropna().values[0] if not df['throughput_upload_local_mean'].iloc[start_idx:end_idx+1].dropna().empty else np.nan

                    rtt_download_local_mean = initial_values.get('rtt_download_local_mean', np.nan)
                    if pd.isna(rtt_download_local_mean):
                        rtt_download_local_mean = df['rtt_download_local_mean'].iloc[start_idx:end_idx+1].dropna().values[0] if not df['rtt_download_local_mean'].iloc[start_idx:end_idx+1].dropna().empty else np.nan

                    rtt_upload_local_mean = initial_values.get('rtt_upload_local_mean', np.nan)
                    if pd.isna(rtt_upload_local_mean):
                        rtt_upload_local_mean = df['rtt_upload_local_mean'].iloc[start_idx:end_idx+1].dropna().values[0] if not df['rtt_upload_local_mean'].iloc[start_idx:end_idx+1].dropna().empty else np.nan

                    throughput_download_local_std = initial_values.get('throughput_download_local_std', np.nan)
                    if pd.isna(throughput_download_local_std):
                        throughput_download_local_std = df['throughput_download_local_std'].iloc[start_idx:end_idx+1].dropna().values[0] if not df['throughput_download_local_std'].iloc[start_idx:end_idx+1].dropna().empty else np.nan

                    throughput_upload_local_std = initial_values.get('throughput_upload_local_std', np.nan)
                    if pd.isna(throughput_upload_local_std):
                        throughput_upload_local_std = df['throughput_upload_local_std'].iloc[startthroughput_u_idx:end_idx+1].dropna().values[0] if not df['throughput_upload_local_std'].iloc[start_idx:end_idx+1].dropna().empty else np.nan

                    rtt_download_local_std = initial_values.get('rtt_download_local_std', np.nan)
                    if pd.isna(rtt_download_local_std):
                        rtt_download_local_std = df['rtt_download_local_std'].iloc[start_idx:end_idx+1].dropna().values[0] if not df['rtt_download_local_std'].iloc[start_idx:end_idx+1].dropna().empty else np.nan

                    rtt_upload_local_std = initial_values.get('rtt_upload_local_std', np.nan)
                    if pd.isna(rtt_upload_local_std):
                        rtt_upload_local_std = df['rtt_upload_local_std'].iloc[start_idx:end_idx+1].dropna().values[0] if not df['rtt_upload_local_std'].iloc[start_idx:end_idx+1].dropna().empty else np.nan

                    event = 1 if start_idx in changepoint_indices and end_idx in changepoint_indices else 0

                    survival_data.append({
                        'client': client,
                        'site': site,
                        'timestamp_start': start_time,
                        'timestamp_end': end_time,
                        'time': duration,
                        'throughput_download': throughput_download_local_mean,
                        'rtt_download': rtt_download_local_mean,
                        'throughput_upload': throughput_upload_local_mean,
                        'rtt_upload': rtt_upload_local_mean,
                        'throughput_download_std': throughput_download_local_std,
                        'rtt_download_std': rtt_download_local_std,
                        'throughput_upload_std': throughput_upload_local_std,
                        'rtt_upload_std': rtt_upload_local_std,
                        'event': event
                    })

    # Converter para DataFrame
    survival_df = pd.DataFrame(survival_data)
    survival_df = pd.get_dummies(survival_df, columns=['client', 'site'])

    for col in survival_df.columns:
        if col.startswith('client_') or col.startswith('site_'):
            survival_df[col] = survival_df[col].astype(int)
    if multivariate:
        survival_df.to_parquet(f'datasets/survival_{data}_mv.parquet', index=False)
    else:
        survival_df.to_parquet(f'datasets/survival_{data}.parquet', index=False)
    return survival_df