import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_changepoints(data, client, site, variable, ylim=None, multivariate=False, 
                 plot_votes=True, plot_probs=True, save_fig=False):
    """
    Plota os valores de uma variável ao longo do tempo com changepoints destacados como linhas verticais,
    opcionalmente incluindo gráficos do número de votos e probabilidades dos votos.

    Parâmetros:
    ----------
    data : str
        Identificador do conjunto de dados (usado para localizar os arquivos de séries temporais).
    client : str
        Identificador do cliente.
    site : str
        Identificador do site.
    variable : str
        Nome da variável a ser plotada.
    ylim : tuple or None, optional
        Limites do eixo Y no formato (y_min, y_max). Se None, os limites serão automáticos.
    multivariate : bool
        Se True, considera os changepoints detectados com a abordagem multivariada do VWCD.
    plot_votes : bool
        Se True, inclui o gráfico com o número de votos.
    plot_probs : bool
        Se True, inclui o gráfico com as probabilidades dos votos.

    Retorna:
    -------
    None
        Exibe os gráficos selecionados alinhados verticalmente.
    """
    # Diretório onde os arquivos com changepoints estão armazenados
    if multivariate:
        input_dir = f'datasets/ts_{data}_cp_mv/'
    else:
        input_dir = f'datasets/ts_{data}_cp/'
    file_name = f"{client}_{site}.parquet"
    file_path = os.path.join(input_dir, file_name)

    if not os.path.exists(file_path):
        print(f"Arquivo para o cliente {client} e site {site} não encontrado.")
        return

    # Carregar o arquivo Parquet
    df = pd.read_parquet(file_path)

    # Verificar se todas as colunas necessárias existem
    if multivariate:
        changepoint_column = 'cp'
        votes_column = 'votes'
        probs_column = 'agg_probs'
    else:
        changepoint_column = f"{variable}_cp"
        votes_column = f"{variable}_votes"
        probs_column = f"{variable}_agg_probs"
    
    # Lista de colunas necessárias baseada nos parâmetros
    required_columns = [variable, changepoint_column]
    if plot_votes:
        required_columns.append(votes_column)
    if plot_probs:
        required_columns.append(probs_column)
    
    if not all(col in df.columns for col in required_columns):
        print(f"Uma ou mais colunas necessárias não estão disponíveis no arquivo.")
        return

    # Obter os timestamps dos changepoints
    changepoints = df['timestamp'][df[changepoint_column] == 1]

    # Determinar o número de subplots necessários
    n_plots = 1 + plot_votes + plot_probs
    
    # Criar figura com os subplots solicitados
    if n_plots > 1:
        # Se mais de um gráfico, usar proporções diferentes
        height_ratios = [3] + [2] * (n_plots - 1)
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3 + 2*n_plots),
                                gridspec_kw={'height_ratios': height_ratios})
        plt.subplots_adjust(hspace=0.3)
        
        # Garantir que axes seja sempre uma lista
        if n_plots == 1:
            axes = [axes]
    else:
        # Se apenas um gráfico, usar proporções padrão
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        axes = [ax]

    # Índice para controlar qual subplot está sendo preenchido
    current_ax = 0

    # Subplot 1: Valores da variável (sempre presente)
    axes[current_ax].plot(df['timestamp'], df[variable], label='Valores', color='gray', alpha=0.7)
    for cp in changepoints:
        axes[current_ax].axvline(x=cp, color='red', linestyle='--', 
                             label='Ponto de mudança' if cp == changepoints.iloc[0] else '')
    axes[current_ax].set_ylabel(variable)
    axes[current_ax].set_title(f"{variable} - {client} - {site}")
    axes[current_ax].legend()
    axes[current_ax].grid(True, alpha=0.3)
    if ylim:
        axes[current_ax].set_ylim(ylim)
    current_ax += 1

    # Subplot 2: Número de votos (se solicitado)
    if plot_votes and current_ax < len(axes):
        axes[current_ax].plot(df['timestamp'], df[votes_column], label='Votos', color='green', marker='o')
        for cp in changepoints:
            axes[current_ax].axvline(x=cp, color='red', linestyle='--')
        axes[current_ax].set_ylabel('Número de votos')
        axes[current_ax].legend()
        axes[current_ax].grid(True, alpha=0.3)
        axes[current_ax].set_ylim(0, 20)
        current_ax += 1

    # Subplot 3: Probabilidades dos votos (se solicitado)
    if plot_probs and current_ax < len(axes):
        axes[current_ax].plot(df['timestamp'], df[probs_column], label='Probabilidade', color='orange', marker='o')
        for cp in changepoints:
            axes[current_ax].axvline(x=cp, color='red', linestyle='--')
        axes[current_ax].set_ylabel('Probabilidade')
        axes[current_ax].legend()
        axes[current_ax].grid(True, alpha=0.3)
        axes[current_ax].set_ylim(0, 1)
        current_ax += 1

    # Configurar o rótulo do eixo x apenas no último subplot
    axes[-1].set_xlabel('Tempo')

    plt.tight_layout()
    if save_fig:
        output_dir = 'imgs'
        os.makedirs(output_dir, exist_ok=True)
        if multivariate:
            output_file = os.path.join(output_dir, f"{client}_{site}_{variable}_mv.png")
        else:
            output_file = os.path.join(output_dir, f"{client}_{site}_{variable}.png")
        plt.savefig(output_file)
    plt.show()


def plot_pairs(data, pairs, survival=True, local_mean=True, changepoints=True, 
               plot_votes=True, plot_probs=True, multivariate=False, thr_max=900,
               rtt_max=250, save_fig=False, filename=None, legend_pos='upper left'):
    """
    Plota os valores de todas as variáveis ao longo do tempo para múltiplos pares (cliente, site),
    com opção de incluir gráficos de votos e probabilidades para cada feature.
    Mantém os eixos x e y alinhados entre os gráficos.

    Parâmetros:
    ----------
    data : str
        Identificador do conjunto de dados.
    pairs : list of tuples
        Lista de pares (cliente, site) a serem plotados.
    survival : bool, default=True
        Se True, plota os valores de 'survival_probability'.
    local_mean : bool, default=True
        Se True, plota as médias locais e limites de Z-score.
    changepoints : bool, default=True
        Se True, plota as linhas verticais nos changepoints.
    plot_votes : bool, default=True
        Se True, inclui gráficos com o número de votos para cada feature.
    plot_probs : bool, default=True
        Se True, inclui gráficos com as probabilidades dos votos para cada feature.
    multivariate : bool, default=False
        Se True, usa a abordagem multivariada do VWCD.
    thr_max : int, default=900
        Valor máximo para o eixo y dos gráficos de throughput
    rtt_max : int, default=250
        Valor máximo para o eixo y dos gráficos de RTT
    save_fig : bool, default=False
        Se True, salva a figura em arquivo
    filename : str, optional
        Nome do arquivo para salvar a figura
    """
    variables = ['throughput_download', 'throughput_upload', 'rtt_download', 'rtt_upload']
    
    # Definir limites y para cada tipo de variável
    y_limits = {
        'throughput_download': (0, thr_max),
        'throughput_upload': (0, thr_max),
        'rtt_download': (0, rtt_max),
        'rtt_upload': (0, rtt_max),
        'votes': (0, 20),
        'probs': (0, 1),
        'survival': (0, 1)
    }
    
    def get_cluster_color(cluster, probability):
        """Retorna a cor do cluster com base na probabilidade."""
        if probability < 0.65:
            intensity = 0.3  # Tom mais claro
        else:
            intensity = 1.0   # Tom mais escuro

        if cluster == 0:  # Tons de vermelho
            return mcolors.to_hex((1, 1 - intensity, 1 - intensity))
        else:  # Tons de azul
            return mcolors.to_hex((1 - intensity, 1 - intensity, 1))

    def create_color_legend():
        """Cria uma legenda de cores mostrando os três tons para cada cluster."""
        legend_elements = []
        
        # Cluster 0 (vermelho)
        legend_elements.append(plt.Line2D([0], [0], color=get_cluster_color(0, 0.5), 
                            label=f'Cluster 0: P < 0.65'))
        legend_elements.append(plt.Line2D([0], [0], color=get_cluster_color(0, 0.9), 
                            label=f'Cluster 0: P >= 0.65'))
        
        # Cluster 1 (azul)
        legend_elements.append(plt.Line2D([0], [0], color=get_cluster_color(1, 0.5), 
                            label=f'Cluster 1: P < 0.65'))
        legend_elements.append(plt.Line2D([0], [0], color=get_cluster_color(1, 0.9), 
                            label=f'Cluster 1: P >= 0.65'))
        
        return legend_elements
    
    # Calcular número de subplots por feature
    plots_per_feature = 1 + (1 if plot_votes else 0) + (1 if plot_probs else 0)
    
    # Calcular número total de linhas por par
    rows_per_pair = len(variables) * plots_per_feature + (1 if survival else 0)
    
    # Calcular número de colunas e linhas necessárias
    max_cols = 5
    num_pairs = len(pairs)
    num_pair_rows = (num_pairs + max_cols - 1) // max_cols  # Arredonda para cima
    
    # Criar figura com tamanho apropriado
    fig = plt.figure(figsize=(24 * min(num_pairs, max_cols), 3 * rows_per_pair * num_pair_rows))
    
    # Criar grid de subplots
    gs = fig.add_gridspec(rows_per_pair * num_pair_rows, max_cols, hspace=0.5)
    
    # Dicionário para armazenar os axes
    axes_dict = {}
    
    # Lista para armazenar todos os timestamps
    all_timestamps = []
    
    # Primeiro, ler todos os dados para determinar o range global do eixo x
    for client, site in pairs:
        if multivariate:
            input_dir = f'datasets/ts_{data}_results_mv'
        else:
            input_dir = f'datasets/ts_{data}_results'
        file_path = os.path.join(input_dir, f"{client}_{site}.parquet")
        
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            all_timestamps.extend(df['timestamp'])
    
    # Determinar limites globais do eixo x
    x_min, x_max = min(all_timestamps), max(all_timestamps)
    
    # Criar axes para cada par e suas variáveis
    for pair_idx, (client, site) in enumerate(pairs):
        # Calcular posição do par atual na grade
        pair_row = pair_idx // max_cols
        pair_col = pair_idx % max_cols
        
        # Offset base para as linhas deste par
        base_row_offset = pair_row * rows_per_pair
        
        # Criar axes para cada variável e seus gráficos associados
        for var_idx, variable in enumerate(variables):
            base_row = base_row_offset + var_idx * plots_per_feature
            
            # Subplot principal da feature
            ax_main = fig.add_subplot(gs[base_row, pair_col])
            axes_dict[(variable, 'main', pair_idx)] = ax_main
            ax_main.set_xlim(x_min, x_max)
            ax_main.set_ylim(*y_limits[variable])
            
            if plot_votes:
                ax_votes = fig.add_subplot(gs[base_row + 1, pair_col])
                axes_dict[(variable, 'votes', pair_idx)] = ax_votes
                ax_votes.set_xlim(x_min, x_max)
                ax_votes.set_ylim(*y_limits['votes'])
                
            if plot_probs:
                ax_probs = fig.add_subplot(gs[base_row + (2 if plot_votes else 1), pair_col])
                axes_dict[(variable, 'agg_probs', pair_idx)] = ax_probs
                ax_probs.set_xlim(x_min, x_max)
                ax_probs.set_ylim(*y_limits['probs'])
        
        # Criar axis para survival se necessário
        if survival:
            survival_row = base_row_offset + rows_per_pair - 1
            ax_surv = fig.add_subplot(gs[survival_row, pair_col])
            axes_dict[('survival', 'main', pair_idx)] = ax_surv
            ax_surv.set_xlim(x_min, x_max)
            ax_surv.set_ylim(*y_limits['survival'])
    
    # Plotar dados para cada par
    for pair_idx, (client, site) in enumerate(pairs):
        if multivariate:
            input_dir = f'datasets/ts_{data}_results_mv'
        else:
            input_dir = f'datasets/ts_{data}_results'
        file_name = f"{client}_{site}.parquet"
        file_path = os.path.join(input_dir, file_name)

        if not os.path.exists(file_path):
            print(f"Arquivo para o cliente {client} e site {site} não encontrado.")
            continue

        df = pd.read_parquet(file_path)
        df = df.sort_values('timestamp')

        # Aplicar a função de cor baseada no cluster e probabilidade
        df['color'] = df.apply(lambda row: get_cluster_color(
            row['cluster'], 
            row['cluster_probability'] if 'cluster_probability' in row else 1.0
        ), axis=1)

        linestyles = {'local_mean': '--', 'changepoints': ':'}

        # Plotar cada variável e seus gráficos associados
        for variable in variables:
            ax_main = axes_dict[(variable, 'main', pair_idx)]
            
            changepoint_column = 'cp' if multivariate else f"{variable}_cp"
            votes_column = 'votes' if multivariate else f"{variable}_votes"
            probs_column = 'agg_probs' if multivariate else f"{variable}_agg_probs"

            # Plotar série temporal com cores baseadas na probabilidade
            for i in range(len(df) - 1):
                ax_main.plot(df['timestamp'].iloc[i:i+2], df[variable].iloc[i:i+2],
                           color=df['color'].iloc[i], linewidth=1)

            # Adicionar legenda de cores apenas no primeiro gráfico do primeiro par
            if pair_idx == 1 and variable == variables[0]:
                legend_elements = create_color_legend()
                ax_main.legend(handles=legend_elements, 
                             loc=legend_pos,  # Posição dentro do gráfico
                             prop={'size': 8},
                             labelspacing=0.2,
                             handlelength=1,
                             handletextpad=0.2
                             )

            if local_mean and f'{variable}_local_mean' in df.columns:
                ax_main.plot(df['timestamp'], df[f'{variable}_local_mean'],
                           linestyle=linestyles['local_mean'],
                           color='black', alpha=0.7, label='Média local')
                ax_main.fill_between(df['timestamp'],
                                   df[f'{variable}_local_mean'] - 2 * df[f'{variable}_local_std'],
                                   df[f'{variable}_local_mean'] + 2 * df[f'{variable}_local_std'],
                                   color='black', alpha=0.1)

            if changepoints and changepoint_column in df.columns:
                changepoints_timestamps = df['timestamp'][df[changepoint_column] == 1]
                for cp in changepoints_timestamps:
                    ax_main.axvline(x=cp, color='green', linestyle=linestyles['changepoints'], alpha=0.8)

            ax_main.set_ylabel(variable)
            ax_main.grid(True, alpha=0.3)
            
            # Plotar votos se solicitado
            if plot_votes and votes_column in df.columns:
                ax_votes = axes_dict[(variable, 'votes', pair_idx)]
                ax_votes.plot(df['timestamp'], df[votes_column], color='green', label='Votos')
                ax_votes.set_ylabel('Votos')
                ax_votes.grid(True, alpha=0.3)
                if changepoints:
                    for cp in changepoints_timestamps:
                        ax_votes.axvline(x=cp, color='green', linestyle=linestyles['changepoints'], alpha=0.8)
            
            # Plotar probabilidades se solicitado
            if plot_probs and probs_column in df.columns:
                ax_probs = axes_dict[(variable, 'agg_probs', pair_idx)]
                ax_probs.plot(df['timestamp'], df[probs_column], color='orange', label='Probabilidade')
                ax_probs.set_ylabel('Probabilidade')
                ax_probs.grid(True, alpha=0.3)
                if changepoints:
                    for cp in changepoints_timestamps:
                        ax_probs.axvline(x=cp, color='green', linestyle=linestyles['changepoints'], alpha=0.8)

        # Adicionar título para a coluna
        axes_dict[(variables[0], 'main', pair_idx)].set_title(f'{client} - {site}', pad=15)

        # Plotar survival se solicitado
        if survival:
            ax_surv = axes_dict[('survival', 'main', pair_idx)]
            ax_surv.plot(df['timestamp'], df['survival_probability'],
                        color='purple', label='Survival', linewidth=1)
            ax_surv.set_xlabel('Tempo')
            ax_surv.set_ylabel('Survival')
            ax_surv.grid(True, alpha=0.3)

    # Adicionar legendas apenas na última coluna de cada linha
    for row in range(num_pair_rows):
        last_col_in_row = min((row + 1) * max_cols, len(pairs)) - 1
        for variable in variables:
            if variable != variables[0]:  # Não adicionar legenda onde já temos a legenda de cores
                axes_dict[(variable, 'main', last_col_in_row)].legend()
        if survival:
            axes_dict[('survival', 'main', last_col_in_row)].legend()

    plt.tight_layout()
    if save_fig:
        output_dir = 'imgs'
        os.makedirs(output_dir, exist_ok=True)
        if multivariate:
            output_file = os.path.join(output_dir, f"{filename}_mv.png")
        else:
            output_file = os.path.join(output_dir, f"{filename}.png")
        plt.savefig(output_file, bbox_inches='tight')
    plt.show()


def plot_decrement(clients, metric, decrement=True, multivariate=False, save_fig=False, filename=None):
    """
    Analisa os pontos de mudança (cp) em arquivos Parquet para uma lista de clientes e uma métrica 
    e plota gráficos do decremento ou incremento da métrica lado a lado.

    Args:
        clients (list of str): Lista com os nomes dos clientes para identificar os arquivos.
        metric (str): Nome da métrica a ser analisada.
        decrement (bool): Indica se a análise é de decremento ou incremento.
        multivariate (bool): Indica se a análise é multivariada.

    Returns:
        dict: Dicionário com as contagens de pontos de mudança por threshold para cada cliente.
    """
    # Diretório dos datasets
    if multivariate:
        dataset_dir = 'datasets/ts_ndt_results_mv'
    else:
        dataset_dir = 'datasets/ts_ndt_results'

    # Coluna de changepoints
    cp_col = 'cp' if multivariate else f'{metric}_cp'

    # Inicializa o dicionário de resultados
    all_results = {}

    # Configura o layout dos subplots
    num_clients = len(clients)
    fig, axes = plt.subplots(1, num_clients, figsize=(5 * num_clients, 4), sharey=True)

    if num_clients == 1:  # Caso especial para apenas um cliente
        axes = [axes]

    # Itera sobre cada cliente
    for idx, client in enumerate(clients):
        # Dicionário para armazenar os resultados por threshold para o cliente
        threshold_counts = {i: 0 for i in range(1, 101)}  # Inicializa thresholds de 1 a 100
        decrementos = []

        # Identifica os arquivos Parquet do cliente
        files = [f for f in os.listdir(dataset_dir) if f.startswith(client) and f.endswith('.parquet')]

        # Itera sobre cada arquivo
        for file in files:
            file_path = os.path.join(dataset_dir, file)
            df = pd.read_parquet(file_path)

            # Itera sobre as entradas do DataFrame
            for i in range(1, len(df)):
                if df.loc[i, cp_col] == 1:  # Verifica se é um ponto de mudança
                    current_mean = df.loc[i-1, f'{metric}_local_mean']
                    next_mean = df.loc[i, f'{metric}_local_mean']

                    if next_mean < current_mean:
                        diff = round(current_mean - next_mean)
                        if diff > 0:  # Armazena todos os decrementos positivos
                            decrementos.append(diff)

        # Calcula as contagens para cada threshold
        for threshold in range(1, 101):
            if decrement:
                threshold_counts[threshold] = sum(1 for d in decrementos if d >= threshold)
            else:
                threshold_counts[threshold] = sum(1 for d in decrementos if d <= threshold)

        # Salva os resultados no dicionário
        all_results[client] = threshold_counts

        # Plota o gráfico no subplot correspondente
        axes[idx].plot(threshold_counts.keys(), threshold_counts.values(), linestyle='-', color='green')
        axes[idx].set_title(f'{client}')
        axes[idx].set_xlabel('Decremento (Mbits/s)')
        axes[idx].set_xticks(range(0, 125, 25))
        axes[idx].set_xlim([0, 100])
        axes[idx].grid(linestyle=':')
        
        if idx == 0:  # Apenas no primeiro gráfico
            axes[idx].set_ylabel('Pontos de mudança')
            axes[idx].set_yticks(range(0, 30, 2))
            axes[idx].set_ylim([0, 10])

    # Ajusta o layout da figura
    plt.tight_layout()
    if save_fig:
        output_dir = 'imgs'
        os.makedirs(output_dir, exist_ok=True)
        if multivariate:
            output_file = os.path.join(output_dir, f"{filename}_mv.png")
        else:
            output_file = os.path.join(output_dir, f"{filename}.png")
        plt.savefig(output_file)
    plt.show()

    return None


def plot_decrement_dual(client_list1, client_list2, metric, decrement=True, multivariate=False, save_fig=False, filename=None):
    """
    Analisa os pontos de mudança (cp) em arquivos Parquet para duas listas de clientes e uma métrica, 
    e plota gráficos em duas colunas: uma para cada lista de clientes.

    Args:
        client_list1 (list of str): Primeira lista de clientes (plotada na primeira coluna, cor vermelha).
        client_list2 (list of str): Segunda lista de clientes (plotada na segunda coluna, cor azul).
        metric (str): Nome da métrica a ser analisada.
        decrement (bool): Indica se a análise é de decremento ou incremento.
        multivariate (bool): Indica se a análise é multivariada.

    Returns:
        dict: Dicionário com as contagens de pontos de mudança por threshold para cada cliente.
    """
    if len(client_list1) != len(client_list2):
        raise ValueError("As duas listas de clientes devem ter o mesmo tamanho.")

    # Diretório dos datasets
    if multivariate:
        dataset_dir = 'datasets/ts_ndt_results_mv'
    else:
        dataset_dir = 'datasets/ts_ndt_results'

    # Coluna de changepoints
    cp_col = 'cp' if multivariate else f'{metric}_cp'

    # Inicializa o dicionário de resultados
    all_results = {}

    # Configura o layout dos subplots
    num_clients = len(client_list1)
    fig, axes = plt.subplots(num_clients, 2, figsize=(10, 4 * num_clients), sharex=True, sharey=True)

    for row, (client1, client2) in enumerate(zip(client_list1, client_list2)):
        for col, (client, color) in enumerate(zip([client1, client2], ['red', 'blue'])):
            # Dicionário para armazenar os resultados por threshold para o cliente
            threshold_counts = {i: 0 for i in range(1, 101)}  # Inicializa thresholds de 1 a 100
            decrementos = []

            # Identifica os arquivos Parquet do cliente
            files = [f for f in os.listdir(dataset_dir) if f.startswith(client) and f.endswith('.parquet')]

            # Itera sobre cada arquivo
            for file in files:
                file_path = os.path.join(dataset_dir, file)
                df = pd.read_parquet(file_path)

                # Itera sobre as entradas do DataFrame
                for i in range(1, len(df)):
                    if df.loc[i, cp_col] == 1:  # Verifica se é um ponto de mudança
                        current_mean = df.loc[i-1, f'{metric}_local_mean']
                        next_mean = df.loc[i, f'{metric}_local_mean']

                        if next_mean < current_mean:
                            diff = round(current_mean - next_mean)
                            if diff > 0:  # Armazena todos os decrementos positivos
                                decrementos.append(diff)

            # Calcula as contagens para cada threshold
            for threshold in range(1, 101):
                if decrement:
                    threshold_counts[threshold] = sum(1 for d in decrementos if d >= threshold)
                else:
                    threshold_counts[threshold] = sum(1 for d in decrementos if d <= threshold)

            # Salva os resultados no dicionário
            all_results[client] = threshold_counts

            # Plota o gráfico no subplot correspondente
            axes[row, col].plot(threshold_counts.keys(), threshold_counts.values(), linestyle='-', color=color)
            axes[row, col].set_title(f'{client}')
            axes[row, col].set_xlabel('Decremento (Mbits/s)')
            axes[row, col].grid(linestyle=':')

            if col == 0:  # Apenas na primeira coluna
                axes[row, col].set_ylabel('Num. de mudanças')

    # Ajusta o layout da figura
    plt.tight_layout()
    if save_fig:
        output_dir = 'imgs'
        os.makedirs(output_dir, exist_ok=True)
        if multivariate:
            output_file = os.path.join(output_dir, f"{filename}_mv.png")
        else:
            output_file = os.path.join(output_dir, f"{filename}.png")
        plt.savefig(output_file)
    plt.show()

    return None