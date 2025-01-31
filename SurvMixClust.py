"""
Autor do algoritmo e do código original: Gabriel Cesário Buginga (2024).
"""
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pandas as pd
from SurvMixClust_utils import X_Xd_from_set, surv_df_from_model,\
    labels_from_model, best_model_keys_from_info_list_K, plot_kms_clusters, cluster_EM,\
    select_bandwidth


class SurvMixClust(BaseEstimator):
    """
    SurvMixClust é um algoritmo de agrupamento para análise de sobrevivência.
    """

    def __init__(self, n_clusters, n_fits=4, max_EM_interations=60, n_jobs=None):
        """
        Parâmetros
        ----------
        n_clusters : int ou lista de ints
            Se for um inteiro, representa o número de clusters que o modelo usará para treino e inferência.
            Se for uma lista de inteiros, o modelo realizará ajustes para cada número de clusters na lista,
            selecionando o melhor ajuste. Mais detalhes na função `model.fit`.
        n_fits : int, opcional
            Número de execuções completamente distintas do algoritmo EM para cada número de clusters em `n_clusters`. 
            O padrão é 5.
        max_EM_interations : int, opcional
            Número máximo de iterações para cada execução completa do algoritmo EM. O padrão é 60.
            O algoritmo EM será interrompido antes de atingir o limite máximo caso a métrica c-index pare de melhorar.
        n_jobs : int, opcional
            Define o número de núcleos a serem usados com `joblib.Parallel`. Se for `None`, o modelo será ajustado de 
            forma sequencial (padrão). Se for `-1`, utiliza todos os núcleos disponíveis. Um número inteiro positivo 
            especifica a quantidade de núcleos a ser utilizada.
        """

        if isinstance(n_clusters, int):
            self.n_clusters = [n_clusters]
        else:
            self.n_clusters = n_clusters
        self.n_fits = n_fits
        self.max_EM_interations = max_EM_interations
        self.n_jobs = n_jobs


        self.global_fixed_bw = (0, 0)
        self.time_max = 0

    def fit(self, X, y):
        """
        Ajusta o modelo ao conjunto de treinamento (X, y).
        O modelo divide os dados em conjunto de treino e validação, gera `n_fits` para cada número de clusters em 
        `n_clusters` e seleciona o melhor modelo com base no desempenho no conjunto de validação.

        Parâmetros
        ----------
        X : np.array ou pd.DataFrame
            Array com formato (n_samples, n_features) contendo os dados de entrada. 
            O modelo funciona melhor com dados já normalizados.
        y : np.array
            Array de tuplas (tempo, evento) contendo o tempo e o status do evento para cada amostra.
            Pode ser gerado usando a função `datasets.y_array(E, T)`.
        """

        # Check that X and y have correct shape.
        X, y = check_X_y(X, y)

        X = pd.DataFrame(data=X)
        data_set = pd.DataFrame(data=X)
        data_set['time'] = y['time']
        data_set['event'] = y['event']

        self.time_max = y['time'].max()

        features_used = data_set.columns[:]
        X, Xd = X_Xd_from_set(data_set, features_used)

        self.global_fixed_bw = select_bandwidth(X)
#         self.global_fixed_bw = [17.52130828203924, 12.733703749619613] # support
        # self.global_fixed_bw = [19.610724293026085,
        #                         65.89859497742401]  # metabric

        n_clusters = self.n_clusters
        number_parallel_tries = self.n_fits
        n_iterations = self.max_EM_interations
        n_jobs = self.n_jobs
        global_fixed_bw = self.global_fixed_bw

        info_list_K = dict()

        for n_cluster in n_clusters:

            if n_jobs is None:
                print(f"Non-parallel processing for {n_cluster} number of clusters.")
                list_list_dicts = list()

                for _ in range(number_parallel_tries):

                    info_list = cluster_EM(X, Xd, X_val, Xd_val,
                                               n_cluster,
                                               n_iterations, global_fixed_bw)
                    list_list_dicts.append(info_list)

                import functools
                info_list = functools.reduce(lambda a, b: a+b, list_list_dicts)

            else:
                print(f"Parallel processing for {n_cluster} number of clusters, n_jobs: {self.n_jobs}.")
                list_list_dicts = Parallel(n_jobs=n_jobs)(delayed(cluster_EM)(X, Xd, X, Xd,
                                                                                  n_cluster,
                                                                                  n_iterations, global_fixed_bw)
                                                          for ite_gen in np.arange(number_parallel_tries))

                import functools
                info_list = functools.reduce(lambda a, b: a+b, list_list_dicts)

            from copy import deepcopy
            info_list_K[(n_cluster)] = deepcopy(info_list)

        sub_experiment_key, idx = best_model_keys_from_info_list_K(info_list_K)

        self.info_list_K = info_list_K

        self.info = info_list_K[sub_experiment_key][idx]
        self.kmfs = info_list_K[sub_experiment_key][idx]['kmfs']
        self.logit = info_list_K[sub_experiment_key][idx]['logit']
        self.global_fixed_bw = info_list_K[sub_experiment_key][idx]['global_fixed_bw']

        self.X_ = 'X'
        self.y_ = 'y'

        # Return the model
        return self

    def predict_surv_df(self, X, n_cluster=None, n_fit=None):
        """
        Utiliza o melhor modelo obtido em `model.fit` para prever a função de sobrevivência para cada amostra em X.
        Permite escolher um modelo específico da lista gerada em `model.fit` especificando `n_cluster` e `n_fit`.

        Parâmetros
        ----------
        X : np.array ou pd.DataFrame
            Array com formato (n_samples, n_features) contendo as amostras. 
            O modelo funciona melhor com dados já normalizados.
        n_cluster : int, opcional
            Número de clusters do modelo específico a ser usado. Se `None`, usa o melhor modelo obtido em `model.fit`.
        n_fit : int, opcional
            Índice da execução do ajuste a ser usado. Se `None`, usa o melhor modelo obtido em `model.fit`.

        Retorna
        -------
        pd.DataFrame
            DataFrame contendo a função de sobrevivência prevista para cada amostra em X. Os índices representam o 
            tempo, e as colunas indicam as amostras.
        """

        # Check if fit had been called.
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X).astype('float32')

        Xd_target = pd.DataFrame(data=X)
    
        if (n_cluster is None) and (n_fit is None):
            surv_df = surv_df_from_model(
            Xd_target.copy(), self.logit, self.kmfs)
            
            return surv_df
        
        else:
            surv_df = surv_df_from_model(
            Xd_target.copy(), self.info_list_K[n_cluster][n_fit]['logit'], self.info_list_K[n_cluster][n_fit]['kmfs'])

            return surv_df

    def predict(self, X, n_cluster=None, n_fit=None):
        """
        Prediz os rótulos clusterizados para cada amostra em X utilizando o modelo ajustado em `model.fit`.

        Parâmetros
        ----------
        X : np.array ou pd.DataFrame
            Array com formato (n_samples, n_features) contendo as amostras. 
            O modelo funciona melhor com dados já normalizados.
        n_cluster : int, opcional
            Número de clusters do modelo específico a ser usado. Se `None`, usa o melhor modelo obtido em `model.fit`.
        n_fit : int, opcional
            Índice da execução do ajuste a ser usado. Se `None`, usa o melhor modelo obtido em `model.fit`.

        Retorna
        -------
        np.array
            Rótulos clusterizados para cada amostra em X.
        """

        # Check if fit had been called.
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation.
        X = check_array(X).astype('float32')

        Xd_target = pd.DataFrame(data=X)

        if (n_cluster is None) and (n_fit is None):
            labels = labels_from_model(
            Xd_target.copy(), self.logit)
            
            return labels
        
        else:
            labels = labels_from_model(
            Xd_target.copy(), self.info_list_K[n_cluster][n_fit]['logit'])

            return labels

    def training_set_labels(self):
        """
        Retorna os rótulos clusterizados para cada amostra no conjunto de treinamento, utilizando o melhor modelo 
        obtido em `model.fit`. Esses rótulos foram gerados pela última iteração do algoritmo EM.

        Retorna
        -------
        np.array
            Rótulos clusterizados para o conjunto de treinamento.
        """
        
        # Check if fit had been called.
        check_is_fitted(self, ['X_', 'y_'])           
        
        return self.info['new_labels']
    
    def score(self, X, y, metric='cindex', n_cluster=None, n_fit=None):
        """
        Calcula a pontuação do modelo utilizando X e y fornecidos.
        Suporta as métricas c-index e Integrated Brier Score (IBS).

        Parâmetros
        ----------
        X : np.array ou pd.DataFrame
            Dados de entrada, com formato (n_samples, n_features). 
            O modelo funciona melhor com dados já normalizados.
        y : np.array
            Array de tuplas (tempo, evento) contendo o tempo e o status do evento para cada amostra.
        n_cluster : int, opcional
            Número de clusters do modelo específico a ser usado. Se `None`, usa o melhor modelo obtido em `model.fit`.
        n_fit : int, opcional
            Índice da execução do ajuste a ser usado. Se `None`, usa o melhor modelo obtido em `model.fit`.
        metric : str, opcional
            Métrica a ser utilizada. `cindex` para c-index (padrão) e `ibs` para Integrated Brier Score.

        Retorna
        -------
        float
            Pontuação do modelo para os dados fornecidos.
        """

        assert (metric in ['cindex', 'c-index',
                'integrated brier score', 'ibs'])
        

        if (n_cluster is None) and (n_fit is None):
            surv_df = self.predict_surv_df(X)
            
        else:
            surv_df = self.predict_surv_df(X, n_cluster=n_cluster, n_fit=n_fit)
        
        
        from pycox.evaluation import EvalSurv
        durations_test, events_test = \
            (y['time']).astype('float32'), (y['event']).astype('float32')
        ev = EvalSurv(surv_df, durations_test, events_test, censor_surv='km')

        if metric in ['cindex', 'c-index']:
            cindex = ev.concordance_td()
            metric_score = cindex
        else:
            time_grid = np.linspace(
                durations_test.min(), durations_test.max(), 100)
            ibs = ev.integrated_brier_score(time_grid)
            metric_score = ibs

        return metric_score

    def plot_clusters(self, n_cluster=None, n_fit=None, show_n_samples=True):
        """
        Gera um gráfico dos resultados clusterizados para o melhor modelo obtido em `model.fit`, 
        ou para o modelo específico definido por `n_cluster` e `n_fit`.

        Parâmetros
        ----------
        n_cluster : int, opcional
            Número de clusters do modelo específico a ser mostrado. Se `None`, utiliza o melhor modelo.
        n_fit : int, opcional
            Índice da execução do ajuste a ser mostrado. Se `None`, utiliza o melhor modelo.
        show_n_samples : bool, opcional
            Se `True`, exibe o número de amostras em cada cluster. O padrão é `True`.

        Retorna
        -------
        plotly.graph_objs._figure.Figure
            Figura do Plotly com os resultados clusterizados. Os intervalos de confiança são obtidos pelo estimador 
            de Kaplan-Meier.
        """

        if (n_cluster is None) and (n_fit is None):
            fig = plot_kms_clusters(
                self.info, self.time_max, show_n_samples=show_n_samples)
        else:
            fig = plot_kms_clusters(
                self.info_list_K[n_cluster][n_fit], self.time_max, show_n_samples=show_n_samples)

        return fig