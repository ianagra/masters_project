"""
Autor do algoritmo e do código original: Gabriel Cesário Buginga (2024).
"""
import time
from datetime import timedelta
import numpy as np
import pandas as pd
from copy import deepcopy
from lifelines import KaplanMeierFitter

def load_obj(name):
    """
    Carrega um objeto armazenado em um arquivo pickle.

    Parâmetros
    ----------
    name : str
        Nome do arquivo (sem extensão) localizado na pasta 'datasets'.

    Retorna
    -------
    object
        Objeto armazenado no arquivo pickle.

    Notas
    -----
    - Útil para carregar dados ou modelos previamente armazenados.
    """
    import pickle
    with open('./datasets/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def y_array(E, T):
    """
    Gera um array estruturado contendo status do evento e tempo para análise de sobrevivência.

    Parâmetros
    ----------
    E : np.array
        Array de valores booleanos indicando se o evento ocorreu (True) ou foi censurado (False).
    T : np.array
        Array de valores float indicando o tempo até o evento ou censura.

    Retorna
    -------
    np.array
        Array estruturado com campos 'event' (booleano) e 'time' (float).

    Notas
    -----
    - Os campos podem ser acessados como y['event'] e y['time'].
    """
    y = [(bool(x), y) for x,y in zip(E,T)]
    y = np.array(y ,  dtype=[('event', '?'), ('time', '<f8')])
    return y

def X_Xd_from_set(subset, features_used):
    """
    Separa um conjunto de dados em X (dados com 'time' e 'event') e Xd (dados sem esses campos).

    Parâmetros
    ----------
    subset : pd.DataFrame
        Conjunto de dados contendo as colunas 'time', 'event' e outras características.
    features_used : list
        Lista de nomes de colunas a serem consideradas.

    Retorna
    -------
    tuple
        (X, Xd): 
        - X: DataFrame com as colunas 'time', 'event' e as características especificadas.
        - Xd: DataFrame com apenas as características especificadas (sem 'time' e 'event').
    """
    
    subset = (subset[[c for c in subset.columns if c in features_used or c in ["time", "event"]]]).reset_index(drop=True)
    subset_d = subset[[n for n in subset.columns if n not in ["time", "event"]]].copy()
    
    return subset, subset_d

def approximate_derivative(t, kmf, global_fixed_bw):
    """
    Aproxima a derivada da função de sobrevivência usando o pacote R `survPresmooth`.

    Parâmetros
    ----------
    t : np.array
        Valores de tempo para calcular a derivada.
    kmf : KaplanMeierFitter
        Objeto KaplanMeierFitter ajustado contendo a função de sobrevivência.
    global_fixed_bw : np.array
        Largura de banda retornada pela função `select_bandwidth`.

    Retorna
    -------
    dict
        Dicionário contendo a estimativa da derivada e a largura de banda utilizada.
    """

    import rpy2.robjects.packages as rpackages
    import rpy2.robjects as robjects
    survPresmooth = rpackages.importr('survPresmooth')
    r_presmooth = survPresmooth.presmooth

    x_est = robjects.FloatVector(t.copy())

    t = robjects.FloatVector(np.array(kmf.durations))
    delta = robjects.FloatVector(np.array(kmf.event_observed))

    X_ = pd.DataFrame({'time': t, 'event':delta})
    t = robjects.FloatVector(X_['time'].values)
    delta = robjects.FloatVector(X_['event'].values)


    fixed_bw = robjects.FloatVector(np.array(global_fixed_bw))
    r_ps = r_presmooth(t, delta, estimand = 'f', bw_selec = 'fixed', fixed_bw = fixed_bw, x_est = x_est) 


    return { 'estimate' : np.array(r_ps.rx2['estimate']), 'bandwidth': list(r_ps.rx2['bandwidth'])}

def select_bandwidth(X, frac=1.):
    """
    Seleciona a largura de banda para estimativas não paramétricas usando o pacote R `survPresmooth`.

    Parâmetros
    ----------
    X : np.array ou pd.DataFrame
        Conjunto de dados com as colunas 'time' e 'event'.
    frac : float, opcional
        Fração de dados a ser usada para calcular a largura de banda. O padrão é 1 (100%).

    Retorna
    -------
    list
        Lista contendo a largura de banda calculada.

    Notas
    -----
    - Se o conjunto de dados for muito grande (> 5000 amostras), é feito um subsample automático.
    """
    
    if len(X) > 5000:
        print(f"\n{4*'#'} Dataset is unecessarily big for bandwidth calculation, subsampling with replacement to 1000. {4*'#'}")
        X = X.sample(n=1000, replace=False).copy()
    
    import rpy2.robjects.packages as rpackages
    import rpy2.robjects as robjects
    survPresmooth = rpackages.importr('survPresmooth')
    r_presmooth = survPresmooth.presmooth

    print(f"\n{8*'#'} Selecting bandwidth {8*'#'}")

    try: 

        print(f"\n{4*'#'} plug-in estimate with 100% of data {4*'#'}")

        X_ = X.copy()

        t = robjects.FloatVector(X_['time'].values)
        delta = robjects.FloatVector(X_['event'].values)

        x_est_python = np.linspace(X_['time'].min(), (X_['time'].max()), num=500)
        x_est = robjects.FloatVector(x_est_python.copy())

        # Only meaningful for density and hazard function estimation. Internally computed when NULL, the default
        r_ps = r_presmooth(t, delta, estimand = 'f', bw_selec = 'plug-in', x_est = x_est)

        print(f"{4*'#'} success {4*'#'}\n")

        return list(r_ps.rx2['bandwidth'])

    except Exception as error:

        print(f"{4*'#'} exception of type {type(error).__name__} ocurred with 100% of data {4*'#'}\n")
        
        max_number_tries = 50
         
        for n_tries in range(max_number_tries):

            try:

                print(f"{4*'#'} plug-in estimate with sub-sampled {int(frac*100)}% of data (n_tries: {n_tries+1}) {4*'#'}")

                if frac == 1.:
                    print(f"{4*'#'} sampling 100% with replacement (n_tries: {n_tries+1}) {4*'#'}")
                    X_ = X.sample(n=int(len(X)), replace=True).copy()
                if frac < 1. :
                    print(f"{4*'#'} sampling {int(frac*100)}% without replacement (n_tries: {n_tries+1}) {4*'#'}")
                    X_ = X.sample(n=int(len(X)*frac), replace=False).copy()                       
                
                t = robjects.FloatVector(X_['time'].values)
                delta = robjects.FloatVector(X_['event'].values)

                x_est_python = np.linspace(X_['time'].min(), (X_['time'].max()), num=500)
                x_est = robjects.FloatVector(x_est_python.copy())

                # Only meaningful for density and hazard function estimation. Internally computed when NULL, the default
                r_ps = r_presmooth(t, delta, estimand = 'f', bw_selec = 'plug-in', x_est = x_est)

                print(f"{4*'#'} success for {list(r_ps.rx2['bandwidth'])} {4*'#'}\n")

                return list(r_ps.rx2['bandwidth'])

            except Exception as sub_error:

                print(f"{4*'#'} exception of type {type(sub_error).__name__} ocurred {4*'#'}\n")
                n_tries += 1
                
def cindex_km_from_model(X_target, Xd_target, logit, kmfs, global_fixed_bw):
    """
    Calcula métricas de desempenho (c-index, Integrated Brier Score, etc.) para o modelo de sobrevivência.

    Parâmetros
    ----------
    X_target : np.array ou pd.DataFrame
        Dados de entrada incluindo 'time' e 'event'.
    Xd_target : np.array ou pd.DataFrame
        Dados de entrada sem 'time' e 'event'.
    logit : LogisticRegression
        Modelo de regressão logística treinado para prever probabilidades dos clusters.
    kmfs : dict
        Dicionário com estimativas Kaplan-Meier para cada cluster.
    global_fixed_bw : np.array
        Largura de banda calculada.

    Retorna
    -------
    dict
        Dicionário contendo:
        - 'cindex': Métrica c-index.
        - 'integrated_brier_score': Integrated Brier Score.
        - 'integrated_nbll': Integrated Negative Binomial Log-likelihood.
    """
    X = X_target.copy()
    Xd = Xd_target.copy()

    logit_proba = logit.predict_proba(Xd)
    ordered_labels = list(np.sort(logit.classes_))

    list_rns = list()
    for l in ordered_labels:

        def survival_function(t):
            return (kmfs[l]['kmf']).survival_function_at_times(t).values            

        X[f'label_{l}_logit'] = logit_proba[:,l]

        t = X['time'].values

        a_d = approximate_derivative(t, kmf = kmfs[l]['kmf'], global_fixed_bw=global_fixed_bw)
        X[f'label_{l}_negative_derivative_survival'] = a_d['estimate']

        X[f'label_{l}_survival'] = survival_function(X['time'].values)

        X[f'label_{l}_rn'] = (X[f'label_{l}_negative_derivative_survival']**(X[f'event']))*\
                             (X[f'label_{l}_survival']**(1-X[f'event']))*\
                             (X[f'label_{l}_logit'])

        list_rns.append(f'label_{l}_rn')

    label_rn_list = [f'label_{l}_rn' for l in ordered_labels]
    ss = X[label_rn_list].sum(axis=1)

    surv_df_list = [kmfs[l]['kmf'].survival_function_ for l in ordered_labels]

    ee = surv_df_list
    indexes = ee[0].index
    for e in ee[1:]:
        indexes = indexes.union(e.index)

    for i,e in enumerate(ee):        
        ee[i] = e.reindex(indexes,method = 'ffill').bfill()        

    idx_all_zero = X[label_rn_list].sum(axis=1)==0

    X.loc[idx_all_zero,label_rn_list] = [1/len(ordered_labels)]*len(ordered_labels)    
    ss.loc[idx_all_zero] = 1

    import functools
    kmfs_list = [functools.reduce(lambda a, b: a+b, [ee[l]*(X[label_rn_list[l]].loc[idx]/ss.loc[idx]) for l in ordered_labels]) for idx in X.index]

    surv_df = pd.concat(kmfs_list, axis=1, ignore_index=True).\
                            bfill().ffill()
    
    # With surv_df we can calculate all needed metrics.
    from pycox.evaluation import EvalSurv
    durations_test, events_test = (X['time']).values.astype('float32'), (X['event']).values.astype('float32') 
    ev = EvalSurv(surv_df, durations_test, events_test, censor_surv='km')
    c_index = ev.concordance_td()
    time_grid = np.linspace(X['time'].min(), X['time'].max(), 200)
    integrated_brier_score = ev.integrated_brier_score(time_grid)
    integrated_nbll = ev.integrated_nbll(time_grid)
 
    return {'cindex': c_index, 'integrated_brier_score': integrated_brier_score, 'integrated_nbll': integrated_nbll}  


def surv_df_from_model(Xd_target, logit, kmfs):
    """
    Gera a função de sobrevivência prevista para cada amostra no conjunto de dados.

    Parâmetros
    ----------
    Xd_target : np.array ou pd.DataFrame
        Dados de entrada sem 'time' e 'event'.
    logit : LogisticRegression
        Modelo treinado para prever probabilidades dos clusters.
    kmfs : dict
        Estimativas Kaplan-Meier para cada cluster.

    Retorna
    -------
    pd.DataFrame
        DataFrame contendo a função de sobrevivência prevista para cada amostra.
    """
    
    X = Xd_target.copy()
    Xd = Xd_target.copy()

    logit_proba = logit.predict_proba(Xd)
    ordered_labels = list(np.sort(logit.classes_))

    list_rns = list()
    for l in ordered_labels:     

        X[f'label_{l}_logit'] = logit_proba[:,l]
        X[f'label_{l}_rn'] = X[f'label_{l}_logit']

        list_rns.append(f'label_{l}_rn')

    label_rn_list = [f'label_{l}_rn' for l in ordered_labels]
    ss = X[label_rn_list].sum(axis=1)

    surv_df_list = [kmfs[l]['kmf'].survival_function_ for l in ordered_labels]

    ee = surv_df_list
    indexes = ee[0].index
    for e in ee[1:]:
        indexes = indexes.union(e.index)

    for i,e in enumerate(ee):        
        ee[i] = e.reindex(indexes,method = 'ffill').bfill()        


    idx_all_zero = X[label_rn_list].sum(axis=1)==0

    X.loc[idx_all_zero,label_rn_list] = [1/len(ordered_labels)]*len(ordered_labels)    
    ss.loc[idx_all_zero] = 1

    import functools
    kmfs_list = [functools.reduce(lambda a, b: a+b, [ee[l]*(X[label_rn_list[l]].loc[idx]/ss.loc[idx]) for l in ordered_labels]) for idx in X.index]

    surv_df = pd.concat(kmfs_list, axis=1, ignore_index=True).\
                            bfill().ffill()

    return surv_df


def labels_from_model(Xd_target, logit):
    """
    Prediz os rótulos clusterizados para cada amostra com base nas probabilidades do modelo.

    Parâmetros
    ----------
    Xd_target : np.array ou pd.DataFrame
        Dados de entrada sem 'time' e 'event'.
    logit : LogisticRegression
        Modelo treinado para prever probabilidades dos clusters.

    Retorna
    -------
    np.array
        Rótulos dos clusters previstos para cada amostra.
    """
    
    X = Xd_target.copy()
    Xd = Xd_target.copy()

    logit_proba = logit.predict_proba(Xd)
    ordered_labels = list(np.sort(logit.classes_))

    list_rns = list()
    for l in ordered_labels:
        
        X[f'label_{l}_logit'] = logit_proba[:,l]
        X[f'label_{l}_rn'] = X[f'label_{l}_logit']
        
        list_rns.append(f'label_{l}_rn')
     
    labels = np.argmax(X[list_rns].values, axis=1)
        
    return labels

def info_flat_from_info_list_K(info_list_K):
    """
    Transforma o dicionário de informações de clusterização em um DataFrame.

    Parâmetros
    ----------
    info_list_K : dict
        Dicionário onde as chaves são números de clusters e os valores são listas de informações.

    Retorna
    -------
    pd.DataFrame
        DataFrame contendo uma linha para cada execução do algoritmo EM com as principais métricas.
    """
    
    columns = ['n_clusters', 'idx', 'cindex',
               'integrated_nbll',
               'integrated_brier_score'
              ]

    info_flat = pd.DataFrame(columns=columns)
    
    for n_clusters in info_list_K.keys():
        for idx, info in enumerate(info_list_K[n_clusters]):
            info['ite'] = info['cindex_by_ite'][-1]['ite']
            info_list_K[n_clusters][idx] = info        

            # Sempre adiciona as métricas do treinamento
            info_flat = info_flat.append(
                pd.DataFrame(
                    data=[[
                        n_clusters, 
                        idx,
                        info['cindex_by_ite'][-1]['cindex'],
                        info['cindex_by_ite'][-1]['integrated_nbll'],
                        info['cindex_by_ite'][-1]['integrated_brier_score']
                    ]],
                    columns=columns
                ),
                ignore_index=True
            )

    return info_flat


def best_model_keys_from_info_list_K(info_list_K):
    """
    Identifica o melhor modelo com base nas métricas de validação.

    Parâmetros
    ----------
    info_list_K : dict
        Dicionário com informações de modelos para diferentes números de clusters.

    Retorna
    -------
    tuple
        Tupla (n_cluster, idx) indicando o número de clusters e o índice do melhor modelo.
    """
    
    info_flat = info_flat_from_info_list_K(info_list_K)
    
    best_model_row = info_flat.sort_values(by=['cindex']).iloc[-1,:]

    sub_experiment_key = best_model_row['n_clusters']
    idx = best_model_row['idx']
    
    return sub_experiment_key, idx

def plot_kms_clusters(info, time_max, show_n_samples=True):
    """
    Gera um gráfico dos resultados clusterizados usando Kaplan-Meier.

    Parâmetros
    ----------
    info : dict
        Informações do modelo para plotar as funções de sobrevivência.
    time_max : float
        Tempo máximo a ser exibido no gráfico.
    show_n_samples : bool, opcional
        Indica se o número de amostras por cluster será exibido. O padrão é True.

    Retorna
    -------
    plotly.graph_objs._figure.Figure
        Gráfico contendo as funções de sobrevivência por cluster.
    """

    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.io as pio
    pio.templates.default = "plotly_white"
    
    title = "Curva de Kaplan-Meier de cada cluster"

    g = go.Figure()

    kmfs = info['kmfs']
    kmfs_p_dict = deepcopy(kmfs)

    for k,v in kmfs.items():

        kmf = v['kmf']    
        kmfs_single = {'confidence' : kmf.confidence_interval_,
                       'survival' : kmf.survival_function_,
                       'sample_size': len(kmf.event_observed)} 

        kmfs_p_dict[k] = kmfs_single

    kmfs_p = kmfs_p_dict

    k = list(kmfs_p.keys())

    trace = dict()
    annotations = list()
    colors = px.colors.qualitative.Set1
    name2color = {individual_sf:colors[i%len(colors)] for i,individual_sf in enumerate(k)}

    for i,cluster_name in enumerate(k):

        survival = kmfs_p_dict[cluster_name]['survival']
        confidences = kmfs_p_dict[cluster_name]['confidence']   

        x = survival.index.values.flatten()
        x_rev = x[::-1]

        y3 = survival.values.flatten()
        y3_upper = confidences.iloc[:,1].values.flatten()
        y3_lower = confidences.iloc[:,0].values.flatten()
        y3_lower = y3_lower[::-1]

        trace[(i,'confidence')] = go.Scatter(
            x=list(x)+list(x_rev),
            y=list(y3_upper)+list(y3_lower),
            fill='toself',
            fillcolor='rgba' + str(name2color[cluster_name])[3:-1]+ ', 0.3)',
            line_color='rgba(255,255,255,0)',
            showlegend=False,
            legendgroup=str(cluster_name),
            name='CI 95%: '+ str(cluster_name),
            hoveron= 'points'
        )

        x_a = ((x.max()-x.min())/(len(k)+1))*(i+1)
        y_a = y3[abs(x-x_a).argmin()]
        
        if show_n_samples:

            annotations.append(dict(x=x_a,
                    y=y_a,
                    ax= 20,
                    ay = -20,
                    text=str(kmfs_p_dict[cluster_name]['sample_size']) + ' amostras',
                    xref="x",
                    yref="y",showarrow=True,
                    font_size = 16,
                    font_color = 'black',
                    arrowhead=7))

        trace[(i,'survival')] = go.Scatter(
            x=x, y=y3,
            line_color=name2color[cluster_name],
            name=str(cluster_name),
            showlegend=True,
            legendgroup=str(cluster_name)
        )

    for i,key in enumerate(trace.keys()):
        o = trace[key]
        g.add_trace(o)
     

    if show_n_samples:
        g.update_layout(annotations = annotations, overwrite = True)

    g.update_yaxes(range=[0,1.0])
    g.update_xaxes(range=[0, time_max])

    g.update_layout(
                    title=dict(
                       text=title,
                        x = 0.5,
                        font_size = 15,
                        xanchor = 'center',
                        yanchor = 'middle'
                    ), xaxis_title = 'Tempo (dias)',
                        yaxis_title = 'Probabilidade',
                    legend = dict(title = 'Clusters'),
                    height = 500, width=950
                   )  
    return g


def cluster_EM(X, Xd, X_val, Xd_val, n_clusters, n_iterations, global_fixed_bw):
    """
    Realiza uma única execução do algoritmo Expectativa-Maximização (EM) para análise de sobrevivência.

    Parâmetros
    ----------
    X : np.array ou pd.DataFrame
        Dados de treinamento com 'time' e 'event'.
    Xd : np.array ou pd.DataFrame
        Dados de treinamento sem 'time' e 'event'.
    X_val : np.array ou pd.DataFrame
        Dados de validação com 'time' e 'event'.
    Xd_val : np.array ou pd.DataFrame
        Dados de validação sem 'time' e 'event'.
    n_clusters : int
        Número de clusters para o algoritmo.
    n_iterations : int
        Número máximo de iterações para o algoritmo.
    global_fixed_bw : np.array
        Largura de banda calculada.

    Retorna
    -------
    list of dict
        Lista contendo informações do ajuste do modelo.
    """

    info_list = list() 

    # Inicialização dos clusters
    new_labels = np.random.randint(0, n_clusters, len(Xd))
    ordered_labels = np.sort(np.unique(new_labels))
    updated_clusters = {(label): list(np.argwhere(new_labels==label).flatten()) for label in ordered_labels}
    X['labels'] = new_labels

    # Inicialização dos medoides
    new_medoids = {k:[] for k in range(n_clusters)}
    for l in ordered_labels:
        new_medoids[l] = Xd.loc[updated_clusters[l],:].mean().values

    # Inicialização dos Kaplan-Meier
    clusters = updated_clusters.copy()
    idxs = list(clusters.keys())
    kmfs = dict()
    for i in idxs:
        time_km, event_km = X.iloc[clusters[i],:]['time'].astype('float32'), X.iloc[clusters[i],:]['event'].astype('float32')
        kmf = KaplanMeierFitter()  
        kmf.fit(time_km, event_km)
        kmfs[i] = {'kmf': kmf}

    # Inicialização das listas de métricas
    iterations = n_iterations
    cindex_by_ite = list()
    
    try:

        # Algoritmo EM principal
        time_algorithm = time.time()
        for ite in np.arange(iterations):

            time_max = time.time()

            print(f'######## iteration: {ite}')
            
            X['labels'] = new_labels  
            
            # Treinamento da regressão logística
            from sklearn.linear_model import LogisticRegression
            logit = LogisticRegression(max_iter=5000)
            logit.fit(Xd, new_labels)
            logit_proba = logit.predict_proba(Xd)

            ###################
            ### Expectation ###
            ###################
            
            # Cálculo das métricas a cada 3 iterações
            if (ite % 3) == 0:
                metrics = cindex_km_from_model(X, Xd, logit, kmfs, global_fixed_bw)
                snap_cindex = metrics['cindex']
                cindex_by_ite.append({
                    'ite': ite, 
                    'cindex': snap_cindex,
                    'integrated_brier_score': metrics['integrated_brier_score'],
                    'integrated_nbll': metrics['integrated_nbll']
                })
                print(f'\t# cindex with train_set at iteration {ite}: {snap_cindex}')

            # Verificação de convergência
            if len(cindex_by_ite) > 2:
                if abs(cindex_by_ite[-2]['cindex'] - cindex_by_ite[-1]['cindex']) < 1e-6:
                    print(f'######## Converged at iteration: {ite} ########')
                    break

            list_rns = list()
            for l in ordered_labels:

                def survival_function(t):
                    return (kmfs[l]['kmf']).survival_function_at_times(t).values

                X[f'label_{l}_logit'] = logit_proba[:,l]

                t = X['time'].values

                a_d = approximate_derivative(t, kmf = kmfs[l]['kmf'], global_fixed_bw=global_fixed_bw)
                X[f'label_{l}_negative_derivative_survival'] = a_d['estimate']
                X[f'label_{l}_survival'] = survival_function(X['time'].values)


                X[f'label_{l}_rn'] = (X[f'label_{l}_negative_derivative_survival']**(X[f'event']))*\
                                     (X[f'label_{l}_survival']**(1-X[f'event']))*\
                                     (X[f'label_{l}_logit'])


                list_rns.append(f'label_{l}_rn')

            X['labels'] = np.argmax(X[list_rns].values, axis=1)   


            ####################
            ### Maximization ###
            ####################

            new_labels = X['labels'].values.copy()
            updated_clusters = {(label): list(np.argwhere(new_labels==label).flatten()) for label in ordered_labels}

            for l in ordered_labels:
                new_medoids[l] = Xd.loc[updated_clusters[l],:].mean().values

            clusters = updated_clusters.copy()
            idxs = list(clusters.keys())
            kmfs = dict()

            empty_kmf = False
            for i in idxs:
                time_km, event_km = X.iloc[clusters[i],:]['time'].astype('float32'), X.iloc[clusters[i],:]['event'].astype('float32')
                if len(time_km) == 0:
                    empty_kmf = True
            if empty_kmf:
                print('EMPTY_kmf, stopped training.')
                break

            for i in idxs:
                time_km, event_km = X.iloc[clusters[i],:]['time'].astype('float32'), X.iloc[clusters[i],:]['event'].astype('float32')
                kmf = KaplanMeierFitter()  
                kmf.fit(time_km, event_km)

                kmfs[i] = {'kmf': kmf}

            print("\tTime elapsed for the entire iteration: ", timedelta(seconds=time.time() - time_max))

        # Salvar informações do modelo
        if len(cindex_by_ite) > 0:  # Garante que temos pelo menos uma métrica
            dict_to_store = {
                'kmfs': deepcopy(kmfs), 
                'logit': logit, 
                'new_labels': new_labels,
                'global_fixed_bw': global_fixed_bw,
                'cindex_by_ite': deepcopy(cindex_by_ite),
                'cindex_last': cindex_by_ite[-1]['cindex']
            }
            info_list.append(dict_to_store)
            
    except Exception as error:
        print("An exception occurred:", type(error).__name__, "–", error)
        print('Error for a single EM run. Skipped.\n\n')
    
    return info_list