"""
Autor do método e do código original: Cleiton Moya de Almeida (2024).

Autor das modificações e da implementação multivariada do método: Ian José Agra Gomes (2025).
"""
import numpy as np
from scipy.stats import shapiro, betabinom, multivariate_normal, betabinom, chi2
from statsmodels.tsa.stattools import adfuller
import time
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def normality_test(y, alpha):
    """
    Realiza o teste de normalidade Shapiro-Wilk.

    Parâmetros:
        y (array-like): Dados da amostra.
        alpha (float): Nível de significância para o teste.

    Retorna:
        bool: True se a hipótese nula (normalidade) não for rejeitada; False caso contrário.
    """
    _, pvalue = shapiro(y)
    return pvalue > alpha


def stationarity_test(y, alpha):
    """
    Realiza o teste de estacionaridade Augmented Dickey-Fuller.

    Parâmetros:
        y (array-like): Série temporal para análise.
        alpha (float): Nível de significância para o teste.

    Retorna:
        bool: True se a hipótese nula (não estacionariedade) for rejeitada; False caso contrário.
    """
    adf = adfuller(y)
    pvalue = adf[1]
    return pvalue < alpha


def logpdf(x, loc, scale):
    """
    Calcula o logaritmo da densidade de probabilidade (log-pdf) para uma distribuição normal.

    Parâmetros:
        x (array-like): Dados da amostra.
        loc (float): Média da distribuição.
        scale (float): Desvio padrão da distribuição.

    Retorna:
        array-like: Valores da log-pdf calculados para os dados.
    """
    c = 1 / np.sqrt(2 * np.pi)
    y = np.log(c) - np.log(scale) - (1 / 2) * ((x - loc) / scale) ** 2
    return y


def loglik(x, loc, scale):
    """
    Calcula a log-verossimilhança para uma distribuição normal.

    Parâmetros:
        x (array-like): Dados da amostra.
        loc (float): Média da distribuição.
        scale (float): Desvio padrão da distribuição.

    Retorna:
        float: Valor da log-verossimilhança calculada.
    """
    n = len(x)
    c = 1 / np.sqrt(2 * np.pi)
    y = n * np.log(c / scale) - (1 / (2 * scale**2)) * ((x - loc) ** 2).sum()
    return y


def vwcd(X, w, w0, ab, p_thr, vote_p_thr, vote_n_thr, y0, yw, aggreg, verbose=False):
    """
    Detecta pontos de mudança em uma série temporal usando o algoritmo Voting Windows Changepoint Detection.

    Parâmetros:
        X (array-like): Série temporal.
        w (int): Tamanho da janela deslizante de votação.
        w0 (int): Janela inicial para estimar parâmetros iniciais.
        ab (float): Hiperparâmetros alfa e beta da distribuição beta-binomial
        p_thr (float): Limiar de probabilidade para o voto de uma janela ser registrado.
        vote_p_thr (float): Limiar de probabilidade para definir um ponto de mudança após a agregação dos votos.
        vote_n_thr (float): Fração mínima da janela que precisa votar.
        y0 (float): Probabilidade a priori da função logística (início da janela).
        yw (float): Probabilidade a priori da função logística (início da janela).
        aggreg (str): Função de agregação para os votos ('posterior' ou 'mean').
        verbose (bool): Se True, exibe informações sobre os pontos de mudança detectados.

    Retorna:
        tuple: 
            - CP (list): Lista de índices dos pontos de mudança detectados.
            - M0 (list): Lista de médias estimadas nas janelas.
            - S0 (list): Lista de desvios padrão estimados nas janelas.
            - elapsedTime (float): Tempo total de execução do algoritmo.
            - vote_counts (array): Array com o número de votos para cada ponto.
            - vote_probs (array): Array com as probabilidades máximas dos votos individuais.
            - agg_probs (array): Array com as probabilidades após agregação dos votos.
    """
    def pos_fun(ll, prior, tau):
        c = np.nanmax(ll)
        lse = c + np.log(np.nansum(prior * np.exp(ll - c)))
        p = ll[tau] + np.log(prior[tau]) - lse
        return np.exp(p)

    def votes_pos(vote_list, prior_v):
        vote_list = np.array(vote_list)
        prod1 = vote_list.prod() * prior_v
        prod2 = (1 - vote_list).prod() * (1 - prior_v)
        p = prod1 / (prod1 + prod2)
        return p

    def logistic_prior(x, w, y0, yw):
        a = np.log((1 - y0) / y0)
        b = np.log((1 - yw) / yw)
        k = (a - b) / w
        x0 = a / k
        y = 1 / (1 + np.exp(-k * (x - x0)))
        return y

    N = len(X)
    vote_n_thr = np.floor(w * vote_n_thr)
    i_ = np.arange(0, w - 3)
    prior_w = betabinom(n=w - 4, a=ab, b=ab).pmf(i_)
    x_votes = np.arange(1, w + 1)
    prior_v = logistic_prior(x_votes, w, y0, yw)

    votes = {i: [] for i in range(N)}
    lcp = 0
    CP = []
    M0 = []
    S0 = []
    
    vote_counts = np.zeros(N)      # Array para armazenar o número de votos
    vote_probs = np.zeros(N)       # Array para armazenar probabilidades individuais dos votos
    agg_probs = np.zeros(N)        # Array para armazenar probabilidades agregadas

    startTime = time.time()
    for n in range(N):
        if n >= w - 1:
            if n == lcp + w0:
                m_w0 = X[n - w0 + 1 : n + 1].mean()
                s_w0 = X[n - w0 + 1 : n + 1].std(ddof=1)
                M0.append(m_w0)
                S0.append(s_w0)

            Xw = X[n - w + 1 : n + 1]
            LLR_h = []
            for nu in range(1, w - 3 + 1):
                x1 = Xw[: nu + 1]
                m1 = x1.mean()
                s1 = x1.std(ddof=1)
                if np.round(s1, 3) == 0:
                    s1 = 0.001
                logL1 = loglik(x1, loc=m1, scale=s1)

                x2 = Xw[nu + 1 :]
                m2 = x2.mean()
                s2 = x2.std(ddof=1)
                if np.round(s2, 3) == 0:
                    s2 = 0.001
                logL2 = loglik(x2, loc=m2, scale=s2)

                llr = logL1 + logL2
                LLR_h.append(llr)

            LLR_h = np.array(LLR_h)
            pos = [pos_fun(LLR_h, prior_w, nu) for nu in range(w - 3)]
            pos = [np.nan] + pos + [np.nan] * 2
            pos = np.array(pos)

            p_vote_h = np.nanmax(pos)
            nu_map_h = np.nanargmax(pos)

            if p_vote_h >= p_thr:
                j = n - w + 1 + nu_map_h
                votes[j].append(p_vote_h)
                vote_counts[j] += 1
                vote_probs[j] = max(vote_probs[j], p_vote_h)

            votes_list = votes[n - w + 1]
            num_votes = len(votes_list)
            if num_votes >= vote_n_thr:
                if aggreg == 'posterior':
                    agg_vote = votes_pos(votes_list, prior_v[num_votes - 1])
                elif aggreg == 'mean':
                    agg_vote = np.mean(votes_list)
                agg_probs[n - w + 1] = agg_vote  # Armazenar probabilidade agregada

                if agg_vote > vote_p_thr:
                    if verbose:
                        print(f'Changepoint at n={n-w+1}, p={agg_vote}, n={num_votes} votes')
                    lcp = n - w + 1
                    CP.append(lcp)

    endTime = time.time()
    elapsedTime = endTime - startTime
    return CP, M0, S0, elapsedTime, vote_counts, vote_probs, agg_probs


def validate_covariance(cov, d, regularization_factor=1e-4):
    """
    Valida e regulariza matriz de covariância
    """
    regularization_factor = regularization_factor * d  # Ajuste baseado na dimensionalidade
    if not isinstance(cov, np.ndarray):
        cov = np.array(cov)
        
    if len(cov.shape) != 2 or cov.shape[0] != d or cov.shape[1] != d:
        return np.eye(d) * regularization_factor
        
    if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
        return np.eye(d) * regularization_factor
        
    try:
        det = np.linalg.det(cov)
        if det <= 1e-10:
            return np.eye(d) * regularization_factor
    except:
        return np.eye(d) * regularization_factor
        
    # Garantir simetria
    cov = (cov + cov.T) / 2
    
    # Adicionar regularização
    return cov + regularization_factor * np.eye(d)


def loglik_mv(X, mean, cov):
    n, d = X.shape
    cov = validate_covariance(cov, d)
    
    try:
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            return -np.inf
            
        inv_cov = np.linalg.inv(cov)
        diff = X - mean
        mahal = np.sum(np.dot(diff, inv_cov) * diff, axis=1)

        # Penalização pela dimensionalidade usando BIC
        bic_penalty = 0.5 * d * np.log(n)
        
        # Normalizar pela dimensionalidade
        const = -0.5 * (d * np.log(2 * np.pi) + logdet)
        return (n * const - 0.5 * np.sum(mahal) - bic_penalty)
        
    except np.linalg.LinAlgError:
        return -np.inf


def vwcd_mv(X, w, w0, ab, p_thr, vote_p_thr, vote_n_thr, y0, yw, aggreg, verbose=False):
    """
    Detecta pontos de mudança em série temporal multivariada usando VWCD.
    
    Parâmetros:
    ----------
    X : array-like
        Série temporal multivariada no formato (N x d)
    w : int
        Tamanho da janela deslizante
    w0 : int
        Tamanho da janela inicial para estimativa
    ab : float
        Parâmetros alfa e beta da distribuição beta-binomial
    p_thr : float
        Limiar de probabilidade para registrar voto
    vote_p_thr : float
        Limiar de probabilidade para definir changepoint
    vote_n_thr : float
        Fração mínima da janela que precisa votar
    y0 : float
        Probabilidade inicial da prior logística
    yw : float
        Probabilidade final da prior logística
    aggreg : str
        Método de agregação ('posterior' ou 'mean')
    verbose : bool
        Se True, imprime informações de debug
        
    Retorna:
    -------
        CP : lista de índices dos changepoints
        M0 : lista de médias originais estimadas nas janelas
        S0 : lista de matrizes de covariância originais estimadas nas janelas
        elapsedTime : tempo total de execução
        vote_counts : contagem de votos
        vote_probs : probabilidades dos votos
        agg_probs : probabilidades agregadas

    """
    def logistic_prior(x, w, y0, yw):
        a = np.log((1 - y0) / y0)
        b = np.log((1 - yw) / yw)
        k = (a - b) / w
        x0 = a / k
        y = 1 / (1 + np.exp(-k * (x - x0)))
        return y

    def votes_pos(vote_list, prior_v, eps=1e-10, max_prob=0.9999, penalty_factor=1.0):
        vote_list = np.clip(np.array(vote_list), eps, max_prob)
        n_votes = len(vote_list)
        
        log_prob1 = np.sum(np.log(vote_list)) / n_votes
        log_prob2 = np.sum(np.log(1 - vote_list)) / n_votes

        log_prob1 += np.log(prior_v) / n_votes
        log_prob2 += np.log(1 - prior_v) / n_votes

        # Usar penalty_factor=1.0 (ou remover a multiplicação) para não exagerar a influência
        log_prob1 *= penalty_factor
        log_prob2 *= penalty_factor

        c = max(log_prob1, log_prob2)
        p = np.exp(log_prob1 - c) / (np.exp(log_prob1 - c) + np.exp(log_prob2 - c))

        return np.clip(p, eps, 1-eps)


    def pos_fun(ll, prior):
        """
        Calcula probabilidade posterior normalizada para cada possível ponto de mudança.
        
        Parâmetros:
        -----------
        ll : array-like
            Log-verossimilhanças para cada ponto candidato
        prior : array-like
            Prior para cada ponto candidato
        """
        # Descartar posições com -inf
        valid_mask = ~np.isinf(ll)
        if not np.any(valid_mask):
            return np.zeros_like(ll)
        
        # Trabalhar apenas com valores válidos
        ll_valid = ll[valid_mask]
        prior_valid = prior[valid_mask]
        
        # Normalização no espaço log para estabilidade numérica
        c = np.max(ll_valid)
        log_sum = c + np.log(np.sum(prior_valid * np.exp(ll_valid - c)))
        
        # Calcular probabilidades no espaço log
        log_prob = ll_valid + np.log(prior_valid) - log_sum
        
        # Converter para probabilidades e garantir soma 1
        prob = np.zeros_like(ll)
        prob[valid_mask] = np.exp(log_prob - np.max(log_prob))  # Evita overflow
        prob = prob / np.sum(prob)
        
        return prob

    # Armazenar o scaler para uso posterior
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    N, d = X_scaled.shape
    vote_n_thr = int(np.floor(w * vote_n_thr))
    
    # Prior usando distribuição beta-binomial
    i_ = np.arange(0, w - 3)
    prior_w = betabinom(n=w - 4, a=ab, b=ab).pmf(i_)
    
    # Prior logístico para votos
    x_votes = np.arange(w + 1)
    prior_v = logistic_prior(x_votes, w, y0, yw)
    
    # Inicializar estruturas
    votes = {i: [] for i in range(N)}
    lcp = 0
    CP = []
    M0 = []  # Vai armazenar médias na escala original
    S0 = []  # Vai armazenar covariâncias na escala original
    
    vote_counts = np.zeros(N)
    vote_probs = np.zeros(N)
    agg_probs = np.zeros(N)

    startTime = time.time()

    for n in range(N):
        if n >= w - 1:
            if n == lcp + w0:
                # Calcular estatísticas na escala original
                x_w0_orig = X[n - w0 + 1 : n + 1]
                m_w0 = np.mean(x_w0_orig, axis=0)
                s_w0 = np.cov(x_w0_orig, rowvar=False)
                M0.append(m_w0)
                S0.append(s_w0)

            # Usar dados padronizados para detecção
            Xw = X_scaled[n - w + 1 : n + 1]
            
            LLR_h = []
            for nu in range(0, w - 3):
                x1 = Xw[:nu + 1]
                x2 = Xw[nu + 1:]
                
                if len(x1) < d + 1 or len(x2) < d + 1:
                    LLR_h.append(-np.inf)
                    continue
                
                m1 = np.mean(x1, axis=0)
                s1 = np.cov(x1, rowvar=False, ddof=1)
                
                m2 = np.mean(x2, axis=0)
                s2 = np.cov(x2, rowvar=False, ddof=1)
                
                logL1 = loglik_mv(x1, m1, s1)
                logL2 = loglik_mv(x2, m2, s2)
                
                llr = logL1 + logL2
                
                LLR_h.append(llr)

            LLR_h = np.array(LLR_h)
            
            if np.all(np.isinf(LLR_h)):
                continue
                
            valid_idx = ~np.isinf(LLR_h)
            if np.sum(valid_idx) > 0 and len(valid_idx) == len(prior_w):
                valid_llr = LLR_h[valid_idx]
                valid_prior = prior_w[valid_idx]
                
                pos = pos_fun(valid_llr, valid_prior)
                
                max_idx = np.argmax(pos)
                p_vote_h = pos[max_idx]
                
                nu_map_h = np.where(valid_idx)[0][max_idx]
                
                if p_vote_h >= p_thr:
                    j = n - w + 1 + nu_map_h
                    votes[j].append(p_vote_h)
                    vote_counts[j] += 1
                    vote_probs[j] = max(vote_probs[j], p_vote_h)
                
                votes_list = votes[n - w + 1]
                num_votes = len(votes_list)
                
                if num_votes >= vote_n_thr:
                    print(f'Votos: {votes_list}; prior: {prior_v[num_votes - 1]}')
                    if aggreg == 'posterior':
                        agg_vote = votes_pos(votes_list, prior_v[num_votes - 1])
                    else:  # aggreg == 'mean'
                        agg_vote = np.mean(votes_list)

                    print(f'Probabilidade agregada: {agg_vote}')
                    agg_probs[n - w + 1] = agg_vote
                    
                    if agg_vote > vote_p_thr:
                        if verbose:
                            print(f"Changepoint at n={n-w+1}, p={agg_vote:.4f}, votes={num_votes}")
                        lcp = n - w + 1
                        CP.append(lcp)

                        # Calcular estatísticas na escala original para o novo segmento
                        if len(CP) > 1:
                            start_idx = CP[-2]
                            end_idx = CP[-1]
                            segment_orig = X[start_idx:end_idx]
                            M0.append(np.mean(segment_orig, axis=0))
                            S0.append(np.cov(segment_orig, rowvar=False))

    endTime = time.time()
    elapsedTime = endTime - startTime

    return CP, M0, S0, elapsedTime, vote_counts, vote_probs, agg_probs