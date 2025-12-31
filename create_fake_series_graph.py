import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# Para reprodutibilidade
np.random.seed(42)

# -----------------------------
# 1. Geração dos dados fictícios
# -----------------------------

N = 100
t = np.arange(N)

# --- DEFINIÇÃO DA REFERÊNCIA (THR-DL) ---
# Pontos onde ocorre a quebra de comportamento (apenas 20 e 60)
ref_change_points = [20, 60]

# Limites dos intervalos para cálculo de média e segmentação:
# Início (0) + Pontos de mudança (20, 60) + Final da série (100)
ref_bounds = [0] + ref_change_points + [N]  # Resulta em [0, 20, 60, 100]

def piecewise_levels(levels, bounds, n):
    """
    Cria uma série por partes constantes baseada nos bounds.
    """
    x = np.zeros(n, dtype=float)
    # Itera sobre os intervalos definidos pelos bounds
    # Se bounds = [0, 20, 60, 100], temos 3 intervalos
    for i in range(len(bounds) - 1):
        # Se não houver nível correspondente na lista 'levels', para ou usa o último
        if i >= len(levels): break
        
        level = levels[i]
        start = bounds[i]
        end = bounds[i+1]
        
        # Proteção de índice
        start = min(start, n)
        end = min(end, n)
        
        x[start:end] = level
    return x

# 1. THR-DL (Referência): Níveis para intervalos [0-20), [20-60), [60-100)
thr_dl = piecewise_levels([50, 80, 30], ref_bounds, N) 
thr_dl += np.random.normal(scale=3.0, size=N)

# 2. RTT-DL: Muda apenas em t=45
rtt_dl_bounds = [0, 45, N]
rtt_dl = piecewise_levels([40, 60], rtt_dl_bounds, N) 
rtt_dl += np.random.normal(scale=2.0, size=N)

# 3. THR-UL: Média constante
thr_ul_bounds = [0, N]
thr_ul = piecewise_levels([40], thr_ul_bounds, N) 
thr_ul += np.random.normal(scale=3.0, size=N)

# 4. RTT-UL: Muda em t=30 e t=75
rtt_ul_bounds = [0, 30, 75, N]
rtt_ul = piecewise_levels([35, 20, 50], rtt_ul_bounds, N) 
rtt_ul += np.random.normal(scale=2.0, size=N)


# -----------------------------
# 2. Cálculo das features por intervalo (DA REFERÊNCIA)
# -----------------------------

interval_labels = [r"$I_1$", r"$I_2$", r"$I_3$"]

metrics = {
    "THR-DL": thr_dl,
    "RTT-DL": rtt_dl,
    "THR-UL": thr_ul,
    "RTT-UL": rtt_ul,
}

features = {lbl: [] for lbl in metrics.keys()}

# Agora iteramos sobre len(ref_bounds) - 1 para pegar os intervalos corretos:
# [0, 20), [20, 60), [60, 100)
for i in range(len(ref_bounds) - 1):
    start = ref_bounds[i]
    end = ref_bounds[i+1]
    idx = slice(start, end) 

    for lbl, series in metrics.items():
        mu = series[idx].mean()
        sd = series[idx].std(ddof=0)
        features[lbl].append((mu, sd))

# Montagem da tabela
col_labels = [
    "Intervalo",
    r"$\mu_{\mathrm{THR-DL}}$", r"$\sigma_{\mathrm{THR-DL}}$",
    r"$\mu_{\mathrm{RTT-DL}}$", r"$\sigma_{\mathrm{RTT-DL}}$",
    r"$\mu_{\mathrm{THR-UL}}$", r"$\sigma_{\mathrm{THR-UL}}$",
    r"$\mu_{\mathrm{RTT-UL}}$", r"$\sigma_{\mathrm{RTT-UL}}$",
]

table_rows = []
metric_order = ["THR-DL", "RTT-DL", "THR-UL", "RTT-UL"]

for i, I_label in enumerate(interval_labels):
    row = [I_label]
    for key in metric_order:
        mu, sd = features[key][i]
        row.append(f"{mu:5.1f}")
        row.append(f"{sd:5.1f}")
    table_rows.append(row)


# -----------------------------
# 3. Plotagem
# -----------------------------

fig = plt.figure(figsize=(10, 8))
gs = GridSpec(3, 1, height_ratios=[1.2, 2.4, 1.3], hspace=0.5)

colors = ['C0', 'C1', 'C2', 'C3']
ref_color = colors[0]

# --- Painel (a) ---
ax_a = fig.add_subplot(gs[0, 0])
ax_a.plot(t, thr_dl, label="THR-DL (Referência)", color=ref_color)

# Linhas verticais apenas nos CHANGE POINTS (20, 60), sem linha no 100
for cp in ref_change_points:
    ax_a.axvline(cp, linestyle="--", color="red", alpha=0.7)

# Ajuste de limites
ymin, ymax = thr_dl.min() - 5, thr_dl.max() + 5
ax_a.set_ylim(ymin, ymax)
ax_a.set_xlim(0, N) # Garante que vai até 100

# Rótulos dos intervalos (usa bounds para calcular o centro)
for i in range(len(ref_bounds) - 1):
    start = ref_bounds[i]
    end = ref_bounds[i+1]
    mid = 0.5 * (start + end)
    ax_a.text(
        mid, 1.02, interval_labels[i],
        transform=ax_a.get_xaxis_transform(),
        ha="center", va="bottom", clip_on=False
    )

ax_a.set_ylabel("THR-DL")
ax_a.set_title(
    "Série temporal da feature de referência",
    fontsize=10, pad=24
)
ax_a.legend(loc="upper right", fontsize=8)

# --- Painel (b) ---
gs_b = GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[1, 0], hspace=0.1)
axes_b = []

metric_series = [
    (thr_dl, "THR-DL", colors[0]),
    (rtt_dl, "RTT-DL", colors[1]),
    (thr_ul, "THR-UL", colors[2]),
    (rtt_ul, "RTT-UL", colors[3]),
]

for i, (series, label, color) in enumerate(metric_series):
    if i == 0:
        ax = fig.add_subplot(gs_b[i, 0])
    else:
        ax = fig.add_subplot(gs_b[i, 0], sharex=axes_b[0])
    axes_b.append(ax)

    ax.plot(t, series, color=color)
    
    # Linhas verticais apenas nos pontos de mudança (sem linha no final)
    for cp in ref_change_points:
        ax.axvline(cp, linestyle="--", color="red", alpha=0.5)

    if i < len(metric_series) - 1:
        ax.tick_params(labelbottom=False)
    else:
        ax.set_xlabel("Tempo (unidade fictícia)")
    
    ax.set_ylabel(label, fontsize=9)

axes_b[0].set_xlim(0, N)
axes_b[0].set_title(
    "Features com comportamentos distintos recortadas pelos mesmos intervalos",
    fontsize=10
)

# --- Painel (c) ---
ax_c = fig.add_subplot(gs[2, 0])
ax_c.axis("off")

table = ax_c.table(
    cellText=table_rows,
    colLabels=col_labels,
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.4)

ax_c.set_title(
    "Estatísticas nos intervalos: $I_1=[0, 20), I_2=[20, 60), I_3=[60, 100)$",
    pad=10, fontsize=10
)

fig.tight_layout()
plt.show()