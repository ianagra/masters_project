import os
import pandas as pd

# Caminho para o diretório com arquivos parquet
diretorio = "datasets/ts_ndt"
# Listar arquivos .parquet
arquivos = [f for f in os.listdir(diretorio) if f.endswith(".parquet")]

# Função para extrair cliente e servidor do nome do arquivo
def extrair_cliente_servidor(nome_arquivo):
    base = nome_arquivo.replace(".parquet", "")
    partes = base.split("_")
    cliente = partes[0]
    servidor = "_".join(partes[1:])
    return cliente, servidor

# Lista para armazenar os dados
dfs = []

# Ler todos os arquivos
for arquivo in arquivos:
    cliente, servidor = extrair_cliente_servidor(arquivo)
    df = pd.read_parquet(os.path.join(diretorio, arquivo))
    df["cliente"] = cliente
    df["servidor"] = servidor
    dfs.append(df)

# Unir todos os DataFrames
dados = pd.concat(dfs, ignore_index=True)

# Garantir que timestamp seja datetime
dados["timestamp"] = pd.to_datetime(dados["timestamp"])

# Ordenar por cliente, servidor e tempo
dados = dados.sort_values(by=["cliente", "servidor", "timestamp"])

# Calcular diferença entre timestamps subsequentes
dados["intervalo_min"] = dados.groupby(["cliente", "servidor"])["timestamp"].diff().dt.total_seconds() / 60

# Remover primeiras linhas de cada grupo (que têm NaN)
intervalos_validos = dados["intervalo_min"].dropna()

# Calcular intervalo médio global
intervalo_medio = intervalos_validos.mean()

# Exibir resultado
print(f"Intervalo médio entre medições subsequentes: {intervalo_medio:.2f} minutos")
