# Análise de Qualidade de Serviço em Redes por Meio de Detecção de Pontos de Mudança e Clusterização Baseada em Sobrevivência

[![Licença MIT](https://img.shields.io/badge/Licença-MIT-blue.svg)](LICENSE)

Repositório oficial da implementação da metodologia proposta no trabalho **"Análise de Qualidade de Serviço em Redes por Meio de Detecção de Pontos de Mudança e Clusterização Baseada em Sobrevivência"**, dissertação de Mestrado em Engenharia de Sistemas e Computação, pela COPPE/UFRJ.

## Descrição

Este repositório contém o código-fonte, datasets e scripts necessários para reproduzir os experimentos do trabalho, que propõe uma metodologia inovadora para análise de desempenho de redes combinando **detecção estatística de pontos de mudança**, **análise de sobrevivência** e **interpretação assistida por modelos de linguagem**.

## Estrutura do Repositório

```
.
├── 📂 datasets/ # Conjuntos de dados utilizados
│ ├── 📂 ts_ndt/ # Séries temporais para cada par cliente-servidor
│ ├── 📂 ts_ndt_cp/ # Séries temporais rotuladas com os pontos de mudança detectados pelo VWCD
│ ├── 📂 ts_ndt_results/ # Séries temporais rotuladas com os pontos de mudança, os clusters e as estatísticas locais
│ ├── 📜 dados_ndt.csv # Extrato do banco de dados de testes NDT para o período analisado
│ ├── 📜 dados_ndt.parquet # Extrato do banco de dados de testes NDT em formato parquet
│ ├── 📜 survival_ndt.parquet # Dataset de sobrevivência rotulado com os clusters e outras informações
│ └── 📜 ts_metadata_ndt.parquet # Informações das séries temporais
│
├── 📂 imgs/ # Imagens dos gráficos gerados
│
├── 📜 cluster_proportions_clients.csv # Tabela contendo as proporções de tempo em cada cluster para cada cliente
├── 📜 cluster_proportions_servers.csv # Tabela contendo as proporções de tempo em cada cluster para cada servidor
├── 📜 coefficients.csv # Tabela contendo os coeficientes da regressão logística associados ao cluster 1 de todas as features
├── 📜 coefficients.json # Tabela de coeficientes em formato JSON, passada para os LLMs
├── 📜 environment.yml # Arquivo de configuração do ambiente conda
├── 📜 survmixclust_thr.pkl # Modelo SurvMixClust treinado.
│
├── 📜 process_results.py # Script com as funções para processar os resultados e rotular os dados
├── 📜 SurvMixClust.py # Script com as funções que implementam o algoritmo SurvMixClust
├── 📜 SurvMixClust_utils.py # Script com as funções auxiliares do algoritmo SurvMixClust
├── 📜 timeseries_processing.py # Script com as funções de processamento das séries temporais
├── 📜 visual_analysis.py # Script com as funções de plotagem dos gráficos
├── 📜 VWCD.py # Script com as funções que implementam o algoritmo VWCD
│
├── 📜 use_example.ipynb # Jupyter notebook contendo a implementação da metodologia proposta
│
├── 📜 LICENSE # Arquivo contendo a licença de uso
└── 📜 README.md # Este arquivo
```

## Pré-requisitos

- **Gerenciador de Pacotes**: Miniconda ou Anaconda instalado
- **Python**: Versão 3.9.x+
- **R**: Versão 4.2.2+
- **Jupyter Notebook**: Incluído no ambiente conda

## Instalação (Linux)

1. Crie e ative o ambiente Conda:
```bash
conda create -n ndtanalysis python=3.9
```
```bash
conda activate ndtanalysis
```

2. Instale as bibliotecas do arquivo `requirements.txt`:
```bash
pip install -r requirements.txt
```

3. Instale o pacote `survPresmooth` do R:
```r
install.packages("survPresmooth")
```
(Dependendo do sistema, pode ser necessário instalar compiladores como GCC, Make etc.)

4. Verifique a instalação:
```bash
python -c "import lifelines, ruptures; print('Ambiente configurado!')"
```

Para uso com Jupyter Notebook, o kernel `ndtanalysis` estará automaticamente disponível.

## Contato

Para dúvidas, entre em contato:
Ian Agra - ian@land.ufrj.br