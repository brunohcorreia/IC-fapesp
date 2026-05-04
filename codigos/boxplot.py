import numpy as np
import pandas as pd

# Carrega o CSV
df = pd.read_csv("base_analise_sentimentos.csv")

# Lista das colunas que queremos gerar os boxplots
colunas_analise = ["num_caracteres", "num_emojis", "num_palavras", "prob_negativo"]

for coluna in colunas_analise:
    print(f"\n{'=' * 50}")
    print(f"📊 GERANDO DADOS PARA A COLUNA: {coluna.upper()}")
    print(f"{'=' * 50}")

    for rede in df["rede"].unique():
        dados = df[df["rede"] == rede][coluna].dropna()

        if len(dados) == 0:
            continue

        q1 = np.percentile(dados, 25)
        mediana = np.percentile(dados, 50)
        q3 = np.percentile(dados, 75)
        iqr = q3 - q1

        # Limites dos bigodes
        limite_inferior = dados[dados >= (q1 - 1.5 * iqr)].min()
        limite_superior = dados[dados <= (q3 + 1.5 * iqr)].max()

        print(f"%%% {rede.upper()} %%%")
        print(f"lower whisker={limite_inferior:.4f},")
        print(f"lower quartile={q1:.4f},")
        print(f"median={mediana:.4f},")
        print(f"upper quartile={q3:.4f},")
        print(f"upper whisker={limite_superior:.4f}")
        print("-" * 30)
