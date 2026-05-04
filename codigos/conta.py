import pandas as pd

# Carrega o seu CSV consolidado
nome_arquivo = "base_analise_sentimentos.csv" 
df = pd.read_csv(nome_arquivo)

# Padroniza tudo para minúsculo antes de contar (evita que 'Instagram' e 'instagram' contem separado)
df['rede'] = df['rede'].astype(str).str.lower()

# Faz a contagem exata de cada rede
contagem = df['rede'].value_counts()

print("\n--- TOTAL DE COMENTÁRIOS POR REDE ---")
print(contagem)
print("-------------------------------------")