import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carrega o CSV consolidado
nome_arquivo = "base_analise_sentimentos.csv" 
df = pd.read_csv(nome_arquivo)

# Seleciona as colunas numéricas que queremos cruzar na matriz
# Você pode remover alguma se achar que não faz sentido para a sua análise final
colunas_numericas = [
    'midia', 'num_caracteres', 'num_palavras', 'num_emojis', 
    'num_mencoes', 'likes', 'repostagens', 'comentarios', 
    'prob_positivo', 'prob_neutro', 'prob_negativo'
]

# Pega apenas as colunas escolhidas e converte para número (por segurança)
df_corr = df[colunas_numericas].apply(pd.to_numeric, errors='coerce')

# Calcula a matriz de correlação (Método de Pearson)
matriz_correlacao = df_corr.corr()

# Configura o tamanho da imagem (deixei grande para caber todos os números bem visíveis)
plt.figure(figsize=(12, 8))

# Gera o Mapa de Calor (Heatmap)
sns.heatmap(matriz_correlacao, 
            annot=True,          # Mostra os valores numéricos dentro dos quadrados
            fmt=".2f",           # Formata os números para ter 2 casas decimais (ex: 0.85)
            cmap="coolwarm",     # Paleta de cores: Azul = Inverso/Negativo | Vermelho = Correlação Positiva
            vmin=-1, vmax=1,     # A correlação matemática sempre vai de -1 a 1
            linewidths=0.5)      # Cria uma linhazinha branca separando os blocos

# Ajustes estéticos de título e eixos
plt.title('Matriz de Correlação das Variáveis (Pearson)', fontsize=16, pad=20)

# Dá uma inclinada nos nomes das colunas embaixo para não sobrepor
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()

# Salva em alta resolução
plt.savefig('matriz_correlacao.png', dpi=300)
plt.close()

print("Matriz de correlação gerada com sucesso!")
print("Arquivo salvo como 'matriz_correlacao.png'")