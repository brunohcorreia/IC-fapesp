import pandas as pd
import matplotlib.pyplot as plt

# Carrega o CSV consolidado
nome_arquivo = "base_analise_sentimentos.csv" 
df = pd.read_csv(nome_arquivo)

# Calcula a porcentagem de mídia por CONTA (coluna 'perfil')
# A média de uma coluna booleana (0 e 1) * 100 resulta na porcentagem
df_porcentagem = df.groupby('perfil')['midia'].mean() * 100

# Configuração do tamanho do gráfico (um pouco mais largo para caber os nomes)
plt.figure(figsize=(10, 6))

# Escolhendo uma cor padrão para as barras
cor_barras = '#457b9d'

# Plota o gráfico de barras
ax = df_porcentagem.plot(kind='bar', color=cor_barras) 

# Ajustes estéticos do título e eixos
plt.title('Porcentagem de Comentários com Mídia por Conta', fontsize=14, pad=15)
plt.xlabel('Conta (Perfil)', fontsize=12)
plt.ylabel('Porcentagem de Comentários (%)', fontsize=12)

# Inclina os nomes das contas em 45 graus para facilitar a leitura
plt.xticks(rotation=45, ha='right')

# Adiciona o valor exato da porcentagem no topo de cada barra
for i, valor in enumerate(df_porcentagem):
    plt.text(i, valor + (df_porcentagem.max() * 0.02), f'{valor:.1f}%', 
             ha='center', fontsize=10, fontweight='bold')

# Ajusta o limite superior do eixo Y para os números não cortarem na borda
plt.ylim(0, df_porcentagem.max() + (df_porcentagem.max() * 0.15))

# Adiciona linhas de grade horizontais sutis
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Garante que nada fique cortado na imagem final
plt.tight_layout()

# Salva o arquivo em alta resolução
plt.savefig('porcentagem_midia_por_conta.png', dpi=300)
plt.close()

print("Gráfico de porcentagem de mídia por conta gerado com sucesso!")
print("Salvo como 'porcentagem_midia_por_conta.png'")