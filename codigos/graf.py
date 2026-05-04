import pandas as pd
import io
import matplotlib.pyplot as plt

# Seus dados em formato CSV
csv_data = """Conta,Rede,ID_Post,Total_Likes,Media_Likes,Total_Comentarios,Negatividade_Media,Positividade_Media,Volume_Posts
BolsonaroSP,Instagram,1,6266,6.266,420.0,0.27274071315210313,0.40655350679554975,1000
BolsonaroSP,Instagram,3,1694,1.694,115.0,0.18577043394674547,0.5065400274691637,1000
BolsonaroSP,Instagram,2,9861,9.861,510.0,0.36045497877383603,0.2702246189948637,1000
BolsonaroSP,Twitter,1,800,0.9975062344139651,0.0,0.5812353005535794,0.14347412007015448,802
BolsonaroSP,Twitter,3,107,0.5169082125603864,0.0,0.3282482431243187,0.20656180085506345,207
BolsonaroSP,Twitter,2,39,0.08533916849015317,0.0,0.5404308754848867,0.09589339913015762,457
GayerGus,Instagram,1,2410,2.41,88.0,0.06358014788432043,0.3447348645767197,1000
GayerGus,Instagram,3,5666,5.666,91.0,0.43137949278519955,0.21681885486631655,1000
GayerGus,Instagram,2,21890,21.89,318.0,0.19374433747527656,0.46007861953997053,1000
GayerGus,Twitter,1,34,0.05492730210016155,0.0,0.20790703734871968,0.27562170394926494,619
GayerGus,Twitter,3,180,0.3415559772296015,0.0,0.36916815298096944,0.11077757993760734,527
GayerGus,Twitter,2,62,0.2719298245614035,0.0,0.345354164445062,0.27626469923211916,228
nikolas_dm,Instagram,1,4089,272.6,147.0,0.10835659484534212,0.6184882063418626,15
nikolas_dm,Instagram,3,7397,7.397,305.0,0.08930726513464467,0.5865787630095146,1000
nikolas_dm,Instagram,2,6980,6.98,334.0,0.22909528893011155,0.45936241589300336,1000
nikolas_dm,Twitter,1,256,0.31295843520782396,0.0,0.5333257113873143,0.09504159366937401,818
nikolas_dm,Twitter,3,903,1.1107011070110702,0.0,0.2843997701332385,0.36668701763241923,813
nikolas_dm,Twitter,2,306,0.3801242236024845,0.0,0.43686935111695196,0.2726543909763073,805"""

# Lê os dados (Lembre-se de trocar para read_csv se estiver lendo do arquivo direto)
df = pd.read_csv(io.StringIO(csv_data))

# Calcula a Neutralidade
df['Neutralidade_Media'] = 1.0 - (df['Positividade_Media'] + df['Negatividade_Media'])

# Agora agrupa pela coluna 'Conta'
df_grouped = df.groupby('Conta')[['Negatividade_Media', 'Positividade_Media', 'Neutralidade_Media']].mean().reset_index()

# Configura o gráfico
df_grouped.set_index('Conta').plot(kind='bar', figsize=(9, 5), color=['#e63946', '#2a9d8f', '#a8dadc'])

# Adiciona títulos, rótulos e ajusta a legenda
plt.title('Média de Sentimentos por Conta', fontsize=14)
plt.ylabel('Proporção Média', fontsize=12)
plt.xlabel('Conta', fontsize=12)

plt.xticks(rotation=0) # Deixa o nome das contas na horizontal
plt.legend(['Negatividade', 'Positividade', 'Neutralidade'], loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
plt.tight_layout()

# Salva a imagem com 300 de DPI na sua pasta
plt.savefig('grafico_sentimentos_por_conta.png', dpi=300)

print("Gráfico 'grafico_sentimentos_por_conta.png' salvo com sucesso!")