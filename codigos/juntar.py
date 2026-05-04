import pandas as pd
from pathlib import Path

# Define a pasta raiz onde estão os resultados
pasta_raiz = Path("resultados")

lista_dfs = []

print("Procurando os arquivos corretos...")

# O rglob agora procura APENAS arquivos que comecem com "analise_sentimento_"
for arquivo in pasta_raiz.rglob("analise_sentimento_*.csv"):
    
    # Quebra o caminho em partes
    partes = arquivo.parts
    
    # A estrutura esperada é: .../conta/id/rede/analise_sentimento_...csv
    # Pegando de trás pra frente (-1 é o arquivo, -2 é a rede, -3 é o id, -4 é a conta)
    if len(partes) >= 4:
        conta = partes[-4]
        id_video = partes[-3]
        rede = partes[-2]
        
        try:
            df = pd.read_csv(arquivo)
            
            # Insere as novas colunas nas primeiras posições (0, 1 e 2)
            df.insert(0, 'perfil', conta)
            df.insert(1, 'rede', rede)
            df.insert(2, 'id_video', id_video)
            
            lista_dfs.append(df)
            print(f"Processado: {arquivo.name} (Perfil: {conta} | Rede: {rede})")
            
        except Exception as e:
            print(f"Erro ao ler o arquivo {arquivo}: {e}")

# Junta tudo e salva se a lista não estiver vazia
if lista_dfs:
    df_consolidado = pd.concat(lista_dfs, ignore_index=True)
    nome_arquivo_final = "base_analise_sentimentos.csv"
    df_consolidado.to_csv(nome_arquivo_final, index=False)
    
    print(f"\nSucesso! Foram unidos {len(lista_dfs)} arquivos perfeitamente.")
    print(f"O arquivo final foi salvo como '{nome_arquivo_final}'.")
else:
    print("\nNenhum arquivo com o padrão 'analise_sentimento_*.csv' foi encontrado.")