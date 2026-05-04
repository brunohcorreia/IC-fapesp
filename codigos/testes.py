import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.stats import mannwhitneyu

# --- CONFIGURAÇÕES DE CAMINHOS ---
PASTA_RESULTADOS = Path("resultados")
PASTA_TESTES = Path("testes")
PASTA_TESTES.mkdir(parents=True, exist_ok=True)

ARQUIVO_ACURACIA = PASTA_TESTES / "acuracia_bert.csv"
ARQUIVO_CONSOLIDADO = PASTA_RESULTADOS / "RELATORIO_GERAL_CONSOLIDADO.csv"

class SuiteDeTestes:
    def __init__(self):
        sns.set_theme(style="whitegrid")

    def gerar_amostra_acuracia(self, n=100):
        """Busca em todas as pastas de resultados e extrai uma amostra aleatória."""
        print(f"--> Criando amostra de {n} posts para validação humana...")
        
        # Encontra todos os arquivos que passaram pela IA
        arquivos_sent = list(PASTA_RESULTADOS.rglob("analise_sentimento_*.csv"))
        
        if not arquivos_sent:
            print("❌ Erro: Nenhum arquivo de sentimento encontrado em 'resultados/'.")
            return

        df_total = pd.concat([pd.read_csv(f) for f in arquivos_sent], ignore_index=True)
        
        # Amostragem Aleatória
        amostra = df_total.sample(n=min(n, len(df_total)), random_state=42)
        
        # Seleciona apenas o essencial para o "olho humano"
        colunas_v = ['usuario', 'texto', 'sentimento_geral']
        df_amostra = amostra[colunas_v].copy()
        df_amostra['meu_gabarito'] = "" # Espaço para você preencher

        df_amostra.to_csv(ARQUIVO_ACURACIA, index=False)
        print(f"✅ Amostra salva em: {ARQUIVO_ACURACIA}")
        print("📢 PRÓXIMO PASSO: Abra o CSV e preencha a coluna 'meu_gabarito' (positivo, neutro, negativo).")

    def calcular_metricas_ia(self):
        """Calcula Acurácia, F1-Score e gera a Matriz de Confusão."""
        if not ARQUIVO_ACURACIA.exists():
            print(f"❌ Erro: Arquivo {ARQUIVO_ACURACIA} não encontrado.")
            return

        df = pd.read_csv(ARQUIVO_ACURACIA).dropna(subset=['meu_gabarito'])
        df = df[df['meu_gabarito'] != ""]

        if len(df) < 5:
            print("❌ Erro: Amostra insuficiente. Preencha o gabarito no CSV.")
            return

        y_true = df['meu_gabarito'].str.lower().str.strip()
        y_pred = df['sentimento_geral'].str.lower().str.strip()

        # Métricas Científicas
        acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        
        # Salva Relatório TXT
        with open(PASTA_TESTES / "metricas_bert_detalhado.txt", "w") as f:
            f.write(f"Relatório de Validação - BERTweet-PT\n")
            f.write(f"Amostra: {len(df)} posts\n")
            f.write(f"Acurácia: {acc:.2%}\n\n")
            f.write(report)

        # Matriz de Confusão
        labels = sorted(list(set(y_true) | set(y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title(f'Matriz de Confusão: Humano vs IA (Acc: {acc:.2%})')
        plt.ylabel('Verdade Real (Humano)')
        plt.xlabel('Predição (BERT)')
        plt.tight_layout()
        plt.savefig(PASTA_TESTES / "matriz_confusao.png", dpi=300)
        
        print(f"✅ Métricas calculadas! Veja os arquivos na pasta '{PASTA_TESTES}'.")

    def teste_significancia_estatistica(self):
        """
        Executa o Teste de Mann-Whitney U para comparar a Negatividade 
        entre Instagram e Twitter de forma cientificamente válida.
        """
        print("--> Executando Teste de Significância (Instagram vs Twitter)...")
        
        # Coleta todas as probabilidades de negatividade por rede
        arquivos_sent = list(PASTA_RESULTADOS.rglob("analise_sentimento_*.csv"))
        
        dados_insta = []
        dados_twitter = []

        for f in arquivos_sent:
            df = pd.read_csv(f)
            if 'instagram' in f.name.lower():
                dados_insta.extend(df['prob_negativo'].dropna().tolist())
            elif 'twitter' in f.name.lower():
                dados_twitter.extend(df['prob_negativo'].dropna().tolist())

        if not dados_insta or not dados_twitter:
            print("❌ Dados insuficientes para comparação estatística.")
            return

        # Mann-Whitney U Test (Não-Paramétrico)
        stat, p = mannwhitneyu(dados_insta, dados_twitter, alternative='two-sided')

        # Interpretação
        resultado = "ESTATISTICAMENTE RELEVANTE" if p < 0.05 else "NÃO RELEVANTE"

        with open(PASTA_TESTES / "teste_estatistico_significancia.txt", "w") as f:
            f.write("=== TESTE DE MANN-WHITNEY U (NEGATIVIDADE) ===\n")
            f.write(f"H0: Não há diferença entre a negatividade do Insta e Twitter.\n")
            f.write(f"H1: Existe diferença significativa entre as redes.\n\n")
            f.write(f"Estatística U: {stat:.4f}\n")
            f.write(f"p-valor: {p:.4e}\n")
            f.write(f"Resultado: {resultado} (nível de confiança de 95%)\n")
            if p < 0.05:
                f.write("\nConclusão: As redes possuem comportamentos emocionais distintos.")
        plt.figure(figsize=(10, 6))
        # Criando um DF temporário para o plot
        df_plot = pd.DataFrame({
            'Negatividade': dados_insta + dados_twitter,
            'Rede': ['Instagram']*len(dados_insta) + ['Twitter']*len(dados_twitter)
        })
        
        sns.violinplot(data=df_plot, x='Rede', y='Negatividade', hue='Rede', 
                       palette=['#E1306C', '#1DA1F2'], inner="quart", legend=False)
        
        plt.title('Distribuição de Negatividade: Instagram vs Twitter', fontsize=14)
        plt.ylabel('Probabilidade de Negatividade (IA)')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Salva com o nome exato que o LaTeX espera
        plt.savefig(PASTA_TESTES / "distribuicao_violino.png", dpi=300)
        plt.close()
        print(f"✅ Teste estatístico concluído! p-valor: {p:.4e}")
    
        
        

def main():
    parser = argparse.ArgumentParser(description="Suite de Testes e Validação Científica")
    parser.add_argument('--run', choices=['amostra', 'metricas', 'estatistica', 'all'], required=True)
    args = parser.parse_args()

    suite = SuiteDeTestes()

    if args.run == 'amostra':
        suite.gerar_amostra_acuracia(n=100)
    elif args.run == 'metricas':
        suite.calcular_metricas_ia()
    elif args.run == 'estatistica':
        suite.teste_significancia_estatistica()
    elif args.run == 'all':
        suite.teste_significancia_estatistica()
        suite.calcular_metricas_ia()

if __name__ == "__main__":
    main()