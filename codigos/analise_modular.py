import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
import emoji
import re
from pathlib import Path
from textblob import TextBlob
from wordcloud import WordCloud
from pysentimiento import create_analyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from matplotlib_venn import venn2 # <--- Adicione isto
from matplotlib_venn import venn3 # Importante para o diagrama de 3 conjuntos
from nltk.corpus import stopwords
import nltk
from collections import Counter
import numpy as np  # Essencial para o Radar
from math import pi # Essencial para o Radar

# --- CONFIGURAÇÕES ---
PASTA_RAW = Path("dados/raw")
PASTA_PROCESSED = Path("dados/processed") # Nova pasta
PASTA_RESULTADOS = Path("resultados")

# --- 1. Gírias e Expressões Informais ---
stop_girias = ['pra', 'pro', 'tá', 'ta', 'tô', 'to', 'vc', 'vcs', 'voce', 'você', 'pq', 'tbm', 'tb', 'bm', 'né', 'ne', 'aí', 'ae', 'ah', 'oh', 'aff', 'blz', 'vlw', 'flw', 'tmj', 'sdd', 'mto', 'mt', 'eh', 'bjs', 'bj', 'top', 'show', 'kra', 'cara', 'mano', 'man', 'tipo', 'so', 'só', 'tudobem', 'td', 'q', 'gnt', 'gente', 'kk', 'kkk', 'kkkk', 'kkkkk', 'kkkkkk', 'kkkkkkk', 'haha', 'hahaha', 'rs']

# --- 2. Verbos de Alta Frequência ---
stop_verbos = ['ser', 'sou', 'é', 'são', 'era', 'eram', 'fui', 'foi', 'foram', 'sendo', 'estar', 'está', 'estão', 'estava', 'estavam', 'tô', 'ter', 'tem', 'têm', 'tinha', 'tinham', 'tenho', 'ir', 'indo', 'vem', 'vir', 'vim', 'voltar', 'volta', 'vai', 'vão', 'vamos', 'fazer', 'faz', 'fez', 'fiz', 'fazendo', 'dar', 'da', 'deu', 'dá', 'dão', 'poder', 'pode', 'podia', 'podem', 'querer', 'quer', 'queria', 'querem', 'falar', 'fala', 'falou', 'falam', 'dizer', 'diz', 'disse', 'ver', 'viu', 'veja', 'olha', 'olhar', 'saber', 'sabe', 'sabia', 'sei', 'achar', 'acho', 'acha', 'ficar', 'fica', 'ficou', 'passar', 'chegar', 'deixar', 'continuar', 'precisa', 'precisar', 'precisou', 'deve']

# --- 3. Conectivos, Pronomes, Advérbios e Tempo ---
stop_conectivos = ['agora', 'hj', 'hoje', 'ontem', 'amanhã', 'amanha', 'dia', 'ano', 'anos', 'hora', 'vez', 'vezes', 'tempo', 'antes', 'depois', 'sempre', 'nunca', 'logo', 'ainda', 'já', 'ja', 'enquanto', 'durante', 'após', 'apos', 'aqui', 'ali', 'lá', 'onde', 'muito', 'pouco', 'mais', 'menos', 'grande', 'pequeno', 'tão', 'tao', 'tanta', 'tanto', 'apenas', 'mesmo', 'nada', 'tudo', 'todo', 'toda', 'todos', 'todas', 'algum', 'alguma', 'alguns', 'coisa', 'desse', 'dessa', 'disso', 'daquele', 'daquela', 'nisso', 'nisto', 'nessa', 'nesse', 'este', 'esta', 'isto', 'outro', 'outra', 'outros', 'outras', 'qualquer', 'quaisquer', 'então', 'entao', 'assim', 'sobre', 'com', 'sem', 'pois', 'porque', 'porquê', 'por', 'que', 'como', 'quando', 'ante', 'talvez', 'sim', 'nao', 'não']

# --- 4. Termos Técnicos e Ruído de Plataforma ---
stop_tecnicos = ['https', 'http', 'tco', 'www', 'com', 'br', 'pt', 'link', 'bio', 'perfil', 'site', 'clique', 'video', 'vídeo', 'foto', 'imagem', 'post', 'story', 'stories', 'retweet', 'rt', 'status', 'null', 'nan', 'twitter', 'instagram', 'facebook', 'youtube', 'whatsapp']

# --- COMBINAÇÃO FINAL ---
STOP_WORDS_EXTRA = stop_girias + stop_verbos + stop_conectivos + stop_tecnicos

class SocialMediaPipe:
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.filename = self.filepath.stem
        self.conta, self.id_conta, self.rede = self._parse_filename()
        self.output_dir = PASTA_RESULTADOS / self.conta / self.id_conta / self.rede
        self.df = None

    def _parse_filename(self):
        """Extrai metadados do nome do arquivo: conta_id_rede.csv"""
        parts = self.filename.split('_')
        if len(parts) >= 3:
            return "_".join(parts[:-2]), parts[-2], parts[-1]
        return "desconhecido", "000", "padrao"

    def preparar_pastas(self):
        """Apenas cria a estrutura de pastas."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{self.conta}] Diretório verificado: {self.output_dir}")

    def carregar_e_normalizar(self):
        """Carrega o CSV e padroniza nomes de colunas."""
        try:
            df_raw = pd.read_csv(self.filepath)
        except Exception as e:
            print(f"Erro ao ler {self.filename}: {e}")
            return False

        # Lógica de Normalização
        rede_lower = self.rede.lower()
        rename_map = {}
        
        if 'instagram' in rede_lower:
            rename_map = {'text': 'conteudo', 'timestamp': 'data', 
                          'likesCount': 'likes', 'repliesCount': 'comentarios'}
        elif 'twitter' in rede_lower:
            rename_map = {'texto': 'conteudo', 'data': 'data', 
                          'likes': 'likes', 'retweets': 'compartilhamentos'}

        self.df = df_raw.rename(columns=rename_map)
        
        # Tratamento de Data
        if 'data' in self.df.columns:
            # Tenta converter automaticamente qualquer formato de data
            self.df['data'] = pd.to_datetime(self.df['data'], errors='coerce', utc=True)
        
        return True

    def executar_processamento(self):
        """
        Lê raw, remove quebras de linha, padroniza com NULL, 
        gera features (incluindo MENÇÕES) e corrige datas.
        """
        import csv 

        PASTA_PROCESSED.mkdir(parents=True, exist_ok=True)
        
        if self.df is None: 
            sucesso = self.carregar_e_normalizar()
            if not sucesso: return

        print(f"--> Processando (Limpeza + Features + Menções) para {self.filename}")
        
        df_proc = self.df.copy()
        
        # 1. Normalização de Usuário
        col_usuario_raw = None
        if 'instagram' in self.rede.lower():
            if 'owner/username' in df_proc.columns: col_usuario_raw = 'owner/username'
            elif 'username' in df_proc.columns: col_usuario_raw = 'username'
        elif 'twitter' in self.rede.lower():
            if 'usuario' in df_proc.columns: col_usuario_raw = 'usuario'
        
        if col_usuario_raw:
            df_proc['usuario'] = df_proc[col_usuario_raw]
        else:
            df_proc['usuario'] = self.conta 

        # 2. Tratamento de Data/Hora
        temp_data_full = pd.to_datetime(df_proc['data'], utc=True, errors='coerce')
        df_proc['data'] = temp_data_full.dt.date
        df_proc['hora'] = temp_data_full.dt.time
        
        # 3. LIMPEZA DE TEXTO
        def limpar_texto_csv(texto):
            if pd.isna(texto): return ""
            txt = str(texto)
            txt = txt.replace('\n', ' ').replace('\r', ' ')
            txt = re.sub(' +', ' ', txt)
            return txt.strip()

        df_proc['conteudo'] = df_proc['conteudo'].apply(limpar_texto_csv)

        # 4. Engenharia de Features (ATUALIZADO COM MENÇÕES)
        def analisar_texto(txt):
            # Regex simples para menções: @ seguido de letras/numeros/underline
            n_mencoes = len(re.findall(r"@\w+", txt))
            
            tem_link = 1 if re.search(r"http\S+|www\S+", txt) else 0
            n_char = len(txt)
            n_palavras = len(txt.split())
            n_emojis = emoji.emoji_count(txt)
            
            return tem_link, n_char, n_palavras, n_emojis, n_mencoes

        metricas = df_proc['conteudo'].apply(analisar_texto)
        
        # Cria as colunas novas
        df_proc[['midia', 'num_caracteres', 'num_palavras', 'num_emojis', 'num_mencoes']] = \
            pd.DataFrame(metricas.tolist(), index=df_proc.index)

        # 5. Padronização final
        rename_map = {
            'conteudo': 'texto',
            'compartilhamentos': 'repostagens'
        }
        df_proc = df_proc.rename(columns=rename_map)

        # Lista Final de Colunas (Adicionado num_mencoes)
        colunas_finais_ordem = [
            'usuario', 'data', 'hora', 'texto', 
            'midia', 'num_caracteres', 'num_palavras', 'num_emojis', 'num_mencoes',
            'likes', 'repostagens', 'comentarios'
        ]

        # Preenchimento de NULL e seleção
        df_final = pd.DataFrame()
        
        for col in colunas_finais_ordem:
            if col in df_proc.columns:
                df_final[col] = df_proc[col].fillna("NULL")
            else:
                df_final[col] = "NULL"

        # 6. Salvar
        nome_saida = f"proc_{self.filename}.csv"
        caminho_saida = PASTA_PROCESSED / nome_saida
        
        df_final.to_csv(caminho_saida, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"    Salvo em: {caminho_saida}")

    def carregar_processado(self):
        """
        Tenta carregar o arquivo da pasta 'processed'.
        Retorna True se conseguir, False se não existir.
        """
        nome_proc = f"proc_{self.filename}.csv"
        caminho_proc = PASTA_PROCESSED / nome_proc
        
        if not caminho_proc.exists():
            print(f"ERRO: Arquivo processado não encontrado: {caminho_proc}")
            print(f"      Rode '--step process' primeiro.")
            return False
            
        # Carrega lidando com os NULLs que criamos
        # 'na_values' converte a string "NULL" de volta para NaN do pandas
        self.df = pd.read_csv(caminho_proc, na_values="NULL")
        
        # Converte data para datetime novamente (pois CSV perdeu o tipo)
        if 'data' in self.df.columns:
            self.df['data'] = pd.to_datetime(self.df['data'], errors='coerce')
            
        return True

    def executar_eda(self):
        """
        Gera 3 gráficos (Timeline, Texto, Correlação) e 1 CSV de resumo
        usando os dados da pasta PROCESSED.
        """
        # Tenta carregar o arquivo processado
        if not self.carregar_processado():
            return

        print(f"--> Gerando EDA Avançada para {self.filename}")
        
        # Garante que as colunas numéricas são números (floats)
        cols_numericas = ['likes', 'comentarios', 'repostagens', 
                          'num_caracteres', 'num_palavras', 'num_emojis', 'num_mencoes']
        
        for col in cols_numericas:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)

        # ARQUIVO 1: Timeline de Engajamento
        if 'data' in self.df.columns:
            plt.figure(figsize=(12, 6))
            df_temp = self.df.sort_values('data')
            
            # Plota Likes
            plt.plot(df_temp['data'], df_temp['likes'], label='Likes', color='blue', alpha=0.7)
            # Plota Comentários (se tiver escala muito diferente, pode ficar pequeno, mas serve para comparar picos)
            plt.plot(df_temp['data'], df_temp['comentarios'], label='Comentários', color='orange', alpha=0.7)
            
            plt.title(f'Timeline de Engajamento - {self.conta}')
            plt.xlabel('Data')
            plt.ylabel('Quantidade')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / "eda_timeline_engajamento.png")
            plt.close()

        # ARQUIVO 2: Distribuição de Texto (Subplots)
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        
        # Histograma Palavras
        sns.histplot(self.df['num_palavras'], kde=True, ax=ax[0], color='teal')
        ax[0].set_title('Distribuição: Nº de Palavras por Post')
        
        # Histograma Emojis
        sns.histplot(self.df['num_emojis'], kde=False, bins=10, ax=ax[1], color='purple')
        ax[1].set_title('Distribuição: Nº de Emojis por Post')
        
        # Histograma Menções
        sns.histplot(self.df['num_mencoes'], kde=False, bins=5, ax=ax[2], color='green')
        ax[2].set_title('Distribuição: Nº de Menções por Post')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "eda_distribuicao_texto.png")
        plt.close()

        # ARQUIVO 3: Matriz de Correlação (Heatmap)
        # Seleciona apenas colunas numéricas relevantes e que tenham variância
        cols_corr = [c for c in cols_numericas if c in self.df.columns and self.df[c].sum() > 0]
        
        if len(cols_corr) > 1:
            plt.figure(figsize=(10, 8))
            corr = self.df[cols_corr].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
            plt.title('Matriz de Correlação')
            plt.tight_layout()
            plt.savefig(self.output_dir / "eda_correlacao_heatmap.png")
            plt.close()
        
        # ARQUIVO 4: Resumo Estatístico (CSV)
        if cols_corr:
            stats = self.df[cols_corr].describe().T
            stats['moda'] = self.df[cols_corr].mode().iloc[0] # Adiciona Moda
            stats.to_csv(self.output_dir / "eda_resumo_estatistico.csv")
            
        print(f"    Gerados 4 arquivos de análise em: {self.output_dir}")

    def executar_sentimento(self):
        """
        Lê processed, aplica modelo BERT (pysentimiento) e salva em resultados
        com colunas de probabilidade e label final.
        """
        # Carrega dados processados (já tratando NULLs)
        if not self.carregar_processado():
            return

        print(f"--> Iniciando Análise de Sentimento (BERT/PT) para {self.filename}")
        print("    (Isso pode demorar um pouco dependendo da quantidade de dados...)")

        # Inicializa o analisador para Português (modelos treinados em tweets PT-BR funcionam bem para redes sociais)
        # task="sentiment" usa o modelo 'pysentimiento/bertweet-pt-sentiment' por padrão
        try:
            analyzer = create_analyzer(task="sentiment", lang="pt")
        except Exception as e:
            print(f"ERRO ao carregar modelo de IA: {e}")
            return

        df_sent = self.df.copy()

        # Função auxiliar para aplicar em cada linha
        def analisar_linha(texto):
            # Se for NULL, vazio ou muito curto, retorna neutro padrão
            if pd.isna(texto) or str(texto) == "NULL" or len(str(texto).strip()) < 2:
                return 0.0, 1.0, 0.0, 'NEU'
            
            try:
                resultado = analyzer.predict(str(texto))
                probs = resultado.probas
                # O output do pysentimiento é POS, NEU, NEG
                return probs.get('POS', 0.0), probs.get('NEU', 0.0), probs.get('NEG', 0.0), resultado.output
            except:
                return 0.0, 1.0, 0.0, 'NEU'

        # Aplica o modelo (cria uma barra de progresso simples visualmente se quiser, mas aqui faremos direto)
        # O resultado é uma série de tuplas, precisamos expandir
        resultados = df_sent['texto'].apply(analisar_linha)
        
        # Cria as novas colunas
        cols_novas = ['prob_positivo', 'prob_neutro', 'prob_negativo', 'sentimento_geral']
        df_sent[cols_novas] = pd.DataFrame(resultados.tolist(), index=df_sent.index)

        # Mapeia os labels em inglês (POS/NEU/NEG) para português, se desejar
        mapa_labels = {'POS': 'positivo', 'NEU': 'neutro', 'NEG': 'negativo'}
        df_sent['sentimento_geral'] = df_sent['sentimento_geral'].map(mapa_labels)

        # Salva o arquivo enriquecido na pasta de RESULTADOS
        nome_saida = f"analise_sentimento_{self.filename}.csv"
        caminho_saida = self.output_dir / nome_saida
        
        import csv
        df_sent.to_csv(caminho_saida, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"    Salvo em: {caminho_saida}")

    def executar_eda_sentimento(self):
        """
        Gera visualizações de Sentimento e ATUALIZA o resumo estatístico
        com métricas de probabilidade (média de positividade, negatividade, etc).
        """
        nome_sent = f"analise_sentimento_{self.filename}.csv"
        caminho_sent = self.output_dir / nome_sent
        
        if not caminho_sent.exists():
            print(f"ERRO: Arquivo de sentimento não encontrado: {caminho_sent}")
            return

        print(f"--> Gerando Gráficos e Atualizando Estatísticas para {self.filename}")
        
        # Carrega dados
        df = pd.read_csv(caminho_sent)
        if 'data' in df.columns:
            df['data'] = pd.to_datetime(df['data'], errors='coerce')
        
        # Garante numéricos
        cols_num = ['likes', 'comentarios', 'repostagens', 'prob_negativo', 'prob_neutro', 'prob_positivo']
        for c in cols_num:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

        paleta_sentimento = {'negativo': '#e74c3c', 'neutro': '#95a5a6', 'positivo': '#2ecc71'}

        # --- PARTE 1: GRÁFICOS (Corrigidos os Warnings) ---

        # GRÁFICO 1: Distribuição Total
        plt.figure(figsize=(8, 5))
        # Correção: adicionei hue e legend=False
        sns.countplot(data=df, x='sentimento_geral', hue='sentimento_geral', legend=False,
                      palette=paleta_sentimento, order=['negativo', 'neutro', 'positivo'])
        plt.title('Distribuição Total dos Sentimentos')
        plt.ylabel('Número de Posts')
        plt.tight_layout()
        plt.savefig(self.output_dir / "sentimento_distribuicao.png")
        plt.close()

        # GRÁFICO 2: Impacto no Engajamento
        if 'likes' in df.columns and df['likes'].sum() > 0:
            plt.figure(figsize=(10, 6))
            q_high = df['likes'].quantile(0.95)
            df_filter = df[df['likes'] < q_high]
            
            # Correção: adicionei hue e legend=False
            sns.boxplot(data=df_filter, x='sentimento_geral', y='likes', hue='sentimento_geral', legend=False,
                        palette=paleta_sentimento, order=['negativo', 'neutro', 'positivo'])
            plt.title('Distribuição de Likes por Sentimento (excluindo outliers)')
            plt.tight_layout()
            plt.savefig(self.output_dir / "sentimento_vs_likes_boxplot.png")
            plt.close()

        # GRÁFICO 3: Timeline
        if 'data' in df.columns:
            plt.figure(figsize=(12, 5))
            df_temp = df.sort_values('data').dropna(subset=['data'])
            # Média móvel
            df_temp['media_movel_neg'] = df_temp['prob_negativo'].rolling(window=5, min_periods=1).mean()
            df_temp['media_movel_pos'] = df_temp['prob_positivo'].rolling(window=5, min_periods=1).mean()
            
            plt.plot(df_temp['data'], df_temp['media_movel_neg'], label='Prob. Negativa', color='#e74c3c')
            plt.plot(df_temp['data'], df_temp['media_movel_pos'], label='Prob. Positiva', color='#2ecc71')
            
            plt.title('Evolução do Tom Emocional')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / "sentimento_timeline.png")
            plt.close()

        # --- PARTE 2: ATUALIZAR RESUMO ESTATÍSTICO ---
        
        # 1. Calcula estatísticas das probabilidades (0 a 1)
        cols_sentimento = ['prob_negativo', 'prob_neutro', 'prob_positivo']
        # Filtra apenas as que existem
        cols_existentes = [c for c in cols_sentimento if c in df.columns]
        
        if cols_existentes:
            stats_sent = df[cols_existentes].describe().T
            stats_sent['moda'] = df[cols_existentes].mode().iloc[0]
            
            # 2. Carrega o arquivo antigo se existir
            arquivo_resumo = self.output_dir / "eda_resumo_estatistico.csv"
            
            if arquivo_resumo.exists():
                try:
                    df_resumo_antigo = pd.read_csv(arquivo_resumo, index_col=0)
                    
                    # Remove linhas antigas de sentimento para não duplicar se rodar 2x
                    linhas_para_manter = [idx for idx in df_resumo_antigo.index if idx not in cols_existentes]
                    df_resumo_antigo = df_resumo_antigo.loc[linhas_para_manter]
                    
                    # Concatena (Junta o antigo com o novo de sentimento)
                    df_final = pd.concat([df_resumo_antigo, stats_sent])
                except Exception as e:
                    print(f"    Aviso: Não consegui ler resumo anterior ({e}). Criando novo.")
                    df_final = stats_sent
            else:
                df_final = stats_sent
            
            # 3. Salva
            df_final.to_csv(arquivo_resumo)
            print(f"    Resumo estatístico ATUALIZADO em: {arquivo_resumo}")
            
            # Extra: Salva contagem bruta (Quantos positivos? Quantos negativos?)
            if 'sentimento_geral' in df.columns:
                contagem = df['sentimento_geral'].value_counts().reset_index()
                contagem.columns = ['sentimento', 'total_posts']
                contagem['porcentagem'] = (contagem['total_posts'] / len(df)) * 100
                contagem.to_csv(self.output_dir / "resumo_contagem_sentimentos.csv", index=False)
                print(f"    Contagem de sentimentos salva em: resumo_contagem_sentimentos.csv")

        print(f"    Gráficos e Tabelas atualizados.")

    def executar_topicos(self, n_topicos=5):
        """
        Gera:
        1. Ranking de Menções (@).
        2. Ranking de Hashtags (#).
        3. Nuvem de Palavras (Imagem PNG).
        4. Tabela de Frequência de palavras.
        5. Modelagem de Tópicos (LDA) -> FORMATO HORIZONTAL (LINHAS).
        """
        if not self.carregar_processado(): return
        print(f"--> Analisando Tópicos, Hashtags e Nuvem para {self.filename}")

        todos_textos_raw = " ".join(self.df['texto'].fillna("").astype(str))
        
        # --- PARTE 1: Ranking de Menções (@) ---
        mencoes = re.findall(r"@[\w\._]+", todos_textos_raw)
        if mencoes:
            df_mencoes = pd.DataFrame(Counter(mencoes).most_common(), columns=['conta', 'frequencia'])
            df_mencoes.to_csv(self.output_dir / f"ranking_mencoes_{self.filename}.csv", index=False)
            print(f"    [1/5] Ranking de menções salvo.")

        # --- PARTE 2: Ranking de Hashtags (#) ---
        hashtags = re.findall(r"#[\w\._]+", todos_textos_raw)
        if hashtags:
            df_hash = pd.DataFrame(Counter(hashtags).most_common(), columns=['hashtag', 'frequencia'])
            df_hash.to_csv(self.output_dir / f"ranking_hashtags_{self.filename}.csv", index=False)
            print(f"    [2/5] Ranking de hashtags salvo.")
        else:
             print(f"    [2/5] Nenhuma hashtag encontrada.")

        # --- PREPARAÇÃO (Stopwords) ---
        try: stop_words = stopwords.words('portuguese')
        except: nltk.download('stopwords', quiet=True); stop_words = nltk.corpus.stopwords.words('portuguese')

        stop_words.extend(STOP_WORDS_EXTRA)

        def limpar_para_topicos(text):
            if pd.isna(text) or str(text) == "NULL": return ""
            txt = re.sub(r'[^\w\s]', '', str(text).lower()) 
            txt = re.sub(r'\d+', '', txt)
            return txt

        textos_limpos = self.df['texto'].apply(limpar_para_topicos)
        texto_gigante_limpo = " ".join(textos_limpos)
        
        # --- PARTE 3: Nuvem de Palavras (Imagem) ---
        if len(texto_gigante_limpo) > 10:
            print(f"    [3/5] Gerando Nuvem de Palavras...")
            wc = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words, colormap='viridis',collocations= False)
            wc.generate(texto_gigante_limpo)
            wc.to_file(self.output_dir / f"nuvem_palavras_{self.filename}.png")
        else:
             print(f"    [3/5] Texto insuficiente para Nuvem de Palavras.")

        # --- PARTE 4: Frequência de Palavras (Tabela) ---
        palavras = [p for p in texto_gigante_limpo.split() if p not in stop_words and len(p) > 2]
        if palavras:
            df_freq = pd.DataFrame(Counter(palavras).most_common(), columns=['palavra', 'frequencia'])
            df_freq.to_csv(self.output_dir / f"frequencia_palavras_total_{self.filename}.csv", index=False)
            print(f"    [4/5] Lista de frequência salva.")
        else:
            return 

        # --- PARTE 5: Modelagem de Tópicos (LDA) - FORMATO HORIZONTAL ---
        print(f"    [5/5] Treinando modelo LDA ({n_topicos} tópicos)...")
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words=stop_words)
        try:
            dtm = vectorizer.fit_transform(textos_limpos)
            lda = LatentDirichletAllocation(n_components=n_topicos, random_state=42)
            lda.fit(dtm)
            
            # Construindo lista de linhas para ficar horizontal
            dados_topicos = []
            feature_names = vectorizer.get_feature_names_out()
            
            for index, topic in enumerate(lda.components_):
                # Pega as top 15 palavras
                top_words = [feature_names[i] for i in topic.argsort()[-15:]]
                # Inverte (mais importantes primeiro)
                top_words = top_words[::-1]
                
                # Cria linha: [Topico_1, palavra1, palavra2, ...]
                linha = [f"Topico_{index+1}"] + top_words
                dados_topicos.append(linha)

            # Cria nomes das colunas: Topico, Palavra_1, Palavra_2...
            colunas = ["Topico"] + [f"Palavra_{i+1}" for i in range(15)]
            
            df_topicos = pd.DataFrame(dados_topicos, columns=colunas)
            df_topicos.to_csv(self.output_dir / f"lda_topicos_{self.filename}.csv", index=False)
            print(f"    Sucesso! Tópicos salvos (orientação horizontal).")
            
        except ValueError:
            print("    ERRO no LDA: Vocabulário insuficiente após limpeza.")

    def executar_extra(self):
        """
        Gera análises avançadas (Versão corrigida para Locale):
        1. Heatmap Temporal.
        2. Dispersão: Tamanho do Texto.
        3. Dispersão: Polarização.
        4. Bigramas e Trigramas.
        5. Outliers.
        """
        # Carrega dados
        arquivo_sent = self.output_dir / f"analise_sentimento_{self.filename}.csv"
        arquivo_proc = PASTA_PROCESSED / f"proc_{self.filename}.csv"
        
        if arquivo_sent.exists():
            df = pd.read_csv(arquivo_sent)
        elif arquivo_proc.exists():
            print("    Aviso: Arquivo de sentimento não achado. Usando dados processados.")
            df = pd.read_csv(arquivo_proc, na_values="NULL")
        else:
            print("    Erro: Nenhum dado encontrado.")
            return

        print(f"--> Gerando Análises Extras para {self.filename}")

        # Tratamento de Data
        if 'data' in df.columns:
            # Cria data completa
            df['data_dt'] = pd.to_datetime(df['data'] + ' ' + df['hora'].astype(str), errors='coerce')
            
            # --- CORREÇÃO DO LOCALE AQUI ---
            # Pegamos o nome em inglês (padrão) e traduzimos manualmente
            df['dia_semana_en'] = df['data_dt'].dt.day_name()
            
            mapa_dias = {
                'Monday': 'Segunda', 'Tuesday': 'Terça', 'Wednesday': 'Quarta',
                'Thursday': 'Quinta', 'Friday': 'Sexta', 'Saturday': 'Sábado',
                'Sunday': 'Domingo'
            }
            df['dia_semana'] = df['dia_semana_en'].map(mapa_dias)
            
            df['hora_int'] = df['data_dt'].dt.hour
        
        # Garante numéricos
        cols_num = ['likes', 'num_palavras', 'prob_negativo']
        for c in cols_num:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

        # --- 1. HEATMAP TEMPORAL ---
        if 'dia_semana' in df.columns and 'hora_int' in df.columns:
            pivot = df.pivot_table(index='dia_semana', columns='hora_int', values='texto', aggfunc='count', fill_value=0)
            
            # Ordena dias da semana em Português
            dias_ordem = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
            pivot = pivot.reindex([d for d in dias_ordem if d in pivot.index])
            
            if not pivot.empty:
                plt.figure(figsize=(12, 6))
                sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt='g')
                plt.title('Mapa de Calor: Frequência de Postagens')
                plt.xlabel('Hora do Dia')
                plt.ylabel('Dia da Semana')
                plt.tight_layout()
                plt.savefig(self.output_dir / "extra_heatmap_horarios.png")
                plt.close()

        # --- 2. SCATTERPLOT: TAMANHO vs LIKES ---
        if 'num_palavras' in df.columns and 'likes' in df.columns:
            plt.figure(figsize=(8, 6))
            sns.regplot(data=df, x='num_palavras', y='likes', scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
            plt.title('Correlação: Tamanho do Texto x Likes')
            plt.tight_layout()
            plt.savefig(self.output_dir / "extra_scatter_tamanho_vs_likes.png")
            plt.close()

        # --- 3. SCATTERPLOT: POLARIZAÇÃO ---
        if 'prob_negativo' in df.columns and 'likes' in df.columns:
            plt.figure(figsize=(8, 6))
            # Ajuste para evitar erro de hue se coluna não existir
            hue_col = 'sentimento_geral' if 'sentimento_geral' in df.columns else None
            sns.scatterplot(data=df, x='prob_negativo', y='likes', alpha=0.6, hue=hue_col, palette='viridis')
            plt.title('Polarização: Negatividade x Likes')
            plt.tight_layout()
            plt.savefig(self.output_dir / "extra_scatter_negatividade_vs_likes.png")
            plt.close()

        # --- 4. BIGRAMAS E TRIGRAMAS ---
        try: stop_words = stopwords.words('portuguese')
        except: nltk.download('stopwords', quiet=True); stop_words = nltk.corpus.stopwords.words('portuguese')
        stop_words.extend(['pra', 'pro', 'que', 'com', 'não', 'uma', 'para', 'https', 'tco'])

        def limpar_ngram(t): return re.sub(r'[^\w\s]', '', str(t).lower())
        textos = df['texto'].fillna('').apply(limpar_ngram) # fillna para evitar crash

        for n, nome in [(2, 'bigramas'), (3, 'trigramas')]:
            try:
                # min_df=2 ignora frases que aparecem só uma vez
                vec = CountVectorizer(ngram_range=(n, n), stop_words=stop_words, min_df=2)
                bow = vec.fit_transform(textos)
                soma_palavras = bow.sum(axis=0) 
                palavras_freq = [(word, soma_palavras[0, idx]) for word, idx in vec.vocabulary_.items()]
                palavras_freq = sorted(palavras_freq, key = lambda x: x[1], reverse=True)
                
                if palavras_freq:
                    df_ngram = pd.DataFrame(palavras_freq, columns=['frase', 'frequencia']).head(50)
                    df_ngram.to_csv(self.output_dir / f"extra_{nome}.csv", index=False)
            except ValueError:
                # Ocorre se o vocabulário for vazio
                pass

        # --- 5. OUTLIERS ---
        if 'likes' in df.columns:
            cols_ver = ['data', 'texto', 'likes', 'comentarios', 'sentimento_geral']
            cols_ver = [c for c in cols_ver if c in df.columns]
            
            if len(df) >= 10:
                top_5 = df.nlargest(5, 'likes')[cols_ver].assign(tipo='TOP 5')
                bottom_5 = df.nsmallest(5, 'likes')[cols_ver].assign(tipo='BOTTOM 5')
                df_outliers = pd.concat([top_5, bottom_5])
                df_outliers.to_csv(self.output_dir / "extra_outliers_posts.csv", index=False)
            
        print(f"    Análises extras salvas em: {self.output_dir}")

class PairComparator:
    """
    Classe dedicada a encontrar e comparar pares de redes sociais.
    Agora com Correlação, Gap de Ódio e Vocabulário Exclusivo.
    """
    def run(self):
        root = Path("resultados")
        if not root.exists():
            print("Pasta 'resultados' não encontrada.")
            return

        print("\n=== INICIANDO COMPARAÇÃO AVANÇADA DE PARES ===\n")
        
        for conta_dir in root.iterdir():
            if not conta_dir.is_dir(): continue
            for id_dir in conta_dir.iterdir():
                if not id_dir.is_dir(): continue
                self.analisar_par(conta_dir.name, id_dir.name, id_dir)

    def analisar_par(self, conta, id_conta, base_path):
        dir_insta = base_path / "instagram"
        dir_twitter = base_path / "twitter"
        
        if not (dir_insta.exists() and dir_twitter.exists()): return 

        print(f"--> Analisando Par: {conta} (ID: {id_conta})")
        
        df_insta = self.carregar_dados(dir_insta)
        df_twitter = self.carregar_dados(dir_twitter)
        
        if df_insta is None or df_twitter is None: return

        df_insta['rede'] = 'Instagram'
        df_twitter['rede'] = 'Twitter'
        
        # DataFrame Conjunto (Vertical)
        df_full = pd.concat([df_insta, df_twitter], ignore_index=True)
        
        # --- ANÁLISE 1: Tabela Básica ---
        resumo = df_full.groupby('rede').agg({
            'texto': 'count',
            'likes': ['sum', 'mean'],
            'comentarios': ['sum', 'mean'],
            'prob_negativo': 'mean'
        }).round(2)
        resumo.to_csv(base_path / f"comparativo_tabela_{conta}_{id_conta}.csv")

        # --- ANÁLISES VISUAIS JÁ EXISTENTES ---
        self.gerar_graficos_barras(df_full, base_path)
        self.gerar_grafico_sentimento(df_full, base_path)
        self.gerar_nuvem_conjunta(df_full, base_path)

        # === NOVIDADES AQUI ===
        
        # 1. Scatter Plot (Correlação de Engajamento)
        self.gerar_scatter_correlacao(df_insta, df_twitter, base_path)
        
        # 2. Gap de Ódio (Diferença Matemática)
        self.calcular_gap_odio(df_insta, df_twitter, base_path)
        
        # 3. Vocabulário Exclusivo
        self.analisar_vocabulario_exclusivo(df_insta, df_twitter, base_path)

        # 4. VENN DIAGRAM VISUAL (NOVO!) <--- ADICIONE AQUI
        try:
            self.gerar_venn_visual(df_insta, df_twitter, base_path)
        except Exception as e:
            print(f"    [!] Erro ao gerar Venn (falta matplotlib-venn?): {e}")

        print("    Todas as análises avançadas foram salvas.")

    # --- MÉTODOS AUXILIARES ---

    def gerar_graficos_barras(self, df, path):
        # (Seu código original de barras aqui...)
        # Vou resumir para não ocupar espaço, mantenha o que você já tinha
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        sns.barplot(data=df, x='rede', y='likes', hue='rede', estimator='mean', ax=ax[0], palette=['#E1306C', '#1DA1F2'])
        sns.barplot(data=df, x='rede', y='comentarios', hue='rede', estimator='mean', ax=ax[1], palette=['#E1306C', '#1DA1F2'])
        plt.tight_layout()
        plt.savefig(path / "comparativo_engajamento.png")
        plt.close()

    def gerar_grafico_sentimento(self, df, path):
        # (Seu código original de sentimento aqui...)
        props = df.groupby(['rede', 'sentimento_geral'], observed=False).size().unstack(fill_value=0)
        props = props.div(props.sum(axis=1), axis=0) * 100
        paleta = {'negativo': '#e74c3c', 'neutro': '#95a5a6', 'positivo': '#2ecc71'}
        ax = props.plot(kind='bar', stacked=True, color=[paleta.get(x, '#333') for x in props.columns], figsize=(8, 6))
        plt.tight_layout()
        plt.savefig(path / "comparativo_sentimentos.png")
        plt.close()

    def gerar_nuvem_conjunta(self, df, path):
        # (Seu código da nuvem conjunta aqui...)
        texto_total = " ".join(df['texto'].astype(str).fillna(""))
        texto_limpo = self.limpar_texto_simples(texto_total)
        if not texto_limpo.strip(): return
        wordcloud = WordCloud(width=1600, height=800, background_color='white', 
                            stopwords=set(nltk.corpus.stopwords.words('portuguese') + STOP_WORDS_EXTRA),
                            collocations=False, max_words=150).generate(texto_limpo)
        wordcloud.to_file(path / "nuvem_palavras_CONJUNTA.png")

    # === NOVOS MÉTODOS ===

    def gerar_scatter_correlacao(self, df_i, df_t, path):
        """
        Cria um gráfico de dispersão comparando Likes do Insta vs Likes do Twitter.
        ATENÇÃO: Isso assume que os dados não estão pareados linha a linha, 
        então comparamos as MÉDIAS ou DISTRIBUIÇÃO.
        Se quisermos parear, precisamos garantir que df_i e df_t tenham o mesmo tamanho.
        Vou truncar para o menor tamanho para permitir o plot.
        """
        min_len = min(len(df_i), len(df_t))
        if min_len < 2: return # Precisa de pelo menos 2 pontos

        # Cria um DF pareado artificialmente (top N posts de cada)
        df_scatter = pd.DataFrame({
            'Likes Instagram': df_i['likes'].head(min_len).values,
            'Likes Twitter': df_t['likes'].head(min_len).values,
            'Sentimento Insta': df_i['prob_negativo'].head(min_len).values,
            'Sentimento Twitter': df_t['prob_negativo'].head(min_len).values
        })

        # Plot 1: Likes vs Likes
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_scatter, x='Likes Instagram', y='Likes Twitter', color='purple', s=100)
        plt.title('Correlação de Engajamento: O viral de um é viral do outro?')
        plt.grid(True, alpha=0.3)
        plt.savefig(path / "extra_scatter_correlacao_likes.png")
        plt.close()

    def calcular_gap_odio(self, df_i, df_t, path):
        """
        Calcula a diferença percentual de negatividade entre as redes.
        """
        neg_insta = df_i['prob_negativo'].mean()
        neg_twitter = df_t['prob_negativo'].mean()
        
        diff = neg_twitter - neg_insta
        diff_perc = (diff * 100)
        
        texto_resultado = (
            f"=== ANÁLISE DO GAP DE ÓDIO ===\n"
            f"Negatividade Média Instagram: {neg_insta:.4f}\n"
            f"Negatividade Média Twitter:   {neg_twitter:.4f}\n"
            f"Diferencial (Twitter - Insta): {diff:.4f} ({diff_perc:.2f}%)\n\n"
            f"Interpretação:\n"
            f"{'O Twitter é MAIS tóxico.' if diff > 0 else 'O Instagram é MAIS tóxico (Surpresa!).'}"
        )
        
        with open(path / "metricas_gap_odio.txt", "w", encoding='utf-8') as f:
            f.write(texto_resultado)

    def analisar_vocabulario_exclusivo(self, df_i, df_t, path):
        """
        Gera dois arquivos de texto com palavras que só existem em uma rede.
        """
        def get_vocab(df):
            texto = " ".join(df['texto'].astype(str).fillna(""))
            limpo = self.limpar_texto_simples(texto)
            return set(limpo.split())

        vocab_insta = get_vocab(df_i)
        vocab_twitter = get_vocab(df_t)
        
        # Diferença de Conjuntos (Set Difference)
        exclusivas_insta = list(vocab_insta - vocab_twitter)
        exclusivas_twitter = list(vocab_twitter - vocab_insta)
        
        # Salva top 50 de cada
        with open(path / "vocabulario_exclusivo.txt", "w", encoding='utf-8') as f:
            f.write(f"=== PALAVRAS EXCLUSIVAS DO INSTAGRAM (Top 50 amostra) ===\n")
            f.write(", ".join(exclusivas_insta[:50]))
            f.write(f"\n\n=== PALAVRAS EXCLUSIVAS DO TWITTER (Top 50 amostra) ===\n")
            f.write(", ".join(exclusivas_twitter[:50]))

    def carregar_dados(self, dir_path):
        files = list(dir_path.glob("analise_sentimento_*.csv"))
        if not files: files = list(dir_path.glob("proc_*.csv"))
        if not files: return None
        df = pd.read_csv(files[0])
        cols = ['likes', 'comentarios', 'prob_negativo', 'prob_positivo', 'num_palavras']
        for c in cols:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        return df

    def limpar_texto_simples(self, texto):
        texto = texto.lower()
        texto = re.sub(r'@\w+', '', texto)
        texto = re.sub(r'http\S+', '', texto)
        texto = re.sub(r'\b[krs]{2,}\b', '', texto)
        texto = re.sub(r'[^a-zA-ZáàâãéèêíïóôõöúçñÁÀÂÃÉÈÊÍÏÓÔÕÖÚÇÑ\s]', '', texto)
        return texto

    def gerar_venn_visual(self, df_i, df_t, path):
        """
        Gera uma imagem de Diagrama de Venn (bolinhas) mostrando a interseção
        dos vocabulários entre Instagram e Twitter.
        """
        # 1. Extrai vocabulários únicos (sets)
        def get_vocab(df):
            texto = " ".join(df['texto'].astype(str).fillna(""))
            limpo = self.limpar_texto_simples(texto)
            return set(limpo.split())

        vocab_insta = get_vocab(df_i)
        vocab_twitter = get_vocab(df_t)
        
        # 2. Configura o gráfico
        plt.figure(figsize=(10, 8))
        plt.title(f"Interseção de Vocabulário: O que é falado nas duas redes?", fontsize=14)
        
        # 3. Gera o Venn
        # set_colors: Insta (Roxo/Rosa), Twitter (Azul)
        venn = venn2([vocab_insta, vocab_twitter], set_labels=('Instagram', 'Twitter'), 
                     set_colors=('#E1306C', '#1DA1F2'), alpha=0.7)
        
        # 4. Estiliza os números (deixa branco e negrito para ler melhor)
        for text in venn.set_labels:
            if text: text.set_fontsize(12)
        for text in venn.subset_labels:
            if text: 
                text.set_fontsize(14)
                text.set_color('white')
                text.set_fontweight('bold')

        # 5. Salva
        plt.savefig(path / "extra_venn_diagram.png")
        plt.close()

class AccountAggregator:
    """
    Classe dedicada a consolidar os dados das 3 coletas (1, 2, 3).
    Gera visão evolutiva, Venn de 3 conjuntos e Correlações.
    """
    def run(self):
        root = Path("resultados")
        if not root.exists(): return
        print("\n=== INICIANDO AGRUPAMENTO POR CONTA (EVOLUÇÃO DOS POSTS) ===\n")
        for conta_dir in root.iterdir():
            if not conta_dir.is_dir(): continue
            self.analisar_conta_completa(conta_dir)

    def analisar_conta_completa(self, conta_path):
        conta_nome = conta_path.name
        print(f"--> Consolidando dados da conta: {conta_nome}")

        dados_para_df = []
        textos_por_id = {} # Dicionário para guardar texto de cada vídeo (para o Venn)

        # 1. Varre as subpastas (1, 2, 3...)
        for id_dir in conta_path.iterdir():
            if not id_dir.is_dir(): continue
            
            # Carrega dados brutos
            df_insta = self.carregar_raw(id_dir / "instagram")
            df_twitter = self.carregar_raw(id_dir / "twitter")
            
            # Pega o texto limpo deste ID (juntando Insta + Twitter)
            textos_deste_id = []

            if df_insta is not None:
                dados_para_df.append({
                    'id_post': id_dir.name, 'rede': 'Instagram',
                    'Média Likes': df_insta['likes'].mean(),
                    'Negatividade': df_insta['prob_negativo'].mean()
                })
                if 'texto' in df_insta.columns:
                    textos_deste_id.extend(df_insta['texto'].dropna().astype(str).tolist())

            if df_twitter is not None:
                dados_para_df.append({
                    'id_post': id_dir.name, 'rede': 'Twitter',
                    'Média Likes': df_twitter['likes'].mean(),
                    'Negatividade': df_twitter['prob_negativo'].mean()
                })
                if 'texto' in df_twitter.columns:
                    textos_deste_id.extend(df_twitter['texto'].dropna().astype(str).tolist())
            
            # Guarda o texto consolidado deste ID para o Venn
            if textos_deste_id:
                texto_unido = " ".join(textos_deste_id)
                # Limpeza simples para garantir palavras únicas
                texto_limpo = re.sub(r'[^a-zA-ZáàâãéèêíïóôõöúçñÁÀÂÃÉÈÊÍÏÓÔÕÖÚÇÑ\s]', '', texto_unido.lower())
                textos_por_id[id_dir.name] = set(texto_limpo.split())

        if not dados_para_df: return

        # 2. Cria DataFrame Mestre e Salva
        df_master = pd.DataFrame(dados_para_df).sort_values(by='id_post')
        df_master.to_csv(conta_path / f"CONSOLIDADO_GERAL_{conta_nome}.csv", index=False)

        # --- GRÁFICOS ---
        
        # A. Evolução (Já existia)
        self.gerar_evolucao_likes(df_master, conta_path, conta_nome)
        
        # B. Venn Triplo (NOVO!)
        # Pega os 3 primeiros IDs encontrados para fazer o Venn
        ids_disponiveis = sorted(list(textos_por_id.keys()))[:3]
        if len(ids_disponiveis) == 3:
            sets = [textos_por_id[i] for i in ids_disponiveis]
            labels = [f"Vídeo {i}" for i in ids_disponiveis]
            self.gerar_venn_triplo(sets, labels, conta_path, conta_nome)
        
        # C. Heatmap de Correlação (NOVO!)
        # Cria um DF numérico agregando tudo
        try:
            df_corr = df_master.pivot_table(index='id_post', columns='rede', values=['Média Likes', 'Negatividade'])
            self.gerar_heatmap_correlacao(df_corr, conta_path)
        except: pass

        self.gerar_radar_contrastante(df_master, conta_path, conta_nome)
        # D. Nuvem Total
        self.gerar_nuvem_total_conta(conta_path, conta_nome)

    # --- MÉTODOS DE GRÁFICOS ---

    def gerar_radar_contrastante(self, df, path, nome):
        try:
            if df.empty: return

            # 1. Preparação dos dados
            cols = ['Likes', 'Comentarios', 'Negatividade', 'Positividade']
            df_plot = df.copy()
            for c in cols:
                # Garante que os dados são numéricos e preenche vazios com zero
                df_plot[c] = pd.to_numeric(df_plot[c], errors='coerce').fillna(0)

            # 2. Normalização INDIVIDUAL por coluna (Min-Max Scaling)
            # Isso é o que resolve o efeito "estilete"
            df_norm = df_plot.copy()
            for col in cols:
                min_v = df_plot[col].min()
                max_v = df_plot[col].max()
                
                if max_v > min_v:
                    # Normaliza entre 0.2 e 1.0 para o ponto não sumir no centro
                    df_norm[col] = 0.2 + 0.8 * (df_plot[col] - min_v) / (max_v - min_v)
                else:
                    # Se não houver variação na métrica, coloca no meio do caminho
                    df_norm[col] = 0.5

            # 3. Configuração do Gráfico Polar
            N = len(cols)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1] # Fecha o polígono
            
            fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
            
            # Ajusta o início para o topo e sentido horário
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)

            # Desenha os eixos e labels
            plt.xticks(angles[:-1], cols, color='black', size=12, fontweight='bold')
            
            # Remove as linhas de grade circulares internas (opcional, para limpar)
            ax.set_rlabel_position(0)
            plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["", "", "", "", ""], color="grey", size=7)
            plt.ylim(0, 1.1)

            # Cores de alto contraste
            cores = ['#FF4500', '#00FF00', '#1E90FF', '#FFD700', '#FF00FF', '#00FFFF']
            
            # 4. Desenha cada vídeo/rede
            for i, (idx, row) in enumerate(df_norm.iterrows()):
                values = row[cols].values.flatten().tolist()
                values += values[:1]
                
                # Legenda identificando vídeo e rede
                label_txt = f"V{row['id_post']} - {row['rede']}"
                
                ax.plot(angles, values, linewidth=3, linestyle='solid', label=label_txt, color=cores[i % len(cores)])
                ax.fill(angles, values, color=cores[i % len(cores)], alpha=0.15)

            plt.title(f"Assinatura de Impacto: {nome}\n(Escala Normalizada por Métrica)", size=16, fontweight='bold', y=1.1)
            
            # Reposiciona a legenda para não cobrir o gráfico
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
            
            plt.tight_layout()
            plt.savefig(path / "extra_radar_contraste.png", bbox_inches='tight', dpi=150)
            plt.close()
            print(f"    [OK] Radar de alto contraste gerado para {nome}.")
            
        except Exception as e:
            print(f"    [!] Erro ao gerar Radar: {e}")

    def gerar_venn_triplo(self, sets_list, labels_list, path, nome):
        """ 
        Gera diagrama de Venn com controle MANUAL de cores para alto contraste.
        Claro nas pontas, Escuro no centro.
        """
        plt.figure(figsize=(10, 10))
        plt.title(f"Interseção de Vocabulário: O que se repete nos 3 vídeos?", fontsize=16)
        
        # Gera o Venn (as cores iniciais não importam, vamos sobrescrever)
        venn = venn3(sets_list, set_labels=labels_list)
        
        # --- MAPA DE CORES DE ALTO CONTRASTE ---
        # IDs explicados: '100' = Só o Conjunto A, '110' = A e B, '111' = A, B e C
        
        cores_personalizadas = {
            # --- PARTES EXCLUSIVAS (Cores Claras/Vibrantes) ---
            '100': '#FFD700',  # Amarelo Ouro (Vídeo 1)
            '010': '#00FFFF',  # Ciano Vibrante (Vídeo 2)
            '001': '#FF69B4',  # Rosa Choque (Vídeo 3)
            
            # --- INTERSEÇÕES DUPLAS (Cores Escuras) ---
            '110': '#4B0082',  # Índigo (Roxo Escuro)
            '101': '#8B0000',  # Vermelho Escuro
            '011': '#006400',  # Verde Escuro
            
            # --- CENTRO (Núcleo Duro) ---
            '111': '#000000'   # Preto Absoluto
        }

        # Aplica as cores manualmente em cada pedaço
        for id_region, cor in cores_personalizadas.items():
            patch = venn.get_patch_by_id(id_region)
            if patch: # Verifica se a região existe (pode ser vazia)
                patch.set_facecolor(cor)
                patch.set_alpha(0.8) # Um pouco de transparência para não ficar "chapado"
                patch.set_edgecolor('white')
                patch.set_linewidth(1)

        # --- AJUSTE DE TEXTO (Legibilidade) ---
        # Como mudamos as cores, precisamos ajustar a cor do texto dos números
        # Texto preto no fundo claro, Texto branco no fundo escuro
        
        ids_claros = ['100', '010', '001']
        
        for id_region in cores_personalizadas.keys():
            label = venn.get_label_by_id(id_region)
            if label:
                label.set_fontsize(13)
                label.set_fontweight('bold')
                # Se for região clara, texto preto. Se for escura, texto branco.
                if id_region in ids_claros:
                    label.set_color('#222222')
                else:
                    label.set_color('white')

        # Ajusta o tamanho dos nomes dos grupos (Vídeo 1, Vídeo 2...)
        for text in venn.set_labels:
            if text:
                text.set_fontsize(14)
                text.set_fontweight('bold')
                text.set_color('#333333')

        plt.tight_layout()
        plt.savefig(path / f"extra_venn_triplo_{nome}.png")
        plt.close()
        print("    Diagrama de Venn (Alto Contraste) salvo.")

    def gerar_heatmap_correlacao(self, df_pivot, path):
        """ Heatmap para ver se Likes e Negatividade estão correlacionados """
        plt.figure(figsize=(8, 6))
        # Calcula correlação entre as colunas
        corr = df_pivot.corr()
        
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title("Mapa de Calor: Correlação entre Métricas")
        plt.tight_layout()
        plt.savefig(path / "extra_heatmap_correlacao.png")
        plt.close()

    def gerar_evolucao_likes(self, df, path, nome):
        try:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df, x='id_post', y='Média Likes', hue='rede', palette=['#E1306C', '#1DA1F2'])
            plt.title(f'Evolução de Likes ({nome})')
            plt.savefig(path / "evolucao_likes_posts.png")
            plt.close()
        except: pass

    def gerar_nuvem_total_conta(self, conta_path, conta_nome):
        # (Seu método de nuvem já existente...)
        # Apenas lembre de usar STOP_WORDS_EXTRA global
        pass 

    def carregar_raw(self, dir_path):
        # (Seu método de carregar CSV já existente...)
        if not dir_path.exists(): return None
        files = list(dir_path.glob("analise_sentimento_*.csv"))
        if not files: files = list(dir_path.glob("proc_*.csv"))
        if not files: return None
        try:
            df = pd.read_csv(files[0])
            for c in ['likes', 'prob_negativo']:
                if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            return df
        except: return None

class Consolidator:
    """
    Gera um DASHBOARD MESTRE consolidando métricas de TODAS as contas e redes.
    Gera CSV, Rankings, Correlações e Análise Textual Global.
    """
    def run(self):
        root = Path("resultados")
        if not root.exists():
            print("Pasta 'resultados' não encontrada.")
            return

        print("\n=== GERANDO RELATÓRIO CONSOLIDADO (DASHBOARD GERAL) ===\n")
        
        lista_resumos = []
        textos_globais = {'Instagram': [], 'Twitter': []} # Para Venn e Nuvem

        # 1. Varredura e Coleta de Dados
        for conta_dir in root.iterdir():
            if not conta_dir.is_dir(): continue
            if "RELATORIO" in conta_dir.name: continue # Pula arquivos soltos
            
            for id_dir in conta_dir.iterdir():
                if not id_dir.is_dir(): continue
                
                for rede_dir in id_dir.iterdir():
                    if not rede_dir.is_dir(): continue
                    
                    # Processa métricas
                    dados, texto_bruto = self.processar_pasta_completa(conta_dir.name, id_dir.name, rede_dir.name, rede_dir)
                    
                    if dados:
                        lista_resumos.append(dados)
                    
                    # Acumula texto para análise global
                    if texto_bruto and rede_dir.name.lower() in textos_globais:
                        textos_globais[rede_dir.name.lower().capitalize()].append(str(texto_bruto))

        if not lista_resumos:
            print("Nenhum dado encontrado para consolidar.")
            return

        # 2. Criação do DataFrame Mestre
        df_final = pd.DataFrame(lista_resumos)
        df_final = df_final.sort_values(by=['Conta', 'Rede'])
        
        # Salva CSV
        caminho_csv = root / "RELATORIO_GERAL_CONSOLIDADO.csv"
        df_final.to_csv(caminho_csv, index=False)
        print(f"✅ Tabela Mestre salva: {caminho_csv}")

        # 3. GERAÇÃO DE GRÁFICOS E INSIGHTS
        print("📊 Gerando gráficos comparativos globais...")
        
        # Configura estilo visual
        sns.set_theme(style="whitegrid")
        palette_redes = {'Instagram': '#E1306C', 'Twitter': '#1DA1F2'}

        # A. Ranking de Engajamento (Quem tem mais Likes totais?)
        self.plot_ranking_engajamento(df_final, root)
        
        # B. Ranking de Negatividade (Quem é mais "tóxico"?)
        self.plot_ranking_negatividade(df_final, root)

        # C. Comparativo de Redes (Boxplot: Insta vs Twitter)
        self.plot_comparativo_redes(df_final, root, palette_redes)

        # D. Correlação (Ódio gera Likes?)
        self.plot_heatmap_correlacao(df_final, root)

        # E. Análise Textual Global (Venn e Nuvem)
        self.analise_textual_global(textos_globais, root)

        print(f"\n🚀 Relatório Completo Finalizado na pasta 'resultados'!")

    def processar_pasta_completa(self, conta, id_conta, rede, pasta):
        """Retorna dicionário de métricas E o texto bruto para acumulação"""
        arquivo = None
        
        # Tenta pegar arquivo com Sentimento (IA)
        files_sent = list(pasta.glob("analise_sentimento_*.csv"))
        if files_sent:
            arquivo = files_sent[0]
        else:
            files_proc = list(pasta.glob("proc_*.csv"))
            if files_proc: arquivo = files_proc[0]
        
        if not arquivo: return None, None

        try:
            df = pd.read_csv(arquivo)
        except: return None, None

        # Garante numéricos
        cols_num = ['likes', 'comentarios', 'prob_negativo', 'num_palavras']
        for c in cols_num:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        
        # Texto para acumulação global
        texto_full = ""
        if 'texto' in df.columns:
            texto_full = " ".join(df['texto'].dropna().astype(str).tolist())

        # Métricas
        resumo = {
            'Conta': conta,
            'Rede': rede.capitalize(), # Padroniza 'Instagram' / 'Twitter'
            'ID_Post': id_conta,
            'Total_Likes': df['likes'].sum(),
            'Media_Likes': df['likes'].mean(),
            'Total_Comentarios': df['comentarios'].sum(),
            'Negatividade_Media': df['prob_negativo'].mean() if 'prob_negativo' in df.columns else 0,
            'Positividade_Media': df['prob_positivo'].mean() if 'prob_positivo' in df.columns else 0,
            'Volume_Posts': len(df)
        }
        
        return resumo, texto_full

    # --- MÉTODOS DE VISUALIZAÇÃO ---

    def plot_ranking_engajamento(self, df, path):
        """Gráfico de barras: Quem soma mais likes no total?"""
        plt.figure(figsize=(12, 6))
        # Agrupa por Conta e soma likes
        df_agg = df.groupby('Conta')['Total_Likes'].sum().sort_values(ascending=False).reset_index()
        
        sns.barplot(data=df_agg, x='Total_Likes', y='Conta', palette='viridis', hue='Conta', legend=False)
        plt.title('Ranking Global de Engajamento (Total de Likes Acumulados)', fontsize=14)
        plt.xlabel('Total de Likes')
        plt.tight_layout()
        plt.savefig(path / "GLOBAL_ranking_engajamento.png", dpi=300)
        plt.close()

    def plot_ranking_negatividade(self, df, path):
        """Gráfico de barras: Qual conta tem a maior média de negatividade?"""
        plt.figure(figsize=(12, 6))
        df_agg = df.groupby('Conta')['Negatividade_Media'].mean().sort_values(ascending=False).reset_index()
        
        sns.barplot(data=df_agg, x='Negatividade_Media', y='Conta', palette='magma', hue='Conta', legend=False)
        plt.title('Ranking de Negatividade (Média de Probabilidade Negativa)', fontsize=14)
        plt.xlabel('Índice de Negatividade (0 a 1)')
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.savefig(path / "GLOBAL_ranking_negatividade.png", dpi=300)
        plt.close()

    def plot_comparativo_redes(self, df, path, cores):
        """Boxplot comparando a distribuição de Likes entre Instagram e Twitter"""
        plt.figure(figsize=(10, 6))
        # Usa escala logarítmica se a diferença for brutal, mas vamos tentar linear primeiro
        sns.boxplot(data=df, x='Rede', y='Media_Likes', palette=cores, hue='Rede', legend=False)
        plt.title('Distribuição de Likes: Instagram vs Twitter (Média por Post)', fontsize=14)
        plt.ylabel('Média de Likes')
        plt.tight_layout()
        plt.savefig(path / "GLOBAL_comparativo_redes_boxplot.png", dpi=300)
        plt.close()

    def plot_heatmap_correlacao(self, df, path):
        """Mapa de calor para ver se Negatividade se correlaciona com Likes"""
        plt.figure(figsize=(8, 6))
        cols_corr = ['Total_Likes', 'Total_Comentarios', 'Negatividade_Media', 'Volume_Posts']
        corr = df[cols_corr].corr()
        
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=1)
        plt.title('Matriz de Correlação Global', fontsize=14)
        plt.tight_layout()
        plt.savefig(path / "GLOBAL_correlacao_heatmap.png", dpi=300)
        plt.close()

    def analise_textual_global(self, textos_dict, path):
        """Gera Nuvem de Palavras Global e Venn Diagram Global"""
        
        # 1. Prepara texto Instagram
        txt_insta = " ".join(textos_dict.get('Instagram', []))
        set_insta = self.limpar_e_criar_set(txt_insta)
        
        # 2. Prepara texto Twitter
        txt_twitter = " ".join(textos_dict.get('Twitter', []))
        set_twitter = self.limpar_e_criar_set(txt_twitter)
        
        if not set_insta and not set_twitter: return

        # --- A. NUVEM DE PALAVRAS GLOBAL (O que todo mundo fala?) ---
        texto_total = txt_insta + " " + txt_twitter
        # Limpeza básica para nuvem
        texto_total = re.sub(r'http\S+|@\w+|[^a-zA-Z\s]', '', texto_total.lower())
        
        try:
            # Usa STOP_WORDS_EXTRA global do arquivo principal
            stops = set(nltk.corpus.stopwords.words('portuguese') + STOP_WORDS_EXTRA)
            wc = WordCloud(width=1600, height=800, background_color='black', 
                          stopwords=stops, max_words=300, collocations=False).generate(texto_total)
            wc.to_file(path / "GLOBAL_nuvem_palavras.png")
            print("    Nuvem de Palavras Global salva.")
        except Exception as e:
            print(f"    Erro na nuvem global: {e}")

        # --- B. VENN GLOBAL (Vocabulário Instagram vs Twitter) ---
        if set_insta and set_twitter:
            plt.figure(figsize=(10, 10))
            plt.title("Universo Instagram vs Universo Twitter (Vocabulário Global)", fontsize=16)
            
            # Cores de Alto Contraste (Solicitado pelo usuário)
            # Insta (Rosa Choque), Twitter (Ciano), Interseção (Preto/Roxo)
            venn = venn2([set_insta, set_twitter], set_labels=('Instagram', 'Twitter'))
            
            # Personalização Manual das Cores
            # '10': Só Insta, '01': Só Twitter, '11': Interseção
            colors = {'10': '#FF1493', '01': '#00BFFF', '11': '#2c3e50'} 
            
            for id_region, color in colors.items():
                patch = venn.get_patch_by_id(id_region)
                if patch:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.8)
                    patch.set_edgecolor('white')
            
            # Ajuste de Texto
            for text in venn.set_labels:
                if text: text.set_fontsize(14); text.set_fontweight('bold')
            
            for text in venn.subset_labels:
                if text: 
                    text.set_fontsize(14); text.set_fontweight('bold'); text.set_color('white')

            plt.savefig(path / "GLOBAL_venn_redes.png", dpi=300)
            plt.close()
            print("    Venn Diagram Global salvo.")

    def limpar_e_criar_set(self, texto):
        if not texto: return set()
        # Limpeza rápida
        texto = texto.lower()
        texto = re.sub(r'http\S+|@\w+|[^a-zA-ZáàâãéèêíïóôõöúçñÁÀÂÃÉÈÊÍÏÓÔÕÖÚÇÑ\s]', '', texto)
        tokens = texto.split()
        # Filtra stopwords
        stops = set(nltk.corpus.stopwords.words('portuguese') + STOP_WORDS_EXTRA)
        return set([t for t in tokens if t not in stops and len(t) > 2])
# --- CONTROLADOR PRINCIPAL ---

def main():
    parser = argparse.ArgumentParser(description="Processador Modular de Redes Sociais")
    
    # Argumento para escolher a função
    parser.add_argument(
        '--step', 
        choices=['setup', 'process', 'eda', 'sentimento', 'eda_sentimento', 'topicos', 'compare', 'relatorio','aggregate' ,'all'],        
        required=True,
        help="Escolha qual etapa executar."
    )
    
    # Argumento opcional para rodar apenas um arquivo específico
    parser.add_argument(
        '--file',
        help="Nome do arquivo específico (opcional). Se vazio, roda em todos.",
        default=None
    )

    args = parser.parse_args()

    if args.step == 'compare':
        comp = PairComparator()
        comp.run()
        return
    
    if args.step == 'aggregate':
        agg = AccountAggregator()
        agg.run()
        return

    if args.step == 'relatorio':  
        cons = Consolidator()
        cons.run()
        return

    # Define quais arquivos processar
    if args.file:
        files = [PASTA_RAW / args.file]
    else:
        files = list(PASTA_RAW.glob("*.csv"))

    if not files:
        print("Nenhum arquivo encontrado.")
        return

    # Loop de Execução
    for f in files:
        pipe = SocialMediaPipe(f)
        
        # Garante que as pastas existem antes de qualquer coisa
        pipe.preparar_pastas() 

        if args.step == 'setup':
            continue 

        elif args.step == 'process':
            pipe.executar_processamento() 
            
        elif args.step == 'eda':
            pipe.executar_eda()

        elif args.step == 'sentimento':     
            pipe.executar_sentimento()
            
        elif args.step == 'edasentimento': 
            pipe.executar_eda_sentimento()

        elif args.step == 'topicos':   
            pipe.executar_topicos()

        elif args.step == 'extra':   # <--- NOVO
            pipe.executar_extra()
            
        elif args.step == 'all':
            # Roda a sequência lógica completa
            pipe.executar_processamento()
            pipe.executar_eda() # Gera estatisticas base
            pipe.executar_sentimento()
            pipe.executar_eda_sentimento() # Atualiza estatisticas com IA
            pipe.executar_topicos()
            pipe.executar_extra()

if __name__ == "__main__":
    main()