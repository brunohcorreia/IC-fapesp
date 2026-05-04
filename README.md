# O MEIO AFETA O DISCURSO? COMO O BOLSONARISMO SE COMUNICA EM DIFERENTES REDES SOCIAIS 
Este repositório contém o arcabouço tecnológico e metodológico desenvolvido para o projeto de análise de dados de redes sociais (Instagram e Twitter). O sistema foi projetado para realizar desde a coleta e normalização de dados brutos até a aplicação de modelos de Inteligência Artificial para análise de sentimento e extração de tópicos latentes.

## 1. Estrutura do Repositório e Navegação

O repositório está organizado de forma a separar a lógica de processamento dos dados brutos e dos resultados gerados. A navegação deve seguir a lógica abaixo:

### 1.1. Diretórios Principais

* **`codigos/`**: Contém todos os scripts em Python responsáveis pela execução do projeto. É aqui que reside a inteligência do sistema, incluindo o motor de processamento e a suíte de testes estatísticos.
* **`resultados/`**: Organiza as saídas geradas pelos scripts. A estrutura interna é hierárquica: `Nome_do_Perfil` -> `ID_da_Conta` -> `Rede_Social`. Dentro de cada pasta final, encontram-se tabelas CSV com as análises individuais e gráficos de desempenho (Timelines, Nuvem de Palavras, Heatmaps).
* **`testes/`**: Destinado exclusivamente à validação científica do modelo. Contém os resultados da acurácia da Inteligência Artificial e os relatórios de significância estatística.
* **`Prints/`**: Armazena evidências visuais das postagens coletadas, servindo como base de verificação manual (Ground Truth) para os dados processados.

### 1.2. Arquivos na Raiz

* **`base_analise_sentimentos.csv`**: Base de dados consolidada e já processada pelo modelo de sentimento.
* **`README.md`**: Este documento de orientação.
* **`arquivos.txt`**: Arvore dos arquivos do projeto

---

## 2. Descrição dos Códigos e Módulos

Os scripts foram desenvolvidos de forma modular para permitir a manutenção e a escalabilidade das análises.

### 2.1. Processamento e Análise (Motor Principal)

* **`analise_modular.py`**: É o script central do projeto. Ele gerencia o fluxo ETL (Extração, Transformação e Carga). Suas funções incluem a normalização de arquivos heterogêneos do Instagram e Twitter, a limpeza de ruídos textuais e a aplicação do modelo **BERTweet-BR** para classificação de sentimentos.
* **`twitter_collector.py`**: Módulo dedicado à extração de dados via API ou técnicas de raspagem, respeitando os limites de taxa das plataformas.

### 2.2. Validação e Estatística

* **`testes.py`**: Script fundamental para o rigor acadêmico da pesquisa. Ele executa o **Teste de Mann-Whitney U**, um método não-paramétrico utilizado para comparar se as diferenças de sentimento entre as redes sociais são estatisticamente significantes (p < 0.05).
* **`metricas.py`**: Calcula indicadores de performance da IA, como Precisão, Recall e F1-Score, comparando a classificação automática com a revisão humana.

### 2.3. Visualização de Dados

* **`graf.py`, `boxplot.py` e `nuvem_se.py`**: Scripts especializados na geração de visualizações técnicas. Eles transformam os dados estatísticos em gráficos de distribuição (Boxplots), diagramas de Venn para comparação de vocabulário e nuvens de palavras baseadas em frequência léxica.

---

## 3. Metodologia e Ferramentas

O projeto utiliza a linguagem **Python** devido à sua robustez em computação científica. As principais bibliotecas e métodos aplicados são:

* **Processamento de Linguagem Natural**: Uso da arquitetura *Transformer* (via `pysentimiento`) para detecção de contexto e ironia em textos curtos.
* **Modelagem de Tópicos**: Aplicação do algoritmo **LDA (Latent Dirichlet Allocation)** para identificar temas recorrentes sem a necessidade de rotulagem prévia.
* **Engenharia de Dados**: Uso de `Pandas` e `NumPy` para manipulação vetorial de dados e tratamento de séries temporais.

---

## 4. Referências Bibliográficas

* **BERT / Transformer**: DEVLIN, Jacob et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv preprint arXiv:1810.04805, 2018.
* **Pysentimiento**: PÉREZ, Juan Manuel; GIUDICI, Juan Carlos; LUQUE, Franco. pysentimiento: A Python Toolkit for Sentiment Analysis and SocialNLP tasks. arXiv preprint arXiv:2106.09462, 2021.
* **LDA**: BLEI, David M.; NG, Andrew Y.; JORDAN, Michael I. Latent dirichlet allocation. Journal of machine Learning research, v. 3, n. Jan, p. 993-1022, 2003.
