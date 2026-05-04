import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import re
import unicodedata

# Carrega o CSV consolidado
nome_arquivo = "base_analise_sentimentos.csv" 
df = pd.read_csv(nome_arquivo)

# Limpa linhas vazias e garante que o texto e o sentimento estejam em formato correto
df = df.dropna(subset=['texto', 'sentimento_geral'])
df['texto'] = df['texto'].astype(str)
df['sentimento_geral'] = df['sentimento_geral'].astype(str).str.lower()

# Lista base de stopwords (palavras que não agregam sentido)
stopwords = set([
    "o", "a", "os", "as", "um", "uma", "uns", "umas", "e", "do", "da", "dos", "das", 
    "de", "em", "para", "com", "que", "nao", "na", "no", "nas", "nos", "se", "por", 
    "como", "mais", "mas", "foi", "ao", "aos", "ou", "seu", "sua", "isso", "esse", "essa", 
    "ele", "ela", "voce", "pra", "ja", "ser", "so", "esta", "tem", "me", "te", "q",
    "vc", "vcs", "tbm", "tb", "pq", "pro", "pras", "pros", "muito", "muita",
    "pelo", "pela", "qual", "quando", "estao", "este", "isto", "nem", "sem", "sobre", 
    "suas", "seus", "tambem", "ter", "ate", "ainda", "vai", "vao", "era", "sao", "sou", 
    "nosso", "nossa", "nossos"
])

# Palavras extras, gírias e lixo de campanhas específicas (como fux, toga, etc)
stopwords_extras = [
    "agora", "mesmo", "todo", "toda", "todos", "todas", "cara", "desse", "dessa", "desses", "dessas",
    "nada", "porque", "bem", "assim", "ver", "fala", "falar", "fazendo", "sempre", "dia", "ano",
    "hoje", "quem", "gente", "outra", "outro", "outros", "outras", "onde", "apenas", "entao",
    "tava", "tao", "ne", "ai", "la", "deu", "dar", "vem", "vou", "vez", "algo", "alguem",
    "qualquer", "pode", "podem", "fazer", "fez", "sabe", "acho", "acha", "coisa", "coisas", "tudo",
    "aqui", "ali", "deve", "disso", "daquilo", "aquilo", "estes", "estas", "sim", "ficar", "diz",
    "dizer", "cara", "ter", "fui", "fomos", "foram", "tinha", "logo", "ai", 
    "fuxhonraatoga", "magnitskynaglobo", "fux", "honra", "honras", "toga", "togas"
]
stopwords.update(stopwords_extras)

# Função para remover acentos
def remover_acentos(txt):
    return ''.join(c for c in unicodedata.normalize('NFD', txt) if unicodedata.category(c) != 'Mn')

# Função de limpeza blindada (remove links, menções, risadas, acentos)
def processar_texto(textos):
    todas_palavras = []
    for texto in textos:
        texto = str(texto).lower()
        texto = re.sub(r'http[s]?://\S+', '', texto)
        texto = re.sub(r'www\.\S+', '', texto)
        texto = re.sub(r'@\w+', '', texto)
        texto = re.sub(r'\b([kakh]{3,}|[rs]{3,})\b', '', texto)
        texto = remover_acentos(texto)
        palavras = re.findall(r'\b[a-z]+\b', texto)
        palavras_limpas = [p for p in palavras if p not in stopwords and len(p) > 2]
        todas_palavras.extend(palavras_limpas)
    return todas_palavras

print("Separando os textos por sentimento...")

# Filtra o DataFrame apenas para os comentários positivos e negativos
textos_positivos = df[df['sentimento_geral'] == 'positivo']['texto']
textos_negativos = df[df['sentimento_geral'] == 'negativo']['texto']

print("Limpando e contando as palavras dos positivos...")
palavras_pos = processar_texto(textos_positivos)
contagem_pos = Counter(palavras_pos)

print("Limpando e contando as palavras dos negativos...")
palavras_neg = processar_texto(textos_negativos)
contagem_neg = Counter(palavras_neg)

# --- 1. NUVEM DE PALAVRAS: POSITIVOS ---
print("Desenhando Nuvem de Positivos...")
wc_pos = WordCloud(width=800, height=400, background_color='white', colormap='winter')
wc_pos.generate_from_frequencies(contagem_pos)

plt.figure(figsize=(10, 5))
plt.imshow(wc_pos, interpolation='bilinear')
plt.axis('off')
plt.title('Palavras Mais Frequentes - Sentimento POSITIVO', fontsize=16, color='darkgreen', pad=15)
plt.tight_layout()
plt.savefig('nuvem_positivos.png', dpi=300)
plt.close()

# --- 2. NUVEM DE PALAVRAS: NEGATIVOS ---
print("Desenhando Nuvem de Negativos...")
wc_neg = WordCloud(width=800, height=400, background_color='white', colormap='tab10')
wc_neg.generate_from_frequencies(contagem_neg)

plt.figure(figsize=(10, 5))
plt.imshow(wc_neg, interpolation='bilinear')
plt.axis('off')
plt.title('Palavras Mais Frequentes - Sentimento NEGATIVO', fontsize=16, color='darkred', pad=15)
plt.tight_layout()
plt.savefig('nuvem_negativos.png', dpi=300)
plt.close()

print("\nSucesso! Arquivos 'nuvem_positivos.png' e 'nuvem_negativos.png' gerados e salvos!")