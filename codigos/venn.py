import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from wordcloud import WordCloud
from collections import Counter
import re
import unicodedata

# Carrega o CSV consolidado
nome_arquivo = "base_analise_sentimentos.csv" 
df = pd.read_csv(nome_arquivo)

df = df.dropna(subset=['texto'])
df['texto'] = df['texto'].astype(str)

# Lista base de stopwords
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

# Adicionei "fux", "honra", "toga" e variações aqui!
stopwords_extras = [
    "agora", "mesmo", "todo", "toda", "todos", "todas", "cara", "desse", "dessa", "desses", "dessas",
    "nada", "porque", "bem", "assim", "ver", "fala", "falar", "fazendo", "sempre", "dia", "ano",
    "hoje", "quem", "gente", "outra", "outro", "outros", "outras", "onde", "apenas", "entao",
    "tava", "tao", "ne", "ai", "la", "deu", "dar", "vem", "vou", "vez", "algo", "alguem",
    "qualquer", "pode", "podem", "fazer", "fez", "sabe", "acho", "acha", "coisa", "coisas", "tudo",
    "aqui", "ali", "deve", "disso", "daquilo", "aquilo", "estes", "estas", "sim", "ficar", "diz",
    "dizer", "cara", "ter", "fui", "fomos", "foram", "tinha", "logo", "ai", "fuxhonraatoga", "magnitskynaglobo",
    "fux", "honra", "toga" # <-- AQUI ESTÃO AS PALAVRAS DA CAMPANHA
]
stopwords.update(stopwords_extras)

def remover_acentos(txt):
    return ''.join(c for c in unicodedata.normalize('NFD', txt) if unicodedata.category(c) != 'Mn')

def processar_texto(textos):
    todas_palavras = []
    for texto in textos:
        texto = str(texto).lower()
        
        # Remove URLs e menções
        texto = re.sub(r'http[s]?://\S+', '', texto)
        texto = re.sub(r'www\.\S+', '', texto)
        texto = re.sub(r'@\w+', '', texto)
        
        # Remove risadas da internet (kkkk, hahaha, rsrsrs, ahaha)
        texto = re.sub(r'\b([kakh]{3,}|[rs]{3,})\b', '', texto)
        
        texto = remover_acentos(texto)
        palavras = re.findall(r'\b[a-z]+\b', texto)
        
        # Ignora palavras que estão na lista de stopwords e que tem menos de 3 letras
        palavras_limpas = [p for p in palavras if p not in stopwords and len(p) > 2]
        todas_palavras.extend(palavras_limpas)
        
    return todas_palavras

print("Limpando textos e removendo palavras de campanha...")

palavras_ig = processar_texto(df[df['rede'].str.lower() == 'instagram']['texto'])
palavras_tw = processar_texto(df[df['rede'].str.lower() == 'twitter']['texto'])

set_ig = set(palavras_ig)
set_tw = set(palavras_tw)

# --- 1. NUVENS DE PALAVRAS ---
print("Gerando Nuvem do Instagram limpa...")
wc_ig = WordCloud(width=800, height=400, background_color='white', colormap='magma', 
                  collocations=False).generate(" ".join(palavras_ig))
plt.figure(figsize=(10, 5))
plt.imshow(wc_ig, interpolation='bilinear')
plt.axis('off')
plt.title('Palavras Mais Frequentes - Instagram', fontsize=16)
plt.tight_layout()
plt.savefig('nuvem_instagram.png', dpi=300)
plt.close()

print("Gerando Nuvem do Twitter limpa...")
wc_tw = WordCloud(width=800, height=400, background_color='white', colormap='Blues', 
                  collocations=False).generate(" ".join(palavras_tw))
plt.figure(figsize=(10, 5))
plt.imshow(wc_tw, interpolation='bilinear')
plt.axis('off')
plt.title('Palavras Mais Frequentes - Twitter', fontsize=16)
plt.tight_layout()
plt.savefig('nuvem_twitter.png', dpi=300)
plt.close()

# --- 2. DIAGRAMA DE VENN ---
print("Gerando Diagrama de Venn...")
plt.figure(figsize=(8, 6))
venn2([set_ig, set_tw], set_labels=('Instagram', 'Twitter'), set_colors=('#E1306C', '#1DA1F2'))
plt.title('Intersecção do Vocabulário (Palavras Únicas)', fontsize=14, pad=15)
plt.tight_layout()
plt.savefig('venn_vocabulario.png', dpi=300)
plt.close()

print("Sucesso! As palavras da campanha do Gayer foram removidas das imagens.")