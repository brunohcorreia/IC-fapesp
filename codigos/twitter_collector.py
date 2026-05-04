import os
import tweepy
from dotenv import load_dotenv
import csv
from datetime import datetime
import time
import logging
import traceback
import sys

# --- CONFIGURAÇÃO ---
ACCOUNTS = ["CarlosBolsonaro", "SF_Moro", "Biakicis", "FlavioBolsonaro", "RenanSantosMBL", "KimKataguiri"] #, "jonesmanoel_PE", "ErikakHilton", "OtoniDepFederal", "EduGiraoOficial", "marcelvanhattem"]
MAX_TWEETS_PER_ACCOUNT = 800
SAVE_FOLDER = "coletas"

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('twitter_monitor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

load_dotenv()

# Configurar Cliente
try:
    client = tweepy.Client(
        bearer_token=os.getenv("TWITTER_BEARER_TOKEN"),
        consumer_key=os.getenv("TWITTER_API_KEY"),
        consumer_secret=os.getenv("TWITTER_API_SECRET"),
        wait_on_rate_limit=True
    )
except Exception as e:
    logger.error(f"Erro na config: {str(e)}")
    sys.exit(1)

def get_highest_bitrate_video(variants):
    if not variants: return None
    mp4_variants = [v for v in variants if v.get('content_type') == 'video/mp4']
    if not mp4_variants: return None
    best_video = sorted(mp4_variants, key=lambda x: x.get('bit_rate', 0), reverse=True)[0]
    return best_video.get('url')

def process_media(media_keys, includes):
    if not media_keys or 'media' not in includes: return []
    media_info = []
    media_map = {m.media_key: m for m in includes['media']}
    for key in media_keys:
        media_item = media_map.get(key)
        if media_item:
            m_data = {'type': media_item.type}
            if media_item.type == 'photo':
                m_data['url'] = media_item.url
            elif media_item.type in ['video', 'animated_gif']:
                if hasattr(media_item, 'variants'):
                    m_data['url'] = get_highest_bitrate_video(media_item.variants)
                else:
                    m_data['url'] = media_item.preview_image_url
            media_info.append(m_data)
    return media_info

def collect_and_save_user_timeline(username):
    """Coleta e salva IMEDIATAMENTE no arquivo a cada lote"""
    try:
        user = client.get_user(username=username)
        if not user.data:
            logger.error(f"Usuário {username} não encontrado.")
            return

        user_id = user.data.id
        logger.info(f"Iniciando coleta para @{username} (ID: {user_id})...")

        # Preparar Arquivo CSV
        if not os.path.exists(SAVE_FOLDER):
            os.makedirs(SAVE_FOLDER)
        
        filename = f"{SAVE_FOLDER}/tweets_{username}_{datetime.now().strftime('%Y%m%d')}.csv"
        file_exists = os.path.isfile(filename)

        # Abre o arquivo em modo 'a' (append/adicionar)
        with open(filename, 'a', newline='', encoding='utf-8') as f:
            fieldnames = ['id', 'created_at', 'text', 'likes', 'retweets', 'media_types', 'media_urls']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # Escreve o cabeçalho apenas se o arquivo for novo
            if not file_exists:
                writer.writeheader()

            # Removido o argumento limit=limit que estava causando erro
            paginator = tweepy.Paginator(
                client.get_users_tweets,
                id=user_id,
                max_results=100,
                tweet_fields=["created_at", "public_metrics", "text", "attachments"],
                expansions=["attachments.media_keys"],
                media_fields=["type", "url", "preview_image_url", "variants", "duration_ms"]
            )

            total_collected = 0

            for response in paginator:
                if not response.data:
                    continue

                includes = response.includes if hasattr(response, 'includes') else {}
                batch_data = []

                for tweet in response.data:
                    media_urls = []
                    media_types = []
                    
                    if tweet.attachments and 'media_keys' in tweet.attachments:
                        processed = process_media(tweet.attachments['media_keys'], includes)
                        for p in processed:
                            media_types.append(p['type'])
                            media_urls.append(p['url'])

                    tweet_data = {
                        'id': tweet.id,
                        'created_at': tweet.created_at,
                        'text': tweet.text.replace('\n', ' ').replace('\r', ''),
                        'likes': tweet.public_metrics['like_count'],
                        'retweets': tweet.public_metrics['retweet_count'],
                        'media_types': ", ".join(media_types),
                        'media_urls': ", ".join([str(u) for u in media_urls if u])
                    }
                    batch_data.append(tweet_data)
                
                # SALVA O LOTE IMEDIATAMENTE
                writer.writerows(batch_data)
                f.flush() # Força a gravação no disco
                
                total_collected += len(batch_data)
                logger.info(f"@{username}: +{len(batch_data)} salvos (Total: {total_collected})")

                if total_collected >= MAX_TWEETS_PER_ACCOUNT:
                    logger.info(f"Limite atingido para @{username}")
                    break

    except Exception as e:
        logger.error(f"Erro crítico ao coletar {username}: {traceback.format_exc()}")

# --- EXECUÇÃO ---
if __name__ == "__main__":
    logger.info("=== Iniciando Monitoramento Seguro ===")
    
    for account in ACCOUNTS:
        collect_and_save_user_timeline(account) 
        logger.info(f"Concluído (ou pausado) para {account}. Aguardando 5s...")
        time.sleep(5)
        
    logger.info("=== Processo Finalizado ===")