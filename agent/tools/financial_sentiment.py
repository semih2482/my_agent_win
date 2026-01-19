# agent/tools/financial_sentiment.py
import requests
from typing import Dict, Any

# Hugging Face Inference API için tokenınızı buraya ekleyin
# veya Transformers kütüphanesini yerel olarak kurun.
# Eğer API kullanıyorsanız:
# HUGGING_FACE_API_TOKEN = os.getenv("HUGGING_FACE_API_TOKEN")
# headers = {"Authorization": f"Bearer {HUGGING_FACE_API_TOKEN}"}
# API_URL = "https://api-inference.huggingface.co/models/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"

# Örnek: Yerel Transformers kütüphanesi ile daha iyi performans
try:
    from transformers import pipeline
    _sentiment_analyzer = pipeline("sentiment-analysis", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    _LOCAL_MODEL = True
except ImportError:
    print("Transformers kütüphanesi bulunamadı. Lütfen 'pip install transformers' komutunu çalıştırın.")
    _LOCAL_MODEL = False

def analyze_financial_sentiment(text: str) -> Dict[str, Any]:
    """
    Verilen metnin finansal duygu analizini yapar.
    """
    if not _LOCAL_MODEL:
        return {"status": "error", "message": "Duygu analizi modeli yüklenemedi. Lütfen transformers kütüphanesini kurun."}

    try:
        if len(text) > 512:
            text = text[:512] # Modeller genellikle metin boyutu sınırı taşır.

        result = _sentiment_analyzer(text)
        sentiment_label = result[0]['label']
        confidence = result[0]['score']

        # Etiketleri daha anlaşılır hale getir
        sentiment_map = {
            'positive': 'Pozitif',
            'negative': 'Negatif',
            'neutral': 'Nötr'
        }

        return {
            "status": "success",
            "result": {
                "sentiment": sentiment_map.get(sentiment_label, 'Bilinmiyor'),
                "confidence": round(confidence, 2)
            }
        }
    except Exception as e:
        return {"status": "error", "message": f"Finansal duygu analizi hatası: {e}"}