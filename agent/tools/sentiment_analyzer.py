# agent/tools/sentiment_analyzer.py
from typing import Dict, Any
import logging

# lazy pipeline
_sentiment_pipeline = None
_pipeline_available = False

TURKISH_SENTIMENT_MODEL = "savasy/bert-base-turkish-sentiment-cased"


def _init_pipeline():
    global _sentiment_pipeline, _pipeline_available
    if _sentiment_pipeline is not None:
        return True
    try:
        from transformers import pipeline


        # Varsayılan İngilizce model yerine, spesifik Türkçe modelini yüklüyoruz.
        print(f"🔹 Türkçe duygu analiz modeli yükleniyor: {TURKISH_SENTIMENT_MODEL}")
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=TURKISH_SENTIMENT_MODEL,
            tokenizer=TURKISH_SENTIMENT_MODEL
        )


        _pipeline_available = True
        print("✅ Türkçe duygu analiz modeli yüklendi.")
        return True
    except Exception as e:
        # Hata mesajını yazdırmak, sorunu anlamak için önemlidir
        logging.error(f"Transformers pipeline (Türkçe) yüklenemedi: {e}")
        _pipeline_available = False
        return False


def analyze_sentiment(text: str, use_transformers: bool = True) -> Dict[str, Any]:
    """
    Dönen yapı:
      {"status":"success","method":"transformers"|"textblob"|"llm","label":"POSITIVE","score":0.95}
    """
    # 1) Transformers pipeline
    if use_transformers and _init_pipeline():
        try:
            res = _sentiment_pipeline(text[:512])
            if isinstance(res, list) and res:
                label = res[0].get("label")
                score = float(res[0].get("score", 0.0))

                # ETİKET EŞLEŞTİRME (savasy/.. modeli için)
                # Bu model 'positive', 'negative', 'neutral' etiketlerini döner.
                sentiment_map = {
                    "positive": "positive",
                    "negative": "negative",
                    "neutral": "neutral"
                }
                sentiment = sentiment_map.get(label.lower(), label.lower())


                return {"status":"success","method":"transformers","sentiment":sentiment,"label":label,"score":score}
        except Exception as e:
            logging.debug(f"transformers sentiment error: {e}")

    # 2) TextBlob fallback
    try:
        from textblob import TextBlob
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0.1:
            sentiment = "positive"
        elif polarity < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        return {"status":"success","method":"textblob","sentiment":sentiment,"score":polarity}
    except Exception as e:
        logging.debug(f"textblob sentiment error: {e}")

    # 3) last resort: simple heuristic
    pos_words = ["iyi","harika","seviyorum","mutlu","memnun","tebrik"]
    neg_words = ["kötü","üzgün","sinir","nefret","hata","korku","endişe"]
    score = 0.0
    lw = text.lower()
    for w in pos_words:
        if w in lw: score += 0.5
    for w in neg_words:
        if w in lw: score -= 0.5
    sentiment = "neutral"
    if score > 0.1: sentiment = "positive"
    elif score < -0.1: sentiment = "negative"
    return {"status":"success","method":"heuristic","sentiment":sentiment,"score":score}
