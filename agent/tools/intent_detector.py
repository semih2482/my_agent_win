import json
import os
import re
from typing import Dict, Any, List, Tuple

import numpy as np

# Lazy-loaded models and embeddings
_embed = None
_intent_data = None
_intent_embeddings = None

# Proje kök dizinini al
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def _get_models():
    """Modelleri yalnızca gerektiğinde yükle."""
    global _embed
    if _embed is None:
        from agent.models.llm import embed
        _embed = embed
    return _embed

def _load_intent_data():
    """Niyet verilerini JSON dosyasından yükle."""
    global _intent_data
    if _intent_data is None:
        file_path = os.path.join(project_root, 'agent', 'intents.json')
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                _intent_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Hata: Niyet dosyası okunamadı: {e}")
            _intent_data = {"intents": []}
    return _intent_data

def _get_intent_embeddings() -> List[Tuple[str, str, np.ndarray]]:
    """
    Niyet örnekleri için embedding'leri bir kere hesapla ve cache'le.
    Her bir örnek için (intent_adı, strateji, embedding) içeren bir liste döner.
    """
    global _intent_embeddings
    embed_func = _get_models()
    intent_data = _load_intent_data()

    if _intent_embeddings is None:
        _intent_embeddings = []
        for intent_info in intent_data.get("intents", []):
            intent_name = intent_info.get("name")
            strategy = intent_info.get("strategy", "reactive") # Varsayılan strateji
            for example in intent_info.get("examples", []):
                try:
                    embedding = np.array(embed_func(example))
                    _intent_embeddings.append((intent_name, strategy, embedding))
                except Exception as e:
                    print(f"'{example}' için embedding oluşturulurken hata: {e}")
    return _intent_embeddings

def detect_intent(user_input: str) -> Dict[str, Any]:
    """
    Kullanıcı isteğinin niyetini ve stratejisini, anlamsal benzerlik kullanarak belirler.

    Dönen dict:
      {
        "intent": "...",
        "strategy": "...",
        "confidence": 0-1,
        "source": "embedding" | "default"
      }
    """
    # Hızlı yol: Çok basit sohbet kalıpları için regex
    chat_pattern = re.compile(r"^\s*(merhaba|selam|naber|nasılsın|günaydın|iyi günler|iyi akşamlar)\s*$", re.IGNORECASE)
    if chat_pattern.match(user_input):
        return {"intent": "chat", "strategy": "reactive", "confidence": 0.95, "source": "regex"}

    embed_func = _get_models()
    all_example_embeddings = _get_intent_embeddings()

    if not all_example_embeddings:
        return {"intent": "unknown", "strategy": "reactive", "confidence": 0.1, "source": "default"}

    try:
        # Kullanıcı girdisinin embedding'ini oluştur
        query_emb = np.array(embed_func(user_input))

        best_intent, best_strategy, best_score = None, None, -1.0

        # Tüm örnek embedding'leri ile karşılaştır
        for intent_name, strategy, example_emb in all_example_embeddings:
            # Normalize edilmiş dot product (kosinüs benzerliği)
            score = np.dot(query_emb, example_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(example_emb))
            if score > best_score:
                best_intent, best_strategy, best_score = intent_name, strategy, score

        # Belirli bir eşik değerinin üzerindeyse niyeti döndür
        # Bu eşik, alakasız girdiler için "unknown" sonucunu garantiler
        confidence_threshold = 0.70
        if best_score > confidence_threshold:
            return {
                "intent": best_intent,
                "strategy": best_strategy,
                "confidence": best_score,
                "source": "embedding"
            }

    except Exception as e:
        print(f"Niyet tespiti sırasında embedding hatası: {e}")
        # Hata durumunda varsayılan yanıta geç
        pass

    # Eşleşme bulunamazsa veya bir hata olursa varsayılan olarak "bilinmeyen" döner, bu da ana ajanın devreye girmesini sağlar
    return {"intent": "unknown", "strategy": "reactive", "confidence": 0.3, "source": "default"}
