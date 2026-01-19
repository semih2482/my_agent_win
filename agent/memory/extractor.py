import json
import re
from agent.models.llm import ask

def extract_json(text: str):
    """
    Metin içindeki ilk geçerli JSON nesnesini veya listesini bulur ve ayrıştırır.
    Farklı formatları (markdown, ham metin) işlemek için birkaç strateji dener.
    """
    # 1. Strateji: Markdown JSON kod bloğunu ara (```json ... ```)
    match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass  # Diğer yöntemlere geç

    # 2. Strateji: Metin içindeki ilk ve son küme parantezi veya köşeli parantez arasını bul
    json_obj_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_obj_match:
        json_str = json_obj_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    json_arr_match = re.search(r"\[.*\]", text, re.DOTALL)
    if json_arr_match:
        json_str = json_arr_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # 3. Strateji: Tüm metni JSON olarak ayrıştırmayı dene
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

def extract_triplets(text: str) -> list[dict]:
    """
    Extracts knowledge triplets from a given text.
    """
    prompt = f"""
Sen bir bilgi mühendisisin. Görevin, sana verilen metinden anlamsal ilişkileri (knowledge triplets) çıkarmak. Çıktın HER ZAMAN [{{"subject": "Varlık 1", "relation": "İLİŞKİ", "object": "Varlık 2"}}, ...] formatında bir JSON listesi olmalıdır. Yanıtın, başka hiçbir metin veya açıklama olmadan, SADECE JSON listesinin kendisi olmalıdır.
Metin: "{text}"
JSON:
"""
    response = ask(prompt)
    json_response = extract_json(response)
    if json_response and isinstance(json_response, list):
        return json_response
    return []
