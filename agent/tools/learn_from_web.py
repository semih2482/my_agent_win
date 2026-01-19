# agent/tools/learn_from_web.py
from agent.tools.web_reader import read_url, summarize_text

TOOL_INFO = {"name": "learn_from_web", "description": "Bir web sayfasının içeriğini okur, özetler ve ana hafızaya kaydeder.", "cacheable": False, "args_schema": {"url": "string"}}

def run(url: str, agent_instance=None):
    """
    URL'deki içeriği okur, özetler ve hafızaya kaydeder.
    Agent ile uyumlu JSON dönüşü sağlar.
    """
    if not agent_instance or not hasattr(agent_instance, 'knowledge_store'):
        return {"status": "error", "message": "Öğrenme için gerekli olan agent hafıza modülü bulunamadı."}

    try:
        content = read_url(url)
        if not content:
            return {"status": "error", "message": f"{url} adresinden içerik okunamadı."}

        summary = summarize_text(content, agent_instance.ask, length="kısa ve öz")
        agent_instance.knowledge_store.add(summary)
        return {"status": "success", "result": f"Özet hafızaya kaydedildi: {summary[:200]}..."}
    except Exception as e:
        return {"status": "error", "message": f"Hata oluştu: {e}"}
