# agent/tools/real_web_search.py
import requests
from bs4 import BeautifulSoup
from agent.tools.web_reader import read_url, summarize_text
from agent.models.llm import ask  # LLM fonksiyonu

DUCKDUCKGO_URL = "https://html.duckduckgo.com/html/"

# Basit bellek içi cache
_search_cache = {}

def real_web_search(query: str, max_results: int = 5):
    query_key = query.strip().lower()
    if query_key in _search_cache:
        return {"status": "success", "result": _search_cache[query_key]}

    try:
        # 1) DuckDuckGo HTML araması yap
        payload = {"q": query}
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.post(DUCKDUCKGO_URL, data=payload, headers=headers, timeout=10)

        if resp.status_code != 200:
            return {"status": "error", "message": f"DuckDuckGo hata kodu: {resp.status_code}"}

        soup = BeautifulSoup(resp.text, "html.parser")
        results = soup.find_all("a", class_="result__a", limit=max_results)

        if not results:
            return {"status": "error", "message": "Aramada sonuç bulunamadı."}

        summaries = []
        for r in results:
            url = r.get("href")
            try:
                page_text = read_url(url)
                page_summary = summarize_text(page_text, llm_ask_function=ask)
                summaries.append(f"- {url}\n  Özet: {page_summary}")
            except Exception as e:
                summaries.append(f"- {url}\n  Hata: {e}")

        final_result = "\n".join(summaries)
        _search_cache[query_key] = final_result  # ✅ Cache’e kaydet
        return {"status": "success", "result": final_result}

    except Exception as e:
        return {"status": "error", "message": str(e)}
