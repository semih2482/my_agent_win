# agent/tools/find_assets.py
import json
from typing import Dict, Any, List

import re
from agent.tools.internet_search import search_and_summarize
from agent.models.llm import ask
# DEPRECATED ADVISOR TOOLS REMOVED
# from agent.tools.investment_advisor import run as get_investment_advice
# from agent.tools.fund_analyst import run as get_fund_advice
# from agent.tools.crypto_advisor import run as get_crypto_advice

# DIRECTLY IMPORT THE CORRECT ANALYST TOOL
# Renamed alias for clarity and to avoid potential conflicts.
from agent.tools.comprehensive_financial_analyst import run as analyze_asset, _get_asset_class

TOOL_INFO = {
    "name": "find_assets",
    "description": "Kullanıcının 'düşük piyasa değerli', 'potansiyeli yüksek' gibi genel kriterlerine göre potansiyel, yüksek büyüme vadeden kripto paraları (low-cap gems) keşfeder. Bu araç analiz yapmaz, sadece keşif yapar ve bir aday listesi sunar.",
    "cacheable": True,
    "args_schema": {
        "query": {
            "type": "string",
            "description": "Kullanıcının varlık bulma isteğini içeren orijinal sorgu (örn: 'potansiyeli yüksek low-cap gem altcoinler')."
        }
    }
}

def _create_search_query(original_query: str) -> List[str]:
    """Kullanıcının 'low-cap gem' gibi taleplerine yönelik uzman arama sorguları listesi oluşturur."""
    print("💡 Kripto para 'gem' keşfi için uzman arama sorguları oluşturuluyor.")

    # Zaman ve trend odaklı anahtar kelimeler ekleyerek sorguları daha dinamik hale getir
    current_year = 2025  # Bu dinamik olarak alınabilir, şimdilik sabit.
    next_year = current_year + 1

    # Çeşitli arama açıları için sorgu şablonları
    query_templates = [
        f"best low-cap crypto gems with 10x potential {current_year}",
        f"undervalued altcoins to watch {current_year}",
        f"top crypto narratives for {next_year} bull run",
        f"most promising new crypto projects with strong fundamentals",
        f"crypto gems discussed on reddit /biz/ {current_year}",
        f"analyst picks for high growth potential crypto {current_year}",
        # Kullanıcının orijinal sorgusunu da dahil ederek özelleştirilmiş bir arama yap
        f"'{original_query}' analysis crypto twitter {current_year}"
    ]

    return query_templates

from concurrent.futures import ThreadPoolExecutor

def _create_extraction_prompt(original_query: str, search_summary: str) -> str:
    """
    LLM'i bir analist gibi davranmaya yönlendirerek, arama sonuçlarından en umut verici
    ve en sık bahsedilen 5 coini belirlemesini isteyen bir prompt oluşturur.
    """
    return f"""
    You are a senior crypto investment analyst. Your task is to analyze the following web search results, which were gathered based on the user's request: "{original_query}".
    Identify the TOP 5 most promising, most frequently mentioned, or most relevant low-cap cryptocurrency gems from the text.

    **INSTRUCTIONS:**
    1.  **Analyze Holistically:** Read the entire text to understand which coins are mentioned most often and in the most positive contexts.
    2.  **Prioritize "Gems":** Focus on assets described as "low-cap," "high-potential," "undervalued," or "10x/100x". Ignore well-established, high-market-cap coins like Bitcoin (BTC) or Ethereum (ETH) unless the context specifically justifies it.
    3.  **Extract Tickers:** Extract only the ticker symbol (e.g., "KAS" for Kaspa, "TAO" for Bittensor).
    4.  **Rank the List:** Present the top 5 tickers as a ranked list.
    5.  **Format:** Your response **MUST** be a single, valid JSON object in the format: `{{"assets": ["TICKER1", "TICKER2", "TICKER3", "TICKER4", "TICKER5"]}}`.
    6.  **No Commentary:** Do not include any explanation, notes, or any text other than the final JSON object.

    **Web Search Results:**
    ---
    {search_summary}
    ---

    **VALID JSON RESPONSE:**
    """

def run(args: Dict[str, Any], agent_instance=None) -> Dict[str, Any]:
    """
    Kullanıcının 'low-cap gem' gibi kriterlerine göre potansiyel kripto paraları keşfetmek için
    çoklu, hedefe yönelik web aramaları yapar ve en umut verici 5 adayı belirler.
    Bu araç analiz yapmaz, sadece keşfeder.
    """
    query = args.get("query")
    if not query:
        return {"status": "error", "message": "find_assets aracı için 'query' argümanı gereklidir."}

    # 1. Uzman Arama Sorguları Oluştur
    search_queries = _create_search_query(query)
    print(f"🔎 Potansiyel 'gem' coinleri bulmak için {len(search_queries)} adet uzman arama yapılıyor...")

    # 2. İnternet Aramalarını Paralel Yap
    all_search_summaries = []
    with ThreadPoolExecutor(max_workers=len(search_queries)) as executor:
        future_to_query = {executor.submit(search_and_summarize, q, max_results=2): q for q in search_queries}
        for future in future_to_query:
            try:
                result = future.result()
                if result.get("status") == "success" and result.get("result"):
                    all_search_summaries.append(result.get("result"))
            except Exception as exc:
                print(f"  -> Arama sorgusu '{future_to_query[future]}' sırasında bir hata oluştu: {exc}")

    if not all_search_summaries:
        return {"status": "error", "message": "Web aramaları sonucunda potansiyel varlıklar hakkında hiçbir bilgi bulunamadı."}

    combined_summary = "\n\n---\n\n".join(all_search_summaries)

    # 3. En Umut Verici Varlıkları Çıkarmak İçin LLM'i Kullan
    extraction_prompt = _create_extraction_prompt(query, combined_summary)

    try:
        json_extractor = agent_instance.extract_json if agent_instance and hasattr(agent_instance, 'extract_json') else None
        response_str = ask(extraction_prompt, max_new_tokens=256).strip()
        
        try:
            start = response_str.index('{')
            end = response_str.rindex('}') + 1
            response_str = response_str[start:end]
        except ValueError:
            return {"status": "error", "message": f"LLM'den gelen yanıtta geçerli bir JSON nesnesi bulunamadı. Yanıt: {response_str}"}
        
        if json_extractor:
            extracted_data = json_extractor(response_str)
        else:
            extracted_data = json.loads(response_str)

        if not extracted_data or "assets" not in extracted_data or not isinstance(extracted_data["assets"], list):
            return {"status": "error", "message": f"LLM'den beklenen formatta varlık listesi alınamadı. Gelen yanıt: {response_str}"}

        asset_codes = extracted_data["assets"]
        print(f"🏆 En umut verici adaylar belirlendi: {asset_codes}")

        return {"status": "success", "result": asset_codes}

    except Exception as e:
        return {"status": "error", "message": f"Varlık kodları ayrıştırılırken veya LLM ile işlenirken bir hata oluştu: {e}"}