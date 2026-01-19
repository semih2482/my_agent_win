# agent/tools/comprehensive_financial_analyst.py
from typing import Dict, Any, List
import re
import requests
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed


from agent.models.llm import ask
from agent.tools.internet_search import search_and_summarize
from agent.tools.financial_sentiment import analyze_financial_sentiment
from agent.tools.technical_analyzer import calculate_technical_indicators

# Derinlemesine analiz için araçlar
from agent.tools.community_tools.critical_web_researcher import run as critical_web_researcher
try:
    from agent.tools.crypto_data_fetcher import fetch_crypto_historical_data
except ImportError:
    # Eğer bu araç yoksa, hata vermemesi için placeholder bir fonksiyon oluştur
    def fetch_crypto_historical_data(symbol):
        print(f"UYARI: crypto_data_fetcher aracı bulunamadı. {symbol} için geçmiş veri çekilemiyor.")
        return {"status": "error", "message": "Crypto data fetcher tool not found."}

@lru_cache(maxsize=1)
def _get_btcturk_tickers() -> set:
    """BtcTurk API'sinden güncel coin sembollerini çeker ve önbelleğe alır."""
    try:
        response = requests.get("https://api.btcturk.com/api/v2/ticker", timeout=10)
        response.raise_for_status()
        raw_data = response.json()
        print(f"[DEBUG] BtcTurk API Yanıtı: {raw_data}") # Hata ayıklama için eklendi
        data = raw_data.get("data", [])
        # Sadece TRY, USDT ve BTC paritelerindeki ilk coini al (örn: 'BTC_TRY' -> 'BTC')
        tickers = {item['pair'].split('_')[0] for item in data if '_' in item['pair']}
        print(f"✅ BtcTurk'ten {len(tickers)} adet güncel coin sembolü çekildi.")
        return tickers
    except requests.exceptions.RequestException as e:
        print(f"UYARI: BtcTurk API'sine erişilemedi: {e}. Coin filtresi devre dışı.")
        return set()

TOOL_INFO = {
    "name": "comprehensive_financial_analyst",
    "description": "ÖNCEDEN BİLİNEN ve adı/sembolü (`query`) belirtilen tek bir finansal varlık (özellikle kripto paralar) hakkında çok adımlı, derinlemesine bir temel, teknik ve duyarlılık analizi yapar. Varlık bilinmiyorsa, önce `find_assets` aracını kullanın.",
    "cacheable": True,
    "args_schema": {
        "query": {
            "type": "string",
            "description": "Analiz edilecek varlığın adı veya sembolü (örn: 'Bitcoin', 'ETH')."
        },
        "investment_horizon": {
            "type": "string",
            "description": "Yatırımcının zaman ufku (örn: 'kısa vade', '6 ay', '2 yıl')."
        },
        "risk_profile": {
            "type": "string",
            "description": "Yatırımcının risk profili (örn: 'düşük risk', 'agresif')."
        }
    }
}



ASSET_MAP = {"bitcoin": "BTC", "ethereum": "ETH"}

@lru_cache(maxsize=128)
def _get_asset_class(code: str) -> str:
    """Basit kurallarla bir varlığın sınıfını (hisse, fon, kripto) tahmin eder."""
    code_upper = code.upper()
    known_cryptos = ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX", "DOT", "MATIC", "RNDR", "KAS", "FET"]
    if code_upper in known_cryptos:
        return "crypto"
    if len(code_upper) > 3 and code_upper.endswith("USD"): # Kripto pariteleri için
        return "crypto"
    if len(code_upper) > 5: # Hisse senetleri için
        return "stock"
    return "unknown"

def _identify_asset(query: str) -> Dict[str, Any] | None:
    """Sorguyu analiz ederek tek bir varlığı ve sınıfını belirler."""
    query_lower = query.lower().strip()
    query_upper = query.upper().strip()
    
    # 1. Doğrudan eşleşme için haritayı kontrol et
    if query_lower in ASSET_MAP:
        symbol = ASSET_MAP[query_lower]
        return {"class": "crypto", "symbol": symbol, "name": query_lower.capitalize()}

    # 2. Sembolleri bul (örn: EREGL, BTC)
    potential_symbols = re.findall(r'\b[A-Z]{2,10}\b', query)
    if potential_symbols:
        symbol = potential_symbols[0]
        asset_class = _get_asset_class(symbol)
        if asset_class != "unknown":
            return {"class": asset_class, "symbol": symbol, "name": symbol}
            
    # 3. Eğer hiçbir şey bulunamazsa, sorgunun kendisini isim olarak kabul et
    return {"class": "crypto", "symbol": query_upper, "name": query_lower.capitalize()}


def _analyze_crypto_deep_dive(asset_info: Dict[str, Any], agent_instance=None) -> Dict[str, Any]:
    """Bir kripto para için derinlemesine, çok adımlı bir analiz yürütür."""
    symbol = asset_info.get("symbol")
    name = asset_info.get("name")
    
    # BtcTurk FİLTRESİ
    btcturk_tickers = _get_btcturk_tickers()
    if btcturk_tickers and symbol not in btcturk_tickers:
        return {"status": "info", "result": f"'{symbol}' sembolü BtcTurk borsasında bulunamadığı için detaylı analiz atlanıyor."}
    # FİLTRE SONU

    llm_func = agent_instance.ask if agent_instance and hasattr(agent_instance, 'ask') else ask
    
    print(f"🚀 KRİPTO DERİN ANALİZİ BAŞLATILIYOR: {name} ({symbol})")
    

    analysis_steps = {
        "fundamentals": (critical_web_researcher, {"query": f"Generate a detailed report on the cryptocurrency '{name} ({symbol})'. Cover its core purpose, underlying technology, the team behind it, and its future roadmap. Validate information from multiple authoritative sources like its official website, technical whitepaper, and reputable crypto analysis sites."}),
        "tokenomics": (search_and_summarize, f"Summarize the tokenomics of {name} ({symbol}) using information from sites like coingecko.com, coinmarketcap.com, or messari.io. Cover max supply, circulating supply, inflation schedule, and token utility."),
        "social_sentiment": (search_and_summarize, f"What is the recent sentiment and key discussion points for {name} ({symbol}) on social platforms like Twitter and Reddit's /r/CryptoCurrency?")
    }

    raw_results = {}
    
    # Veri Toplama Adımlarını Paralel Çalıştır
    with ThreadPoolExecutor(max_workers=len(analysis_steps) + 2) as executor:
        future_to_step = {}

        # Temel, Tokenomik ve Sosyal adımlarını planla
        for step_name, (tool_func, tool_args) in analysis_steps.items():
            if step_name == "fundamentals":
                 future_to_step[executor.submit(tool_func, args=tool_args, agent_instance=agent_instance)] = step_name
            else:
                 future_to_step[executor.submit(tool_func, tool_args, llm_ask_function=llm_func)] = step_name

        # Geçmiş fiyat verisini çek
        future_to_step[executor.submit(fetch_crypto_historical_data, symbol)] = "historical_data"
        
        # Fiyat tahmini aracını planla
        if agent_instance and 'price_forecaster' in agent_instance.available_tools:
            forecaster_tool = agent_instance.available_tools['price_forecaster']['func']
            future_to_step[executor.submit(forecaster_tool, args={"ticker": symbol, "days_to_forecast": 90}, agent_instance=agent_instance)] = "price_forecast"

        # Tüm adımların tamamlanmasını bekle
        for future in as_completed(future_to_step):
            step_name = future_to_step[future]
            try:
                raw_results[step_name] = future.result()
                print(f"   ✅ {step_name.replace('_', ' ').title()} adımı tamamlandı.")
            except Exception as exc:
                raw_results[step_name] = {"status": "error", "message": f"'{step_name}' adımı başarısız: {exc}"}
                print(f"   ❌ {step_name.replace('_', ' ').title()} adımı başarısız: {exc}")

    # Toplanan Ham Verileri İşle ve Yapılandır
    processed_results = {"varlik_bilgisi": f"{name} ({symbol})"}
    
    # Temel Analiz
    res = raw_results.get("fundamentals", {})
    processed_results["temel_analiz_raporu"] = res.get("result", f"Hata: {res.get('message', 'Bilinmeyen hata')}") if res.get("status") == "success" else f"Hata: {res.get('message', 'İşlem başarısız')}"

    # Tokenomik Analizi
    res = raw_results.get("tokenomics", {})
    processed_results["tokenomik_ozeti"] = res.get("result", f"Hata: {res.get('message', 'Bilinmeyen hata')}") if res.get("status") == "success" else f"Hata: {res.get('message', 'İşlem başarısız')}"

    # Sosyal Duyarlılık
    res = raw_results.get("social_sentiment", {})
    if res.get("status") == "success":
        social_summary = res.get("result", "")
        sentiment_res = analyze_financial_sentiment(social_summary)
        processed_results["sosyal_medya_ozeti"] = social_summary
        processed_results["sosyal_duyarlilik_skoru"] = sentiment_res.get("result", "Hesaplanamadı")
    else:
        processed_results["sosyal_medya_ozeti"] = f"Hata: {res.get('message', 'İşlem başarısız')}"
        processed_results["sosyal_duyarlilik_skoru"] = "Hesaplanamadı"

    # Teknik Analiz
    res = raw_results.get("historical_data", {})
    if res.get("status") == "success":
        prices = res.get("result", {}).get("prices", [])
        if prices:
            tech_res = calculate_technical_indicators(prices)
            processed_results["teknik_analiz_ozeti"] = tech_res.get("result", {}).get("summary", "Özet oluşturulamadı.")
        else:
            processed_results["teknik_analiz_ozeti"] = "Geçmiş fiyat verisi bulunamadığı için teknik analiz yapılamadı."
    else:
        processed_results["teknik_analiz_ozeti"] = f"Teknik analiz verisi çekilemedi: {res.get('message')}"

    # Fiyat Tahmini
    res = raw_results.get("price_forecast", {})
    processed_results["fiyat_tahmin_ozeti"] = res.get("result", f"Hata: {res.get('message', 'Bilinmeyen hata')}") if res.get("status") == "success" else f"Hata: {res.get('message', 'İşlem başarısız')}"

    return {"status": "success", "result": processed_results}


def run(args: Dict[str, Any], agent_instance=None) -> Dict[str, Any]:
    """
    Belirtilen tek bir varlık için derinlemesine analiz sürecini başlatır ve
    sonuçları sentezleyerek bütünsel bir yatırım tezi oluşturur.
    """
    query = args.get("query", "")
    investment_horizon = args.get("investment_horizon", "belirtilmedi")
    risk_profile = args.get("risk_profile", "belirtilmedi")
    llm_func = agent_instance.ask if agent_instance and hasattr(agent_instance, 'ask') else ask

    # ADIM 1: Varlığı Tanımla
    asset_info = _identify_asset(query)
    if not asset_info:
        return {"status": "info", "result": "Analiz için geçerli bir varlık adı veya sembolü bulunamadı. Lütfen sorgunuzu kontrol edin."}

    # ADIM 2: Varlık Sınıfına Göre Analiz Stratejisi Seç
    asset_class = asset_info.get("class")
    analysis_result = None

    if asset_class == "crypto":
        analysis_result = _analyze_crypto_deep_dive(asset_info, agent_instance)
    else:
        # Şimdilik sadece kripto paralar için derinlemesine analizi destekliyoruz
        return {"status": "info", "result": f"'{asset_info.get('name')}' bir kripto para olarak tanımlanmadı. Şu anda derinlemesine analiz sadece kripto paralar için desteklenmektedir."}

    if not analysis_result or analysis_result.get("status") != "success":
        return analysis_result or {"status": "error", "message": "Analiz sırasında bilinmeyen bir hata oluştu."}

    # ADIM 3: NİHAİ SENTEZ
    # Toplanan tüm yapılandırılmış verileri LLM'e göndererek bir yatırım tezi oluşturmasını iste
    synthesis_prompt = f"""
Sen bir kıdemli yatırım analistisin. Görevin, sana sunulan yapılandırılmış verileri kullanarak '{asset_info.get('name')}' adlı kripto para için kapsamlı bir yatırım tezi oluşturmak.

Yatırımcının Profili:
- Zaman Ufku: {investment_horizon}
- Risk Profili: {risk_profile}

Analiz Raporları:
---
**Varlık Bilgisi:**
{analysis_result['result'].get('varlik_bilgisi')}

---
**1. Temel Analiz Raporu (Proje, Teknoloji, Ekip, Yol Haritası):**
{analysis_result['result'].get('temel_analiz_raporu')}

---
**2. Tokenomik Özeti (Arz, Enflasyon, Kullanım Alanı):**
{analysis_result['result'].get('tokenomik_ozeti')}

---
**3. Sosyal Medya Analizi:**
- Özet: {analysis_result['result'].get('sosyal_medya_ozeti')}
- Duyarlılık Skoru: {analysis_result['result'].get('sosyal_duyarlilik_skoru')}

---
**4. Teknik Analiz Özeti:**
{analysis_result['result'].get('teknik_analiz_ozeti')}

---
**5. Fiyat Tahmin Özeti (Gelecek 90 Gün):**
{analysis_result['result'].get('fiyat_tahmin_ozeti')}

---

**GÖREVİN:**
Yukarıdaki tüm verileri birleştirerek aşağıdaki formatta bir yatırım tezi oluştur:

**1. Genel Değerlendirme:** Varlık hakkında bir paragraflık genel bir özet ve yatırımcının profiline uygun olup olmadığına dair ilk izlenim.

**2. Güçlü Yönler (Potansiyel):** Projenin temel analizi, tokenomik yapısı veya topluluk gücünden kaynaklanan en önemli avantajları ve potansiyeli.

**3. Zayıf Yönler ve Riskler:** Teknik analizdeki zayıf sinyaller, sosyal medyadaki olumsuz görüşler, projenin temelindeki veya tokenomik yapısındaki riskler.

**4. Yatırım Tezi ve Sonuç:** Belirtilen yatırımcı profiline (zaman ufku ve risk iştahı) göre bu varlığa yatırım yapmanın mantıklı olup olmadığına dair net bir sonuç. Olası bir potansiyel veya risk senaryosunu özetle. Cevabını 'Bu bir yatırım tavsiyesi değildir. Kendi araştırmanızı yapmanız esastır.' uyarısıyla bitir.
"""

    print("🧠 Tüm veriler toplandı. Nihai sentez için LLM'e soruluyor...")
    try:
        final_synthesis = llm_func(synthesis_prompt, max_new_tokens=4096)
        return {"status": "success", "result": final_synthesis}
    except Exception as e:
        print(f"HATA: Nihai sentez sırasında hata oluştu: {e}")
        return {"status": "error", "message": f"Nihai sentez başarısız: {e}", "partial_results": analysis_result['result']}

