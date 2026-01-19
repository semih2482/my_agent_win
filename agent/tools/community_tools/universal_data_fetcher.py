# agent/tools/community_tools/universal_data_fetcher.py
from typing import Dict, Any

# İlgili tüm alt fonksiyonları import edelim
from agent.tools.stock_data_fetcher import fetch_stock_data
from agent.tools.crypto_data_fetcher import fetch_crypto_data
from agent.tools.fund_data_fetcher import fetch_fund_data

TOOL_INFO = {
    "name": "universal_data_fetcher",
    "description": "Herhangi bir finansal varlık (hisse senedi, kripto para, yatırım fonu) için temel verileri tek bir yerden çeker. Girdi olarak varlığın sembolünü alır (örn: 'EREGL', 'BTC', 'AFA').",
    "cacheable": True,
    "args_schema": {
        "type": "object",
        "properties": {
            "symbol": {"type": "string"}
        },
        "required": ["symbol"]
    }
}

def _get_asset_class(symbol: str) -> str:
    """
    Basit kurallarla bir varlığın sınıfını (hisse, fon, kripto) tahmin eder.
    Bu fonksiyon, comprehensive_financial_analyst'tan buraya taşınarak merkezileştirildi.
    """
    symbol_upper = symbol.upper()
    # Bilinen kripto paralar veya kripto formatı (örn: BTC-USD)
    known_cryptos = ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX", "DOT", "MATIC"]
    if symbol_upper in known_cryptos or '-' in symbol:
        return "crypto"
    # Fon kodları genellikle 3 harflidir
    if len(symbol_upper) == 3:
        return "fund"
    # Hisse senedi sembolleri genellikle 4-5 harflidir ve sonunda .IS olabilir
    if len(symbol_upper) >= 4:
        return "stock"
    return "unknown"

def run(args: str | dict, agent_instance=None) -> Dict[str, Any]:
    """
    Aracın ana çalışma fonksiyonu. Varlık sınıfını belirler ve ilgili veri çekme fonksiyonunu çağırır.
    """
    symbol = ""
    if isinstance(args, str):
        symbol = args
    elif isinstance(args, dict):
        symbol = args.get("symbol")

    if not symbol:
        return {"status": "error", "message": "Varlık sembolü ('symbol') belirtilmedi."}

    asset_class = _get_asset_class(symbol)
    print(f"🏛️ Universal Fetcher: '{symbol}' sembolü '{asset_class}' olarak algılandı.")

    if asset_class == "stock":
        # fetch_stock_data zaten dict bekliyor, uyumlu.
        return fetch_stock_data(symbol)
    elif asset_class == "crypto":
        # fetch_crypto_data string bekliyor.
        return fetch_crypto_data(symbol)
    elif asset_class == "fund":
        # fetch_fund_data string bekliyor.
        return fetch_fund_data(symbol)
    else:
        # Bilinmeyen bir varlık sınıfı için tüm fetcher'ları sırayla deneyelim
        print(f"   -> Varlık sınıfı bilinmiyor, tüm kaynaklar deneniyor...")
        result = fetch_stock_data(symbol)
        if result.get("status") == "success":
            return result
        result = fetch_crypto_data(symbol)
        if result.get("status") == "success":
            return result
        result = fetch_fund_data(symbol)
        if result.get("status") == "success":
            return result

        return {"status": "error", "message": f"'{symbol}' sembolü için bilinen hiçbir veri kaynağında (hisse, kripto, fon) veri bulunamadı."}