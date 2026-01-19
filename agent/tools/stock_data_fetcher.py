# agent/tools/stock_data_fetcher.py
import yfinance as yf
import json
from typing import Dict, Any

def fetch_stock_data(args: str | dict) -> Dict[str, Any]:
    """
    Belirli bir hisse senedi sembolünün güncel ve özet verilerini Yahoo Finance'dan çeker.
    """
    symbol = ""
    if isinstance(args, str):
        symbol = args
    elif isinstance(args, dict):
        symbol = args.get("symbol", "")
    try:
        # Türkiye'deki hisseler için Yahoo Finance genellikle ".IS" sonekini gerektirir.
        # Eğer sembol zaten bir sonek içermiyorsa, ekleyelim.
        symbol_for_yf = symbol.upper()
        if '.' not in symbol_for_yf:
            symbol_for_yf += ".IS"
        ticker = yf.Ticker(symbol_for_yf)
        info = ticker.info

        if not info or 'regularMarketPrice' not in info:
            return {"status": "error", "message": f"'{symbol}' sembolü için veri bulunamadı. Lütfen sembolü kontrol edin."}

        # Güncel fiyat ve temel verileri çek
        current_price = info.get('regularMarketPrice')
        volume = info.get('regularMarketVolume')
        market_cap = info.get('marketCap')

        # Ek bilgi (açılış, yüksek, düşük)
        open_price = info.get('regularMarketOpen')
        day_high = info.get('dayHigh')
        day_low = info.get('dayLow')

        return {
            "status": "success",
            "result": {
                "symbol": symbol.upper(),
                "name": info.get('shortName', 'Bilinmiyor'),
                "current_price": round(current_price, 2),
                "open_price": round(open_price, 2) if open_price else None,
                "day_high": round(day_high, 2) if day_high else None,
                "day_low": round(day_low, 2) if day_low else None,
                "volume": volume,
                "market_cap": market_cap,
                "source": "Yahoo Finance"
            }
        }
    except Exception as e:
        return {"status": "error", "message": f"Hisse senedi verisi çekme hatası: {e}"}