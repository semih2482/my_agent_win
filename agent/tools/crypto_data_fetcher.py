# agent/tools/crypto_data_fetcher.py
import ccxt
import time
import pandas as pd
from typing import Dict, Any, List

# API'ye bağlanmak için bir borsa seçelim (Binance en kapsamlısıdır)
# Public (herkese açık) veri çektiğimiz için API anahtarına gerek yok.
exchange = ccxt.binance()

def fetch_crypto_data(args: str | dict) -> Dict[str, Any]:
    """
    Belirli bir kripto para sembolünün güncel verilerini çeker.
    Sembol 'BTC', 'ETH' gibi olmalı. Otomatik olarak '/USDT' ekler.
    """
    symbol = ""
    if isinstance(args, str):
        symbol = args
    elif isinstance(args, dict):
        symbol = args.get("symbol", "")
    try:
        # Kullanıcı 'BTC' yazarsa 'BTC/USDT' yap
        if '/' not in symbol:
            symbol = f"{symbol.upper()}/USDT"
        else:
            symbol = symbol.upper()

        print(f"🔄 Kripto verisi çekiliyor: {symbol}")

        # Güncel ticker (fiyat) bilgisini çek
        ticker = exchange.fetch_ticker(symbol)

        return {
            "status": "success",
            "result": {
                "symbol": symbol,
                "current_price": ticker.get('last'),
                "high_24h": ticker.get('high'),
                "low_24h": ticker.get('low'),
                "volume_24h": ticker.get('baseVolume'),
                "change_24h_percent": ticker.get('percentage'),
                "source": "ccxt (Binance)"
            }
        }
    except ccxt.BadSymbol:
        return {"status": "error", "message": f"'{symbol}' sembolü borsada bulunamadı."}
    except Exception as e:
        return {"status": "error", "message": f"Kripto verisi çekme hatası: {e}"}

def fetch_crypto_historical_data(symbol: str, timeframe: str = '1d', days: int = 90) -> Dict[str, Any]:
    """
    Teknik analiz için geçmiş fiyat verilerini (kapanış fiyatları) çeker.
    """
    try:
        if '/' not in symbol:
            symbol = f"{symbol.upper()}/USDT"
        else:
            symbol = symbol.upper()

        print(f"🔄 Kripto geçmiş verisi çekiliyor: {symbol} (son {days} gün)")

        # Gerekli milisaniye cinsinden zaman damgası
        since = exchange.milliseconds() - (days * 24 * 60 * 60 * 1000)

        # OHLCV (Open, High, Low, Close, Volume) verisini çek
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since)

        if not ohlcv:
            return {"status": "error", "message": "Geçmiş veri bulunamadı."}

        # Veriyi pandas DataFrame'e çevirip sadece 'Close' (kapanış) fiyatlarını al
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # technical_analyzer'ın beklediği gibi, en yeniden en eskiye doğru
        prices = df['close'].tolist()[::-1]

        return {"status": "success", "result": {"prices": prices}}
    except Exception as e:
        return {"status": "error", "message": f"Kripto geçmiş verisi çekme hatası: {e}"}