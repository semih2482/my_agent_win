# agent/tools/technical_analyzer.py
from typing import List, Dict, Any
import pandas as pd

def calculate_technical_indicators(prices: List[float]) -> Dict[str, Any]:
    """
    Verilen fiyat listesine göre RSI ve MACD gibi teknik göstergeleri hesaplar.
    - prices: Güncel fiyattan geçmişe doğru sıralanmış fiyat listesi.
    """
    if len(prices) < 35: # MACD için yeterli veri (26 + 9)
        return {
            "status": "error",
            "message": "Teknik analiz için yetersiz veri (en az 35 gün gerekli)."
        }

    try:
        # Fiyatları doğru sıraya koy (geçmişten günümüze)
        df = pd.DataFrame({'price': prices[::-1]})

        # RSI (14 günlük)
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        last_rsi = rsi.iloc[-1]

        # MACD (12, 26, 9 günlük)
        exp12 = df['price'].ewm(span=12, adjust=False).mean()
        exp26 = df['price'].ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal

        last_macd = macd.iloc[-1]
        last_signal = signal.iloc[-1]
        last_histogram = histogram.iloc[-1]

        rsi_yorum = "Nötr"
        if pd.notna(last_rsi):
            if last_rsi > 70:
                rsi_yorum = "Aşırı Alım Bölgesinde (Satış baskısı gelebilir)"
            elif last_rsi < 30:
                rsi_yorum = "Aşırı Satım Bölgesinde (Alış fırsatı olabilir)"
            else:
                rsi_yorum = f"Nötr Bölgede ({round(last_rsi, 2)})"

        macd_yorum = "Nötr"
        if pd.notna(last_histogram):
            if last_histogram > 0:
                macd_yorum = "Pozitif (AL Sinyali)"
            else:
                macd_yorum = "Negatif (SAT Sinyali)"

        return {
            "status": "success",
            "result": {
                "rsi_14": round(last_rsi, 2) if pd.notna(last_rsi) else "Hesaplanamadı",
                "macd": {
                    "value": round(last_macd, 4) if pd.notna(last_macd) else "Hesaplanamadı",
                    "signal": round(last_signal, 4) if pd.notna(last_signal) else "Hesaplanamadı",
                    "histogram": round(last_histogram, 4) if pd.notna(last_histogram) else "Hesaplanamadı"
                },

                "summary": (
                    f"Teknik Görünüm: RSI '{rsi_yorum}'. "
                    f"MACD histogramı '{macd_yorum}'."
                )
            }
        }
    except Exception as e:
        return {"status": "error", "message": f"Teknik gösterge hesaplama hatası: {e}"}