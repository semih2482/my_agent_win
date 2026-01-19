# agent/tools/community_tools/price_forecaster.py
import yfinance as yf
import pandas as pd
from prophet import Prophet
import json

# Gerekli kütüphanelerin kurulu olduğundan emin olun:
# pip install yfinance prophet

# Teknik analiz için gerekli fonksiyonu import edelim
from agent.tools.technical_analyzer import calculate_technical_indicators

TOOL_INFO = {
    "name": "price_forecaster",
    "description": "ÖNCEDEN BİLİNEN ve sembolü (`ticker`) belirtilen bir hisse senedi veya kripto para için gelecekteki fiyatları tahmin eder. Varlık bilinmiyorsa, önce `find_assets` aracını kullanın. Girdi: {'ticker': 'SEMBOL', 'days_to_forecast': GÜN_SAYISI}",
    "cacheable": True,
    "args_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string"},
            "days_to_forecast": {"type": "integer"}
        },
        "required": ["ticker", "days_to_forecast"]
    }
}

def run(args: str | dict, agent_instance=None) -> dict:
    """
    Aracın ana çalışma fonksiyonu. Prophet modelini kullanarak fiyat tahmini yapar.
    """
    try:
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return {"status": "error", "message": "Girdi geçerli bir JSON formatında değil. Örnek: {'ticker': 'BTC-USD', 'days_to_forecast': 30}"}

        ticker = args.get("ticker")
        days = int(args.get("days_to_forecast", 30))

        if not ticker:
            return {"status": "error", "message": "Ticker sembolü belirtilmedi."}

        # Kripto paralar için ticker'ı düzelt (örn: BTC -> BTC-USD)
        known_cryptos = ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX", "DOT", "MATIC"]
        if ticker.upper() in known_cryptos and not ticker.endswith("-USD"):
            ticker = f"{ticker.upper()}-USD"

        print(f"📈 Fiyat tahmini başlatılıyor: {ticker} için {days} gün...")

        # 1. GÜNCELLENMİŞ VERİ ÇEKME: Varlık, Endeks ve Döviz Kurunu Birlikte Çek
        # Modelin daha akıllı olması için piyasa endeksi (XU100) ve Dolar/TL kurunu harici faktör olarak ekliyoruz.
        tickers_to_download = [ticker, 'XU100.IS', 'TRY=X']
        data = yf.download(tickers_to_download, period="2y", progress=False)['Close']
        if data.empty:
            return {"status": "error", "message": f"'{ticker}' için geçmiş veri bulunamadı. Sembolü kontrol edin (Hisse senetleri için sonuna '.IS' eklemeyi unutmayın)."}

        # 2. GÜNCELLENMİŞ VERİ HAZIRLAMA: Tüm verileri birleştir ve Prophet formatına getir
        data.rename(columns={
            ticker: 'y', # Tahmin edilecek ana hedef
            'XU100.IS': 'market_index', # Harici regresör 1
            'TRY=X': 'currency_rate'  # Harici regresör 2
        }, inplace=True)

        # Teknik Göstergeleri Hesapla ve Veriye Ekle
        # Prophet'in anlayabilmesi için göstergeleri geçmiş her gün için hesaplamamız gerekiyor.
        # Basitlik adına, kapanış fiyatları üzerinden RSI ve MACD histogramını ekleyelim.
        close_prices = data['y'].tolist()
        if len(close_prices) > 35: # Teknik analiz için yeterli veri var mı?
            tech_indicators = calculate_technical_indicators(close_prices[::-1]) # Fiyatları doğru sırada gönder
            if tech_indicators.get("status") == "success":
                # DataFrame'e eklemek için göstergeleri pandas Serisine çevir
                rsi_series = pd.Series(tech_indicators.get("raw_results", {}).get("rsi_values"), index=data.index, name="rsi")
                macd_hist_series = pd.Series(tech_indicators.get("raw_results", {}).get("macd_histogram_values"), index=data.index, name="macd_hist")
                data = pd.concat([data, rsi_series, macd_hist_series], axis=1)
                print("   -> Teknik göstergeler (RSI, MACD) tahmin modeline eklendi.")

        df_prophet = data.reset_index().rename(columns={'Date': 'ds'})
        df_prophet.dropna(inplace=True) # Eksik verileri olan satırları temizle

        # 3. GÜNCELLENMİŞ MODEL OLUŞTURMA: Harici Regresörleri Ekle
        model = Prophet(daily_seasonality=True)
        model.add_regressor('market_index')
        model.add_regressor('currency_rate')
        if 'rsi' in df_prophet.columns: model.add_regressor('rsi')
        if 'macd_hist' in df_prophet.columns: model.add_regressor('macd_hist')

        # Modeli eğit
        model.fit(df_prophet)

        # 4. Gelecek için DataFrame oluşturma ve Tahmin Yapma
        # ÖNEMLİ: Geleceği tahmin etmek için regresörlerin gelecekteki değerlerine de ihtiyacımız var.
        # Gerçek bir modelde bunları da tahmin etmemiz gerekir, ancak basitlik adına son bilinen değerleri geleceğe taşıyacağız.
        future = model.make_future_dataframe(periods=days)

        regressor_columns = ['ds', 'market_index', 'currency_rate']
        if 'rsi' in df_prophet.columns: regressor_columns.append('rsi')
        if 'macd_hist' in df_prophet.columns: regressor_columns.append('macd_hist')
        future = pd.merge(future, df_prophet[regressor_columns], on='ds', how='left')
        future.fillna(method='ffill', inplace=True) # Son bilinen değerlerle doldur

        forecast = model.predict(future)

        # 5. Sonucu Formatlama
        # Sadece tahmin edilen günlerin sonuçlarını alalım
        forecast_values = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days)

        last_actual_price = df_prophet['y'].iloc[-1]
        predicted_price_in_x_days = forecast_values['yhat'].iloc[-1]

        summary = (f"'{ticker}' için {days} günlük çok faktörlü fiyat tahmini (piyasa, kur ve teknik göstergeler dahil edilerek) tamamlandı. "
                   f"Son bilinen kapanış fiyatı: {last_actual_price:.4f}. "
                   f"{days} gün sonraki tahmini fiyat: {predicted_price_in_x_days:.4f}. "
                   f"(Tahmin aralığı: {forecast_values['yhat_lower'].iloc[-1]:.4f} - {forecast_values['yhat_upper'].iloc[-1]:.4f}).")

        return {"status": "success", "result": summary, "forecast_data": forecast_values.to_dict('records')}

    except Exception as e:
        return {"status": "error", "message": f"Tahmin sırasında bir hata oluştu: {e}"}