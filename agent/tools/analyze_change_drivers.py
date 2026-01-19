# agent/tools/analyze_change_drivers.py

import yfinance as yf
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
from typing import Dict, Any

def analyze_drivers(symbol: str,
                    index_symbol: str = 'XU100.IS',
                    currency_symbol: str = 'TRY=X',
                    period: str = '3y') -> Dict[str, Any]:
    """
    Belirtilen hisse senedinin (symbol) günlük fiyat değişimlerinin,
    belirtilen piyasa endeksi (index_symbol) ve döviz kuru (currency_symbol)
    günlük değişimleri ile ilişkisini çoklu doğrusal regresyon kullanarak analiz eder.
    Fark alınmış seriler üzerinde çalışır ve otokorelasyonu azaltmayı hedefler.

    Dönen yapı: {"status": "success", "result": {"summary": "...", "adj_r_squared": ..., "dw_test": ...}}
    """
    try:
        print(f"🔄 Sürücü analizi için veriler çekiliyor: {symbol}, {index_symbol}, {currency_symbol} ({period})")

        # Veri Çekme
        data = {}
        symbols = {'asset': symbol, 'index': index_symbol, 'currency': currency_symbol}
        dfs = {}

        for name, sym in symbols.items():
            ticker = yf.Ticker(sym)
            df_raw = ticker.history(period=period)
            if df_raw.empty or 'Close' not in df_raw.columns:
                return {"status": "error", "message": f"{sym} için fiyat verisi ('Close') çekilemedi."}
            # Timezone kaldır ve sadece Kapanış fiyatını al
            dfs[name] = df_raw['Close'].tz_localize(None).to_frame(name=f'{name}_Fiyat')
            print(f"✅ {sym} verisi çekildi.")

        # Birleştirme
        df = pd.merge(dfs['asset'], dfs['index'], left_index=True, right_index=True, how='inner')
        df = pd.merge(df, dfs['currency'], left_index=True, right_index=True, how='inner')

        if df.empty:
             return {"status": "error", "message": "Veriler birleştirilemedi (tarih uyuşmazlığı?)."}

        # Fark Alma
        df_diff = df.diff(1).dropna()

        if df_diff.empty:
            return {"status": "error", "message": "Fark alma sonrası veri kalmadı."}
        if df_diff.isnull().values.any():
            return {"status": "error", "message": "Fark alınmış veride NaN değerler var."}

        # Sütunları regresyon formülü için yeniden adlandır
        df_diff.rename(columns={
            'asset_Fiyat': 'Asset_Diff',
            'index_Fiyat': 'Index_Diff',
            'currency_Fiyat': 'Currency_Diff'
        }, inplace=True)

        # Regresyon Modeli
        print("📊 Fark alınmış değişkenlerle regresyon modeli çalıştırılıyor...")
        model_diff = smf.ols('Asset_Diff ~ Index_Diff + Currency_Diff', data=df_diff).fit()

        # Sonuçları Ayıklama
        adj_r_kare = model_diff.rsquared_adj
        dw_test = sm.stats.stattools.durbin_watson(model_diff.resid)

        # Katsayıları ve p-değerlerini güvenli bir şekilde al
        params = model_diff.params
        pvalues = model_diff.pvalues

        katsayi_index = params.get('Index_Diff', np.nan)
        p_degeri_index = pvalues.get('Index_Diff', np.nan)
        katsayi_currency = params.get('Currency_Diff', np.nan)
        p_degeri_currency = pvalues.get('Currency_Diff', np.nan)

        # Metinsel Özeti Oluşturma
        summary_lines = []

        # Index Yorumu
        if p_degeri_index < 0.05:
            direction_index = "pozitif" if katsayi_index > 0 else "negatif"
            summary_lines.append(f"{index_symbol} endeksindeki günlük değişim ile {symbol} fiyatındaki günlük değişim arasında istatistiksel olarak anlamlı ve {direction_index} bir ilişki vardır.")
        else:
            summary_lines.append(f"{index_symbol} endeksindeki günlük değişim ile {symbol} fiyatındaki günlük değişim arasında anlamlı bir ilişki bulunamamıştır.")

        # Currency Yorumu
        if p_degeri_currency < 0.05:
            direction_currency = "pozitif" if katsayi_currency > 0 else "negatif"
            summary_lines.append(f"{currency_symbol} kurundaki günlük değişim ile {symbol} fiyatındaki günlük değişim arasında (piyasa etkisi kontrol edildikten sonra) istatistiksel olarak anlamlı ve {direction_currency} bir ilişki vardır.")
        else:
            summary_lines.append(f"{currency_symbol} kurundaki günlük değişim ile {symbol} fiyatındaki günlük değişim arasında (piyasa etkisi kontrol edildikten sonra) anlamlı bir ilişki bulunamamıştır.")

        # Model Uyumu Yorumu
        summary_lines.append(f"Model, {symbol}'in günlük fiyat değişimlerinin yaklaşık %{adj_r_kare*100:.1f}'ini açıklamaktadır (Adj. R-kare).")

        # DW Test Yorumu
        if 1.5 < dw_test < 2.5:
             summary_lines.append(f"Durbin-Watson testi ({dw_test:.2f}), modeldeki otokorelasyon sorununun büyük ölçüde çözüldüğünü göstermektedir.")
        else:
             summary_lines.append(f"UYARI: Durbin-Watson testi ({dw_test:.2f}), modelde hala otokorelasyon sorunu olabileceğini göstermektedir.")

        final_summary = " ".join(summary_lines)
        print(f"✅ Analiz tamamlandı. Özet: {final_summary}")

        return {
            "status": "success",
            "result": {
                "summary": final_summary,
                "adj_r_squared": round(adj_r_kare, 3),
                "dw_test": round(dw_test, 3),
                "index_coeff": round(katsayi_index, 4) if not np.isnan(katsayi_index) else None,
                "index_pvalue": round(p_degeri_index, 4) if not np.isnan(p_degeri_index) else None,
                "currency_coeff": round(katsayi_currency, 4) if not np.isnan(katsayi_currency) else None,
                "currency_pvalue": round(p_degeri_currency, 4) if not np.isnan(p_degeri_currency) else None,
            }
        }

    except Exception as e:
        print(f"❌ Sürücü analizi hatası: {e}")
        # Hatanın detayını görmek için traceback faydalı olabilir
        # import traceback
        # traceback.print_exc()
        return {"status": "error", "message": f"Sürücü analizi sırasında hata oluştu: {str(e)}"}