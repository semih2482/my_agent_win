# agent/tools/fund_data_fetcher.py
import requests
import json
from typing import Dict, Any
from datetime import datetime, timedelta

TEFAS_API_URL = "https://www.tefas.gov.tr/api/db/son_veriler"
TEFAS_HISTORY_API_URL = "https://www.tefas.gov.tr/api/DB/BindHistory"

def fetch_fund_data(args: str | dict, fetch_history: bool = False) -> Dict[str, Any]:
    """
    TEFAS üzerinden yatırım fonu verilerini çeker.
    `fund_code`: Örneğin, 'TTE', 'AFA' gibi fon kodları.
    """
    fund_code = ""
    if isinstance(args, str):
        fund_code = args
    elif isinstance(args, dict):
        fund_code = args.get("symbol", "")
    try:
        payload = {
            "sorgu": {
                "FonKod": fund_code.upper(),
                "Tarih": "" # En güncel veriyi almak için boş bırakılır.
            }
        }

        response = requests.post(TEFAS_API_URL, json=payload, headers={"Content-Type": "application/json"}, timeout=10) # 10 saniye timeout
        response.raise_for_status() # HTTP hatalarını kontrol et

        data = response.json()

        if not data or 'veri' not in data or not data['veri']:
            return {"status": "error", "message": f"'{fund_code}' kodlu fon için TEFAS'ta veri bulunamadı."}

        fund_info = data['veri'][0] # İlk fonu al (kod tek olduğu için)

        # Gerekli verileri çek
        price = fund_info.get('Fiyat', 'N/A')
        daily_change = fund_info.get('Degisim', 'N/A')
        fund_name = fund_info.get('FonAd', 'Bilinmiyor')
        fund_type = fund_info.get('FonTuru', 'Bilinmiyor')

        result_data = {
            "code": fund_code.upper(),
            "name": fund_name,
            "price": price,
            "daily_change": daily_change,
            "type": fund_type,
            "source": "TEFAS"
        }

        return {
            "status": "success",
            "result": result_data
        }
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"TEFAS API isteği başarısız oldu: {e}"}
    except Exception as e:
        return {"status": "error", "message": f"Yatırım fonu verisi çekme hatası: {e}"}

def fetch_fund_historical_data(fund_code: str, days: int = 90) -> Dict[str, Any]:
    """
    TEFAS üzerinden bir fonun geçmiş fiyat verilerini çeker.
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        start_str = start_date.strftime('%d.%m.%Y')
        end_str = end_date.strftime('%d.%m.%Y')

        payload = {
            "fontip": "YAT", # Yatırım Fonları
            "sfontur": "",
            "fonkod": fund_code.upper(),
            "bastarih": start_str,
            "sontarih": end_str,
        }

        response = requests.post(TEFAS_HISTORY_API_URL, data=payload, timeout=10) # 10 saniye timeout
        response.raise_for_status()

        data = response.json()
        if not data or 'data' not in data or not data['data']:
            return {"status": "error", "message": f"'{fund_code}' için geçmiş veri bulunamadı."}

        # Fiyatları çek (en yeniden en eskiye)
        prices = [item['Fiyat'] for item in data['data']]

        return {"status": "success", "result": {"prices": prices}}

    except Exception as e:
        return {"status": "error", "message": f"Fon geçmiş verisi çekme hatası: {e}"}