# agent/community_tools/get_current_weather.py
import requests

# Her araç bu standart yapıyı takip etmelidir.
TOOL_INFO = {
    "name": "get_current_weather",
    "description": "Belirtilen bir şehir için güncel hava durumu bilgisini verir. Girdi: şehir adı (örn: 'Istanbul')."
}

def run(args: str | dict, agent_instance=None) -> dict:
    """
    Aracın ana çalışma fonksiyonu.
    """
    city = ""
    if isinstance(args, dict):
        # Eğer girdi bir sözlük ise, 'city' anahtarını arar.
        city = args.get("city", "")
    elif isinstance(args, str):
        # Eğer girdi bir string ise, doğrudan onu şehir olarak kullanır.
        city = args

    if not city:
        return {"status": "error", "message": "Şehir adı belirtilmedi."}

    try:
        # wttr.in, API anahtarı gerektirmeyen basit bir hava durumu servisidir.
        response = requests.get(f"https://wttr.in/{city}?format=j1")
        response.raise_for_status()
        data = response.json()

        current = data.get('current_condition', [{}])[0]
        weather_report = f"{city} için hava durumu: {current.get('temp_C')}°C, {current.get('weatherDesc', [{}])[0].get('value')}. Hissedilen sıcaklık: {current.get('FeelsLikeC')}°C."
        return {"status": "success", "result": weather_report}
    except Exception as e:
        return {"status": "error", "message": f"Hava durumu bilgisi alınırken hata oluştu: {e}"}