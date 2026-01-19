# agent/tools/web_reader.py
import requests
import trafilatura
from bs4 import BeautifulSoup

def read_url(url: str, timeout: int = 15):
    """URL içeriğini indirip temiz metin döndürür. Trafılatura başarısızsa fallback yapar."""
    # ANSI renk kodları
    OKCYAN = "\033[96m"
    ENDC = "\033[0m"
    try:
        # trafilatura'nın requests ile timeout kullanarak çalışmasını sağlayalım
        print(f"{OKCYAN}  -> 📖 '{url}' okunuyor...{ENDC}")
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
        response.raise_for_status()
        downloaded = response.text

        if downloaded:
            text = trafilatura.extract(downloaded)
            if text:
                return text

        # Trafılatura metin çıkaramazsa, BeautifulSoup ile fallback yapalım
        soup = BeautifulSoup(downloaded, "html.parser")
        title = soup.title.string.strip() if soup.title else ""
        desc_tag = soup.find("meta", attrs={"name": "description"})
        desc = desc_tag["content"].strip() if desc_tag and desc_tag.get("content") else ""

        return f"{title}\n{desc}" if (title or desc) else downloaded[:3000]

    except Exception as e:
        return f"URL okuma hatası: {e}"


def summarize_text(text, llm_ask_function):
    """Verilen metni LLM kullanarak özetler."""
    if not text:
        return "Özetlenecek içerik bulunamadı."
    try:
        # Context'i aşmamak için metni sınırla
        limited_text = text[:4000]
        prompt = (
            "Aşağıdaki metni Türkçe olarak, ana fikirlerini koruyarak kısa ve öz bir şekilde özetle:\n\n"
            f"{limited_text}\n\n"
            "Özet:"
        )
        return llm_ask_function(prompt, max_new_tokens=1024)
    except Exception as e:
        return f"Özetleme sırasında hata oluştu: {e}"
