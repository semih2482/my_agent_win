# agent/tools/philosophy_learner.py
import os
from ddgs import DDGS
from typing import Dict, Any

def learn_and_save(topic: str, category: str) -> Dict[str, Any]:
    """
    Belirtilen bir konuyu DuckDuckGo kullanarak internette araştırır,
    toplanan bilgileri birleştirir ve belirtilen kategoride bir dosyaya kaydeder.
    """
    print(f"🧠 '{topic}' konusu ({category}) hakkında bilgi toplanıyor...")
    
    try:
        # 1. DDGS ile konuyu araştır ve metin parçalarını topla
        results = []
        with DDGS(timeout=20) as ddgs:
            search_results = ddgs.text(f"What is {topic}? Key concepts, history, and major figures.", max_results=5)
            if search_results:
                results = list(search_results)

        if not results:
            message = f"'{topic}' hakkında arama sonucu bulunamadı."
            print(f"❌ {message}")
            return {"status": "error", "message": message}

        print(f"🔗 {len(results)} adet kaynak bulundu. İçerik birleştiriliyor...")

        # 2. Toplanan metin parçalarını (snippet) birleştir
        content_snippets = []
        for item in results:
            snippet = item.get("body")
            if snippet:
                content_snippets.append(f"--- KAYNAK: {item.get('title')}\nURL: {item.get('href')}\n\n{snippet}")

        if not content_snippets:
            message = f"'{topic}' için öğrenilecek içerik bulunamadı."
            print(f"❌ {message}")
            return {"status": "error", "message": message}

        full_content = "\n\n".join(content_snippets)

        # 3. Öğrenilen bilgileri dosyaya kaydet
        # Dosya adını konudan türet (boşlukları _ ile değiştir ve küçük harf yap)
        file_name = f"{topic.replace(' ', '_').lower()}.txt"
        
        # Kayıt dizinini oluştur
        save_directory = f"/mnt/d/my_agent_win/data/{category}"
        os.makedirs(save_directory, exist_ok=True)
        
        file_path = os.path.join(save_directory, file_name)

        print(f"💾 Bilgiler '{file_path}' dosyasına kaydediliyor...")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"# Konu: {topic.title()}\n\n")
            f.write(full_content)

        success_message = f"'{topic}' konusu başarıyla öğrenildi ve '{file_path}' dosyasına kaydedildi."
        print(f"✅ {success_message}")

        return {"status": "success", "file_path": file_path}

    except Exception as e:
        error_message = f"Öğrenme ve kaydetme sırasında bir hata oluştu: {e}"
        print(f"❌ {error_message}")
        return {"status": "error", "message": str(e)}
