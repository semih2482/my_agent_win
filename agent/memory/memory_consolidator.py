# agent/memory/memory_consolidator.py

import os
import sys

# Proje kök dizinini Python yoluna ekleyerek import hatalarını çöz
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agent import config
from agent.models.llm import ask
from agent.memory.knowledge_store import VectorKnowledgeStore

TOOL_INFO = {
    "name": "memory_consolidator",
    "description": "Son 24 saat içinde eklenen dağınık notları okur, bunları özetleyerek daha yoğun ve anlamlı yeni notlar oluşturur ve eski notları temizler. Bu, agent'ın hafızasını düzenli tutar.",
    "cacheable": False,
    "args_schema": {}
}

def run(args: dict = None, agent_instance=None) -> dict:
    """
    Hafıza birleştirme işlemini çalıştırır.
    """
    try:
        print("🧠 Hafıza Birleştirme ve Özetleme işlemi başlatılıyor...")
        knowledge_store = VectorKnowledgeStore(db_path=config.MEMORY_DB_PATH)

        # 1. Son 24 saatteki notları al
        recent_notes = knowledge_store.get_documents_since(days=1)
        if not recent_notes:
            message = "Son 24 saatte özetlenecek yeni bir not bulunamadı."
            print(f"✅ {message}")
            return {"status": "success", "result": message}

        print(f"  -> Özetlenmek üzere {len(recent_notes)} adet yeni not bulundu.")
        note_ids_to_delete = [note[0] for note in recent_notes]
        combined_text = "\n\n---\n\n".join([note[1] for note in recent_notes])

        # 2. LLM'e özetleme ve sentezleme prompt'unu gönder
        prompt = f"""
        Aşağıda, son 24 saat içinde kaydedilmiş bir dizi not bulunmaktadır. Görevin, bu notları analiz etmek ve içlerindeki en önemli, birbiriyle ilişkili ve gelecekte tekrar kullanılabilecek ana fikirleri, gerçekleri ve sonuçları çıkararak daha yoğun ve özetlenmiş yeni notlar oluşturmaktır.

        - Tekrarlanan, önemsiz veya geçici (örn: "dosya yazıldı") bilgileri atla.
        - Birbiriyle ilişkili bilgileri tek bir anlamlı not altında birleştir.
        - Çıktın, her biri kendi başına anlamlı olan, madde imi (-) ile ayrılmış bir dizi yeni not olmalıdır.

        İŞLENECEK NOTLAR:
        ---
        {combined_text}
        ---

        YENİ, ÖZETLENMİŞ VE BİRLEŞTİRİLMİŞ NOTLAR (madde imli liste olarak):
        """

        print("  -> LLM'e notları özetlemesi için gönderiliyor...")
        summarized_response = ask(prompt, max_new_tokens=2048)

        # Yanıttan madde imli notları ayıkla
        new_notes = [note.strip() for note in summarized_response.split('-') if len(note.strip()) > 20]

        if not new_notes:
            return {"status": "error", "message": "LLM'den geçerli bir özet alınamadı."}

        # 3. Yeni, özetlenmiş notları hafızaya ekle
        print(f"  -> {len(new_notes)} adet yeni, özetlenmiş not hafızaya ekleniyor...")
        for note in new_notes:
            knowledge_store.add(f"📝 Özetlenmiş Anı: {note}")

        # 4. Eski, detaylı notları hafızadan sil
        print(f"  -> {len(note_ids_to_delete)} adet eski not temizleniyor...")
        deleted_count = knowledge_store.delete_by_ids(note_ids_to_delete)

        final_message = f"Hafıza başarıyla birleştirildi. {len(recent_notes)} eski not, {len(new_notes)} yeni özet nota dönüştürüldü. {deleted_count} not silindi."
        print(f"✅ {final_message}")
        return {"status": "success", "result": final_message}

    except Exception as e:
        error_message = f"Hafıza birleştirme sırasında bir hata oluştu: {e}"
        print(f"❌ HATA: {error_message}")
        return {"status": "error", "message": error_message}