# migrate.py (Tavsiye Edilen Güvenli Versiyon)

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import agent.config as config
from agent.memory.knowledge_store import VectorKnowledgeStore
from agent.memory.personal_vector_store import PersonalVectorStore
from agent.memory.knowledge_graph import KnowledgeGraphStore
from agent.memory.extractor import extract_triplets
from agent.models.llm import load_model
from tqdm import tqdm # İlerleme çubuğu için bunu import edin
import sys # Hata detayları için

CACHE_FILE = os.path.join(config.PROJECT_ROOT, "data", "migration_cache.json")

def load_cache():
    """Geçici önbellek dosyasını yükler."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_to_cache(cache_data):
    """Veriyi geçici önbellek dosyasına kaydeder."""
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)


def migrate():
    """
    Migrates all notes from VectorKnowledgeStore and PersonalVectorStore
    to the KnowledgeGraphStore with error handling and progress bar.
    """
    # --- YENİ GÜVENLİK ADIMI ---
    # Taşıma işlemine başlamadan önce eski Bilgi Grafiği veritabanını sil.
    # Bu, sütun uyuşmazlığı gibi hataları önler ve her zaman temiz bir başlangıç sağlar.
    if os.path.exists(config.KG_DB_PATH):
        print(f"Eski Bilgi Grafiği veritabanı ({config.KG_DB_PATH}) bulundu ve siliniyor...")
        os.remove(config.KG_DB_PATH)
        print("Eski veritabanı temizlendi.")

    print("Loading model for migration...")
    try:
        load_model() # LLM'i yükle
    except Exception as e:
        print(f"Kritik Hata: LLM yüklenemedi. 'extractor' çalışamaz. Hata: {e}")
        return # Model yüklenemezse devam etmenin anlamı yok.

    print("Model loaded. Starting migration to Knowledge Graph...")

    # Initialize stores
    try:
        vks = VectorKnowledgeStore(db_path=config.MEMORY_DB_PATH)
        pvs = PersonalVectorStore(store_path=config.PERSONAL_STORE_PATH)
        kgs = KnowledgeGraphStore(db_path=config.KG_DB_PATH)
    except Exception as e:
        print(f"Kritik Hata: Veritabanı mağazaları başlatılamadı: {e}")
        return

    # Get all documents
    vks_docs = vks.get_all_document_texts()
    pvs_docs = pvs.get_all_document_texts()
    all_docs = vks_docs + pvs_docs

    if not all_docs:
        print("Migrate edilecek hiçbir döküman bulunamadı. İşlem tamamlandı.")
        return

    print(f"Found {len(all_docs)} total documents to migrate.")
    print("Extracting triplets... (This may take a long time)")

    # --- YENİ: PARALEL İŞLEME VE ÖNBELLEKLEME MANTIĞI ---
    cache = load_cache()
    failed_docs = 0
    all_triplets = []

    # Sadece önbellekte olmayan dökümanları işle
    docs_to_process = [doc for doc in all_docs if doc not in cache and doc and len(doc.strip()) >= 20]

    # Önbellekte olanları doğrudan ekle
    for doc in all_docs:
        if doc in cache and cache[doc]:
            all_triplets.extend(cache[doc])

    print(f"Önbellekte {len(all_docs) - len(docs_to_process)} adet döküman bulundu. {len(docs_to_process)} adet yeni döküman paralel olarak işlenecek.")

    def process_document(doc):
        """Tek bir dökümanı işleyen worker fonksiyonu."""
        try:
            # Asıl LLM çağrısı burada
            triplets = extract_triplets(doc)
            return doc, triplets
        except Exception as e:
            # Hata durumunda dökümanı ve hatayı döndür
            return doc, f"HATA: {e}"

    # ThreadPoolExecutor ile paralel işleme
    with ThreadPoolExecutor(max_workers=config.RESEARCHER_MAX_WORKERS) as executor:
        # Görevleri havuza gönder
        future_to_doc = {executor.submit(process_document, doc): doc for doc in docs_to_process}

        # tqdm ile ilerleme çubuğu oluştur
        for future in tqdm(as_completed(future_to_doc), total=len(docs_to_process), desc="Extracting Triplets (Parallel)"):
            doc, result = future.result()
            if isinstance(result, list):
                cache[doc] = result # Sonucu önbelleğe al
                if result:
                    all_triplets.extend(result)
            else: # Hata oluştu
                print(f"\n[HATA] Döküman işlenemedi: {doc[:50]}... Hata: {result}", file=sys.stderr)
                failed_docs += 1

            # Her birkaç işlemde bir önbelleği diske kaydetmek, uzun işlemlerde veri kaybını önler
            if len(cache) % 10 == 0:
                save_to_cache(cache)

    print(f"\nExtraction complete. {failed_docs} döküman işlenirken hata oluştu.")

    # Tüm toplanan triplet'leri tek seferde veritabanına yaz
    if all_triplets:
        try:
            print(f"\n[Final Write] Toplam {len(all_triplets)} adet bilgi veritabanına yazılıyor...")
            kgs.add_triplets(all_triplets)
        except Exception as e:
            print(f"Kritik Hata: Triplet'ler Knowledge Graph'a yazılırken hata oluştu: {e}")

    print("Tüm eski notlar (Knowledge ve Personal) Knowledge Graph'a başarıyla aktarıldı.")
    save_to_cache(cache) # Son kez önbelleği kaydet

if __name__ == "__main__":
    migrate()