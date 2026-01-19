# personal_vector_store.py

import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import uuid
from datetime import datetime

class PersonalVectorStore:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", store_path="personal_store"):
        self.model = SentenceTransformer(model_name)
        self.store_path = store_path
        self.index_path = os.path.join(store_path, "index.faiss")
        self.meta_path = os.path.join(store_path, "meta.json")
        self.queue_path = os.path.join(store_path, "research_queue.txt") # Araştırma kuyruğu
        os.makedirs(store_path, exist_ok=True)

        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)

        self.metadata = []
        if os.path.exists(self.meta_path) and os.path.exists(self.index_path):
            self._load()

    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def _load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def add(self, text: str, metadata: dict, _add_to_queue: bool = True):
        """Yeni kişisel bilgiyi meta verilerle ekle ve araştırma kuyruğuna yaz."""
        if not text.strip():
            print("Metin boş olamaz.")
            return

        note_id = str(uuid.uuid4())
        now_iso = datetime.now().isoformat()

        # Gelen metadata'yı kopyala ve not ID'sini ekle
        full_metadata = metadata.copy()
        full_metadata['id'] = note_id
        full_metadata['text'] = text # Metni de meta veriye ekleyelim
        full_metadata['created_at'] = now_iso
        full_metadata['last_accessed_at'] = now_iso

        # Vektör veritabanına ekle
        embedding = self.model.encode([text])
        self.index.add(np.array(embedding, dtype="float32"))
        self.metadata.append(full_metadata)
        self._save()

        # Araştırma kuyruğuna ekle (SADECE _add_to_queue True ise)
        if _add_to_queue:
            try:
                topic = metadata.get("topic", "Genel")
                with open(self.queue_path, "a", encoding="utf-8") as f:
                    f.write(f"[{topic}] {text}\n")
            except Exception as e:
                print(f"Araştırma kuyruğuna yazılırken hata oluştu: {e}")


    def search(self, query: str, top_k: int = 5, metadata_filter: dict = None):
        """Sorguya en yakın kişisel bilgileri getir. Meta veriye göre filtreleyebilir."""
        if self.index.ntotal == 0:
            return []

        # Perform a wider search to have more candidates for filtering
        search_k = top_k * 5 if metadata_filter else top_k
        search_k = min(search_k, self.index.ntotal)

        query_vec = self.model.encode([query])
        D, I = self.index.search(np.array(query_vec, dtype="float32"), search_k)

        updated = False
        now_iso = datetime.now().isoformat()
        results = []
        for rank, i in enumerate(I[0]):
            if i < len(self.metadata):
                meta = self.metadata[i]
                # Son erişim zamanını güncelle
                if meta.get('last_accessed_at') != now_iso:
                    self.metadata[i]['last_accessed_at'] = now_iso
                    updated = True

                if metadata_filter:
                    match = all(meta.get(key) == value for key, value in metadata_filter.items())
                    if match:
                        results.append((meta, float(D[0][rank])))
                else:
                    results.append((meta, float(D[0][rank])))

        if updated:
            self._save()

        return results[:top_k]

    def summarize_personal_knowledge(self):
        """Kişisel bilgi özetini döndür (isteğe bağlı, persona’ya dahil etmek için)"""
        if not self.metadata:
            return "Henüz kişisel bilgi eklenmedi."
        combined = "\n".join([f'[{item.get("topic", "Genel")}] {item.get("text", "")}' for item in self.metadata[-10:]])
        return combined

    def rebuild_from_meta(self):
        """meta.json dosyasını okur ve FAISS indeksini yeniden oluşturur."""
        if not os.path.exists(self.meta_path):
            return

        try:
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

            self.index.reset()
            if self.metadata:
                texts_to_encode = [item["text"] for item in self.metadata]
                embeddings = self.model.encode(texts_to_encode)
                self.index.add(np.array(embeddings, dtype="float32"))

            self._save()
        except Exception as e:
            print(f"İndeks yeniden oluşturulurken hata: {e}")

    def get_all_topics(self):
        """Depodaki tüm benzersiz konu başlıklarını listeler."""
        if not self.metadata:
            return []
        return sorted(list(set(item["topic"] for item in self.metadata if "topic" in item)))

    def get_notes_by_topic(self, topic: str):
        """Belirli bir konudaki tüm notları döndürür."""
        if not self.metadata:
            return []
        return [item for item in self.metadata if item.get("topic") == topic]

    def delete_by_topic(self, topic: str):
        """Bir konuya ait tüm notları siler ve indeksi yeniden oluşturur."""
        initial_count = len(self.metadata)
        # Belirtilen konuya ait olmayan tüm notları tut
        self.metadata = [note for note in self.metadata if note.get("topic") != topic]

        if len(self.metadata) < initial_count:
            print(f"'{topic}' konusuna ait notlar silindi. İndeks yeniden oluşturuluyor...")
            self.rebuild_from_meta() # Değişiklik sonrası indeksi yeniden senkronize et
            return True
        return False

    def delete_note_by_id(self, note_id: str):
        """Verilen ID'ye sahip notu siler ve indeksi yeniden oluşturur."""
        initial_count = len(self.metadata)
        self.metadata = [note for note in self.metadata if note.get("id") != note_id]

        if len(self.metadata) < initial_count:
            print(f"Not (ID: {note_id}) metadata'dan silindi. İndeks yeniden oluşturuluyor...")
            self.rebuild_from_meta() # İndeksi yeniden senkronize et
            return True
        return False

    def get_all_document_texts(self) -> list[str]:
        """
        meta.json'daki tüm notların metinlerini bir liste olarak döndürür.
        """
        if not self.metadata:
            return []

        # Sadece 'text' alanlarını alıp bir liste yap
        return [item["text"] for item in self.metadata if "text" in item]