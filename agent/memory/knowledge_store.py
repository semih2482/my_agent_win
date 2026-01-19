# agent/memory/vector_store.py
# Bu dosya, hem KnowledgeStore'un hem de VectorMemory'nin yerini alır.

import sqlite3
import numpy as np
import faiss
from typing import List, Tuple, Any
from datetime import datetime, timedelta
from agent.models.llm import embed, _EMBED_DIM # Model dosyanızdan boyutu alıyoruz

class VectorKnowledgeStore:
    """
    Hem kalıcı SQL depolamayı (SQLite) hem de hızlı anlamsal aramayı (FAISS)
    birleştiren tek, güçlü hafıza sınıfı.
    """

    def __init__(self, db_path: str = "data/memory.sqlite", dim: int = _EMBED_DIM):
        self.db_path = db_path
        self.dim = dim
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_table()

        # FAISS index'i, SQL veritabanından her başlatmada "in-memory" olarak yüklenir.
        # Bu, diske bağımlı kırılgan .index dosyalarından çok daha güvenlidir.
        self.index = faiss.IndexFlatL2(dim)
        self._load_index_from_db()

    def _create_table(self):
        """Notu ve embedding'ini aynı tabloda saklar."""
        cursor = self._conn.cursor()
        cursor.execute("PRAGMA table_info(vector_notes)")
        columns = [info[1] for info in cursor.fetchall()]

        if not columns:
            # Tablo hiç yok, baştan oluştur
            cursor.execute("""
                 CREATE TABLE vector_notes (
                     id INTEGER PRIMARY KEY,
                     content TEXT NOT NULL,
                     embedding BLOB NOT NULL,
                     created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                     last_accessed_at DATETIME
                 );
             """)
        else:
            # Tablo var, eksik sütunları ekle
            if 'created_at' not in columns:
                print("🔹 Veritabanı şeması güncelleniyor: 'created_at' sütunu ekleniyor...")
                cursor.execute("ALTER TABLE vector_notes ADD COLUMN created_at DATETIME")
                cursor.execute("UPDATE vector_notes SET created_at = ? WHERE created_at IS NULL", (datetime.now().isoformat(),))
            if 'last_accessed_at' not in columns:
                print("🔹 Veritabanı şeması güncelleniyor: 'last_accessed_at' sütunu ekleniyor...")
                cursor.execute("ALTER TABLE vector_notes ADD COLUMN last_accessed_at DATETIME")

        self._conn.commit()

    def _vec_to_blob(self, vec: np.ndarray) -> bytes:
        return vec.astype("float32").tobytes()

    def _blob_to_vec(self, blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype="float32")

    def _load_index_from_db(self):
        """Veritabanındaki tüm vektörleri FAISS'e (RAM'e) yükler."""
        print("🧠 Anlamsal hafıza (FAISS) SQL'den yükleniyor...")
        with self._conn:
            cursor = self._conn.execute("SELECT id, embedding FROM vector_notes")
            rows = cursor.fetchall()
            if not rows:
                print("🔹 Hafıza boş, yeni index başlatıldı.")
                return

            # FAISS'in "ID"leri ile SQL'in "ID"lerinin eşleşmesi için
            # IndexIDMap kullanmak en sağlam yoldur.
            ids = np.array([row[0] for row in rows], dtype='int64')
            embeddings = np.array([self._blob_to_vec(row[1]) for row in rows]).astype('float32')

            # Boyut uyumluluğunu kontrol et
            if embeddings.shape[1] != self.dim:
                print(f"HATA: Veritabanı boyutu ({embeddings.shape[1]}) ile model boyutu ({self.dim}) uyumsuz.")
                return

            self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dim))
            self.index.add_with_ids(embeddings, ids)
            print(f"✅ {len(rows)} adet anı hafızaya yüklendi.")

    def add(self, content: str) -> dict:
        """
        Yeni bir not ekler. Hem SQL'e (kalıcı) hem de FAISS'e (anlık) yazar.
        """
        try:
            vec = embed(content)
            vec_blob = self._vec_to_blob(vec)
            now_iso = datetime.now().isoformat()

            with self._conn:
                cursor = self._conn.execute(
                    "INSERT INTO vector_notes (content, embedding, created_at, last_accessed_at) VALUES (?, ?, ?, ?)",
                    (content, vec_blob, now_iso, now_iso)
                )
                new_id = cursor.lastrowid

            # Canlı (in-memory) FAISS indeksine de ekle
            vec_np = vec.astype('float32').reshape(1, -1)
            id_np = np.array([new_id], dtype='int64')
            self.index.add_with_ids(vec_np, id_np)

            return {"status": "success", "message": "Anı başarıyla eklendi.", "id": new_id}
        except Exception as e:
            return {"status": "error", "message": f"Anı ekleme hatası: {e}"}

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float, str]]:
        """
        Sorguya en yakın notları FAISS kullanarak arar ve SQL'den çeker.
        :return: List of (content, distance, created_at)
        """
        if self.index.ntotal == 0:
            return [] # Hafıza boşsa arama yapma

        q_vec = embed(query).astype('float32').reshape(1, -1)
        distances, ids = self.index.search(q_vec, top_k)

        results = []
        if len(ids[0]) == 0:
            return []

        # FAISS'ten dönen ID'leri kullanarak SQL'den metin içeriklerini çek
        # Bu, notları ayrı bir .txt dosyasında tutmaktan çok daha güvenlidir.
        id_list = tuple(ids[0].astype(int))
        id_list_placeholder = ','.join('?' for _ in id_list)
        query_sql = f"SELECT id, content, created_at FROM vector_notes WHERE id IN ({id_list_placeholder})"

        # ID'leri doğru sırayla almak için bir sözlük kullan
        with self._conn:
            id_map = {row[0]: (row[1], row[2]) for row in self._conn.execute(query_sql, id_list)}

            # Son erişim zamanını güncelle
            now_iso = datetime.now().isoformat()
            update_sql = f"UPDATE vector_notes SET last_accessed_at = ? WHERE id IN ({id_list_placeholder})"
            self._conn.execute(update_sql, (now_iso, *id_list))
            self._conn.commit()

        # FAISS'in döndürdüğü sırayla sonuçları oluştur
        for i, idx in enumerate(ids[0]):
            if idx in id_map:
                content, created_at = id_map[idx]
                dist = distances[0][i]
                results.append((content, float(dist), created_at))

        return results

    def get_documents_since(self, days: int = 1) -> List[Tuple[int, str]]:
        """
        Veritabanından belirtilen gün sayısından daha yeni olan tüm notları alır.
        :return: List of (id, content)
        """
        try:
            since_date = (datetime.now() - timedelta(days=days)).isoformat()
            with self._conn:
                cursor = self._conn.execute(
                    "SELECT id, content FROM vector_notes WHERE created_at >= ?",
                    (since_date,)
                )
                return cursor.fetchall()
        except Exception as e:
            print(f"HATA: (get_documents_since) Son dökümanlar alınamadı: {e}")
            return []

    def delete_by_ids(self, ids_to_delete: List[int]) -> int:
        """
        Verilen ID listesine göre notları hem SQL'den hem de FAISS'ten siler.
        """
        if not ids_to_delete:
            return 0
        try:
            # 1. FAISS indeksinden bu ID'lere karşılık gelen vektörleri kaldır
            ids_selector = faiss.IDSelectorBatch(np.array(ids_to_delete, dtype='int64'))
            removed_count = self.index.remove_ids(ids_selector)
            print(f"FAISS indeksinden {removed_count} vektör kaldırıldı.")

            # 2. SQL veritabanından bu notları sil
            with self._conn:
                id_placeholders = ','.join('?' for _ in ids_to_delete)
                cursor = self._conn.execute(
                    f"DELETE FROM vector_notes WHERE id IN ({id_placeholders})",
                    tuple(ids_to_delete)
                )
                deleted_rows = cursor.rowcount
                self._conn.commit()
                print(f"SQL veritabanından {deleted_rows} satır silindi.")

            return deleted_rows
        except Exception as e:
            print(f"HATA: (delete_by_ids) Notlar silinirken bir hata oluştu: {e}")
            return -1


    def delete_by_content(self, content_substring: str) -> int:
        """
        Belirtilen bir alt metni içeren tüm notları hem SQL'den hem de FAISS'ten siler.
        Bu, yanlış veya ilgisiz bilgileri temizlemek için kullanışlıdır.
        """
        try:
            # 1. Alt metni içeren notların ID'lerini ve içeriklerini SQL'den bul
            with self._conn:
                cursor = self._conn.execute(
                    "SELECT id, content FROM vector_notes WHERE content LIKE ?",
                    (f'%{content_substring}%',)
                )
                rows_to_delete = cursor.fetchall()

            if not rows_to_delete:
                print(f"'{content_substring}' içeren silinecek not bulunamadı.")
                return 0

            ids_to_delete = [row[0] for row in rows_to_delete]
            print(f"Silinecek notlar (ID'ler): {ids_to_delete}")

            # 2. FAISS indeksinden bu ID'lere karşılık gelen vektörleri kaldır
            # remove_ids bir ID seçici nesnesi bekler
            ids_selector = faiss.IDSelectorBatch(np.array(ids_to_delete, dtype='int64'))
            removed_count = self.index.remove_ids(ids_selector)
            print(f"FAISS indeksinden {removed_count} vektör kaldırıldı.")

            # 3. SQL veritabanından bu notları sil
            with self._conn:
                id_placeholders = ','.join('?' for _ in ids_to_delete)
                cursor = self._conn.execute(
                    f"DELETE FROM vector_notes WHERE id IN ({id_placeholders})",
                    tuple(ids_to_delete)
                )
                print(f"SQL veritabanından {cursor.rowcount} satır silindi.")

            return cursor.rowcount
        except Exception as e:
            print(f"HATA: (delete_by_content) Notlar silinirken bir hata oluştu: {e}")
            return -1

    def get_all_document_texts(self) -> list[str]:
        """
        Veritabanındaki (SQL) tüm notların metin içeriklerini bir liste olarak döndürür.
        """
        try:
            with self._conn:
                cursor = self._conn.execute("SELECT content FROM vector_notes")

                # fetchall() bir liste döner, örn: [('not 1',), ('not 2',)]
                rows = cursor.fetchall()

                # Bu [(tuple)] listesini [string] listesine çevir
                return [row[0] for row in rows]
        except Exception as e:
            print(f"HATA: (get_all_document_texts) SQL'den tüm dökümanlar alınamadı: {e}")
            return []

    def close(self):
        """Veritabanı bağlantısını kapatır."""
        self._conn.close()