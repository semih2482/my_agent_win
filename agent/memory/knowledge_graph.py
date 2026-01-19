# /mnt/d/my_agent_win/agent/memory/knowledge_graph.py
import sqlite3
import re
import threading
from datetime import datetime
from agent.models.llm import ask


class KnowledgeGraphStore:
    """
    Bilgi grafiğini (knowledge graph) bir SQLite veritabanında yönetir.
    İlişkisel bilgileri (subject, relation, object) depolar ve sorgular.
    """

    def __init__(self, db_path: str):
        """
        Veritabanı bağlantısını başlatır ve tabloyu oluşturur.
        Thread-local storage kullanarak her thread için ayrı bir bağlantı yönetir.
        """
        self.db_path = db_path
        self.local = threading.local()
        self._get_conn()  # Ana thread için bağlantıyı başlat

    def _get_conn(self) -> sqlite3.Connection:
        """
        Mevcut thread için bir veritabanı bağlantısı oluşturur veya döndürür.
        Bu, ThreadPoolExecutor ile kullanımda thread güvenliği sağlar.
        """
        if not hasattr(self.local, "conn"):
            self.local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.create_table()
        return self.local.conn

    def create_table(self):
        """Veritabanında 'knowledge_graph' tablosunu oluşturur."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_graph (
                subject TEXT,
                relation TEXT,
                object TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_accessed_at DATETIME,
                UNIQUE(subject, relation, object) ON CONFLICT IGNORE
            )
        ''')
        conn.commit()

    def add_triplets(self, triplets: list[dict]):
        """Verilen üçlüleri (triplets) veritabanına ekler."""
        if not triplets:
            return
        conn = self._get_conn()
        cursor = conn.cursor()
        now_iso = datetime.now().isoformat()
        # Gelen dict listesini, created_at ve last_accessed_at için aynı değeri kullanarak tuple listesine çevir
        data_to_insert = [
            (t["subject"], t["relation"], t["object"], now_iso, now_iso)
            for t in triplets
            if "subject" in t and "relation" in t and "object" in t
        ]
        if not data_to_insert:
            return

        cursor.executemany(
            "INSERT OR IGNORE INTO knowledge_graph (subject, relation, object, created_at, last_accessed_at) VALUES (?, ?, ?, ?, ?)",
            data_to_insert,
        )
        conn.commit()

    def query(self, keyword: str) -> list[tuple[str, str, str]]:
        """
        Bir anahtar kelimeye göre bilgi grafiğini sorgular.
        Anahtar kelimenin subject veya object alanlarında geçtiği kayıtları arar.
        """
        conn = self._get_conn()
        cursor = conn.cursor()
        # rowid'yi de seçerek güncelleme yapmayı kolaylaştırıyoruz
        cursor.execute(
            "SELECT rowid, subject, relation, object, created_at FROM knowledge_graph WHERE subject LIKE ? OR object LIKE ?",
            (f'%{keyword}%', f'%{keyword}%')
        )
        rows = cursor.fetchall()

        # Son erişim zamanını güncelle
        if rows:
            now_iso = datetime.now().isoformat()
            row_ids = tuple(row[0]for row in rows)
            id_placeholders = ','.join('?' for _ in row_ids)
            cursor.execute(f"UPDATE knowledge_graph SET last_accessed_at = ? WHERE rowid IN ({id_placeholders})", (now_iso, *row_ids))
            conn.commit()

        return [(row[1], row[2], row[3]) for row in rows] # Orijinal formatı koru

    def query_as_text(self, prompt: str) -> str:
        """
        Bir kullanıcı istemini LLM ile analiz ederek anahtar kelimeler çıkarır,
        bu kelimelerle veritabanını sorgular ve sonucu okunabilir bir metin olarak döndürür.
        Bu fonksiyon, agent'ın karar verme mekanizmasına doğrudan girdi sağlar.
        """
        keyword_prompt = f"Aşağıdaki metinden en önemli 1-2 anahtar kelimeyi veya varlık adını çıkar. Sadece kelimeleri, virgülle ayırarak yaz. Metin: '{prompt}'"
        try:
            keywords_str = ask(keyword_prompt, max_new_tokens=32).strip()
            keywords = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]
        except Exception as e:
            print(f"[KnowledgeGraph] Anahtar kelime çıkarılırken hata: {e}")
            return ""

        if not keywords: return ""
        all_results = set()
        for keyword in keywords:
            results = self.query(keyword)
            for res in results:
                all_results.add(res)
        if not all_results: return ""
        return "\n".join([f"- {s} {r} {o}." for s, r, o in all_results])

    def close(self):
        """Veritabanı bağlantısını kapatır."""
        if hasattr(self.local, "conn"):
            self.local.conn.close()
