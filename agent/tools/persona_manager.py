import re
import sqlite3
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import numpy as np
from agent.models.llm import ask, embed

# optional encryption
try:
    from cryptography.fernet import Fernet
    _CRYPTO_AVAILABLE = True
except Exception:
    _CRYPTO_AVAILABLE = False

DEFAULT_DB = "data/persona.sqlite"
REDACTION_PATTERNS = [
    (r"\b(?:\d{10,15})\b", "<PHONE>"),
    (r"\b\d{5,7}\b", "<PIN>"),
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "<EMAIL>"),
    (r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "<IP>"),
]

_SIM_THRESHOLD = 0.82

class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"

class PersonaManager:
    """
    Kullanıcı mesajlarından özellik / trait çıkarır ve DB'ye kaydeder.
    Opt-in ve hassas veri maskesi içerir.
    """

    def __init__(self, db_path = DEFAULT_DB, encrypt_key: Optional[bytes] = None, retention_days: int = 365):
        self.db_path = db_path
        self.retention_days = retention_days
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._cur = self._conn.cursor()
        self._create_tables()

        # encryption
        if encrypt_key and _CRYPTO_AVAILABLE:
            self._fernet = Fernet(encrypt_key)
        else:
            self._fernet = None

    # DATABASE & REDACTION
    def _create_tables(self):
        self._cur.execute("""
        CREATE TABLE IF NOT EXISTS persona_traits (
            id INTEGER PRIMARY KEY,
            trait TEXT NOT NULL,
            embedding BLOB,
            source_text BLOB,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """)
        self._cur.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            k TEXT PRIMARY KEY,
            v TEXT
        );
        """)
        self._conn.commit()

    def _redact(self, text: str) -> str:
        out = text
        for pattern, repl in REDACTION_PATTERNS:
            out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
        if len(out) > 2000:
            out = out[:2000] + "..."
        return out

    def _encrypt(self, data: str) -> bytes:
        if self._fernet:
            return self._fernet.encrypt(data.encode("utf-8"))
        return data.encode("utf-8")

    def _decrypt(self, data: bytes) -> str:
        if self._fernet:
            return self._fernet.decrypt(data).decode("utf-8")
        return data.decode("utf-8")

    def _vec_to_blob(self, vec: np.ndarray) -> bytes:
        return vec.tobytes()

    def _blob_to_vec(self, blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype="float32")

    # TRAIT EKLEME & ÖZET
    def add_trait(self, trait_text: str, source_text: str = "") -> Dict[str, Any]:
        trait_text_clean = self._redact(trait_text.strip())
        if not trait_text_clean:
            return {"status": "error", "message": "Boş trait."}

        # 1. Adım: Birebir aynı metin var mı diye kontrol et (En Hızlı)
        self._cur.execute("SELECT id, trait FROM persona_traits WHERE trait = ?", (trait_text_clean,))
        existing = self._cur.fetchone()
        if existing:
            _id, trait = existing
            return {"status": "exists", "message": "Aynı trait zaten var.", "id": _id, "trait": trait}

        # 2. Adım: Anlamsal olarak benzer bir özellik var mı diye kontrol et (Daha Akıllı)
        vec = embed(trait_text_clean)
        if isinstance(vec, list):
            vec = np.array(vec, dtype="float32")

        # Veritabanındaki tüm embedding'leri çek
        self._cur.execute("SELECT id, trait, embedding FROM persona_traits")
        all_traits = self._cur.fetchall()

        for existing_id, existing_trait, existing_emb_blob in all_traits:
            if existing_emb_blob:
                existing_vec = self._blob_to_vec(existing_emb_blob)
                # Kosinüs benzerliğini hesapla
                similarity = np.dot(vec, existing_vec) / (np.linalg.norm(vec) * np.linalg.norm(existing_vec))
                if similarity > _SIM_THRESHOLD:
                    return {"status": "exists_semantically",
                            "message": f"Anlamsal olarak benzer bir özellik zaten mevcut: '{existing_trait}' (Benzerlik: {similarity:.2f})",
                            "id": existing_id}

        # 3. Adım: Özellik yeniyse, veritabanına ekle

        emb_blob = self._vec_to_blob(vec.astype("float32"))
        encrypted_source = self._encrypt(source_text) if source_text else b""
        self._cur.execute(
            "INSERT INTO persona_traits (trait, embedding, source_text) VALUES (?, ?, ?)",
            (trait_text_clean, emb_blob, encrypted_source)
        )
        self._conn.commit()
        return {"status": "success", "message": "Trait eklendi."}

    def extract_and_add_from_message(self, message: str) -> Dict[str, Any]:
        """
        Mesajdan LLM ile 3-6 trait çıkarır ve DB'ye ekler.
        """
        redacted = self._redact(message)
        prompt = (
            "Aşağıdaki kullanıcı mesajından 3-6 tane kısa, tek-ifadeli 'trait' veya tercih çıkar. "
            "Her birini virgülle ayır. Traitler kısa ve açık olsun (örn. 'kahve sever', 'gece kuşu', 'python geliştirici').\n\n"
            f"Mesaj: {redacted}\n\nTraitler (virgülle ayrılmış):"
        )
        try:
            resp = ask(prompt, max_new_tokens=80)
            candidate_text = resp.strip().split("\n")[0]
            candidates = [c.strip().strip(",-.") for c in re.split(r",|\t|;|\n", candidate_text) if c.strip()]
            added = []
            for c in candidates:
                if 2 <= len(c) <= 150:
                    res = self.add_trait(c, source_text=message)
                    added.append({"trait": c, "result": res})
            if added:
                print(f"{Colors.OKGREEN}[Persona Güncellendi]: {added}{Colors.ENDC}")
            return {"status": "success", "added": added}
        except Exception as e:
            print(f"{Colors.WARNING}[Persona Extraction Hatası]: {e}{Colors.ENDC}")
            return {"status": "error", "message": str(e)}

    def get_traits(self, limit: int = 50) -> List[Dict[str, Any]]:
        self._cur.execute("SELECT id, trait, source_text, timestamp FROM persona_traits ORDER BY timestamp DESC LIMIT ?", (limit,))
        rows = []
        for r in self._cur.fetchall():
            id_, trait, src, ts = r
            try:
                src_dec = self._decrypt(src) if src else ""
            except Exception:
                src_dec = "<encrypted>"
            rows.append({"id": id_, "trait": trait, "source": src_dec, "timestamp": ts})
        return rows

    def summarize_persona(self, max_chars: int = 600) -> str:
        traits = self.get_traits(limit=40)
        if not traits:
            return ""
        trait_list = [t["trait"] for t in traits]
        prompt = (
            "Aşağıdaki kısa kullanıcı özelliklerini al ve Türkçe, "
            "doğal bir şekilde 2-4 cümle ile özetle. Kişisel hassas veriler çıkartılmasın.\n\n"
            f"{', '.join(trait_list)}\n\nÖzet:"
        )
        try:
            summary = ask(prompt, max_new_tokens=120)
            if len(summary) > max_chars:
                return summary[:max_chars] + "..."
            return summary
        except Exception:
            return ", ".join(trait_list[:10])

    def purge_old(self):
        cutoff = datetime.utcnow() - timedelta(days=self.retention_days)
        self._cur.execute("DELETE FROM persona_traits WHERE timestamp < ?", (cutoff.isoformat(),))
        self._conn.commit()

    def close(self):
        self._conn.commit()
        self._conn.close()
