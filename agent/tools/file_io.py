import os
import shutil

def write_text(filepath: str, content: str):
    """
    Verilen metni belirtilen dosya yoluna yazar.
    - Mevcut dosya varsa önce yedek oluşturur.
    - Hataları JSON formatında döner, CLI/agent için parse edilebilir.
    """
    try:
        # Dosya klasörü yoksa oluştur
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Mevcut dosya varsa yedek al
        if os.path.exists(filepath):
            backup_filepath = f"{filepath}.bak"
            shutil.copyfile(filepath, backup_filepath)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        return {"status": "success", "message": f"Metin '{filepath}' dosyasına başarıyla yazıldı."}

    except Exception as e:
        return {"status": "error", "message": f"Dosya yazma hatası: {e}"}


def read_text(filepath: str) -> dict:
    """
    Belirtilen dosya yolundan metni okur.
    - Dosya yoksa hata mesajı döner.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return {"status": "success", "content": f.read()}
    except FileNotFoundError:
        return {"status": "error", "message": f"Dosya bulunamadı: {filepath}"}
    except Exception as e:
        return {"status": "error", "message": f"Dosya okuma hatası: {e}"}
