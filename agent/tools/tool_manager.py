# agent/tools/tool_manager.py
import os
import glob
import shutil

QUARANTINE_PATH = os.path.join(os.path.dirname(__file__), 'quarantine_tools')
COMMUNITY_PATH = os.path.join(os.path.dirname(__file__), 'community_tools')

def review_tools(tool_filename: str = None):
    """
    Karantinadaki araçları yönetir.
    - Parametre verilmezse: Karantinadaki araçları listeler.
    - `tool_filename` verilirse: Belirtilen aracın kodunu gösterir ve onay sorar.
    """
    if not os.path.exists(QUARANTINE_PATH):
        return {"status": "success", "result": "Karantinada bekleyen araç bulunmuyor."}

    # Sadece karantinadaki araçları listele
    if not tool_filename:
        quarantined_files = [os.path.basename(f) for f in glob.glob(os.path.join(QUARANTINE_PATH, "*.py"))]
        if not quarantined_files:
            return {"status": "success", "result": "Karantinada bekleyen araç bulunmuyor."}
        return {"status": "success", "result": "Onay bekleyen araçlar:\n" + "\n".join(quarantined_files)}

    # Belirli bir aracı incele ve onayla
    source_path = os.path.join(QUARANTINE_PATH, tool_filename)
    dest_path = os.path.join(COMMUNITY_PATH, tool_filename)

    if not os.path.exists(source_path):
        return {"status": "error", "message": f"'{tool_filename}' karantinada bulunamadı."}

    with open(source_path, 'r', encoding='utf-8') as f:
        code = f.read()

    print("\n--- ONAY BEKLEYEN ARAÇ KODU ---")
    print(f"Dosya: {tool_filename}")
    print("---------------------------------")
    print(code)
    print("---------------------------------")

    try:
        approval = input("Bu aracı onaylayıp `community_tools` klasörüne taşımak istiyor musunuz? (yes/no): ").lower().strip()
        if approval == 'yes':
            shutil.move(source_path, dest_path)
            return {"status": "success", "result": f"'{tool_filename}' onaylandı ve taşındı. Değişikliklerin etkili olması için ajanı yeniden başlatın."}
        return {"status": "success", "result": "Araç onayı iptal edildi."}
    except (KeyboardInterrupt, EOFError):
        return {"status": "success", "result": "Onay işlemi kullanıcı tarafından iptal edildi."}