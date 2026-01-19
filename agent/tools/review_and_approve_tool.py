# agent/tools/review_and_approve_tool.py
import os
import shutil

# Bu dosyanın bulunduğu dizini temel alarak karantina ve topluluk araçları dizinlerinin mutlak yollarını belirle
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
QUARANTINE_DIR = os.path.join(TOOLS_DIR, 'quarantine_tools')
COMMUNITY_DIR = os.path.join(TOOLS_DIR, 'community_tools')

TOOL_INFO = {
    "name": "review_and_approve_tool",
    "description": "Karantinadaki araçları listeler, kodlarını gösterir ve onaylama veya reddetme işlemlerini yapar. Onaylanan araçlar community_tools klasörüne taşınır."
}

def run(args: dict | str, agent_instance=None) -> dict:
    """
    Karantina dizinindeki araçları yönetir.

    Args:
        args (dict or str): Gerçekleştirilecek eylemi ve dosya adını içeren sözlük veya metin.
                     Eğer metin ise format: '<action> <tool_filename>'
                     'action': 'list', 'review', 'approve', veya 'reject' olabilir.
                     'tool_filename': Üzerinde işlem yapılacak aracın dosya adı.
    """
    if isinstance(args, str):
        parts = args.split(maxsplit=1)
        action = parts[0] if parts else None
        tool_filename = parts[1] if len(parts) > 1 else None
    elif isinstance(args, dict):
        action = args.get('action')
        tool_filename = args.get('tool_filename')
    else:
        action = None
        tool_filename = None

    if not action:
        return {"status": "error", "message": "Eylem belirtilmedi. 'list', 'review', 'approve' veya 'reject' kullanın."}

    if not os.path.exists(QUARANTINE_DIR):
        os.makedirs(QUARANTINE_DIR)
        return {"status": "success", "result": "Karantina dizini mevcut değildi ve oluşturuldu. İncelenecek araç yok."}

    if action == 'list':
        try:
            tools = [f for f in os.listdir(QUARANTINE_DIR) if f.endswith('.py')]
            if not tools:
                return {"status": "success", "result": "Şu anda karantinada hiç araç yok."}
            return {"status": "success", "result": f"Karantinadaki araçlar: {', '.join(tools)}"}
        except Exception as e:
            return {"status": "error", "message": f"Araçlar listelenirken hata oluştu: {e}"}

    if action in ['review', 'approve', 'reject']:
        if not tool_filename:
            return {"status": "error", "message": "Bu eylem için bir 'tool_filename' gereklidir."}

        source_path = os.path.join(QUARANTINE_DIR, tool_filename)
        if not os.path.exists(source_path):
            return {"status": "error", "message": f"Araç '{tool_filename}' karantinada bulunamadı."}

        if action == 'review':
            try:
                with open(source_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return {"status": "success", "result": f"--- {tool_filename} için kod ---\n\n{content}"}
            except Exception as e:
                return {"status": "error", "message": f"Araç kodu okunurken hata oluştu: {e}"}

        if action == 'approve':
            try:
                if not os.path.exists(COMMUNITY_DIR):
                    os.makedirs(COMMUNITY_DIR)
                destination_path = os.path.join(COMMUNITY_DIR, tool_filename)
                shutil.move(source_path, destination_path)
                return {
                    "status": "success",
                    "result": f"Araç '{tool_filename}' onaylandı ve community_tools klasörüne taşındı. Araçlar yeniden yükleniyor.",
                    "special_action": "reload_tools"
                }
            except Exception as e:
                return {"status": "error", "message": f"Araç onaylanırken hata oluştu: {e}"}

        if action == 'reject':
            try:
                os.remove(source_path)
                return {"status": "success", "result": f"Araç '{tool_filename}' reddedildi ve silindi."}
            except Exception as e:
                return {"status": "error", "message": f"Araç reddedilirken hata oluştu: {e}"}

    return {"status": "error", "message": "Geçersiz eylem belirtildi. 'list', 'review', 'approve' veya 'reject' kullanın."}
