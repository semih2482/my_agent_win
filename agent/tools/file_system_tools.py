# /mnt/d/my_agent_win/agent/tools/file_system_tools.py
import os
from typing import Dict, Any, List

TOOL_INFO = {
    "name": "file_system",
    "description": "Dosya sistemi işlemleri yapmak için bir araç seti. Dosyaları okuyabilir, yazabilir ve dizinleri listeleyebilirsiniz.",
    "cacheable": True,
    "sub_tools": {
        "read_file": {
            "description": "Belirtilen dosyanın içeriğini okur.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Okunacak dosyanın tam yolu."}
                },
                "required": ["file_path"]
            }
        },
        "write_file": {
            "description": "Belirtilen dosyaya içerik yazar. Dosya mevcutsa üzerine yazılır, değilse oluşturulur.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Yazılacak dosyanın tam yolu."},
                    "content": {"type": "string", "description": "Dosyaya yazılacak içerik."}
                },
                "required": ["file_path", "content"]
            }
        },
        "list_directory": {
            "description": "Belirtilen dizinin içeriğini listeler.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "İçeriği listelenecek dizinin yolu."}
                },
                "required": ["path"]
            }
        }
    }
}

def run(args: dict, agent_instance=None) -> dict:
    """
    Gelen argümanlara göre ilgili dosya sistemi işlemini çalıştırır.
    """
    if not isinstance(args, dict):
        return {"status": "error", "message": "Argümanlar bir sözlük olmalıdır."}

    operation = args.get("operation")
    op_args = args.get("args", {})

    if not operation:
        return {"status": "error", "message": "Yapılacak işlem ('operation') belirtilmedi."}

    if operation == "read_file":
        return read_file(**op_args)
    elif operation == "write_file":
        return write_file(**op_args)
    elif operation == "list_directory":
        return list_directory(**op_args)
    else:
        return {"status": "error", "message": f"Bilinmeyen işlem: {operation}"}

def read_file(file_path: str) -> Dict[str, Any]:
    """
    Belirtilen dosyanın içeriğini okur.

    Args:
        file_path (str): Okunacak dosyanın tam yolu.

    Returns:
        dict: İşlemin durumu ve dosya içeriği veya hata mesajı.
    """
    if not os.path.isabs(file_path):
        # Güvenlik için sadece mutlak yollara izin ver
        # ve projenin içinde olduğundan emin ol.
        # Bu kontrolü daha sonra ekleyebiliriz.
        # For now, let's resolve it relative to the project root.
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        file_path = os.path.join(project_root, file_path)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return {"status": "success", "result": content}
    except FileNotFoundError:
        return {"status": "error", "message": f"Dosya bulunamadı: {file_path}"}
    except Exception as e:
        return {"status": "error", "message": f"Dosya okunurken hata oluştu: {e}"}

def write_file(file_path: str, content: str) -> Dict[str, Any]:
    """
    Belirtilen dosyaya içerik yazar.

    Args:
        file_path (str): Yazılacak dosyanın tam yolu.
        content (str): Dosyaya yazılacak içerik.

    Returns:
        dict: İşlemin durumu ve başarı veya hata mesajı.
    """
    if not os.path.isabs(file_path):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        file_path = os.path.join(project_root, file_path)

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return {"status": "success", "result": f"Dosya başarıyla yazıldı: {file_path}"}
    except Exception as e:
        return {"status": "error", "message": f"Dosya yazılırken hata oluştu: {e}"}

def list_directory(path: str) -> Dict[str, Any]:
    """
    Belirtilen dizinin içeriğini listeler.

    Args:
        path (str): İçeriği listelenecek dizinin yolu.

    Returns:
        dict: İşlemin durumu ve dizin içeriği veya hata mesajı.
    """
    if not os.path.isabs(path):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        path = os.path.join(project_root, path)

    try:
        if not os.path.isdir(path):
            return {"status": "error", "message": f"Dizin bulunamadı: {path}"}
        
        items = os.listdir(path)
        return {"status": "success", "result": items}
    except Exception as e:
        return {"status": "error", "message": f"Dizin listelenirken hata oluştu: {e}"}
