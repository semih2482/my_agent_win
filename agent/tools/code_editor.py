# agent/tools/code_editor.py
import ast
import os
import re
import shutil

# `patch` kütüphanesini import etmeye çalış, yoksa hata yönetimi için hazır ol
try:
    import patch
    _PATCH_AVAILABLE = True
except ImportError:
    _PATCH_AVAILABLE = False

TOOL_INFO = {
    "name": "code_editor",
    "description": "Dosyaları okumak, yazmak, kod eklemek ve diff formatındaki yamaları uygulamak için kullanılır.",
    "cacheable": False,
    "args_schema": {
        "action": "'read_file' | 'rewrite_file' | 'apply_patch' | 'append_code' | 'refactor_rename_function' | 'apply_diff'",
        "file_path": "string",
        "new_content": "string (for rewrite_file)",
        "pattern": "string (for apply_patch)",
        "replacement": "string (for apply_patch)",
        "new_code": "string (for append_code)",
        "diff_content": "string (for apply_diff, unified diff format)",
        "old_function_name": "string (for refactor_rename_function)",
        "new_function_name": "string (for refactor_rename_function)"
    }
}

def run(args: dict, agent_instance=None) -> dict:
    """Wrapper function to call the appropriate code editor function."""
    action = args.get('action')
    file_path = args.get('file_path')

    if not action or not file_path:
        return {"status": "error", "message": "Missing 'action' or 'file_path' in arguments."}

    if action == "read_file":
        return read_file(file_path)
    elif action == "rewrite_file":
        new_content = args.get('new_content')
        if new_content is None:
            return {"status": "error", "message": "Missing 'new_content' for rewrite_file action."}
        return rewrite_file(file_path, new_content)
    elif action == "apply_patch":
        pattern = args.get('pattern')
        replacement = args.get('replacement')
        if not pattern or replacement is None:
            return {"status": "error", "message": "Missing 'pattern' or 'replacement' for apply_patch action."}
        return apply_patch(file_path, pattern, replacement)
    elif action == "append_code":
        new_code = args.get('new_code')
        if new_code is None:
            return {"status": "error", "message": "Missing 'new_code' for append_code action."}
        return append_code(file_path, new_code)
    elif action == "apply_diff":
        diff_content = args.get('diff_content')
        if diff_content is None:
            return {"status": "error", "message": "Missing 'diff_content' for apply_diff action."}
        return apply_diff(file_path, diff_content)
    elif action == "refactor_rename_function":
        old_function_name = args.get('old_function_name')
        new_function_name = args.get('new_function_name')
        if not old_function_name or not new_function_name:
            return {"status": "error", "message": "Missing 'old_function_name' or 'new_function_name' for refactor_rename_function action."}
        return refactor_rename_function(file_path, old_function_name, new_function_name)
    else:
        return {"status": "error", "message": f"Unknown action: {action}"}

def read_file(file_path: str) -> dict:
    try:
        if not os.path.exists(file_path):
            return {"status": "error", "message": f"File not found: {file_path}"}
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return {"status": "success", "result": content}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def rewrite_file(file_path: str, new_content: str) -> dict:
    try:
        if os.path.exists(file_path):
            shutil.copy(file_path, file_path + ".bak")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        if file_path.endswith(".py"):
            os.system(f"ruff format {file_path}")
        return {"status": "success", "result": f"File rewritten: {file_path} (backup saved)"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def apply_patch(file_path: str, pattern: str, replacement: str) -> dict:
    try:
        if not os.path.exists(file_path):
            return {"status": "error", "message": f"File not found: {file_path}"}
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        new_content, count = re.subn(pattern, replacement, content, flags=re.MULTILINE)
        if count == 0:
            return {"status": "error", "message": f"No matches for pattern: {pattern}"}

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        if file_path.endswith(".py"):
            os.system(f"ruff format {file_path}")

        return {"status": "success", "result": f"Patched {count} occurrence(s) in {file_path}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def append_code(file_path: str, new_code: str, check_duplicate=False) -> dict:
    try:
        if check_duplicate and os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                if new_code in f.read():
                    return {"status": "warning", "message": "Code already exists, not appended."}

        with open(file_path, "a", encoding="utf-8") as f:
            f.write("\n" + new_code + "\n")
        if file_path.endswith(".py"):
            os.system(f"ruff format {file_path}")
        return {"status": "success", "result": f"Appended code to {file_path}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def apply_diff(file_path: str, diff_content: str) -> dict:
    """
    Bir dosyaya birleşik diff formatında bir yama uygular.
    """
    if not _PATCH_AVAILABLE:
        return {"status": "error", "message": "Yama uygulamak için 'patch' kütüphanesi gerekli. Lütfen 'pip install patch' ile kurun."}

    if not os.path.exists(file_path):
        return {"status": "error", "message": f"Dosya bulunamadı: {file_path}"}

    try:
        # Yama uygulamadan önce bir yedek oluştur
        shutil.copy(file_path, file_path + ".bak")

        # Yamayı string olarak oluştur
        patch_set = patch.fromstring(diff_content.encode('utf-8'))

        # Yamayı uygula (root, dosya sisteminin kökünü belirtir)
        # Proje içindeki göreceli yolları doğru yönetmek için dosyanın bulunduğu dizini root olarak ayarlıyoruz.
        if patch_set.apply(strip=1, root=os.path.dirname(file_path)):
            # Başarılı olursa dosyayı formatla
            if file_path.endswith(".py"):
                os.system(f"ruff format {file_path}")
            return {"status": "success", "result": f"Diff yaması '{file_path}' dosyasına başarıyla uygulandı."}
        else:
            # Yedekten geri dön
            shutil.move(file_path + ".bak", file_path)
            return {"status": "error", "message": "Diff yaması uygulanamadı. Dosya içeriği yama ile eşleşmiyor olabilir."}

    except Exception as e:
        return {"status": "error", "message": f"Diff yaması uygulanırken bir hata oluştu: {e}"}

# AST Tabanlı Yeniden Düzenleme Araçları

class FunctionRenamer(ast.NodeTransformer):
    def __init__(self, old_name, new_name):
        self.old_name = old_name
        self.new_name = new_name

    def visit_FunctionDef(self, node):
        if node.name == self.old_name:
            node.name = self.new_name
        self.generic_visit(node)
        return node

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == self.old_name:
            node.func.id = self.new_name
        self.generic_visit(node)
        return node

def refactor_rename_function(file_path: str, old_function_name: str, new_function_name: str) -> dict:
    """
    Bir dosya içindeki bir fonksiyonu ve çağrıldığı yerleri AST kullanarak güvenli bir şekilde yeniden adlandırır.
    Bu, basit metin değiştirmekten çok daha güvenilir bir yöntemdir.
    """
    try:
        if not os.path.exists(file_path) or not file_path.endswith(".py"):
            return {"status": "error", "message": "Dosya bulunamadı veya Python dosyası değil."}

        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()

        tree = ast.parse(source_code)
        renamer = FunctionRenamer(old_name=old_function_name, new_name=new_function_name)
        new_tree = renamer.visit(tree)
        ast.fix_missing_locations(new_tree)

        new_code = ast.unparse(new_tree)

        # Değiştirilen kodla dosyanın üzerine yaz
        shutil.copy(file_path, file_path + ".bak") # Yedek oluştur
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_code)

        # Sonucu otomatik olarak formatla
        os.system(f"ruff format {file_path}")

        return {"status": "success", "result": f"'{old_function_name}' fonksiyonu '{new_function_name}' olarak {file_path} içinde başarıyla yeniden düzenlendi."}

    except Exception as e:
        return {"status": "error", "message": f"AST ile yeniden düzenleme başarısız oldu: {e}"}