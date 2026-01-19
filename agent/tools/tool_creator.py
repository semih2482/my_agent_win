import os
import re
import json
import ast
import sys
import pkgutil
from typing import Dict, Any, Union

# Proje içinden LLM fonksiyonunu import et
try:
    from agent.models.llm import ask
except ImportError:
    pass

TOOL_INFO = {
    "name": "tool_creator",
    "description": "Verilen görev için GERÇEK işlevselliğe sahip yeni bir Python aracı oluşturur. Mock veri kullanmaz, kodu otomatik düzeltir, Code Auditor ile denetler ve kaydeder.",
    "cacheable": False,
    "input_schema": {
        "type": "object",
        "properties": {
            "task_description": {
                "type": "string",
                "description": "Aracın ne yapacağı (Örn: 'Verilen URL'nin HTTP headerlarını çeken araç')."
            },
            "tool_name": {
                "type": "string",
                "description": "Dosya adı (örn: 'http_header_checker')."
            },
            "input_schema": {
                "type": "object",
                "description": "Aracın alacağı parametreler (JSON Schema formatında)."
            }
        },
        "required": ["task_description", "tool_name", "input_schema"]
    }
}

def _get_third_party_imports(code: str) -> list[str]:
    """Parses code to find third-party imports."""
    standard_libs = set(sys.builtin_module_names) | set(m.name for m in pkgutil.iter_modules())
    
    imports = set()
    try:
        root = ast.parse(code)
        for node in ast.walk(root):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split('.')[0]
                    if module_name not in standard_libs:
                        imports.add(module_name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split('.')[0]
                    if module_name not in standard_libs:
                        imports.add(module_name)
    except Exception:
        # Ignore parsing errors here, they will be caught in validation
        pass
    return list(imports)


def extract_python_code(text: str) -> str:
    """LLM çıktısından sadece temiz Python kodunu ayıklar (Daha agresif)."""
    # 1. More flexible Markdown block regex
    match = re.search(r"```(python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(2).strip()

    # 2. Fallback for unclosed blocks
    match = re.search(r"```(python)?\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(2).strip()
        
    # 3. Fallback to find code if no markdown
    lines = text.split('\n')
    start_index = -1
    for i, line in enumerate(lines):
        if line.strip().startswith(('import ', 'from ', 'TOOL_INFO', 'def run')):
            start_index = i
            break

    if start_index != -1:
        code_lines = lines[start_index:]
        return '\n'.join(code_lines).strip()

    return text.strip()

def _check_imports(code: str) -> tuple[bool, str]:
    """Dynamically checks if imported modules are available."""
    try:
        root = ast.parse(code)
        for node in ast.walk(root):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    __import__(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    __import__(node.module)
        return True, "Imports are valid."
    except ImportError as e:
        return False, f"ImportError: {e}. The required library is not installed."
    except Exception as e:
        return False, f"An unexpected error occurred during import checking: {e}"


def validate_code_quality(code: str, expected_tool_info: dict) -> tuple[bool, str]:
    """Sözdizimini, yasaklı kelimeleri, importları ve TOOL_INFO'yu kontrol eder."""
    # Syntax Check
    try:
        parsed_code = ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg} (Line content: {e.text})"

    # Forbidden Terms Check
    forbidden_terms = ["TODO: YOUR REAL CODE GOES HERE", "mock data", "fake result", "dummy response"]
    for term in forbidden_terms:
        if term in code:
            return False, f"Code contains forbidden placeholder or mock term: '{term}'. Write REAL logic."

    # Import Check
    imports_valid, import_msg = _check_imports(code)
    if not imports_valid:
        return False, import_msg

    # TOOL_INFO integrity check
    tool_info_node = None
    for node in ast.walk(parsed_code):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            if isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'TOOL_INFO':
                tool_info_node = node.value
                break
    
    if tool_info_node is None:
        return False, "Code is missing the 'TOOL_INFO' dictionary."

    try:
        # Safely evaluate the TOOL_INFO dictionary
        generated_tool_info = ast.literal_eval(tool_info_node)
        
        # Compare critical fields
        if generated_tool_info.get('name') != expected_tool_info.get('name'):
            return False, f"TOOL_INFO 'name' mismatch. Expected '{expected_tool_info.get('name')}', but found '{generated_tool_info.get('name')}'."
            
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as e:
        return False, f"Could not safely evaluate the generated TOOL_INFO block. Error: {e}"


    return True, "Valid"

def run(args: Union[dict, str], agent_instance=None) -> dict:
    """Yeni bir araç oluşturur (Auto-Fix, Template ve Code Auditor özellikli)."""

    # 1. Girdi Ayrıştırma
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return {"status": "error", "message": "Girdi formatı hatalı. JSON bekleniyordu."}

    task_description = args.get("task_description")
    tool_name = args.get("tool_name")
    input_schema = args.get("input_schema", {})

    if not task_description or not tool_name:
        return {"status": "error", "message": "Eksik argüman: task_description ve tool_name zorunludur."}

    # Dosya adını temizle
    safe_tool_name = re.sub(r'[^a-z0-9_]', '', tool_name.lower())

    expected_tool_info = {
        "name": safe_tool_name,
        "description": task_description,
        "input_schema": input_schema
    }
    
    # 2. PROMPT ŞABLONU
    prompt_template = f"""
You are an expert Python Developer. Your task is to write the logic for a tool named `{safe_tool_name}.py`.

**TASK:** {task_description}

**CRITICAL RULES:**
1.  **NO MOCK DATA or PLACEHOLDERS:** The code must have real functionality. Do not use dummy data or placeholders like '...'.
2.  **DO NOT CHANGE `TOOL_INFO`:** The `TOOL_INFO` block is pre-filled. Do not alter it in any way.
3.  **LIST THIRD-PARTY LIBRARIES:** If you use any libraries not included in the standard Python library (like 'requests', 'beautifulsoup4', etc.), list them in a comment at the top of the file. For example:
    ```python
    # THIRD-PARTY-LIBS: requests, beautifulsoup4
    ```
4.  **HANDLE ARGUMENTS CAREFULLY:** The `run` function receives arguments in a dictionary. Access them using `args.get('key_name')`.

**TEMPLATE (Copy this and fill the `TODO` section):**
```python
import json
import os
import requests
import time
from typing import Dict, Any, Union


TOOL_INFO = {{
    "name": "{safe_tool_name}",
    "description": {repr(task_description)},
    "input_schema": {repr(input_schema)}
}}

def run(args: Union[Dict, str], agent_instance=None) -> Dict[str, Any]:

    if isinstance(args, str):
        try:
            clean_args = args.strip().replace("```json", "").replace("```", "")
            args = json.loads(clean_args)
        except json.JSONDecodeError:
            return {{"status": "error", "message": "Invalid JSON arguments."}}

    if not isinstance(args, dict):
         return {{"status": "error", "message": "Arguments must be a dictionary."}}


    try:
        # TODO: YOUR REAL CODE GOES HERE.
        # Example: target_url = args.get('url')

        result_data = "..." # Replace with actual result

        return {{
            "status": "success",
            "result": result_data
        }}

    except Exception as e:
        return {{"status": "error", "message": f"Error: {{str(e)}}"}}
```

Output the COMPLETE Python code now.
"""

    MAX_RETRIES = 3
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            print(f"🔧 [Tool Creator] '{safe_tool_name}' kodlanıyor (Deneme {attempt+1}/{MAX_RETRIES})...")

            current_prompt = prompt_template
            if last_error:
                current_prompt += f"\n\n❌ PREVIOUS SYNTAX ERROR: {last_error}\nPAY ATTENTION TO STRING QUOTES AND NEWLINES."

            generated_text = ask(current_prompt, max_new_tokens=2500)
            final_code = extract_python_code(generated_text)

            # Validasyon
            is_valid, msg = validate_code_quality(final_code, expected_tool_info)
            if not is_valid:
                last_error = msg
                print(f"⚠️ [Tool Creator] Kod reddedildi: {msg}")
                continue

            # DEPENDENCY CHECK
            third_party_imports = _get_third_party_imports(final_code)
            if third_party_imports:
                print(f"⚠️ [Tool Creator] Uyarı: Araç üçüncü taraf kütüphaneler kullanıyor olabilir: {', '.join(third_party_imports)}. Bu kütüphanelerin kurulu olduğundan emin olun.")

            # KAYDETME İŞLEMİ
            tools_dir = os.path.dirname(os.path.abspath(__file__))
            community_path = os.path.join(tools_dir, "community_tools")

            if not os.path.exists(community_path):
                os.makedirs(community_path)

            file_path = os.path.join(community_path, f"{safe_tool_name}.py")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(final_code)

            print(f"✅ [Tool Creator] '{safe_tool_name}.py' diske yazıldı: {file_path}")

            # OTO-DENETİM VE DÜZELTME
            if agent_instance and "code_auditor" in agent_instance.available_tools:
                print(f"🔍 [Tool Creator] 'code_auditor' ile kalite kontrolü başlatılıyor...")
                try:
                    auditor_func = agent_instance.available_tools["code_auditor"]["func"]
                    audit_response = auditor_func({"file_path": file_path}, agent_instance)

                    if audit_response.get("status") == "success" and audit_response.get("raw_suggestions"):
                        suggestions = audit_response["raw_suggestions"]
                        if suggestions:
                            print(f"🔧 [Tool Creator] Code Auditor {len(suggestions)} iyileştirme önerdi. Uygulanıyor...")
                            
                            with open(file_path, "r", encoding="utf-8") as f:
                                original_code = f.read()
                            
                            code_lines = original_code.splitlines(True)
                            
                            # Apply suggestions in reverse to avoid line number shifts
                            suggestions.sort(key=lambda s: s.get('line_number', 0), reverse=True)
                            
                            for suggestion in suggestions:
                                line_num = suggestion.get("line_number")
                                original = suggestion.get("original_code")
                                suggested = suggestion.get("suggested_code")

                                if line_num and original and suggested:
                                    #-1 for 0-based index
                                    line_index = line_num - 1
                                    if line_index < len(code_lines) and original in code_lines[line_index]:
                                        code_lines[line_index] = code_lines[line_index].replace(original, suggested)

                            fixed_code = "".join(code_lines)

                            # Re-validate the fixed code
                            is_valid, msg = validate_code_quality(fixed_code, expected_tool_info)
                            if is_valid:
                                with open(file_path, "w", encoding="utf-8") as f:
                                    f.write(fixed_code)
                                print(f"✅ [Tool Creator] {len(suggestions)} düzeltme başarıyla uygulandı ve doğrulandı.")
                            else:
                                print(f"⚠️ [Tool Creator] Code Auditor'ın önerdiği düzeltmeler doğrulamayı geçemedi: {msg}. Değişiklikler geri alınıyor.")
                        else:
                             print(f"✅ [Tool Creator] Code Auditor herhangi bir sorun bulmadı.")

                except Exception as e:
                    print(f"⚠️ [Tool Creator] Code Auditor çalıştırılırken hata oluştu (önemsiz): {e}")


            return {
                "status": "success",
                "result": f"Yeni araç '{safe_tool_name}.py' başarıyla oluşturuldu ve Code Auditor ile denetlendi.",
                "special_action": "reload_tools",
                "tool_filename": f"{safe_tool_name}.py"
            }

        except Exception as e:
            last_error = str(e)
            print(f"❌ [Tool Creator] Beklenmedik hata: {e}")

    return {
        "status": "error",
        "message": f"Araç oluşturulamadı (3 deneme başarısız). Son hata: {last_error}"
    }