import json
import os
import sys
from typing import Dict, Any

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agent.models.llm import ask

TOOL_INFO = {
    "name": "code_auditor",
    "description": "Analyzes a given Python file and suggests improvements to make it more robust, efficient, and professional. It identifies potential bugs, style issues, and areas for refactoring.",
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The absolute path to the Python file to be audited."
            }
        },
        "required": ["file_path"]
    }
}

def run(args: Dict[str, Any], agent_instance: Any = None) -> Dict[str, Any]:
    """
    Runs the code audit tool.
    """
    # AKILLI GİRDİ KONTROLÜ
    # Bu araç, LLM yanlışlıkla bir web görevi için onu seçerse diye
    # gelen argümanları kontrol etmeli ve görevi reddetmelidir.
    if not isinstance(args, dict) or "file_path" not in args:
        return {
            "status": "error",
            "message": "Bu araç sadece bir dosya yolunu ('file_path') analiz edebilir. Girdi, bu araca uygun değil. Belki de 'tool_creator' ile yeni bir araç oluşturulmalıdır."
        }

    file_path = args.get("file_path")

    if not file_path or not os.path.isabs(file_path):
        return {"status": "error", "message": "Mutlak bir dosya yolu ('file_path') sağlanmalıdır."}

    if not os.path.exists(file_path):
        return {"status": "error", "message": f"File not found at path: {file_path}"}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code_content = f.read()
    except Exception as e:
        return {"status": "error", "message": f"Error reading file: {str(e)}"}

    if not code_content.strip():
        return {"status": "success", "result": "The file is empty. No audit needed."}

    prompt = f"""
You are an expert Python code reviewer and software architect. Your task is to perform a thorough audit of the following Python code.

Analyze the code for the following aspects:
1.  **Potential Bugs:** Look for logical errors, race conditions, unhandled exceptions, and other potential bugs.
2.  **Performance Issues:** Identify inefficient code, unnecessary computations, and opportunities for optimization.
3.  **Readability and Style:** Check for adherence to PEP 8, good naming conventions, and overall code clarity. Suggest improvements for better readability.
4.  **Refactoring Opportunities:** Identify complex functions or classes that could be simplified, broken down, or redesigned for better modularity and maintainability.
5.  **Security Vulnerabilities:** Look for common security issues like injection vulnerabilities, insecure handling of data, etc. (if applicable).

Provide your feedback as a JSON object. The JSON object should contain a single key "suggestions", which is a list of suggestion objects. Each suggestion object must have the following keys:
- `line_number`: The starting line number for the code block the suggestion applies to.
- `original_code`: The exact block of code that needs improvement.
- `suggestion_type`: A category for the suggestion (e.g., "Bug", "Performance", "Style", "Refactoring", "Security").
- `description`: A clear and concise explanation of the issue and why the improvement is needed.
- `suggested_code`: The new, improved code block.

If you find no issues, return a JSON object with an empty "suggestions" list: `{{"suggestions": []}}`.

Here is the code from the file `{file_path}`:
```python
{code_content}
```

Your response must be ONLY the JSON object, with no other text or explanation.
"""

    try:
        print(f"[Code Auditor] Analyzing file: {file_path}")
        llm_response = ask(prompt, max_new_tokens=4096)

        # Basic cleanup to find the JSON object
        json_match = llm_response[llm_response.find('{'):llm_response.rfind('}')+1]

        if not json_match:
            return {"status": "error", "message": "LLM did not return a valid JSON object."}

        suggestions = json.loads(json_match)

        if "suggestions" not in suggestions or not isinstance(suggestions["suggestions"], list):
            return {"status": "error", "message": "LLM response is not in the expected format (missing 'suggestions' list)."}

        if not suggestions["suggestions"]:
            return {"status": "success", "result": f"No improvement suggestions found for {file_path}."}

        # Format the result for better readability
        formatted_suggestions = []
        for s in suggestions["suggestions"]:
            formatted_suggestions.append(
                f"Line {s['line_number']} ({s['suggestion_type']}):\n"
                f"  - Description: {s['description']}\n"
                f"  - Original Code:\n```python\n{s['original_code']}\n```\n"
                f"  - Suggested Code:\n```python\n{s['suggested_code']}\n```\n"
            )

        result_text = f"Found {len(suggestions['suggestions'])} improvement suggestions for {file_path}:\n\n" + "\n".join(formatted_suggestions)

        # Return both the raw suggestions for programmatic use and the formatted text for display
        return {
            "status": "success",
            "result": result_text,
            "raw_suggestions": suggestions["suggestions"]
        }

    except json.JSONDecodeError as e:
        return {"status": "error", "message": f"Failed to parse LLM response as JSON. Error: {e}. Response: {llm_response}"}
    except Exception as e:
        return {"status": "error", "message": f"An unexpected error occurred during code audit: {str(e)}"}

if __name__ == '__main__':
    # For testing the tool directly
    test_file_path = os.path.abspath(os.path.join(project_root, 'agent', 'core', 'agent.py'))
    if not os.path.exists(test_file_path):
        print(f"Test file not found: {test_file_path}")
    else:
        test_args = {"file_path": test_file_path}
        result = run(test_args)
        print(json.dumps(result, indent=2, ensure_ascii=False))