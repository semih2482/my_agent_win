# /mnt/d/my_agent_win/tests/test_tools/test_code_auditor.py
import pytest
import json
from agent.tools import code_auditor

def test_code_auditor_success(mock_llm_ask, tmp_path):
    """
    code_auditor aracının başarılı bir şekilde çalıştığı senaryoyu test eder.
    - mock_llm_ask: conftest.py'den gelen ve LLM'i taklit eden fixture.
    - tmp_path: pytest'in sağladığı geçici bir dosya yolu.
    """
    # 1. Test için geçici bir Python dosyası oluştur
    test_file = tmp_path / "test_script.py"
    test_file.write_text("def my_func():\n    pass")

    # 2. Mock LLM'in bu test için ne döndüreceğini belirle
    mock_response = {
        "suggestions": [
            {
                "line_number": 1,
                "original_code": "def my_func():",
                "suggestion_type": "Style",
                "description": "Add a docstring.",
                "suggested_code": "def my_func():\n    \"\"\"This is a docstring.\"\"\""
            }
        ]
    }
    mock_llm_ask.return_value = json.dumps(mock_response)

    # 3. Aracı çalıştır
    args = {"file_path": str(test_file)}
    result = code_auditor.run(args)

    # 4. Sonuçları doğrula
    assert result["status"] == "success"
    assert "Found 1 improvement suggestion" in result["result"]
    assert len(result["raw_suggestions"]) == 1
    assert result["raw_suggestions"][0]["suggestion_type"] == "Style"

    # LLM'in doğru prompt ile çağrıldığını kontrol et
    mock_llm_ask.assert_called_once()
    call_args, _ = mock_llm_ask.call_args
    assert "def my_func():" in call_args[0] # Prompt'un kodumuzu içerdiğini doğrula

def test_code_auditor_file_not_found():
    """Dosya bulunamadığında aracın doğru hata mesajını döndürdüğünü test eder."""
    args = {"file_path": "/non/existent/file.py"}
    result = code_auditor.run(args)

    assert result["status"] == "error"
    assert "File not found" in result["message"]