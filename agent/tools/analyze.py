# agent/tools/analyze.py
import json
import re
import io
import csv
import statistics
from typing import Dict, Any, Callable
from agent.models.llm import ask # LLM fonksiyonunu import et
from agent.tools.web_reader import read_url # Tutarlılık için web_reader'ı kullan

# Helper function for robust JSON parsing from LLM output
def _parse_llm_json_output(response: str) -> Dict[str, Any]:
    """
    Extracts and parses a JSON object from the LLM's text response.
    Handles markdown blocks and raw JSON text.
    """
    json_str = None
    # 1. Strateji: Markdown JSON kod bloğunu ara
    match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        # 2. Strateji: Metin içindeki ilk ve son küme parantezi arasını bul
        obj_match = re.search(r"\{.*\}", response, re.DOTALL)
        if obj_match:
            json_str = obj_match.group(0)

    if not json_str:
        return {"status": "error", "message": "No valid JSON object found in the response.", "raw_response": response}

    try:
        # Yüklemeden önce string'i temizle (örn: kaçış karakterleri)
        cleaned_json_str = json.loads(f'"{json_str}"')[1:-1] if json_str.startswith('"') and json_str.endswith('"') else json_str
        return {"status": "success", "data": json.loads(cleaned_json_str)}
    except json.JSONDecodeError as e:
        # Ham string ile son bir deneme yap
        try:
            return {"status": "success", "data": json.loads(json_str)}
        except json.JSONDecodeError:
            return {"status": "error", "message": f"Failed to decode JSON: {e}", "raw_response": json_str}

# 1. Deeper Text Analysis
def _analyze_text(content: str, topic: str = "general", llm_ask_function: Callable = ask) -> Dict[str, Any]:
    """Analyzes a piece of text on a specific topic."""
    prompt = (
        f"Aşağıdaki metni '{topic}' odağında Türkçe olarak analiz et. "
        "Cevabın, başka hiçbir metin veya açıklama olmadan, SADECE aşağıdaki şemaya uygun bir JSON nesnesi olmalıdır:\n"
        "{\n"
        '  "summary": "Konuyla ilgili kısa özet (2-3 cümle)",\n'
        '  "scores": {"relevance": 0-10, "clarity": 0-10, "depth": 0-10},\n'
        '  "strengths": ["Metnin konuyla ilgili güçlü yönleri"],\n'
        '  "weaknesses": ["Metnin konuyla ilgili zayıf yönleri"],\n'
        '  "suggestions": ["Geliştirme için öneriler"]\n'
        "}\n"
    )
    response = llm_ask_function(prompt + f"\n\nMetin:\n{content}")
    return _parse_llm_json_output(response)

# Main analyze function (router)
def analyze(content: str, analysis_type: str = "text", topic: str = "general", llm_ask_function: Callable = ask) -> Dict[str, Any]:
    """
    Analyzes content based on the specified analysis type.
    Acts as a router to different specialized analysis functions.
    """
    if analysis_type == "text":
        return _analyze_text(content, topic=topic, llm_ask_function=llm_ask_function)
    # elif analysis_type == "financial":
    #     return _analyze_financial_data(content, llm_ask_function=llm_ask_function)
    else:
        return {"status": "error", "message": f"Analysis type '{analysis_type}' not supported."}