# agent/tools/calculator.py
import ast
import operator as op
import math
from typing import Dict, Any
from agent.models.llm import ask

# izin verilen işlemler
_ALLOWED_BINOPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.FloorDiv: op.floordiv,
}

_ALLOWED_UNARYOPS = {
    ast.UAdd: lambda x: x,
    ast.USub: lambda x: -x,
}

_ALLOWED_FUNCS = {
    'sin': math.sin,
    'cos': math.cos,
    'tan': math.tan,
    'sqrt': math.sqrt,
    'log': math.log,
    'ln': math.log,
    'pow': math.pow,
    'abs': abs,
}

_ALLOWED_CONSTS = {
    'pi': math.pi,
    'e': math.e
}


def _eval_node(node):
    """AST düğümünü güvenli şekilde değerlendirir (rekürsif)."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):  # Python 3.8+
        return node.value

    if isinstance(node, ast.Num):  # legacy
        return node.n

    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        op_type = type(node.op)
        if op_type in _ALLOWED_BINOPS:
            return _ALLOWED_BINOPS[op_type](left, right)
        raise ValueError(f"İzin verilmeyen binary operator: {op_type}")

    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        op_type = type(node.op)
        if op_type in _ALLOWED_UNARYOPS:
            return _ALLOWED_UNARYOPS[op_type](operand)
        raise ValueError(f"İzin verilmeyen unary operator: {op_type}")

    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            fname = node.func.id
            if fname in _ALLOWED_FUNCS:
                args = [_eval_node(a) for a in node.args]
                return _ALLOWED_FUNCS[fname](*args)
        raise ValueError("Yalnızca izin verilen fonksiyon çağrılabilir.")

    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_CONSTS:
            return _ALLOWED_CONSTS[node.id]
        raise ValueError(f"İzin verilmeyen isim: {node.id}")

    raise ValueError(f"İşlem anlaşılamadı: {type(node)}")


def calculate(expression: str, llm_ask_function=None, explain: bool = False) -> Dict[str, Any]:
    """
    Güvenli hesaplama yapar.
    - expression: matematiksel ifade (örn. "sin(pi/2) + 2**3")
    - explain: True ise (ve llm_ask_function verilmişse) LLM'den adım adım açıklama ister.
    Dönen dict:
    {"status": "success", "result": <number>, "explanation": <string, optional>}
    """
    llm_ask_function = llm_ask_function or ask

    try:
        parsed = ast.parse(expression, mode='eval')
        value = _eval_node(parsed)
        # tam sayıysa int döndür
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        out = {"status": "success", "result": value}

        if explain:
            try:
                prompt = (
                    f"Bu matematiksel ifadeyi adım adım açıkla (Türkçe). "
                    f"İfade: {expression}\n"
                    f"Sonuç: {value}\n\n"
                    "Mümkünse her adımı kısa maddeler halinde yaz."
                )
                explanation = llm_ask_function(prompt, max_new_tokens=220)
                out["explanation"] = explanation
            except Exception:
                out["explanation"] = "Açıklama alınamadı."

        return out

    except ZeroDivisionError:
        return {"status": "error", "message": "Bir sayı sıfıra bölünemez."}
    except Exception as e:
        return {"status": "error", "message": f"Hesaplama hatası: {e}"}
