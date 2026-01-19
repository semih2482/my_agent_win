# agent/tools/community_tools/portfolio_manager.py
import json
import os
from typing import Dict, Any, List

# Proje kök dizinini al
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
PORTFOLIO_FILE = os.path.join(project_root, 'data', 'portfolio.json')

TOOL_INFO = {
    "name": "portfolio_manager",
    "description": "Kullanıcının yatırım portföyünü yönetir. Varlık ekleme, çıkarma, görüntüleme ve yeniden dengeleme önerileri sunar. Girdi: {'action': 'add'|'remove'|'view'|'rebalance', 'payload': {...}}",
    "cacheable": False,
    "args_schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["add", "remove", "view", "rebalance"]},
            "payload": {"type": "object"}
        },
        "required": ["action"]
    }
}

def _load_portfolio() -> List[Dict[str, Any]]:
    """Portföyü JSON dosyasından yükler."""
    if not os.path.exists(PORTFOLIO_FILE):
        return []
    try:
        with open(PORTFOLIO_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []

def _save_portfolio(portfolio: List[Dict[str, Any]]):
    """Portföyü JSON dosyasına kaydeder."""
    os.makedirs(os.path.dirname(PORTFOLIO_FILE), exist_ok=True)
    with open(PORTFOLIO_FILE, 'w', encoding='utf-8') as f:
        json.dump(portfolio, f, indent=2, ensure_ascii=False)

def add_asset(payload: dict) -> Dict[str, Any]:
    """Portföye yeni bir varlık ekler veya mevcut varlığın miktarını günceller."""
    symbol = payload.get("symbol")
    quantity = float(payload.get("quantity", 0))
    purchase_price = float(payload.get("price", 0))

    if not symbol or quantity <= 0:
        return {"status": "error", "message": "Geçerli bir sembol ve miktar belirtilmelidir."}

    portfolio = _load_portfolio()
    found = False
    for asset in portfolio:
        if asset.get("symbol") == symbol:
            # Varlık zaten var, miktar ve ortalama maliyeti güncelle
            old_total_cost = asset.get("quantity", 0) * asset.get("avg_cost", 0)
            new_total_cost = quantity * purchase_price
            total_quantity = asset.get("quantity", 0) + quantity
            asset["quantity"] = total_quantity
            asset["avg_cost"] = (old_total_cost + new_total_cost) / total_quantity
            found = True
            break

    if not found:
        portfolio.append({
            "symbol": symbol,
            "quantity": quantity,
            "avg_cost": purchase_price
        })

    _save_portfolio(portfolio)
    return {"status": "success", "result": f"{quantity} adet {symbol} portföye eklendi."}

def remove_asset(payload: dict) -> Dict[str, Any]:
    """Portföyden bir varlığı çıkarır."""
    symbol = payload.get("symbol")
    quantity_to_remove = float(payload.get("quantity", 0))

    if not symbol or quantity_to_remove <= 0:
        return {"status": "error", "message": "Geçerli bir sembol ve miktar belirtilmelidir."}

    portfolio = _load_portfolio()
    asset_found = False
    for i, asset in enumerate(portfolio):
        if asset.get("symbol") == symbol:
            asset_found = True
            if asset["quantity"] > quantity_to_remove:
                asset["quantity"] -= quantity_to_remove
                _save_portfolio(portfolio)
                return {"status": "success", "result": f"{quantity_to_remove} adet {symbol} portföyden çıkarıldı."}
            else:
                # Eğer satılan miktar mevcut miktardan fazla veya eşitse, varlığı tamamen kaldır
                del portfolio[i]
                _save_portfolio(portfolio)
                return {"status": "success", "result": f"{symbol} varlığı portföyden tamamen kaldırıldı."}

    if not asset_found:
        return {"status": "error", "message": f"Portföyde '{symbol}' bulunamadı."}

def view_portfolio(payload: dict = None) -> Dict[str, Any]:
    """Mevcut portföyü görüntüler."""
    portfolio = _load_portfolio()
    if not portfolio:
        return {"status": "success", "result": "Portföyünüz şu anda boş."}

    summary = "Mevcut Portföy:\n"
    for asset in portfolio:
        summary += f"- Sembol: {asset['symbol']}, Miktar: {asset['quantity']}, Ortalama Maliyet: {asset.get('avg_cost', 'N/A'):.4f}\n"
    return {"status": "success", "result": summary}

def rebalance_portfolio(payload: dict, agent_instance=None) -> Dict[str, Any]:
    """
    Portföyü analiz eder ve yeniden dengeleme önerileri sunar.
    Bu, sistemin en akıllı parçasıdır.
    """
    if not agent_instance:
        return {"status": "error", "message": "Analiz için agent örneği gerekli."}

    portfolio = _load_portfolio()
    if not portfolio:
        return {"status": "success", "result": "Portföyünüz boş olduğu için yeniden dengeleme yapılamıyor."}

    print("⚖️ Portföy yeniden dengeleme analizi başlatılıyor...")
    analysis_results = []
    # Her varlığı analiz etmek için comprehensive_financial_analyst aracını kullan
    analyzer_tool = agent_instance.available_tools.get('comprehensive_financial_analyst', {}).get('func')
    if not analyzer_tool:
        return {"status": "error", "message": "Kapsamlı analiz aracı bulunamadı."}

    for asset in portfolio:
        symbol = asset['symbol']
        print(f"   -> Mevcut varlık analiz ediliyor: {symbol}")
        # Analiz aracına sorguyu gönder
        analysis = analyzer_tool(query=f"{symbol} için detaylı analiz", agent_instance=agent_instance)
        analysis_results.append({"symbol": symbol, "analysis": analysis.get("result", "Analiz başarısız.")})

    # Tüm analiz sonuçlarını birleştirip LLM'e nihai öneriyi oluşturması için sor
    synthesis_prompt = f"""
    Sen bir portföy yöneticisisin. Mevcut portföydeki varlıklar için aşağıdaki analizler yapıldı.
    Bu analizleri ve özellikle fiyat tahminlerini dikkate alarak, portföyü yeniden dengelemek için bir strateji öner.
    Önerilerin net olmalı: "Şu kadar X sat, yerine Y al" gibi.

    Mevcut Portföy: {json.dumps(portfolio, indent=2, ensure_ascii=False)}

    Varlık Analizleri:
    ---
    {json.dumps(analysis_results, indent=2, ensure_ascii=False)}
    ---

    Yeniden Dengeleme Önerisi:
    """
    final_recommendation = agent_instance.ask(synthesis_prompt, max_new_tokens=1024)
    return {"status": "success", "result": final_recommendation}


def run(args: Dict[str, Any], agent_instance=None) -> Dict[str, Any]:
    """Ana yönlendirici fonksiyon."""
    action = args.get("action")
    payload = args.get("payload", {})

    if action == "add":
        return add_asset(payload)
    elif action == "remove":
        return remove_asset(payload)
    elif action == "view":
        return view_portfolio(payload)
    elif action == "rebalance":
        return rebalance_portfolio(payload, agent_instance)
    else:
        return {"status": "error", "message": f"Bilinmeyen eylem: {action}"}