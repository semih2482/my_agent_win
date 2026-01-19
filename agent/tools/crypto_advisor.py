# agent/tools/crypto_advisor.py
from typing import Dict, Any
from agent.tools.comprehensive_financial_analyst import run as analyze_investment_query

def run(query: str, investment_horizon: str = "belirtilmedi", risk_profile: str = "belirtilmedi", include_macro: bool = False, agent_instance=None) -> dict:
    """
    DEPRECATED: Bu araç artık doğrudan kullanılmamaktadır.
    Tüm analiz istekleri yeni `comprehensive_financial_analyst` aracına yönlendiriliyor.
    """
    print("--- UYARI: `get_crypto_advice` aracı eskidir ve `analyze_investment_query` aracına yönlendiriliyor. ---")

    # include_macro gibi özel bir parametre varsa, bunu sorguya ekleyerek niyetin kaybolmamasını sağlayalım.
    if include_macro:
        query += " (makro analiz dahil)"

    return analyze_investment_query(query, investment_horizon, risk_profile, agent_instance=agent_instance)
