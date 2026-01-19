# agent/tools/fund_analyst.py
from typing import Dict, Any
from agent.tools.comprehensive_financial_analyst import run as analyze_investment_query

def run(fund_code: str, investment_horizon: str = "belirtilmedi", risk_profile: str = "belirtilmedi", agent_instance=None) -> Dict[str, Any]:
    """
    DEPRECATED: Bu araç artık doğrudan kullanılmamaktadır.
    Tüm analiz istekleri yeni `comprehensive_financial_analyst` aracına yönlendiriliyor.
    """
    print("--- UYARI: `get_fund_advice` aracı eskidir ve `analyze_investment_query` aracına yönlendiriliyor. ---")

    # Sorguyu yeniden oluşturarak orijinal niyeti koru
    query = f"{fund_code} fonu için analiz"

    return analyze_investment_query(query, investment_horizon, risk_profile, agent_instance=agent_instance)