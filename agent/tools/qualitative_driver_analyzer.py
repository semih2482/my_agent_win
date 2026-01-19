# agent/tools/qualitative_driver_analyzer.py
from typing import Dict, Any, List
from agent.models.llm import ask

def analyze_qualitative_drivers(asset_name: str, news_summary: str, historical_prices: List[float], agent_instance=None) -> Dict[str, Any]:
    """
    Uses an LLM to analyze the correlation between news/social media summary and price movements.
    """
    llm_func = agent_instance.ask if agent_instance and hasattr(agent_instance, 'ask') else ask

    # Identify significant price changes
    if len(historical_prices) < 2:
        return {"status": "info", "result": "Not enough historical data to analyze price changes."}

    price_change_percent = ((historical_prices[-1] - historical_prices[0]) / historical_prices[0]) * 100
    price_change_summary = f"The price has changed by {price_change_percent:.2f}% over the period."

    prompt = f"""
    As a financial analyst, your task is to identify the potential drivers behind the price movements of a financial asset based on recent news and social media discussions.

    Asset: {asset_name}

    Summary of recent news and discussions:
    ---
    {news_summary}
    ---

    Recent price movement: {price_change_summary}

    Based on the provided summary, please answer the following:
    1.  What are the main positive factors (bullish drivers) mentioned in the news that could be driving the price up?
    2.  What are the main negative factors (bearish drivers) mentioned that could be driving the price down?
    3.  Provide a concluding synthesis on whether the overall sentiment and news point towards a positive or negative outlook for the asset.

    Present your analysis in a clear, concise, and structured manner.
    """

    try:
        response = llm_func(prompt)
        return {"status": "success", "result": response}
    except Exception as e:
        return {"status": "error", "message": f"Failed to get analysis from LLM: {str(e)}"}
