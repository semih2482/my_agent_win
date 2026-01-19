#tool_registry.py
import os
import importlib.util
from agent.tools import code_editor

from agent.tools.web_reader import read_url, summarize_text
from agent.tools.internet_search import search_urls, search_for_snippets
from agent.tools.learn_from_web import learn_from_web
from agent.tools.analyze import analyze
from agent.tools.entity_extractor import extract_entities
from agent.tools.knowledge_graph import build_knowledge_graph
from agent.tools.sentiment_analyzer import analyze_sentiment as sentiment_analysis
from agent.tools.persona_manager import PersonaManager
from agent.tools.code_completion import complete_code
from agent.tools.chat import chat_function, ask_for_clarification
from agent.models.llm import ask
from agent.tools.stock_data_fetcher import fetch_stock_data
from agent.tools.fund_data_fetcher import fetch_fund_data
from agent.tools.financial_sentiment import analyze_financial_sentiment
from agent.tools.investment_advisor import get_investment_advice
from agent.tools.fund_analyst import get_fund_advice
from agent.tools.multimodal_tools import analyze_image, analyze_video, analyze_pdf, analyze_text, analyze_file
from agent.tools.comprehensive_financial_analyst import analyze_investment_query



# Örnek persona manager ve knowledge store
persona_manager = PersonaManager()
# knowledge_store = KnowledgeStore()  # Eğer varsa

tools = {
    "search_urls": search_urls,
    "search_for_snippets": search_for_snippets,
    "read_url": read_url,
    "summarize_text": lambda x: summarize_text(x, llm_ask_function=ask),
    "learn_from_web": lambda x: learn_from_web(x, knowledge_store=None, llm_ask_function=ask),
    "analyze": lambda x: analyze(x, llm_ask_function=ask),
    "extract_entities": extract_entities,
    "knowledge_graph": build_knowledge_graph,
    "sentiment": lambda x: sentiment_analysis(x, llm_ask_function=ask),
    "extract_traits": lambda x: persona_manager.extract_and_add_from_message(x),
    "add_trait": lambda x: persona_manager.add_trait(x),
    "complete_code": lambda x: complete_code(x, language="python", llm_ask_function=ask),
    "chat": chat_function,
    "ask_for_clarification": ask_for_clarification,
    "read_file": code_editor.read_file,
    "rewrite_file": code_editor.rewrite_file,
    "apply_patch": code_editor.apply_patch,
    "append_code": code_editor.append_code,
    "fetch_stock_data": fetch_stock_data,
    "fetch_fund_data": fetch_fund_data,
    "analyze_financial_sentiment": analyze_financial_sentiment,
    "analyze_investment_query": analyze_investment_query,

    "analyze_image": analyze_image,
    "analyze_video": analyze_video,
    "analyze_pdf": analyze_pdf,
    "analyze_text": analyze_text,
    "analyze_file": analyze_file,
}

def load_community_tools():
    community_tools_dir = os.path.join(os.path.dirname(__file__), '..', 'tools', 'community_tools')
    for filename in os.listdir(community_tools_dir):
        if filename.endswith('.py') and not filename.startswith('__'):
            module_name = f"agent.tools.community_tools.{filename[:-3]}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, os.path.join(community_tools_dir, filename))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Dosya adıyla aynı ada sahip bir fonksiyon arayın
                function_name = filename[:-3]
                if hasattr(module, function_name):
                    tools[function_name] = getattr(module, function_name)
                    print(f"[Tool Loaded]: '{function_name}' from community_tools loaded successfully.")
                # Alternatif olarak, `run` veya `execute` gibi standart bir fonksiyon adı arayabilirsiniz
                elif hasattr(module, 'run'):
                    tools[function_name] = getattr(module, 'run')
                    print(f"[Tool Loaded]: '{function_name}' (as run) from community_tools loaded successfully.")
                elif hasattr(module, 'execute'):
                    tools[function_name] = getattr(module, 'execute')
                    print(f"[Tool Loaded]: '{function_name}' (as execute) from community_tools loaded successfully.")

            except Exception as e:
                print(f"Failed to load tool from {filename}: {e}")

load_community_tools()
