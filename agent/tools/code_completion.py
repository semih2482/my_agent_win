# agent/tools/code_completion.py
from agent.models.llm import ask

def complete_code(code_snippet: str, language="python", llm_ask_function=None):
    llm_ask_function = llm_ask_function or ask
    prompt = f"Bu {language} kod parçasını tamamla:\n\n{code_snippet}\n\nTamamlanmış kod:"
    return llm_ask_function(prompt, max_new_tokens=512)
