# agent/tools/chat.py
from agent.models.llm import ask
from typing import Dict, Any

TOOL_INFO = {
    "name": "chat",
    "description": "Kullanıcıdan ek bilgi veya açıklama istemek için kullanılır. Bir sonraki adıma geçmeden önce belirsizliği giderir.",
    "cacheable": False,
    "args_schema": {"message": "string"}
}

def run(args: dict, agent_instance=None) -> dict:
    """Wrapper function to ask for clarification."""
    message = args.get('message')
    if not message:
        return {"status": "error", "message": "Missing 'message' in arguments."}
    return ask_for_clarification(message)

def chat_function(prompt: str):
    """Legacy function, not used by the planner directly."""
    return ask(prompt, max_new_tokens=200)

def ask_for_clarification(question: str) -> Dict[str, Any]:
    """
    Kullanıcıya bir soru sorar ve yanıt bekler.
    Bu araç, agent'in eksik veya belirsiz bilgiler için kullanıcıdan açıklama istemesi gerektiğinde kullanılır.
    """
    print(f"❓ Agent'tan Açıklama İsteği: {question}")
    # This status will be handled by the UI loop to get user input.
    # For now, the planner will stop, and the agent will return this to the user.
    return {"status": "clarification_needed", "question": question, "result": f"Kullanıcıdan açıklama bekleniyor: {question}"}