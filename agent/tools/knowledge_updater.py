# agent/tools/knowledge_updater.py

from agent import config

TOOL_INFO = {
    "name": "update_knowledge",
    "description": "Updates a piece of knowledge in the agent's memory. It deletes the old information and adds the new, corrected version.",
    "input_schema": {
        "type": "object",
        "properties": {
            "old_knowledge_text": {
                "type": "string",
                "description": "The exact text of the outdated or incorrect knowledge to be deleted."
            },
            "updated_knowledge_text": {
                "type": "string",
                "description": "The new, corrected, and updated version of the knowledge to be added."
            }
        },
        "required": ["old_knowledge_text", "updated_knowledge_text"]
    }
}

def run(old_knowledge_text: str, updated_knowledge_text: str, agent_instance=None) -> dict:
    """
    Updates a piece of knowledge in the agent's memory.
    """
    if not agent_instance or not hasattr(agent_instance, 'knowledge_store'):
        return {"status": "error", "message": "Bilgi güncelleme için gerekli olan agent hafıza modülü bulunamadı."}

    if not old_knowledge_text or not updated_knowledge_text:
        return {"status": "error", "message": "Both old_knowledge_text and updated_knowledge_text must be provided."}

    try:
        knowledge_store = agent_instance.knowledge_store
        # Delete the old knowledge
        deleted_count = knowledge_store.delete_by_content(old_knowledge_text)
        if deleted_count <= 0:
            print(f"Warning: Could not find and delete the old knowledge: '{old_knowledge_text}'")

        # Add the new knowledge
        result = knowledge_store.add(updated_knowledge_text)
        if result["status"] == "success":
            message = f"Knowledge updated successfully. Replaced '{old_knowledge_text}' with '{updated_knowledge_text}'."
            return {"status": "success", "message": message}
        else:
            return {"status": "error", "message": f"Failed to add updated knowledge. Reason: {result['message']}"}

    except Exception as e:
        return {"status": "error", "message": f"An unexpected error occurred during knowledge update: {e}"}
