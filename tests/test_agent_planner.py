# test_agent_planner.py

from unittest.mock import MagicMock
import agent.memory.store as store
import agent.ui.cli as cli

# 🔹 KnowledgeStore’u mockla (DB açılmasın)
store.KnowledgeStore = MagicMock()

# 🔹 Mock memory ve policy sınıfları
from agent.memory.store import MemoryStore as ShortTermMemory
from agent.memory.vectore_memory import VectorMemory
from agent.memory.store import KnowledgeStore as LongTermMemory
from agent.rl.reward import RewardSignal
from agent.policy.tool_policy import ToolPolicy
from agent.policy.prompt_policy import PromptPolicy
from agent.planner.planner import Planner
from agent.ui.cli import AVAILABLE_TOOLS, Colors, run_agent

# 🔹 Mock memory ve policy instance
short_term_memory = ShortTermMemory()
vector_memory = VectorMemory()
long_term_memory = LongTermMemory()
reward_signal = RewardSignal()
tool_policy = ToolPolicy(tools=AVAILABLE_TOOLS)
prompt_policy = PromptPolicy(list(AVAILABLE_TOOLS.keys()))
planner = Planner(AVAILABLE_TOOLS)

# 🔹 Mock ask fonksiyonu
def mock_ask(prompt, max_new_tokens=512):
    return '{"thought":"Planner test düşüncesi","action":"none","input":""} Mock yanıt'

# 🔹 Global olarak cli modülündeki ask fonksiyonunu mockla
cli.ask = mock_ask

# 🔹 Mock tool fonksiyonları
for t in AVAILABLE_TOOLS:
    AVAILABLE_TOOLS[t]["func"] = lambda x=None, **kwargs: {"status":"success","result":f"Mock sonuç for {x}"}

# 🔹 Mock reflect_and_note
def mock_reflect_and_note(user_prompt, final_response, vector_memory, long_term_memory):
    print(f"[Mock note] Kullanıcı: {user_prompt} → Asistan: {final_response}")

cli.reflect_and_note = mock_reflect_and_note

# 🔹 Test çalıştır
if __name__ == "__main__":
    test_prompt = "Web sayfasını özetle ve not al."
    run_agent(test_prompt)
