# test_run_agent_mock.py

from agent.models.llm import load_model
load_model()  # modeli hazırla

from unittest.mock import MagicMock
import agent.memory.store as store

store.KnowledgeStore = MagicMock()

from agent.ui.cli import run_agent
from agent.memory.store import MemoryStore as ShortTermMemory
from agent.memory.vectore_memory import VectorMemory
from agent.memory.store import KnowledgeStore as LongTermMemory
from agent.rl.reward import RewardSignal
from agent.policy.tool_policy import ToolPolicy
from agent.policy.prompt_policy import PromptPolicy
from agent.planner.planner import Planner
from agent.ui.cli import AVAILABLE_TOOLS, Colors

# 🔹 Mock memory ve policy
short_term_memory = ShortTermMemory()
vector_memory = VectorMemory()
long_term_memory = LongTermMemory()
reward_signal = RewardSignal()

# PromptPolicy için basit mock prompt listesi
prompt_list = [
    "Kullanıcı ile doğal sohbet",
    "Araç kullanım talimatları"
]
prompt_policy = PromptPolicy(prompts=prompt_list)

tool_policy = ToolPolicy(tools=AVAILABLE_TOOLS)
planner = Planner(AVAILABLE_TOOLS)

# 🔹 Mock ask fonksiyonu
def ask(prompt, max_new_tokens=512):
    # JSON formatında ve "none" action döner
    return '{"thought":"Bu bir mock düşünce","action":"none","input":""} Mock yanıt'

# 🔹 Mock tool fonksiyonları
for t in AVAILABLE_TOOLS:
    AVAILABLE_TOOLS[t]["func"] = lambda x=None, **kwargs: {"status": "success", "result": f"Mock sonuç for {x}"}

# 🔹 Mock reflect_and_note
def reflect_and_note(user_prompt, final_response, vector_memory, long_term_memory):
    print(f"[Mock note] Kullanıcı: {user_prompt} → Asistan: {final_response}")

# 🔹 Test agent
if __name__ == "__main__":
    prompt = "Web sayfası özetle ve bana not al."
    run_agent(prompt)
