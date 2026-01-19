# tests/test_planner.py

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent.planner.tools_registry import tools
from agent.planner.planner import Planner

# Planner başlat
planner = Planner(tools)

# Test amaçlı goal
goal = "Webden Albert Einstein hakkında bilgi öğren ve persona güncelle"

# Çalıştır
result = planner.plan_and_execute(goal, tools)

# Çıktıyı göster
print("=== Planner Output ===")
print(result)
