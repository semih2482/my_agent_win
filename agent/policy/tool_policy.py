# agent/policy/tool_policy.py

import random
import numpy as np
import json
import os
from typing import List, Dict, Optional, Union
from agent.models.llm import embed

class ToolPolicy:
    """
    RL / contextual bandit tabanlı araç seçimi.
    - tools: dict veya liste (tool isimleri)
    - epsilon: keşif oranı
    - internal: q_values (ödül tahmini), counts ve tool embedding ortalamaları
    """

    def __init__(self, tools: Union[List[str], Dict[str, object]], epsilon: float = 0.2, data_path: str = 'data/tool_policy_data.json'):
        if isinstance(tools, dict):
            self.tools = list(tools.keys())
        else:
            self.tools = list(tools)

        self.epsilon = epsilon
        self.data_path = data_path
        self._alpha = 0.2  # running average factor for embeddings
        self._beta = 1.0   # similarity weight when scoring

        # Verileri yükle veya başlat
        self._load_data()
        self._initialize_tools()

    def _load_data(self):
        """Kayıtlı verileri dosyadan yükler."""
        if os.path.exists(self.data_path):
            with open(self.data_path, 'r') as f:
                data = json.load(f)
                self.q_values = data.get("q_values", {})
                self.counts = data.get("counts", {})
                self.tool_embeds = {k: np.array(v) if v is not None else None for k, v in data.get("tool_embeds", {}).items()}
        else:
            self.q_values: Dict[str, float] = {}
            self.counts: Dict[str, int] = {}
            self.tool_embeds: Dict[str, Optional[np.ndarray]] = {}

    def _save_data(self):
        """Verileri dosyaya kaydeder."""
        data = {
            "q_values": self.q_values,
            "counts": self.counts,
            "tool_embeds": {k: v.tolist() if v is not None else None for k, v in self.tool_embeds.items()}
        }
        with open(self.data_path, 'w') as f:
            json.dump(data, f, indent=4)

    def _initialize_tools(self):
        """Yeni araçları iç belleğe ekler."""
        for tool in self.tools:
            if tool not in self.q_values:
                self.q_values[tool] = 0.0
                self.counts[tool] = 0
                self.tool_embeds[tool] = None

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Kosinüs benzerliği hesaplar."""
        if a is None or b is None:
            return 0.0
        # Paydadaki 1e-10, sıfıra bölme hatasını önler.
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    def select(self, context: Optional[Union[str, np.ndarray]] = None, return_score: bool = False):
        """
        Araç seçer. Context string veya direkt embedding vektörü olabilir.
        `return_score=True` ise (tool, score) tuple'ı döndürür.
        """
        # Keşif (epsilon) tetiklendiğinde rastgele seçim
        if random.random() < self.epsilon:
            choice = random.choice(self.tools)
            return (choice, self.q_values.get(choice, 0.0)) if return_score else choice

        # Eğer context yoksa veya hiç tool embed yoksa saf q_values seçimi
        if context is None or all(v is None for v in self.tool_embeds.values()):
            # En yüksek Q-value'ya sahip aracı seç
            best_tool = max(self.q_values, key=self.q_values.get)
            return (best_tool, self.q_values.get(best_tool, 0.0)) if return_score else best_tool

        # Eğer context string gelmişse embed et, yoksa direkt kullan
        ctx_emb = context if isinstance(context, np.ndarray) else embed(context)

        best_score = -float("inf")
        best_tool = None
        for tool in self.tools:
            # Ödül değeri ile bağlamsal benzerlik skorunu birleştir
            sim = self._cosine_sim(ctx_emb, self.tool_embeds.get(tool))
            score = self.q_values.get(tool, 0.0) + self._beta * sim

            if score > best_score:
                best_score = score
                best_tool = tool

        return (best_tool, best_score) if return_score else best_tool

    def update(self, tool: str, reward: float, context: Optional[Union[str, np.ndarray]] = None):
        """
        Ödüle göre Q-value ve context gömülmesini günceller.
        """
        if tool not in self.q_values:
            self.q_values[tool] = 0.0
            self.counts[tool] = 0
            self.tool_embeds[tool] = None

        # Q-value güncelle (running average)
        self.counts[tool] += 1
        n = self.counts[tool]
        old_q = self.q_values[tool]
        self.q_values[tool] = old_q + (reward - old_q) / n

        # Context embedding ortalamasını güncelle
        if context is not None:
            try:
                ctx_emb = context if isinstance(context, np.ndarray) else embed(context)
                if self.tool_embeds[tool] is None:
                    self.tool_embeds[tool] = ctx_emb
                else:
                    self.tool_embeds[tool] = (1 - self._alpha) * self.tool_embeds[tool] + self._alpha * ctx_emb
            except Exception:
                # embed hatası
                pass

        self._save_data() # Her güncellemede veriyi kaydet