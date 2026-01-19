# agent/policy/prompt_policy.py

import random
import numpy as np
import json
import os
from typing import List, Dict, Optional, Union
from agent.models.llm import embed


class PromptPolicy:
    """
    Context-aware prompt selection.
    prompts: list of prompt templates (string)
    epsilon: keşif oranı
    """

    def __init__(self, prompts: List[str], epsilon: float = 0.2, data_path: str = 'data/prompt_policy_data.json'):
        self.prompts = prompts
        self.epsilon = epsilon
        self.data_path = data_path
        self._alpha = 0.1  # Öğrenme oranı (learning rate)
        self._gamma = 0.9  # Gelecek ödüller için indirim faktörü (discount factor)
        self._similarity_weight = 1.0  # embedding similarity weight

        # Verileri yükle veya başlat
        self._load_data()
        self._initialize_prompts()

    def _load_data(self):
        """Kayıtlı verileri dosyadan yükler."""
        if os.path.exists(self.data_path):
            with open(self.data_path, 'r') as f:
                data = json.load(f)
                self.q_values = data.get("q_values", {})
                self.counts = data.get("counts", {})
                self.prompt_embeds = {k: np.array(v) if v is not None else None for k, v in data.get("prompt_embeds", {}).items()}
        else:
            self.q_values: Dict[str, float] = {}
            self.counts: Dict[str, int] = {}
            self.prompt_embeds: Dict[str, Optional[np.ndarray]] = {}

    def _save_data(self):
        """Verileri dosyaya kaydeder."""
        data = {
            "q_values": self.q_values,
            "counts": self.counts,
            "prompt_embeds": {k: v.tolist() if v is not None else None for k, v in self.prompt_embeds.items()}
        }
        with open(self.data_path, 'w') as f:
            json.dump(data, f, indent=4)

    def _initialize_prompts(self):
        """Yeni prompt'ları iç belleğe ekler."""
        for prompt in self.prompts:
            if prompt not in self.q_values:
                self.q_values[prompt] = 0.0
                self.counts[prompt] = 0
                self.prompt_embeds[prompt] = None

    def _get_prompt_embed(self, prompt: str):
        if self.prompt_embeds.get(prompt) is None:
            try:
                self.prompt_embeds[prompt] = embed(prompt)
            except Exception:
                self.prompt_embeds[prompt] = None
        return self.prompt_embeds[prompt]

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Kosinüs benzerliği hesaplar, sıfıra bölme hatasını önler."""
        if a is None or b is None:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    def select(self, context: Optional[str] = None) -> str:
        if random.random() < self.epsilon:
            return random.choice(self.prompts)

        # fallback: highest q_value
        if context is None:
            return max(self.q_values, key=self.q_values.get)

        try:
            ctx_emb = embed(context)
        except Exception:
            return max(self.q_values, key=self.q_values.get)

        best = None
        best_score = -float("inf")
        for p in self.prompts:
            p_emb = self._get_prompt_embed(p)
            sim = self._cosine_sim(ctx_emb, p_emb) if p_emb is not None else 0.0
            score = self.q_values.get(p, 0.0) + self._similarity_weight * sim
            if score > best_score:
                best_score = score
                best = p

        # Eğer hiçbir şey seçilemezse, varsayılan olarak en yüksek Q-value'ya sahip olanı döndür
        if best is None:
            return max(self.q_values, key=self.q_values.get)

        return best

    def update(self, prompt: str, reward: float, context: Optional[str] = None):
        """
        Ödüle göre Q-değerini ve prompt gömülmesini günceller.
        Q-değeri güncellemesi için standart Q-learning formülü kullanılır.
        """
        if prompt not in self.q_values:
            self.q_values[prompt] = 0.0
            self.counts[prompt] = 0
            self.prompt_embeds[prompt] = None

        # Q-değeri güncellemesi
        self.counts[prompt] += 1
        old_q = self.q_values[prompt]
        # Standart Q-learning güncellemesi: Q(s,a) <- Q(s,a) + alpha * (reward - Q(s,a))
        new_q = old_q + self._alpha * (reward - old_q)

        # Q-değerinin aşırı negatif olmasını engelleyerek kararlılığı artır.
        self.q_values[prompt] = max(-1.0, new_q)

        if context:
            try:
                ctx_emb = embed(context)
                p_emb = self._get_prompt_embed(prompt)
                if p_emb is None:
                    self.prompt_embeds[prompt] = ctx_emb
                else:
                    # Gömülmeyi bağlam yönünde yavaşça güncelle
                    self.prompt_embeds[prompt] = (1 - 0.05) * p_emb + 0.05 * ctx_emb
            except Exception:
                pass

        self._save_data() # Her güncellemede veriyi kaydet