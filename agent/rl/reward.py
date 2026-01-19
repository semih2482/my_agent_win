# agent/rl/reward.py

from typing import Optional

class RewardSignal:
    """
    Kullanıcı geri bildirimi + metriklerden reward üretir.
    Feedback, latency, tool error, sentiment ve retry sayısını birleştirir.
    """

    def __init__(self):
        # Ödül sinyali ağırlıkları (Önem sırasına göre ayarlanmıştır)
        self.W_FEEDBACK = 2.0  # Kullanıcı geri bildirimi en yüksek öncelik
        self.W_ERROR = 1.5     # Hataları güçlü cezalandır
        self.W_RETRY = 1.0     # Verimsizliği cezalandır
        self.W_LATENCY = 0.5   # Hızı teşvik et
        self.W_SENTIMENT = 0.0 # (Şimdilik devre dışı)
        pass

    def from_feedback(self, feedback: str) -> float:
        feedback = feedback.lower()
        if feedback in ["yes", "👍", "good", "correct"]:
            return 1.0
        elif feedback in ["no", "👎", "bad", "wrong"]:
            return -1.0
        return 0.0

    def from_latency(self, start_time: float, end_time: float) -> float:
        latency = end_time - start_time
        # Yeni LLM ayarlarınla (30-60 saniye sürebildiği için) toleransı biraz artıralım
        if latency < 5.0:  # 5 saniyeden hızlı: Mükemmel
            return 0.5
        elif latency < 15.0: # 15 saniyeden hızlı: Kabul edilebilir
            return 0.0
        return -0.5

    def from_tool_error(self, error: Optional[str]) -> float:
        """Araç hatalarını cezalandır"""
        if not error:
            return 0.0
        return -1.0 # Hata varsa daha büyük ceza ver (önceki -0.5 yerine)

    def from_sentiment(self, text: str) -> float:
        """
        Basit pozitif/negatif kelime tabanlı sentiment reward.
        Şu an için reward toplamına dahil edilmemektedir (W_SENTIMENT = 0.0).
        """
        positive_words = ["iyi", "harika", "teşekkür", "👍", "başarılı"]
        negative_words = ["kötü", "hata", "👎", "başarısız", "çöktü"]
        reward = 0.0
        for w in positive_words:
            if w in text.lower():
                reward += 0.5
        for w in negative_words:
            if w in text.lower():
                reward -= 0.5
        return reward

    def from_retry(self, retries: int, max_retries: int = 3) -> float:
        """
        Retry sayısına göre reward azalt.
        0 retry → +1, max retry → -1 civarı.
        """
        if retries <= 0:
            return 1.0
        penalty = min(retries / max_retries, 1.0) # Penalty'nin 1'i geçmesini engelle
        return 1.0 - 2 * penalty # retry arttıkça reward düşer (max retry = -1.0)

    def total_reward(
        self,
        feedback: str,
        start_time: float,
        end_time: float,
        error: Optional[str] = None,
        user_text: str = "",
        retries: int = 0
    ) -> float:
        """
        Tüm reward sinyallerini ağırlıklandırarak birleştirir ve tek float döndürür.
        """
        r_feedback = self.from_feedback(feedback)
        r_latency = self.from_latency(start_time, end_time)
        r_error = self.from_tool_error(error)
        r_sentiment = self.from_sentiment(user_text) # Hesapla ama ağırlığı 0
        r_retry = self.from_retry(retries)

        # Ağırlıklandırılmış Toplam
        total = (r_feedback * self.W_FEEDBACK) + \
                (r_latency * self.W_LATENCY) + \
                (r_error * self.W_ERROR) + \
                (r_retry * self.W_RETRY) + \
                (r_sentiment * self.W_SENTIMENT) # Ağırlık 0 olduğu için şimdilik etkisiz.


        return total