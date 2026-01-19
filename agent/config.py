# agent/config.py

import os

# Hugging Face cache dizini.
HF_HOME = "D:/huggingface"

# ANA MODEL (GPU)
#MODEL_REPO_ID = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
#MODEL_FILENAME = "mistral-7b-instruct-v0.2.Q4_0.gguf"
MODEL_REPO_ID = "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF"
MODEL_FILENAME = "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"

# HIZLI MODEL (CPU)
CPU_MODEL_REPO_ID = "QuantFactory/Phi-3-mini-4k-instruct-GGUF"
CPU_MODEL_FILENAME = "Phi-3-mini-4k-instruct.Q4_K_S.gguf"
# Model Parametreleri
# GPU'ya yüklenecek katman sayısı.
# -1: Tüm katmanları yükle (yeterli VRAM varsa en hızlısı).
#  0: Sadece CPU kullan.
# Pozitif bir sayı (örn: 18): Belirtilen sayıda katmanı GPU'ya yükle.
N_GPU_LAYERS = -1 # 4GB VRAM için ayarlandı

# ÇOKLU GPU AYARI
# Modeli birden fazla GPU'ya bölmek için kullanılır.
# Eğer bu ayar kullanılırsa, N_GPU_LAYERS ayarı göz ardı edilir.
# Örn: [0.5, 0.5] -> Katmanları iki GPU arasında eşit böl.
GPU_SPLIT_WEIGHTS = [0.5, 0.5]

# GPU Modeli için parametreler
GPU_N_CTX = 4096  # Context penceresi boyutu (4GB VRAM için ayarlandı)
GPU_N_BATCH = 512 # Paralel işleme için batch boyutu (4GB VRAM için ayarlandı)

# CPU Modeli için parametreler
CPU_N_CTX = 2048
CPU_N_BATCH = 512



GPU_N_THREADS = 8
CPU_N_THREADS = 8




# Paralel İşleme Parametreleri
RESEARCHER_MAX_WORKERS = 5 # critical_web_researcher için ana worker sayısı
SUMMARY_MAX_WORKERS = 4    # /ozetle komutu için worker sayısı

# Üretkenlik Parametreleri (GPU)
GPU_TEMPERATURE = 0.2
GPU_TOP_P = 0.9
GPU_REPEAT_PENALTY = 1.1

# Üretkenlik Parametreleri (CPU)
# Daha hızlı ve küçük model için biraz daha fazla yaratıcılığa izin verilebilir.
CPU_TEMPERATURE = 0.3
CPU_TOP_P = 0.9
CPU_REPEAT_PENALTY = 1.1


# Proje ana dizinini belirle
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Veritabanı ve diğer veri yolları
MEMORY_DB_PATH = os.path.join(PROJECT_ROOT, "data", "memory.sqlite")
PERSONA_DB_PATH = os.path.join(PROJECT_ROOT, "data", "persona.sqlite")
KG_DB_PATH = os.path.join(PROJECT_ROOT, "data", "knowledge_graph.sqlite")
PERSONAL_STORE_PATH = os.path.join(PROJECT_ROOT, "data", "personal_store")
KNOWLEDGE_STORE_PATH = os.path.join(PROJECT_ROOT, "data", "knowledge_store")

# Diğer yapılandırma ayarları buraya eklenebilir
# Örn: DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# CLI Renkleri
class Colors:
    HEADER = "[95m"
    OKBLUE = "[94m"
    OKCYAN = "[96m"
    OKGREEN = "[92m"
    WARNING = "[93m"
    FAIL = "[91m"
    ENDC = "[0m"
    BOLD = "[1m"
