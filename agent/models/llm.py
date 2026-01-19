# agent/models/llm.py

import os
import sys
import requests
import numpy as np
from tqdm import tqdm
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from agent import config
from agent.config import Colors
import threading
import multiprocessing
from huggingface_hub import hf_hub_download


LLM_GPU = None  # Ana model (GPU üzerinde çalışacak)
LLM_CPU = None  # Hızlı model (CPU üzerinde çalışacak)


# Paralel thread'lerin aynı anda GPU'ya yüklenmesini önler, CUDA hatalarını engeller.
gpu_lock = threading.Lock()

def _get_cuda_supported_gpu_indices() -> list[int]:
    """Sistemdeki CUDA destekli NVIDIA GPU'ların indekslerini döndürür."""
    try:
        import torch
        if not torch.cuda.is_available():
            print(f"{Colors.WARNING}Uyarı: PyTorch için CUDA mevcut değil. GPU kullanımı devre dışı.{Colors.ENDC}")
            return []

        device_count = torch.cuda.device_count()
        cuda_devices = []
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            # Intel, AMD gibi entegre veya harici CUDA olmayan GPU'ları filtrele
            if "nvidia" in device_name.lower():
                cuda_devices.append(i)
                print(f"{Colors.OKGREEN}✅ CUDA Destekli GPU bulundu: [{i}] {device_name}{Colors.ENDC}")
            else:
                print(f"{Colors.WARNING}⚠️ Uyumsuz GPU bulundu ve atlanıyor: [{i}] {device_name}{Colors.ENDC}")
        return cuda_devices
    except ImportError:
        print(f"{Colors.WARNING}Uyarı: PyTorch kurulu değil. GPU tespiti yapılamıyor.{Colors.ENDC}")
        return []

_embed_model = None
_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_EMBED_DIM = 384

def _download_single_model(repo_id: str, filename: str, model_path: str):
    """
    Belirtilen modeli huggingface_hub kullanarak indirir.
    Kimlik doğrulaması gerektiren (gated) modelleri destekler.
    """
    model_dir = os.path.dirname(model_path)
    os.makedirs(model_dir, exist_ok=True)

    # Eğer dosya zaten varsa, indirmeyi atla.
    if os.path.exists(model_path):
        print(f"✅ Model zaten mevcut: {os.path.basename(model_path)}.")
        return

    print(f"Model indiriliyor: {filename} (repo: {repo_id})")
    try:
        # hf_hub_download, kaydedilmiş token'ı otomatik olarak kullanır.
        # Dosyayı doğrudan projemizdeki 'models' klasörüne indirir.
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=model_dir,
            local_dir_use_symlinks=False,  # Dosyayı kopyala, sembolik link oluşturma
            resume_download=True,
        )
        print(f"✅ Model başarıyla indirildi: {model_path}")

    except Exception as e:
        error_message = str(e)
        if "401" in error_message or "Gated" in error_message:
            print(f"\n❌ Model indirilemedi: {filename}. Hata: Yetkilendirme Hatası (401).")
            print("Lütfen Hugging Face web sitesinden model lisansını kabul ettiğinizden ve 'huggingface-cli login' ile giriş yaptığınızdan emin olun.")
        else:
            print(f"\n❌ Model indirilemedi: {filename}. Hata: {e}")

        # İndirme başarısız olduysa kısmi dosyayı temizle
        if os.path.exists(model_path):
            os.remove(model_path)
        raise RuntimeError(f"Model indirilemedi: {filename}") from e

def download_model():
    """Yapılandırmada belirtilen her iki modeli de indirir."""
    # Ana GPU modelini indir
    gpu_model_path = os.path.join(config.PROJECT_ROOT, "models", config.MODEL_FILENAME)
    _download_single_model(config.MODEL_REPO_ID, config.MODEL_FILENAME, gpu_model_path)

    # Hızlı CPU modelini indir
    cpu_model_path = os.path.join(config.PROJECT_ROOT, "models", config.CPU_MODEL_FILENAME)
    _download_single_model(config.CPU_MODEL_REPO_ID, config.CPU_MODEL_FILENAME, cpu_model_path)

def load_model():
    """Hem GPU hem de CPU modellerini belleğe yükler."""
    global LLM_GPU, LLM_CPU

    default_threads = max(1, multiprocessing.cpu_count() // 2)

    # GPU modelini yükle
    gpu_model_path = os.path.join(config.PROJECT_ROOT, "models", config.MODEL_FILENAME)
    if os.path.exists(gpu_model_path):
        if LLM_GPU is None:
            print("Ana model (GPU) yükleniyor...")
            cuda_devices = _get_cuda_supported_gpu_indices()

            # Çoklu GPU veya tek GPU için parametreleri ayarla
            llama_params = {
                "model_path": gpu_model_path,
                "n_ctx": config.GPU_N_CTX,
                "n_batch": config.GPU_N_BATCH,
                "n_threads": getattr(config, 'GPU_N_THREADS', default_threads),
                "verbose": False, # Gereksiz başlangıç loglarını kapatır.
                "f16_kv": True    # VRAM kullanımını azaltır ve hızı artırır.
            }

            # Sadece birden fazla CUDA destekli GPU varsa modeli böl
            if hasattr(config, 'GPU_SPLIT_WEIGHTS') and config.GPU_SPLIT_WEIGHTS and len(cuda_devices) > 1:
                print(f"{Colors.OKCYAN}✅ Çoklu GPU modu aktif. Model {len(cuda_devices)} GPU arasında bölünüyor...{Colors.ENDC}")
                # llama-cpp-python, tensor_split'i otomatik olarak mevcut GPU'lara dağıtır.
                # Belirli GPU'ları hedeflemek gerekmez, sadece oranları vermek yeterlidir.
                llama_params["tensor_split"] = config.GPU_SPLIT_WEIGHTS[:len(cuda_devices)] # Sadece mevcut GPU sayısı kadar oran kullan
            elif cuda_devices:
                # Tek bir CUDA GPU'su varsa, tüm katmanları ona yükle
                print(f"{Colors.OKCYAN}✅ Tek GPU modu aktif. Model GPU {cuda_devices[0]}'a yükleniyor...{Colors.ENDC}")
                llama_params["n_gpu_layers"] = -1 # -1 tüm katmanları yükle demektir
            else:
                # Hiç CUDA GPU'su yoksa, CPU'da çalıştır
                print(f"{Colors.WARNING}⚠️ CUDA destekli GPU bulunamadı. Model CPU üzerinde çalışacak.{Colors.ENDC}")
                llama_params["n_gpu_layers"] = 0

            LLM_GPU = Llama(**llama_params)
            print(f"✅ Ana model başarıyla yüklendi.")
        else:
            print("ℹ️ Ana model (GPU) zaten yüklü.")
    else:
        print("❌ Ana model (GPU) dosyası bulunamadı. Lütfen önce indirin.")

    # CPU modelini yükle
    cpu_model_path = os.path.join(config.PROJECT_ROOT, "models", config.CPU_MODEL_FILENAME)
    if os.path.exists(cpu_model_path):
        if LLM_CPU is None:
            print("Hızlı model (CPU) yükleniyor...")
            LLM_CPU = Llama(
                model_path=cpu_model_path,
                n_ctx=config.CPU_N_CTX,
                n_gpu_layers=0,
                n_batch=config.CPU_N_BATCH, # Config'den gelen değeri kullan
                n_threads=getattr(config, 'CPU_N_THREADS', default_threads),
                verbose=False
            )
            print("✅ Hızlı model (CPU) başarıyla yüklendi.")
        else:
            print("ℹ️ Hızlı model (CPU) zaten yüklü.")
    else:
        print("❌ Hızlı model (CPU) dosyası bulunamadı. Lütfen önce indirin.")

def ask(prompt: str, max_new_tokens: int = 256) -> str:
    """Ana GPU modeline bir prompt gönderir ve yanıt alır."""
    global LLM_GPU
    if LLM_GPU is None:
        raise RuntimeError("Ana model (GPU) yüklenmedi. load_model() çağırın!")

    with gpu_lock:
        try:
            # ÖNEMLİ: 'reset=True' eklemek, her çağrının
            # taze bir 'n_batch' ayarıyla başlamasını sağlar ve hızı korur.
            # LLM_GPU.reset()

            response = LLM_GPU.create_completion(
                prompt,
                max_tokens=max_new_tokens,
                temperature=config.GPU_TEMPERATURE,
                top_p=config.GPU_TOP_P,
                repeat_penalty=config.GPU_REPEAT_PENALTY,
            )
            text = response['choices'][0]['text'].strip()
            return text
        except Exception as e:
            return f"[HATA] Ana model (GPU) yanıt veremedi: {e}"

def ask_fast_cpu(prompt: str, max_new_tokens: int = 128) -> str:
    """Hızlı CPU modeline bir prompt gönderir ve yanıt alır."""
    global LLM_CPU
    if LLM_CPU is None:
        raise RuntimeError("Hızlı model (CPU) yüklenmedi. load_model() çağırın!")

    try:
        # print(">>> Hızlı LLM'e (CPU) soruluyor...")
        response = LLM_CPU.create_completion(
            prompt,
            max_tokens=max_new_tokens,
            temperature=config.CPU_TEMPERATURE,
            top_p=config.CPU_TOP_P,
            repeat_penalty=config.GPU_REPEAT_PENALTY,
        )
        # print("<<< Hızlı LLM'den (CPU) yanıt alındı.")
        text = response['choices'][0]['text'].strip()
        return text
    except Exception as e:
        return f"[HATA] Hızlı model (CPU) yanıt veremedi: {e}"



def _load_embed_model(model_name: str = _EMBED_MODEL_NAME):
    global _embed_model
    if _embed_model is None:
        try:
            print("🔹 Embedding modeli yükleniyor...")
            _embed_model = SentenceTransformer(model_name)
            print("✅ Embedding modeli yüklendi.")
        except Exception as e:
            print(f"[embed yükleme hatası] {e}")
            _embed_model = None

def embed(text: str):
    """
    Text'i embedding'e çevirir.
    """
    global _embed_model
    if _embed_model is None:
        _load_embed_model()

    if _embed_model is None:
        raise RuntimeError(
            "Embedding modeli yüklenemedi. "
            "Lütfen 'pip install sentence-transformers' komutunu çalıştırdığından "
            "ve internet bağlantından emin ol."
        )

    vec = _embed_model.encode([text], convert_to_numpy=True)[0]
    return vec.astype("float32")
