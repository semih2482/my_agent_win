import os
import pdfplumber
from PIL import Image

# 🔹 Modelleri "Lazy Loading" ile yükle
_summarizer = None
_image_captioner = None

def get_summarizer():
    global _summarizer
    if _summarizer is None:
        from transformers import pipeline
        import torch
        _summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)
    return _summarizer

def get_image_captioner():
    global _image_captioner
    if _image_captioner is None:
        from transformers import pipeline
        import torch
        _image_captioner = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning", device=0 if torch.cuda.is_available() else -1)
    return _image_captioner

_whisper_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        import torch
        # Performans ve doğruluk dengesi için "base" modelini seçiyoruz.
        # Daha yüksek doğruluk için "medium" veya "large" kullanılabilir.
        _whisper_model = whisper.load_model("base")
    return _whisper_model

def describe_image(image: Image.Image) -> str:
    """
    Verilen bir görseli analiz ederek içeriğini metin olarak açıklar.
    Örn: "A brown dog is running on the grass."
    """
    try:
        image_captioner = get_image_captioner()
        # Modeli kullanarak görselden metin üret
        captions = image_captioner(image)
        return captions[0]['generated_text']
    except Exception as e:
        return f"Görsel açıklama hatası: {e}"

def analyze_image(file_path: str) -> dict:
    """Görseli OCR, nesne tanıma ve açıklama ile analiz et"""
    try:
        import pytesseract
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img).strip()
        description = describe_image(img)
        summary = "Metin bulunamadı"
        if text:
            summarizer = get_summarizer()
            summary = summarizer(text, max_length=60, min_length=10, do_sample=False)[0]['summary_text']

        return {
            "status": "success",
            "type": "image",
            "description": description,
            "ocr_text": text,
            "summary": summary,
        }
    except ImportError:
        return {"status": "error", "message": "Pytesseract kütüphanesi kurulu değil. Lütfen 'pip install pytesseract' ile kurun ve Tesseract OCR'ı sisteminize yükleyin."}

def analyze_video(file_path: str, scene_interval: int = 5) -> dict:
    """Videoyu sahne ve ses bazlı analiz et"""
    import moviepy.editor as mp
    import torch
    try:
        video = mp.VideoFileClip(file_path)
        duration = video.duration
        audio_text = "Ses bulunamadı veya analiz edilemedi."
        scene_objects = []

        # Ses analizi (Whisper STT entegrasyonu)
        audio_path = None
        try:
            if video.audio:
                whisper_model = get_whisper_model()
                audio_path = "temp_audio.wav"
                video.audio.write_audiofile(audio_path, verbose=False, logger=None)
                result = whisper_model.transcribe(audio_path, fp16=torch.cuda.is_available())
                audio_text = result["text"]
        except Exception as audio_e:
            audio_text = f"Ses analizi hatası: {audio_e}"
        finally:
            # Geçici ses dosyasını sil
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)

        # Sahne tabanlı nesne tespiti
        for t in range(0, int(duration), scene_interval):
            frame = video.get_frame(t)
            frame_image = Image.fromarray(frame) # Çerçeveyi PIL görseline çevir
            description = describe_image(frame_image)
            scene_objects.append({"time_sec": t, "description": description})

        return {
            "status": "success",
            "type": "video",
            "duration_sec": duration,
            "audio_summary": audio_text,
            "scene_objects": scene_objects
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

_spacy_model = None

def get_spacy_model():
    global _spacy_model
    if _spacy_model is None:
        import spacy
        try:
            # Performans için küçük bir model seçiyoruz.
            _spacy_model = spacy.load("en_core_web_sm")
        except OSError:
            # Modeli bulamazsa, kullanıcıya nasıl yükleyeceği konusunda net bir hata mesajı ver.
            raise ImportError("spaCy 'en_core_web_sm' modeli kurulu değil. Lütfen terminalde 'python -m spacy download en_core_web_sm' komutunu çalıştırarak modeli indirin.")
    return _spacy_model

def analyze_pdf(file_path: str) -> dict:
    """PDF metnini oku, özetle ve kavramları çıkar"""
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""

        summary = "PDF boş veya metin çıkarılamadı"
        if text:
            summary = get_summarizer()(text[:3000], max_length=120, min_length=30, do_sample=False)[0]['summary_text']

        # Kavram Çıkarımı (Entity Extraction)
        entities = []
        try:
            if text:
                nlp = get_spacy_model()
                # spaCy'nin varsayılan karakter limitini (1,000,000) aşmamak için metni kırp
                doc = nlp(text[:999_000])
                entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        except Exception as e:
            entities = [{"error": str(e)}]

        return {
            "status": "success",
            "type": "pdf",
            "summary": summary,
            "char_count": len(text),
            "entities": entities
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def analyze_text(text: str) -> dict:
    """Serbest metni analiz et: özet, duygu, kavramlar"""
    try:
        from textblob import TextBlob
        sentiment_data = TextBlob(text).sentiment
        sentiment = {
            "polarity": sentiment_data.polarity,
            "subjectivity": sentiment_data.subjectivity
        }
    except Exception as e:
        sentiment = {"error": f"Duygu analizi hatası: {e}"}

    try:
        summary = get_summarizer()(text, max_length=80, min_length=20, do_sample=False)[0]['summary_text']
    except Exception as e:
        summary = f"Özetleme hatası: {e}"

    # Kavram Çıkarımı (Entity Extraction)
    entities = []
    try:
        nlp = get_spacy_model()
        doc = nlp(text)
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    except Exception as e:
        entities = [{"error": str(e)}]

    return {
        "status": "success",
        "type": "text",
        "sentiment": sentiment,
        "summary": summary,
        "entities": entities
    }

def analyze_file(file_path: str) -> dict:
    """Dosya türünü otomatik algıla ve uygun analiz fonksiyonunu çağır"""
    if not os.path.exists(file_path):
        return {"status": "error", "message": "Dosya bulunamadı."}

    ext = os.path.splitext(file_path)[-1].lower()
    if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        return analyze_image(file_path)
    elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
        return analyze_video(file_path)
    elif ext in [".pdf"]:
        return analyze_pdf(file_path)
    elif ext in [".txt"]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return analyze_text(f.read())
        except Exception as e:
            return {"status": "error", "message": str(e)}
    else:
        return {"status": "error", "message": f"Desteklenmeyen dosya türü: {ext}"}

_image_generator = None

def get_image_generator():
    global _image_generator
    if _image_generator is None:
        from diffusers import StableDiffusionPipeline
        import torch
        try:
            # Performans ve kalite dengesi için v1.5 modelini seçiyoruz.
            model_id = "runwayml/stable-diffusion-v1-5"
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
            _image_generator = pipe
        except ImportError:
            raise ImportError("`diffusers` kütüphanesi kurulu değil. Lütfen `pip install diffusers transformers accelerate` komutları ile kurun.")
        except Exception as e:
            # Örneğin, internet bağlantısı yoksa veya model indirilemezse
            raise RuntimeError(f"Görüntü üretme modeli yüklenemedi: {e}")
    return _image_generator

def generate_image(prompt: str, output_path: str) -> dict:
    """
    Verilen bir metin (prompt) kullanarak bir görsel üretir ve belirtilen yola kaydeder.
    """
    try:
        # Çıktı yolunun geçerli olduğundan emin ol
        if not output_path.lower().endswith(".png"):
            return {"status": "error", "message": "Çıktı yolu '.png' ile bitmelidir."}

        generator = get_image_generator()
        image = generator(prompt).images[0]
        image.save(output_path)
        return {"status": "success", "message": f"Görsel başarıyla '{output_path}' yoluna kaydedildi."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def text_to_speech(text: str, output_path: str, lang: str = 'tr') -> dict:
    """
    Verilen metni ses dosyasına dönüştürür ve belirtilen yola kaydeder.
    """
    try:
        from gtts import gTTS
    except ImportError:
        raise ImportError("`gTTS` kütüphanesi kurulu değil. Lütfen `pip install gTTS` komutu ile kurun.")

    try:
        # Çıktı yolunun geçerli olduğundan emin ol
        if not output_path.lower().endswith(".mp3"):
            return {"status": "error", "message": "Çıktı yolu '.mp3' ile bitmelidir."}

        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(output_path)
        return {"status": "success", "message": f"Ses dosyası başarıyla '{output_path}' yoluna kaydedildi."}
    except Exception as e:
        return {"status": "error", "message": str(e)}
