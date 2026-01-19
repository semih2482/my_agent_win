import os
import sys
import re
import json
import time
import traceback
import select # Non-blocking input için
from datetime import datetime, timedelta
import tty # Non-blocking input için
import termios # Non-blocking input için
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agent import config
from agent.config import Colors
from agent.models.llm import ask, ask_fast_cpu, embed
from agent.memory.knowledge_store import VectorKnowledgeStore
from agent.memory.knowledge_graph import KnowledgeGraphStore
from agent.memory.extractor import extract_triplets
from agent.policy.tool_policy import ToolPolicy
from agent.policy.prompt_policy import PromptPolicy
from agent.rl.reward import RewardSignal
from agent.planner.planner import Planner
from agent.tools.persona_manager import PersonaManager
from agent.memory.personal_vector_store import PersonalVectorStore
from agent.tools.intent_detector import detect_intent
from agent.tools import knowledge_updater

# CLI renkleri
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"

DEBUG = True

def debug_print(msg):
    if DEBUG:
        print(msg)

def _repair_and_parse_json(json_str: str) -> dict | list | None:
    """
    Bir JSON string'ini ayrıştırmayı dener. Yalnızca temel temizlik yapar.
    """
    try:
        return json.loads(json_str.strip())
    except json.JSONDecodeError as e:
        debug_print(f"[JSON Onarım] Ayrıştırma başarısız: {e}. String: '{json_str[:100]}...'")
        return None

def extract_json(text: str) -> dict | list | None:
    """
    Metin içindeki ilk geçerli JSON nesnesini veya listesini bulur ve ayrıştırır.
    LLM'in eklediği metinleri ve markdown'ı yok saymak için tasarlanmıştır.
    """
    # En içteki JSON nesnesini veya listesini açgözlü olmayan bir şekilde bulmaya çalışan regex
    match = re.search(r'(\{.*\})|(\[.*\])', text, re.DOTALL)
    
    if match:
        json_str = match.group(0)
        first_brace = json_str.find('{')
        first_bracket = json_str.find('[')
        
        start_pos = -1
        

        if first_brace != -1 and first_bracket != -1:
            start_pos = min(first_brace, first_bracket)
        elif first_brace != -1:
            start_pos = first_brace
        else:
            start_pos = first_bracket

            if start_pos != -1:
            last_brace = json_str.rfind('}')
            last_bracket = json_str.rfind(']')
            end_pos = max(last_brace, last_bracket)

            if end_pos > start_pos:
                json_str = json_str[start_pos:end_pos+1]
                

                parsed_json = _repair_and_parse_json(json_str)
                if parsed_json is not None:
                    return parsed_json

    debug_print(f"[extract_json] Regex ile metinde geçerli bir JSON bloğu bulunamadı. Ham metin deneniyor.")

    return _repair_and_parse_json(text)

def smart_truncate(text: str, max_len: int, context_prompt: str = "") -> str:
    """
    Metni, belirtilen maksimum uzunluğu aşıyorsa, hızlı CPU modelini kullanarak akıllıca özetler.
    Eğer özetleme başarısız olursa, metni basitçe kırpar.
    """
    if len(text) <= max_len:
        return text

    print(f"{Colors.WARNING}[Akıllı Kırpma]: Metin ({len(text)} karakter) {max_len} karakter sınırını aşıyor. Özetleniyor...{Colors.ENDC}")

    context_info = f"Bu özet, '{context_prompt}' ana görevi için kullanılacak." if context_prompt else ""
    prompt = f"""Aşağıdaki metni, en önemli bilgileri koruyarak yaklaşık {max_len // 2} karaktere sığacak şekilde özetle. {context_info}

METİN:
{text}

ÖZET:"""
    try:
        return ask(prompt, max_new_tokens=1024)
    except Exception as e:
        print(f"{Colors.FAIL}[Akıllı Kırpma Hatası]: Özetleme başarısız oldu: {e}. Metin basitçe kırpılıyor.{Colors.ENDC}")
        return text[:max_len] + "\n...[METİN KIRPILDI]..."
class Agent:
    def _check_for_interrupt(self) -> bool:
        """
        Kullanıcının 'd' ve ardından Enter tuşuna basıp basmadığını non-blocking şekilde kontrol eder.
        Linux/macOS üzerinde çalışır.
        """
        if not sys.stdin.isatty():
            return False

        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):

                buffered_input = ""
                while select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                    buffered_input += sys.stdin.read(1)


                    return True
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        return False

    def __init__(self, available_tools, non_cacheable_tools, reload_tools_func=None):
        self.persona_mgr = PersonaManager(db_path=config.PERSONA_DB_PATH, encrypt_key=None, retention_days=365)
        self.personal_store = PersonalVectorStore(store_path=config.PERSONAL_STORE_PATH)
        self.knowledge_store = VectorKnowledgeStore(db_path=config.MEMORY_DB_PATH)
        self.knowledge_graph = KnowledgeGraphStore(db_path=config.KG_DB_PATH)
        self.short_term_memory = deque(maxlen=20)

        self.available_tools = available_tools

        self.available_tools["update_knowledge"] = {
            "func": knowledge_updater.run,
            "description": knowledge_updater.TOOL_INFO["description"],
            "input_schema": knowledge_updater.TOOL_INFO["input_schema"]
        }
        self.non_cacheable_tools = non_cacheable_tools
        self.reload_tools_func = reload_tools_func

        self.tool_policy = ToolPolicy(tools=self.available_tools)
        self.prompt_policy = PromptPolicy(prompts=["default_prompt"])
        self.reward_signal = RewardSignal()
        self.planner = Planner(tools=self.available_tools, max_retries=2)

        self.response_cache = {}

        self.action_history = deque(maxlen=5)
        self.stuck_counter = 0

    def _log_tool_action(self, thought, tool_name, tool_input, reward=None):
        """Araç seçimlerini RL policy için logla."""
        debug_print(f"{Colors.OKCYAN}\n[POLICY-LOG] Thought: {thought}{Colors.ENDC}")
        debug_print(f"{Colors.OKCYAN}[POLICY-LOG] Action: {tool_name}, Input: {tool_input}{Colors.ENDC}")
        if reward is not None:
            debug_print(f"{Colors.OKCYAN}[POLICY-LOG] Reward: {reward}{Colors.ENDC}")

    def _reflect_and_note(self, user_message: str, response: str):
        """Yanıt sonrası önemli noktaları otomatik not eder."""
        importance = 0
        if not isinstance(user_message, str):
            return

        keywords = ["yarın", "haftaya", "unutma", "seviyorum", "istemiyorum", "benim", "adresim", "telefonum"]
        if any(kw in user_message.lower() for kw in keywords):
            importance += 1

        importance_queries = ["kişisel bilgi", "görev", "talimat", "tercih"]
        for q in importance_queries:
            q_vec = embed(q)
            u_vec = embed(user_message)
            sim = float(np.dot(q_vec, u_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(u_vec)))
            if sim > 0.80:
                importance += 1

        if importance > 0:
            note_text = f"📝 Kullanıcı dedi ki → {user_message}"
            result = self.knowledge_store.add(note_text)
            debug_print(f"{Colors.OKGREEN}[Auto-Note]: {note_text}{Colors.ENDC} ({result['status']})")

            try:
    
                triplets = extract_triplets(note_text)

                # 2. Knowledge Graph'a ekle
                if triplets:
                    self.knowledge_graph.add_triplets(triplets)
                    debug_print(f"[Dual-Write]: {len(triplets)} triplet Knowledge Graph'a eklendi.")
            except Exception as e:
                debug_print(f"[Dual-Write Hatası]: Knowledge Graph'a yazarken hata: {e}")


    def _choose_strategy(self, intent_info: dict) -> str:
        """Niyete göre en uygun stratejiyi seçer."""
        strategy = intent_info.get("strategy", "reactive")
        intent = intent_info.get("intent")
        source = intent_info.get("source")
        confidence = intent_info.get("confidence", 0)

        print(
            f"{Colors.OKCYAN}💡 Strateji: '{strategy}' (Niyet: {intent}, Kaynak: {source}, Güven: {confidence:.2f}){Colors.ENDC}"
        )
        return strategy

    def _get_llm_decision(self, user_prompt: str, persona_text: str, past_knowledge: str, last_observation: str, personal_knowledge: str) -> dict | None:
        """LLM'e danışarak bir sonraki adım için araç kararı alır."""

        tool_anti_patterns = {
            "code_auditor": "Web sitelerini, URL'leri veya metinleri analiz etmek için DEĞİL, sadece mevcut bir Python dosyasının kod kalitesini denetlemek için kullanılır.",
            "code_editor": "Geçici veri depolamak için KULLANILMAZ. Sadece kullanıcı açıkça bir dosyayı kalıcı olarak oluşturmak veya değiştirmek istediğinde kullanılır.",
            "internet_search": "Daha spesifik bir araştırma aracı (örn: 'critical_web_researcher') göreve daha uygunsa, bu genel aracı kullanmaktan kaçın.",
            "tool_creator": "Mevcut araçlardan herhangi birinin zaten yapabildiği bir görevi yerine getirmek için KULLANILMAZ. Sadece tamamen yeni bir yetenek gerektiğinde kullanılır."
        }

        tools_list = []
        for name, props in self.available_tools.items():
            description = props["description"]

            if name in tool_anti_patterns:

                tools_list.append(
                    f'- `{name}`: {description}\n'
                    f'  **Uygun Değil:** {tool_anti_patterns[name]}'
                )
            else:
                tools_list.append(f'- `{name}`: {description}')

        tools_string = "\n".join(tools_list)

        approval_input_example = '`{"tool_filename": "test.py"}`'

        observation_text = ""
        if last_observation:

            last_observation = smart_truncate(last_observation, 3500, context_prompt=user_prompt)
            observation_text = f"**ÖNCEKİ ADIMIN GÖZLEMİ:**\n{last_observation}\n"

        conversation_history = "\n".join([f"- {msg['role']}: {msg['content']}" for msg in reversed(self.short_term_memory)])
        if not conversation_history:
            conversation_history = "Konuşma geçmişi boş."

        past_knowledge = smart_truncate(past_knowledge, 1500)
        personal_knowledge = smart_truncate(personal_knowledge, 1500)
        prompt = f'''Sen, bir görevi tamamlamak için doğru araçları seçmesi ve sonuçları eleştirel bir gözle analiz etmesi gereken zeki ve otonom bir ajansın. Bir görevi adımlara ayırabilir, araçları art arda çalıştırabilirsin. Cevabını HER ZAMAN JSON formatında ver.

**BİLGİ KAYNAKLARI (ÖNCELİK SIRASINA GÖRE)**

**1. KULLANICININ KİŞİSEL NOTLARI (EN YÜKSEK ÖNCELİK):**
Bu notlar, doğrudan kullanıcıyla ilgili veya kullanıcının daha önce "unutma" dediği en önemli bilgilerdir. Kararlarını verirken ve cevaplarını oluştururken **her zaman ilk olarak bu bilgilere başvur ve en yüksek ağırlığı bu bilgilere ver**.
{personal_knowledge or "Bu konuda ilgili kişisel not bulunmuyor."}

**2. YAPISAL BİLGİ GRAFİĞİ (YÜKSEK ÖNCELİK):**
Bunlar, doğrulanmış ve birbiriyle ilişkilendirilmiş kesin gerçeklerdir. Kişisel notlardan sonra en güvenilir bilgi kaynağın budur.
{self.knowledge_graph.query_as_text(user_prompt) or "Bu konuda yapısal bilgi bulunmuyor."}

**3. GEÇMİŞ BİLGİLER VE GENEL NOTLAR (ORTA ÖNCELİK):**
Bunlar, daha önceki konuşmalardan, araştırmalardan ve genel gözlemlerden elde edilmiş bilgilerdir. Yukarıdaki kaynaklarda bilgi yoksa veya ek bağlam gerekiyorsa bu notları kullan.
{past_knowledge or "Geçmiş bilgi veya genel not bulunmuyor."}

**SON KONUŞMA GEÇMİŞİ (EN YENİDEN ESKİYE):**
{conversation_history}

**KULLANICI PROFİL ÖZETİ:**
{persona_text}

{observation_text}**MEVCUT ARAÇLAR:**
{tools_string}

**GÖREV VE KURALLAR:**

**0. KURAL: DOĞRUDAN ARAÇ ÇAĞRISINI TESPİT ET (EN YÜKSEK ÖNCELİK - SÜPER KURAL)**
    *   **İLK OLARAK,** kullanıcının isteğinin, mevcut araçlardan birini doğrudan adıyla çağırıp çağırmadığını kontrol et. Örneğin: "`review_and_approve_tool` aracını kullanarak ... onayla" veya "`internet_search` ile ... araştır".
    *   **EĞER BÖYLE BİR DURUM VARSA,** başka hiçbir kuralı düşünme. `action` olarak kullanıcının belirttiği araç adını, `input` olarak da o aracın girdisini yaz. Bu kural, diğer tüm analizlerden önce gelir.

**1. KURAL: OLUŞTURULAN ARACI ONAYLA (DÖNGÜYÜ KIRMAK İÇİN EN ÖNEMLİ KURAL!)**
    *   Eğer `ÖNCEKİ ADIMIN GÖZLEMİ` alanı, bir aracın yeni oluşturulduğunu ve onaylanması gerektiğini belirtiyorsa (örneğin, "Yeni araç 'dosya_adi.py' başarıyla oluşturuldu" ve "onaylamak için" gibi ifadeler içeriyorsa), başka hiçbir kuralı düşünme. Bu, sonraki adımdır.
    *   `action` olarak `"review_and_approve_tool"` seç.
    *   `input` olarak, gözlem metninden çıkardığın araç dosyasının adını (`tool_filename`) ver. Örneğin, gözlem "Yeni araç 'test.py' oluşturuldu..." ise, input `{{"tool_filename": "test.py"}}` olmalıdır.
    *   Bu durumda başka hiçbir aracı (özellikle `tool_creator`'ı) KESİNLİKLE kullanma. Bu kural, sonsuz döngüye girmeyi önlemek için kritik öneme sahiptir.

**2. KURAL: GÖREV İÇİN EN UYGUN ARACI SEÇ**
    *   Yukarıdaki kurallar geçerli değilse, kullanıcının ana görevini (`ANA GÖREV`) analiz et.
    *   `MEVCUT ARAÇLAR` listesini dikkatlice incele ve görevi en iyi şekilde yerine getirebilecek aracı bul.
    *   **Eğer uygun bir araç varsa,** `action` olarak o aracın adını seç ve `input` alanını doldur.

**2.5 KURAL: ARAÇ HATASINDAN DERS ÇIKAR (TEKRARLANAN HATALARI ÖNLEMEK İÇİN)**
    *   Eğer `ÖNCEKİ ADIMIN GÖZLEMİ` bir araç hatası içeriyorsa (örneğin, "unexpected keyword argument", "Missing 'query'", "sadece bir dosya yolunu analiz edebilir"), bu hatayı tekrarlama.
    *   Hata veren aracı veya benzer şekilde çalışması muhtemel diğer araçları (örneğin, bir URL beklemeyen başka bir dosya aracı) tekrar denemekten kaçın.
    *   Görevi bu hataları göz önünde bulundurarak yeniden değerlendir. Eğer kalan araçlardan hiçbiri görevi yapamıyorsa, doğrudan **3. KURAL**'a geç ve `tool_creator` ile yeni bir araç oluştur.

**3. KURAL: YENİ ARAÇ OLUŞTUR (GEREKİYORSA)**
    *   **SADECE VE SADECE** `MEVCUT ARAÇLAR` listesinde görevi yerine getirebilecek HİÇBİR araç yoksa ve görev yeni, yeniden kullanılabilir bir yetenek gerektiriyorsa (örneğin, "bir Python scripti yaz", "bir API'ye bağlanan bir fonksiyon oluştur", "belirli bir analizi yapan bir araç yap" gibi), o zaman `tool_creator` aracını kullan.
    *   Eğer bir önceki adımda var olmayan bir araç kullanmaya çalıştıysan (gözlemde "böyle bir araç mevcut değil" yazıyorsa), bu, yeni bir araç oluşturman gerektiğinin güçlü bir işaretidir.
    *   `action` olarak `"tool_creator"` seç.
    *   `input` olarak, `tool_creator` aracının şemasına uygun bir JSON nesnesi sağla. Bu nesne `task_description`, `tool_name` ve `input_schema` içermelidir.

**4. KURAL: SOHBET ET (ARAÇ GEREKMİYORSA)**
    *   Eğer istek basit bir selamlama, sohbet veya araç gerektirmeyen bir soru ise, `action: "none"` kullan ve `response` alanında cevap ver.

**5. KURAL: Eleştirel Düşün ve Analiz Et**
    * Bir aracı çalıştırdıktan sonra elde ettiğin sonuçları körü körüne kabul etme.
    * Bilgiler arasında **tutarsızlık, çelişki veya mantıksızlık** var mı diye kontrol et.
    * Eğer bir tutarsızlık bulursan, bunu `thought` kısmında belirt.

**6. KURAL: Bilgiyi Doğrula**
    * `internet_search` gibi bir araçla önemli bir bilgi bulduysan, **hemen sonuca varma**.
    * Bulduğun bilgiyi doğrulamak için **ikinci bir `internet_search` çalıştır**.
    * İki kaynak uyuşuyorsa, bilgiyi doğrulanmış kabul et ve `action: "none"` ile nihai cevabını `response` alanında ver.

**7. KURAL: JSON Formatında Cevap Ver**
    *   Cevabın **her zaman** aşağıdaki şemaya uygun, geçerli bir JSON nesnesi olmalıdır.
    *   `thought`: Eylemini seçerken ne düşündüğünü açıklayan kısa bir metin.
    *   `action`: "MEVCUT ARAÇLAR" listesinden seçilen aracın adı (string). Eğer hiçbir araç gerekmiyorsa `"none"`.
    *   `input`: Seçilen aracın girdisi (string). Eğer araç argüman gerektirmiyorsa, bu alanı boş bir metin (`""`) olarak ayarla.
    *   `response`: Sadece `action` değeri `"none"` olduğunda kullanılır. Kullanıcıya verilecek nihai cevabı içeren **tek bir metin (string)** olmalıdır.
    *   **GÖREVİ BİTİRME:** Eğer bir önceki adımın gözlemi (`ÖNCEKİ ADIMIN GÖZLEMİ`) görevin başarıyla tamamlandığını gösteriyorsa (örn: "Araç onaylandı", "Dosya başarıyla yazıldı", "İşlem tamamlandı") ve ana görevi tamamlamak için yapacak başka bir adım kalmadıysa, görevi bitirmek için `action: "none"` kullan ve kullanıcıya nihai bir cevap ver.

**ANA GÖREV:** "{user_prompt}"

**Cevap (sadece JSON formatında):**
'''
        print(f"{Colors.OKBLUE}🤔 Düşünüyor... (LLM'e soruluyor){Colors.ENDC}")
        ai_response = ask(prompt, max_new_tokens=1024).strip()
        debug_print(f"{Colors.WARNING}[Ham Model Cevabı]: {ai_response}{Colors.ENDC}")

        parsed_json = extract_json(ai_response)
        return parsed_json, ai_response

    def _check_for_contradictions(self, new_observation: str, force_check: bool = False):
        """
        Yeni bir gözlemi mevcut bilgiyle karşılaştırır ve çelişkileri tespit edip çözmeye çalışır.
        Ayrıca bilginin eskiliğini (staleness) kontrol eder.
        """

            return None, None

        print(f"{Colors.OKCYAN}🔍 Bilgi doğrulama (çelişki ve eskilik) kontrolü başlatılıyor...{Colors.ENDC}")


        related_knowledge = self.knowledge_store.search(new_observation, top_k=1)

        if not related_knowledge:
            print("  -> İlgili geçmiş bilgi bulunamadı.")
            return None, None

        existing_knowledge_text, _, created_at_str = related_knowledge[0]

        is_stale = False
        if created_at_str:
            try:
                created_at = datetime.fromisoformat(created_at_str)
                if datetime.now() - created_at > timedelta(days=7):
                    is_stale = True
                    print(f"{Colors.WARNING}  -> Tespit edilen ilgili bilgi 7 günden eski. Güncellik kontrolü yapılacak.{Colors.ENDC}")
            except (ValueError, TypeError):
                pass


        prompt = f"""
        You are a fact-checking expert. Your task is to identify contradictions between a new piece of information and existing knowledge.

        **Existing Knowledge:**
        ---
        {existing_knowledge_text}
        ---

        **New Information:**
        ---
        {new_observation}
        ---

        **Instructions:**
        1.  **Staleness:** The "Existing Knowledge" is {'OLD' if is_stale else 'RECENT'}. If it is OLD, be more critical and favor the "New Information" if it seems more current.
        2.  **Compare:** Carefully compare the "New Information" with the "Existing Knowledge".
        2.  **Identify Contradiction:** Is there a direct contradiction or a significant factual inconsistency between the two?
        3.  **JSON Response:** Provide your answer in a strict JSON format with the following keys:
            *   `"contradiction_found"`: (boolean) `true` if there is a contradiction, otherwise `false`.
            *   `"confidence_score"`: (float, 0.0 to 1.0) How confident you are about the contradiction.
            *   `"explanation"`: (string) A brief explanation of why you think there is or isn't a contradiction.
            *   `"more_accurate_info"`: (string, "new", "existing", or "mixed") Which piece of information seems more accurate or reliable? If both have value, choose "mixed".
            *   `"updated_knowledge"`: (string) If a contradiction is found, provide a new, corrected, and comprehensive text that merges the valuable information from both sources and resolves the inconsistency. If no contradiction, this should be `null`.

        **Example Response (Contradiction Found):**
        ```json
        {{
          "contradiction_found": true,
          "confidence_score": 0.95,
          "explanation": "The existing knowledge states the capital of Australia is Sydney, while the new information correctly identifies it as Canberra.",
          "more_accurate_info": "new",
          "updated_knowledge": "The capital of Australia is Canberra. Sydney is its largest city, but not the capital."
        }}
        ```

        **Example Response (No Contradiction):**
        ```json
        {{
          "contradiction_found": false,
          "confidence_score": 0.99,
          "explanation": "Both pieces of information discuss similar topics but do not present conflicting facts.",
          "more_accurate_info": "mixed",
          "updated_knowledge": null
        }}
        ```

        Provide ONLY the JSON response.
        """

        try:
            response_str = ask(prompt, max_new_tokens=1024)
            analysis = extract_json(response_str)

            if not analysis or not isinstance(analysis, dict):
                print(f"  -> Çelişki analizi için LLM'den geçerli JSON alınamadı. Yanıt: {response_str}")
                return None, None

            if (analysis.get("contradiction_found") and analysis.get("confidence_score", 0) > 0.75) or is_stale:
                print(f"{Colors.WARNING}  -> Çelişki bulundu! Güven: {analysis['confidence_score']}{Colors.ENDC}")
                print(f"  -> Açıklama: {analysis['explanation']}")

                updated_knowledge = analysis.get("updated_knowledge")
                if updated_knowledge:
                    return existing_knowledge_text, updated_knowledge
            else:
                print("  -> Anlamlı bir çelişki bulunamadı.")

        except Exception as e:
            print(f"{Colors.FAIL}  -> Çelişki kontrolü sırasında hata: {e}{Colors.ENDC}")

        return None, None

    def _reflect_and_synthesize(self, observation: str, source_tool: str):
        """
        Bir gözlemden (genellikle bir araç çıktısından) proaktif olarak bilgi çıkarır ve
        yapısal olarak Bilgi Grafiği'ne (Knowledge Graph) kaydeder.
        """
        if not observation or len(observation) < 250 or source_tool not in ["internet_search", "comprehensive_financial_analyst", "critical_web_researcher"]:
            return

        print(f"{Colors.OKCYAN}🤔 Gözlemden öğreniliyor ({source_tool})...{Colors.ENDC}")

        prompt = f"""
        You are a knowledge engineering expert. Your task is to extract key, reusable facts from the following text, which is an observation from a tool's output.
        Extract the information as a list of "Subject-Relation-Object" triplets.
        Focus on facts that are likely to be useful in the future. Ignore trivial details or process confirmations (e.g., "file written successfully").

        RULES:
        - The output MUST be a JSON list of objects.
        - Each object must have "subject", "relation", and "object" keys.
        - If no significant facts can be extracted, return an empty list `[]`.
        - Do not add any commentary. Respond ONLY with the JSON list.

        OBSERVATION TEXT:
        ---
        {observation}
        ---

        JSON TRIPLETS:
        """
        try:
            response_str = ask(prompt, max_new_tokens=1024)
            triplets = extract_json(response_str)

            if triplets and isinstance(triplets, list):
                self.knowledge_graph.add_triplets(triplets)
                print(f"{Colors.OKGREEN}  -> ✅ {len(triplets)} adet yeni bilgi Bilgi Grafiği'ne eklendi.{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.FAIL}  -> ❌ Bilgi sentezleme sırasında hata: {e}{Colors.ENDC}")

    def run(self, user_prompt: str):
        if user_prompt in self.response_cache:
            print(f"{Colors.OKCYAN}⚡ Hızlı Yanıt (Önbellekten){Colors.ENDC}")
            final_response = self.response_cache[user_prompt]
            print(f"{Colors.OKGREEN}\nFinal Yanıtı:\n{final_response}{Colors.ENDC}")
            self.short_term_memory.append({"role": "user", "content": user_prompt})
            self.short_term_memory.append({"role": "agent", "content": final_response})
            return


        try:
            intent_info = detect_intent(user_prompt)
        except Exception as e:
            print(f"{Colors.FAIL}Niyet tespiti sırasında bir hata oluştu: {e}{Colors.ENDC}")
            intent_info = {"intent": "chat", "strategy": "reactive", "confidence": 0.0, "source": "error"}


        if intent_info.get("intent") == "chat" and intent_info.get("source") == "regex":
            print(f"{Colors.OKCYAN}⚡ Hızlı Sohbet Yolu (CPU) aktif...{Colors.ENDC}")
            chat_prompt = f"<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n{user_prompt}</s>\n<|assistant|>"
            try:
                final_response = ask_fast_cpu(chat_prompt, max_new_tokens=512)
                print(f"{Colors.OKGREEN}\nFinal Yanıtı:\n{final_response}{Colors.ENDC}")
                self.short_term_memory.append({"role": "user", "content": user_prompt})
                self.short_term_memory.append({"role": "agent", "content": final_response})
                self.response_cache[user_prompt] = final_response
                return
            except Exception as e:
                debug_print(f"{Colors.WARNING}[Hızlı Sohbet Yolu Hatası]: {e}{Colors.ENDC}")


        max_retries = 2
        MAX_STEPS = 10
        start_time = time.time()
        self.short_term_memory.append({"role": "user", "content": user_prompt})

        print(f"{Colors.OKCYAN}🌀 Bilgi toplama adımları paralel olarak başlatılıyor...{Colors.ENDC}")

        tasks = {}
        results = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            try:
                self.persona_mgr.extract_and_add_from_message(user_prompt)
            except (Exception, UnicodeEncodeError) as e:
                debug_print(f"{Colors.WARNING}[Persona Trait Hatası]: {e}{Colors.ENDC}")

            tasks[executor.submit(self.persona_mgr.summarize_persona)] = "persona"

            tasks[executor.submit(self.knowledge_graph.query_as_text, user_prompt)] = "knowledge_graph"

            for future in as_completed(tasks):
                task_name = tasks[future]
                try:
                    results[task_name] = future.result()
                except Exception as exc:
                    print(f'{Colors.FAIL}{task_name} oluşturulurken bir hata oluştu: {exc}{Colors.ENDC}')
                    results[task_name] = None


        persona_text = results.get("persona", "Kullanıcı profili özeti bulunamadı.")


        personal_knowledge_results = results.get("personal", [])
        personal_knowledge_text = "\n".join([f"- {item[0]['text']}" for item in personal_knowledge_results])
        if personal_knowledge_text:
            print(f"{Colors.OKGREEN}[Personal Store Hit]: {len(personal_knowledge_results)} ilgili not bulundu.{Colors.ENDC}")


        past_knowledge_results = results.get("knowledge", [])

        past_knowledge_text = "\n".join([f"- {item[0]}" for item in past_knowledge_results]) if past_knowledge_results else ""
        if past_knowledge_text:
            print(f"{Colors.OKGREEN}[VectorKnowledgeStore Hit]: {len(past_knowledge_results)} ilgili anı bulundu.{Colors.ENDC}")


        knowledge_graph_text = results.get("knowledge_graph", "")
        if knowledge_graph_text:

            print(f"{Colors.OKGREEN}[KnowledgeGraph Hit]: Yapısal bilgi bulundu.{Colors.ENDC}")

        print(f"{Colors.OKCYAN}✅ Paralel bilgi toplama tamamlandı.{Colors.ENDC}")


        strategy = self._choose_strategy(intent_info)

        final_response = "Üzgünüm, isteğinizi işlerken bir sorun oluştu."
        tool_result = None
        thought = ""
        tool_name = ""
        if strategy == "planner":
            try:
                print(f"{Colors.HEADER}--- Görev Planlanıyor ve Yürütülüyor ---{Colors.ENDC}")
                planner_result = self.planner.plan_and_execute(user_prompt)

                if planner_result and planner_result.get("status") == "clarification_needed":
                    final_response = planner_result.get('question', 'Sizden ek bilgi bekliyorum.')
                    tool_result = planner_result
                elif planner_result and planner_result.get("status") == "success":
                    plan_output = planner_result.get('result') or json.dumps(planner_result, indent=2, ensure_ascii=False)


                    synthesis_prompt = f"""
                    Bir kullanıcı sorusuna yanıt vermek için bir dizi eylem gerçekleştirdin. Şimdi, bu eylemlerin sonucunu kullanarak kullanıcıya kapsamlı ve doğrudan bir yanıt oluştur.

                    Kullanıcının Orijinal Sorusu: "{user_prompt}"

                    Gerçekleştirdiğin Eylemlerin Sonucu:
                    ---
                    {plan_output}
                    ---

                    Yukarıdaki bilgilere dayanarak, kullanıcının sorusuna doğrudan ve sohbet havasında bir yanıt ver. Teknik detayları veya dosya yazma gibi ara adımları değil, sadece nihai sonucu ve cevabı vurgula.
                    """
                    print(f"{{Colors.OKCYAN}}🔄 Nihai yanıt sentezleniyor...{{Colors.ENDC}}")
                    final_response = ask(synthesis_prompt, max_new_tokens=1024)
                    tool_result = planner_result
                else:
                    error_message = planner_result.get('message', 'Bilinmeyen hata.')
                    final_response = f"Planlayıcı görevi tamamlayamadı: {error_message}"
                    tool_result = {"status": "error", "message": error_message}

                thought = "Görev, Planner tarafından planlandı ve yürütüldü."
                tool_name = "planner"
            except Exception as e:
                print(f"{Colors.FAIL}Planner çalıştırılırken beklenmedik bir hata oluştu: {e}{Colors.ENDC}")
                traceback.print_exc()
                final_response = f"Sistemsel bir hata nedeniyle görev tamamlanamadı: {e}"
                tool_result = {"status": "error", "message": str(e)}

        else:
            last_observation = ""
            tool_input = ""
            for step in range(MAX_STEPS):
                print(f"{Colors.HEADER}--- Adım {step + 1}/{MAX_STEPS} ---{Colors.ENDC}")

                if self._check_for_interrupt():
                    print(f"\n{Colors.FAIL}🛑 Görev kullanıcı tarafından iptal edildi.{Colors.ENDC}")
                    final_response = "Görev iptal edildi. Yeni bir komut bekliyorum."
                    tool_result = {"status": "cancelled", "message": "Görev kullanıcı tarafından iptal edildi."}
                    break

                decision = None
                raw_response = ""


                approval_needed_phrases = [

                ]
                if last_observation and any(phrase in last_observation for phrase in approval_needed_phrases):

                    if match:

                        tool_filename = next((g for g in match.groups() if g is not None), None)
                        if tool_filename:
                            print(f"{Colors.OKCYAN}🔧 Kural tabanlı döngü kırma: '{tool_filename}' aracı onaylanacak.{Colors.ENDC}")
                            decision = {
                                "thought": "Bir önceki adımda bir araç oluşturuldu ve şimdi onaylanması gerekiyor. Döngüyü kırmak için 'review_and_approve_tool' aracını kullanıyorum.",
                                "action": "review_and_approve_tool",
                                "input": {"action": "approve", "tool_filename": tool_filename.strip()}
                            }


                if decision is None:
                    for attempt in range(max_retries):
                        decision, raw_response = self._get_llm_decision(
                            user_prompt=user_prompt,
                            persona_text=persona_text,
                            past_knowledge=past_knowledge_text,
                            last_observation=last_observation,
                            personal_knowledge=personal_knowledge_text
                        )
                        if decision and isinstance(decision, dict):
                            break
                        print(f"{Colors.WARNING}[Deneme {attempt + 1}/{max_retries}] LLM'den geçerli bir JSON kararı alınamadı. Tekrar deneniyor...{Colors.ENDC}")
                        time.sleep(1)

                if not decision or not isinstance(decision, dict):
                    print(f"{Colors.FAIL}LLM'den geçerli bir JSON kararı alınamadı. Ham yanıt: {raw_response}{Colors.ENDC}")
                    final_response = "Üzgünüm, bir karar veremedim. Lütfen tekrar dener misin?"
                    break

                thought = decision.get("thought", "Düşünce belirtilmedi.")
                action = decision.get("action", "none")

                if isinstance(action, list) and action:
                    print(f"{Colors.WARNING}[Düzeltme]: LLM'den eylem listesi alındı, ilk eylem '{action[0]}' kullanılıyor.{Colors.ENDC}")
                    action = action[0]

                tool_input = decision.get("input")
                tool_name = action

                print(f"{Colors.OKCYAN}Düşünce: {thought}{Colors.ENDC}")

                current_decision_summary = ""
                if action != "none":
                    action_to_check = action
                    input_to_check = str(tool_input or "")
                    input_summary = (input_to_check[:75] + '..') if len(input_to_check) > 75 else input_to_check
                    current_decision_summary = f"{action_to_check}({input_summary})"

                    if current_decision_summary in self.action_history:
                        self.stuck_counter += 1
                        print(f"{Colors.WARNING}⚠️  Döngü Uyarısı: Aynı eylem '{current_decision_summary}' tekrar ediliyor. (Sayaç: {self.stuck_counter}){Colors.ENDC}")
                    else:
                        self.stuck_counter = 0

                    if self.stuck_counter >= 2:
                        print(f"{Colors.FAIL}🛑 Döngü Tespit Edildi! Ajan aynı eylemde takılı kaldı. Görev sonlandırılıyor.{Colors.ENDC}")
                        final_response = "Bir döngüye girdiğimi fark ettim ve ilerleme kaydedemiyorum. Lütfen görevi farklı bir şekilde ifade etmeyi deneyin."
                        # HATA DÜZELTME: 'break' yerine 'return' kullanarak fonksiyondan tamamen çık.
                        print(f"{Colors.OKGREEN}\nFinal Yanıtı:\n{final_response}{Colors.ENDC}")
                        self.short_term_memory.append({"role": "agent", "content": final_response})
                        return

                if current_decision_summary:
                    self.action_history.append(current_decision_summary)

                if action == "none":
                    print(f"{Colors.OKGREEN}✅ Eylem Gerekmiyor. Görev tamamlandı.{Colors.ENDC}")
                    final_response = decision.get("response", "Görevi tamamladım ama bir yanıt üretemedim.")
                    tool_result = {"status": "success"}
                    break

                if action in self.available_tools:
                    print(f"{Colors.OKBLUE}Eylem: '{action}' aracı çalıştırılıyor...{Colors.ENDC}")
                    print(f"{Colors.OKBLUE}Girdi: {tool_input}{Colors.ENDC}")
                    try:
                        if action == "tool_creator" and isinstance(tool_input, dict):
                            print(f"{Colors.OKCYAN}🧠 'tool_creator' için hafıza taranıyor...{Colors.ENDC}")

                            task_description = tool_input.get("task_description")
                            if task_description:

                                personal_notes_results = self.personal_store.search(task_description, top_k=3)
                                personal_notes_text = "\n".join([item[0]['text'] for item in personal_notes_results])
                                if personal_notes_text:
                                    print(f"{Colors.OKGREEN}   -> Kişisel notlardan ilgili bilgiler bulundu. Göreve ekleniyor...{Colors.ENDC}")


                                relevant_knowledge_results = self.knowledge_store.search(task_description, top_k=2)
                                knowledge_text = "\n".join([item[0] for item in relevant_knowledge_results])
                                if knowledge_text:
                                    print(f"{Colors.OKGREEN}   -> Genel hafızadan ilgili bilgiler bulundu. Göreve ekleniyor...{Colors.ENDC}")


                                combined_knowledge = ""
                                if personal_notes_text:
                                    combined_knowledge += f"KULLANICININ BU KONUYLA İLGİLİ KİŞİSEL NOTLARI (ÖNCELİKLİ):\n{personal_notes_text}\n\n"
                                if knowledge_text:
                                    combined_knowledge += f"GENEL BİLGİ (YARDIMCI OLABİLİR):\n{knowledge_text}"


                                if combined_knowledge:
                                    enhanced_description = f"""
GÖREV: {task_description}

Bu görevi yaparken SANA YARDIMCI OLMASI İÇİN DAHA ÖNCE ÖĞRENDİĞİMİZ BİLGİLER ŞUNLAR:
{combined_knowledge}
"""
                                    tool_input['task_description'] = enhanced_description
                            else:
                                print(f"{Colors.WARNING}   -> 'tool_creator' girdisinde 'task_description' bulunamadı, hafıza taraması atlanıyor.{Colors.ENDC}")

                        tool_func = self.available_tools[action]["func"]

                        result = tool_func(args=tool_input, agent_instance=self)

                        if isinstance(result, dict):
                            if result.get("status") == "error":
                                last_observation = f"Araç hatası: {result.get('message')}"
                                tool_result = result
                                print(f"{Colors.FAIL}Araç '{action}' bir hata döndürdü: {result.get('message')}{Colors.ENDC}")
                            else:
                                last_observation = result.get("result", json.dumps(result, ensure_ascii=False))
                                tool_result = result

                                if action == "code_auditor" and result.get("raw_suggestions"):
                                    print(f"{Colors.OKGREEN}🔧 'code_auditor' önerileri bulundu. Otomatik düzeltme başlıyor...{Colors.ENDC}")
                                    suggestions = result["raw_suggestions"]

                                    # Girdi'den dosya yolunu güvenli bir şekilde çıkar
                                    file_path_to_fix = None
                                    try:
                                        # Girdi bir dict ise
                                        if isinstance(tool_input, dict):
                                            file_path_to_fix = tool_input.get("file_path")
                                        # Girdi bir string ise (JSON string'i olabilir)
                                        elif isinstance(tool_input, str):
                                            try:
                                                # JSON string'i olarak ayrıştırmayı dene
                                                input_dict = json.loads(tool_input)
                                                file_path_to_fix = input_dict.get("file_path")
                                            except (json.JSONDecodeError, TypeError):

                                                if os.path.exists(tool_input):
                                                    file_path_to_fix = tool_input

                                        if not file_path_to_fix or not os.path.isabs(file_path_to_fix):
                                            raise ValueError(f"Geçerli bir mutlak dosya yolu bulunamadı. Girdi: {tool_input}")

                                        fixes_applied = 0
                                        try:
                                            with open(file_path_to_fix, 'r', encoding='utf-8') as f:
                                                original_content = f.read()

                                            current_content = original_content
                                            for suggestion in suggestions:
                                                original_code = suggestion.get("original_code")
                                                suggested_code = suggestion.get("suggested_code")

                                                if original_code and suggested_code and original_code in current_content:

                                                    current_content = current_content.replace(original_code, suggested_code)
                                                    fixes_applied += 1
                                                    print(f"  -> Düzeltme uygulandı: {suggestion.get('description', 'Açıklama yok')}")
                                                else:
                                                    print(f"{Colors.WARNING}  -> Düzeltme atlandı: Orijinal kod dosyada bulunamadı veya öneri eksik.{Colors.ENDC}")

                                            if fixes_applied > 0:
                                                with open(file_path_to_fix, 'w', encoding='utf-8') as f:
                                                    f.write(current_content)

                                                correction_summary = f"'{os.path.basename(file_path_to_fix)}' dosyasına {fixes_applied} adet otomatik düzeltme başarıyla uygulandı."
                                                print(f"{Colors.OKGREEN}✅ {correction_summary}{Colors.ENDC}")
                                                last_observation = f"{last_observation}\n\nOTOMATİK DÜZELTME RAPORU:\n{correction_summary}"

                                        except FileNotFoundError:
                                            print(f"{Colors.FAIL}  -> Hata: Düzeltilecek dosya bulunamadı: {file_path_to_fix}{Colors.ENDC}")
                                        except Exception as e:
                                            print(f"{Colors.FAIL}  -> Otomatik düzeltme sırasında bir hata oluştu: {e}{Colors.ENDC}")

                                    except Exception as e:
                                        print(f"{Colors.FAIL}  -> 'code_auditor' girdisinden dosya yolu çıkarılamadı: {e}{Colors.ENDC}")



                                if result.get("special_action") == "reload_tools":
                                    if self.reload_tools_func:
                                        self.reload_tools_func(self)
                                    else:
                                        print(f"{Colors.WARNING}[System]: Araçların yeniden yüklenmesi istendi ancak yeniden yükleme fonksiyonu mevcut değil.{Colors.ENDC}")


                                if action == "critical_web_researcher" and result.get("status") == "success":
                                    chunks = result.get("chunks", [])
                                    if chunks:
                                        print(f"{Colors.OKGREEN}🧠 Araştırma sonuçları hafızaya ekleniyor...{Colors.ENDC}")
                                        for chunk in chunks:
                                                note_text = f"Araştırma sonucu ({chunk.get('sub_topic', 'Bilinmeyen Alt Başlık')}): {chunk['summary']}"
                                                self.knowledge_store.add(note_text)
                                                try:
                                                    # Triplet'leri çıkar ve Knowledge Graph'a ekle
                                                    triplets = extract_triplets(note_text)
                                                    if triplets:
                                                        self.knowledge_graph.add_triplets(triplets)
                                                        debug_print(f"[Dual-Write]: {len(triplets)} triplet (from research) added to Knowledge Graph.")
                                                except Exception as e:
                                                    debug_print(f"[Dual-Write Error]: Failed to write to Knowledge Graph from research result: {e}")
                                    else:
                                        print(f"{Colors.WARNING}critical_web_researcher başarıyla tamamlandı ancak 'chunks' bulunamadı.{Colors.ENDC}")

                        else:
                            last_observation = str(result)
                            tool_result = {"status": "success", "result": last_observation}

                        print(f"{Colors.WARNING}Gözlem: {last_observation}{Colors.ENDC}")

                        self._reflect_and_synthesize(last_observation, action)




                        old_knowledge, new_knowledge = self._check_for_contradictions(last_observation)
                        if old_knowledge and new_knowledge:
                            print(f"{Colors.OKGREEN}🔧 Hafıza düzeltiliyor...{Colors.ENDC}")
                            update_result = self.available_tools["update_knowledge"]["func"](
                                old_knowledge_text=old_knowledge,
                                updated_knowledge_text=new_knowledge
                            )
                            print(f"{Colors.OKGREEN}  -> {update_result.get('message')}{Colors.ENDC}")

                            last_observation = f"Bilgi düzeltildi. Yeni bilgi: {new_knowledge}"


                    except Exception as e:
                        print(f"{Colors.FAIL}'{action}' aracı çalıştırılırken hata: {e}{Colors.ENDC}")
                        traceback.print_exc()
                        last_observation = f"Hata: {e}"
                        tool_result = {"status": "error", "message": str(e)}
                else:

                    print(f"{Colors.FAIL}Bilinmeyen eylem: '{action}'. Ajan durumu yeniden değerlendirecek.{Colors.ENDC}")
                    last_observation = (
                        f"HATA: Bir önceki adımda '{action}' adında bir araç seçmeye çalıştım "
                        f"ancak böyle bir araç mevcut değil. Bu görevi tamamlamak için ya mevcut araçlardan "
                        f"farklı birini seçmeliyim ya da bu işi yapacak yeni bir aracı 'tool_creator' ile oluşturmalıyım."
                    )
                    tool_result = {"status": "error", "message": f"Bilinmeyen eylem: '{action}'"}
                    continue


                if tool_result and tool_result.get("status") == "error":
                    error_message = tool_result.get("message", "Bilinmeyen bir araç hatası.")
                    print(f"{Colors.FAIL}Araç hatası oluştu: {error_message}. Ajan durumu yeniden değerlendirecek.{Colors.ENDC}")
                    last_observation = f"HATA: Bir önceki adımda '{action}' aracını çalıştırırken şu hatayı aldım: '{error_message}'. Bu, ya aracın yanlış seçildiği ya da argümanların hatalı olduğu anlamına gelir. Görevi tamamlamak için farklı bir araç veya farklı argümanlar denemeliyim."
                    continue

                time.sleep(1)
            else:
                print(f"{Colors.WARNING}Maksimum adım sayısına ({MAX_STEPS}) ulaşıldı.{Colors.ENDC}")
                final_response = last_observation if last_observation else "Görevi tamamlayamadım."

        print(f"{Colors.OKBLUE}✅ Sonuçlar işleniyor ve hafıza güncelleniyor...{Colors.ENDC}")
        end_time = time.time()
        ctx_emb = embed(user_prompt)

        retries_used = 0
        if tool_result:
            retries_used = tool_result.get("retries", 0) if strategy == "planner" else 0

        total_reward = self.reward_signal.total_reward(
            feedback="👍" if tool_result and tool_result.get("status") == "success" else "👎",
            start_time=start_time,
            end_time=end_time,
            error=(tool_result.get("message") if tool_result and tool_result.get("status") == "error" else None),
            user_text=user_prompt,
            retries=retries_used
        )

        self.tool_policy.update(tool_name or "none", total_reward, context=ctx_emb)
        self.prompt_policy.update("default_prompt", total_reward)
        self._log_tool_action(
            thought=thought,
            tool_name=tool_name or "none",
            tool_input=tool_input,
            reward=total_reward
        )

        self.knowledge_store.add(f"Kullanıcı: {user_prompt}\nAsistan: {final_response}")
        self.short_term_memory.append({"role": "agent", "content": final_response})


        try:

            note_text = f"Kullanıcı: {user_prompt}\nAsistan: {final_response}"
            triplets = extract_triplets(note_text)


            if triplets:
                self.knowledge_graph.add_triplets(triplets)
                debug_print(f"[Dual-Write]: {len(triplets)} triplet Knowledge Graph'a eklendi.")
        except Exception as e:
            debug_print(f"[Dual-Write Hatası]: Knowledge Graph'a yazarken hata: {e}")


        self._reflect_and_note(user_prompt, final_response)

        financial_tools = ["get_investment_advice", "get_fund_advice", "find_assets", "get_crypto_advice"]
        if tool_result and tool_name in financial_tools and tool_result.get("status") == "success":
            note_text = f"💡 Finansal Tavsiye Notu ({tool_name}): Kullanıcının '{user_prompt}' isteğine karşılık şu analiz sunuldu:\n{final_response}"
            self.knowledge_store.add(note_text)
            debug_print(f"{Colors.OKGREEN}[Auto-Finance-Note]: Finansal tavsiye hafızaya kaydedildi.{Colors.ENDC}")

            try:

                triplets = extract_triplets(note_text)


                if triplets:
                    self.knowledge_graph.add_triplets(triplets)
                    debug_print(f"[Dual-Write]: {len(triplets)} triplet Knowledge Graph'a eklendi.")
            except Exception as e:
                debug_print(f"[Dual-Write Hatası]: Knowledge Graph'a yazarken hata: {e}")


        if tool_result and tool_result.get("status") == "success" and (tool_name or "none") not in self.non_cacheable_tools:
            self.response_cache[user_prompt] = final_response
            debug_print(f"{Colors.OKGREEN}[Cache-Save]: Yanıt, '{user_prompt}' anahtarıyla önbelleğe kaydedildi.{{Colors.ENDC}}")

        print(f"{Colors.OKGREEN}\nFinal Yanıtı:\n{{final_response}}{{Colors.ENDC}}")

# Forcing recompile to fix stale cache issue.

