import os
import sys
import re
import glob
import importlib.util
import json
import time
import traceback
import concurrent.futures

from agent.core import agent
from agent.models.llm import ask


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agent import config
from agent.config import Colors
from agent.core.agent import Agent

# Felsefe modu için durum değişkeni
PHILOSOPHY_MODE_ENABLED = False

AVAILABLE_TOOLS = {}
NON_CACHEABLE_TOOLS = set()

def _load_all_tools():
    """
    Internal function to load tools. Clears and repopulates the global tool dictionaries.
    Loads all tools from `agent/tools` and `agent/tools/community_tools` directories dynamically.
    """
    global AVAILABLE_TOOLS, NON_CACHEABLE_TOOLS
    AVAILABLE_TOOLS.clear()
    NON_CACHEABLE_TOOLS.clear()

    tools_dirs = [
        os.path.join(os.path.dirname(__file__), '..', 'tools'),
        os.path.join(os.path.dirname(__file__), '..', 'tools', 'community_tools')
    ]

    for tools_dir in tools_dirs:
        if not os.path.exists(tools_dir):
            os.makedirs(tools_dir)
            print(f"{Colors.OKCYAN}[System]: Created directory `{os.path.basename(tools_dir)}`.{Colors.ENDC}")

        for tool_file in glob.glob(os.path.join(tools_dir, "*.py")):
            if "__init__" in tool_file:
                continue

            try:
                # Modülü her zaman yeniden yükle
                relative_path = os.path.relpath(tool_file, project_root)
                module_name = os.path.splitext(relative_path)[0].replace(os.path.sep, '.')

                module = importlib.import_module(module_name)
                importlib.reload(module)

                tool_info = None
                if hasattr(module, 'get_tool_info'):
                    tool_info = module.get_tool_info()
                elif hasattr(module, 'TOOL_INFO') and hasattr(module, 'run'):
                    tool_info = {
                        "name": module.TOOL_INFO.get("name", module_name),
                        "description": module.TOOL_INFO["description"],
                        "args_schema": module.TOOL_INFO.get("args_schema", {}),
                        "func": module.run,
                        "cacheable": module.TOOL_INFO.get("cacheable", True)
                    }

                if tool_info:
                    tool_name = tool_info["name"]
                    AVAILABLE_TOOLS[tool_name] = {
                        "func": tool_info["func"],
                        "description": tool_info["description"],
                        "args_schema": tool_info.get("args_schema", {})
                    }
                    if not tool_info.get("cacheable", True):
                        NON_CACHEABLE_TOOLS.add(tool_name)
                    print(f"{Colors.OKGREEN}[Tool Loaded]: '{tool_name}' tool loaded successfully.{Colors.ENDC}")
            except Exception as e:
                print(f"{Colors.FAIL}[Tool Load Error]: Failed to load '{tool_file}'. Error: {e}{Colors.ENDC}")

    return AVAILABLE_TOOLS, NON_CACHEABLE_TOOLS

def get_tools_and_reload_function(agent_instance):
    """
    Reloads all tools and updates the provided agent instance with the new set.
    """
    print(f"\n{Colors.HEADER}🔄 Araçlar yeniden yükleniyor...{Colors.ENDC}")
    new_available_tools, new_non_cacheable_tools = _load_all_tools()

    # Agent nesnesinin kendi araç listelerini doğrudan güncelle
    agent_instance.available_tools = new_available_tools
    agent_instance.non_cacheable_tools = new_non_cacheable_tools

    # Agent'ın içindeki policy ve planner'ı da yeni araçlarla güncelle
    from agent.policy.tool_policy import ToolPolicy
    agent_instance.tool_policy = ToolPolicy(tools=agent_instance.available_tools)
    if hasattr(agent_instance, 'planner'):
        agent_instance.planner.tools = agent_instance.available_tools
    print(f"{Colors.OKGREEN}✅ Araçlar başarıyla yeniden yüklendi ve ajan güncellendi.{Colors.ENDC}\n")

def reload_agent_logic():
    """
    Agent'ın ana mantığını içeren modülleri yeniden yükler.
    Bu, agent.py'deki prompt değişikliklerini anında yansıtır.
    """
    print(f"{Colors.OKCYAN}🧠 Agent mantığı yeniden yükleniyor...{Colors.ENDC}")
    try:
        # Yeniden yüklenecek ana modülleri belirt
        modules_to_reload = [
            'agent.core.agent',
            'agent.models.llm',
            'agent.planner.planner',
            'agent.memory.knowledge_graph',
            'agent.memory.extractor'
        ]
        for module_name in modules_to_reload:
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
                print(f"  -> Modül '{module_name}' yeniden yüklendi.")

        from agent.core.agent import Agent
        print(f"{Colors.OKGREEN}✅ Agent mantığı başarıyla yeniden yüklendi.{Colors.ENDC}")
        return Agent
    except Exception as e:
        print(f"{Colors.FAIL}❌ Agent mantığı yeniden yüklenirken kritik hata: {e}{Colors.ENDC}")
        traceback.print_exc()
        return None

def run_philosophy_mode(user_prompt: str, agent: Agent):
    """Felsefe modunu çalıştırır ve 3 aşamalı prompt zincirini uygular."""
    try:
        print(f"{Colors.HEADER}--- Felsefe Modu Aktif ---{Colors.ENDC}")

        print(f"{Colors.OKBLUE}🌀 1. Adım: Soru analiz ediliyor ve içsel durum belirleniyor...{Colors.ENDC}")
        prompt1 = f"Kullanıcının şu sorusunu analiz et: '[{user_prompt}]'. Bu soruya vereceğin cevabın içsel durumunu [Yoğunluk], [Belirsizlik], [Amaç_Mesafesi] olarak üç kelimeyle tanımla. Sadece bu üç kelimeyi ve değerlerini yaz. Örneğin: [Yoğunluk: Düşük], [Belirsizlik: Yüksek], [Amaç_Mesafesi: Yakın]"
        hissiyat = ask(prompt1, max_new_tokens=50).strip()
        print(f"{Colors.OKCYAN}   İçsel Durum: {hissiyat}{Colors.ENDC}")

        print(f"{Colors.OKBLUE}🌀 2. Adım: İçsel duruma uygun felsefi akım seçiliyor...{Colors.ENDC}")
        from agent.models.llm import ask
        prompt2 = f"İç durumun '{hissiyat}'. Bu duruma en uygun felsefi akım hangisidir? Cevap olarak SADECE felsefi akımın adını yaz. Örneğin: Stoacılık"
        felsefi_akim = ask(prompt2, max_new_tokens=50).strip()
        felsefi_akim = felsefi_akim.split(':')[-1].strip().replace('.', '')
        print(f"{Colors.OKCYAN}   Seçilen Felsefe: {felsefi_akim}{Colors.ENDC}")

        try:
            print(f"{Colors.OKBLUE}   Ek Adım: '{felsefi_akim}' hakkında bilgi tazeleniyor...{Colors.ENDC}")
            if 'learn_philosophy' in AVAILABLE_TOOLS:
                AVAILABLE_TOOLS['learn_philosophy']['func'](felsefi_akim)
            else:
                print(f"{Colors.WARNING}   'learn_philosophy' aracı bulunamadı. Bu adım atlanıyor.{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.FAIL}   Bilgi tazeleme sırasında beklenmedik bir hata oluştu: {e}{Colors.ENDC}")

        print(f"{Colors.OKBLUE}🌀 3. Adım: '{felsefi_akim}' modunda cevap üretiliyor...{Colors.ENDC}")
        prompt3 = f"Bir anlığına '{felsefi_akim}' felsefesini benimsemiş bir varlık ol. Bu felsefenin bir temsilcisi gibi davranarak ve onun dünya görüşünü ve üslubunu kullanarak kullanıcının '[{user_prompt}]' sorusuna doğrudan bir cevap ver. Felsefeyi açıklama, sadece o felsefeye göre yaşa ve cevapla."
        final_response = ask(prompt3, max_new_tokens=1024)

        print(f"{Colors.OKGREEN}\nFelsefi Yanıt:\n{final_response}{Colors.ENDC}")

        agent.short_term_memory.add_message(role="user", content=user_prompt)
        agent.short_term_memory.add_message(role="agent", content=final_response)
        agent.knowledge_store.add(f"Kullanıcı (Felsefe Modu): {user_prompt}\nAsistan ({felsefi_akim}): {final_response}")

    except Exception as e:
        print(f"{Colors.FAIL}Felsefe modu sırasında bir hata oluştu: {e}{Colors.ENDC}")
        traceback.print_exc()

#def handle_proactive_assistant(agent: Agent):
    """Kişisel notlardaki yeni konuları proaktif olarak araştırır."""
    try:
        research_queue_path = agent.personal_store.queue_path
        if os.path.exists(research_queue_path) and os.path.getsize(research_queue_path) > 0:
            with open(research_queue_path, 'r+', encoding='utf-8') as f:
                lines = f.readlines()
                topic_to_research = lines[0].strip()

                if topic_to_research:
                    if not sys.stdout.isatty():
                        print(f"\n{Colors.OKCYAN}[Proaktif Asistan]: Etkileşimli olmayan modda çalışıyor. Proaktif araştırma atlanıyor.{Colors.ENDC}")
                        # Kuyruktaki konuyu temizlemeden çık, böylece interaktif bir oturumda tekrar denenebilir.
                        return

                    print(f"\n{Colors.HEADER}[Proaktif Asistan]:{Colors.ENDC} Kişisel notlarınızda yeni bir konu fark ettim: '{topic_to_research}'")
                    user_consent = input(f"{Colors.BOLD}Bu konuda derinlemesine bir araştırma yapıp öğrenmemi ister misiniz? (e/h): {Colors.ENDC}").lower()

                    if user_consent == 'e':
                        topic_lower = topic_to_research.lower()
                        research_tool_name = 'critical_web_researcher'

                        crypto_keywords = ['bitcoin', 'ethereum', 'crypto', 'kripto', 'btc', 'eth', 'solana', 'xrp']
                        stock_keywords = ['hisse', 'fon', 'borsa', 'eregl', 'thyao', 'bist', '.is']
                        philosophy_keywords = ['felsefe', 'stoacılık', 'varoluşçuluk', 'nihilizm', 'platon', 'sokrates', 'nietzsche']
                        security_keywords = ['güvenlik', 'siber', 'zafiyet', 'hacker', 'owasp', 'vulnerability']

                        if any(kw in topic_lower for kw in security_keywords):
                            research_tool_name = 'critical_web_researcher'
                        elif any(kw in topic_lower for kw in crypto_keywords):
                            research_tool_name = 'get_crypto_advice'
                        elif any(kw in topic_lower for kw in stock_keywords):
                            research_tool_name = 'get_investment_advice'
                        elif any(kw in topic_lower for kw in philosophy_keywords):
                            research_tool_name = 'learn_philosophy'

                        print(f"{Colors.OKCYAN}💡 Araştırma için en uygun araç olarak '{research_tool_name}' seçildi.{Colors.ENDC}")

                        if research_tool_name in AVAILABLE_TOOLS:
                            research_func = AVAILABLE_TOOLS[research_tool_name]['func']
                        elif 'web_search' in AVAILABLE_TOOLS:
                            print(f"{Colors.WARNING}   '{research_tool_name}' aracı bulunamadı. Varsayılan 'web_search' aracına geçiliyor.{Colors.ENDC}")
                            research_func = AVAILABLE_TOOLS['web_search']['func']
                        else:
                            print(f"{Colors.FAIL}[System]: Ne '{research_tool_name}' ne de varsayılan 'web_search' aracı bulunamadı. Araştırma işlemi iptal ediliyor.{Colors.ENDC}")
                            # Araştırmayı iptal et ama konuyu kuyrukta bırak ki araçlar düzelince tekrar denenebilsin.
                            return # Fonksiyondan güvenli bir şekilde çık

                        # Araç fonksiyonunu hata kontrolü yapan bir sarmalayıcı ile çağır
                        def safe_research_wrapper(query: str):
                            """Araç fonksiyonunu çalıştırır ve olası hataları yakalar."""
                            try:
                                result = research_func(query)
                                # Araçların standart bir {status: '...', ...} formatında yanıt döndürdüğünü varsayıyoruz.
                                if isinstance(result, dict) and result.get('status') == 'error':
                                    print(f"\n{Colors.FAIL}[Araç Hatası - {research_tool_name}]: {result.get('message', 'Bilinmeyen bir hata oluştu.')}{Colors.ENDC}")
                                    return None # Hata durumunda None döndürerek işlemi durdur
                                return result
                            except Exception as e:
                                print(f"\n{Colors.FAIL}[Araç Çalıştırma Hatası - {research_tool_name}]: Araç çalıştırılırken beklenmedik bir istisna oluştu: {e}{Colors.ENDC}")
                                traceback.print_exc()
                                return None

                        print(f"{Colors.HEADER}--- Doğrudan Kritik Araştırmacı Başlatılıyor: '{topic_to_research}' ---{Colors.ENDC}")

                        # safe_research_wrapper'ın sadece tek bir parametre aldığını biliyoruz (query)
                        research_result_dict = safe_research_wrapper(topic_to_research)

                        final_report = None
                        chunks_to_save = [] # Bu, 'critical_web_researcher'dan dönen parçaları tutacak

                        if research_result_dict and research_result_dict.get('status') == 'success':
                            final_report = research_result_dict.get('result')
                            # "chunks" anahtarını critical_web_researcher'dan alıyoruz (daha önce eklemiştik)
                            chunks_to_save = research_result_dict.get('chunks', [])
                        else:
                            print(f"{Colors.FAIL}[System]: Kritik Araştırmacı bir hata döndürdü.{Colors.ENDC}")


                        if final_report:
                            # Raporun tamamını basmak terminali boğabilir, kısa bir özetini basalım
                            print(f"\n{Colors.HEADER}--- Araştırma Raporu (Özet) ---{Colors.ENDC}")
                            print(final_report[:1000] + "\n... (raporun tamamı hafızaya kaydediliyor) ...")
                            print(f"{Colors.HEADER}---------------------------------{Colors.ENDC}")


                            try:
                                print(f"{Colors.OKBLUE}🌀 Araştırma raporu parçalara ayrılıyor ve hafızaya kaydediliyor...{Colors.ENDC}")
                                note_topic = topic_to_research.split(']')[0].strip('[') if ']' in topic_to_research else "Araştırma Sonucu"

                                # Parçaları 'critical_web_researcher'dan aldık (chunks_to_save)
                                if not chunks_to_save:
                                    print(f"{Colors.WARNING}[System]: Araştırmacı 'chunks' döndürmedi. Raporun tamamı parçalanıyor...{Colors.ENDC}")
                                    # 'chunks' gelmezse, final raporunu manuel olarak parçalayan 'fallback' kodu
                                    report_chunks_text = [chunk.strip() for chunk in final_report.split('\n\n') if chunk.strip()]
                                else:
                                    print(f"  -> Araştırmacıdan {len(chunks_to_save)} adet hazır parça (chunk) alındı.")
                                    # 'chunks_to_save' listesi [{'sub_topic': '...', 'summary': '...'}] formatında
                                    report_chunks_text = [f"Araştırma Konusu: {item['sub_topic']}\nAraştırma Özeti: {item['summary']}" for item in chunks_to_save]

                                if not report_chunks_text:
                                    print(f"{Colors.WARNING}[System]: Rapor içeriği boş veya parçalara ayrılamadı.{Colors.ENDC}")
                                else:
                                    print(f"  -> Rapor, {len(report_chunks_text)} adet anlamlı anıya bölündü.")
                                    for i, chunk in enumerate(report_chunks_text):
                                        if len(chunk) > 50:
                                            agent.personal_store.add(text=chunk, topic=note_topic)
                                            agent.knowledge_store.add(chunk)
                                            print(f"  -> Anı {i+1}/{len(report_chunks_text)} '{note_topic}' konusuna ve Ana Beyne eklendi.")
                                    print(f"{Colors.OKGREEN}[Hafıza]: Araştırma sonucu {len(report_chunks_text)} parça halinde başarıyla kaydedildi.{Colors.ENDC}")

                            except Exception as e:
                                print(f"{Colors.FAIL}[System]: Araştırma sonucu hafızaya kaydedilirken bir hata oluştu: {e}{Colors.ENDC}")


                            remaining_lines = lines[1:]
                            f.seek(0)
                            f.truncate()
                            f.writelines(remaining_lines)
                        else:
                            print(f"\n{Colors.WARNING}[System]: Araştırma tamamlanamadı. Konu, bir sonraki deneme için kuyrukta bekletiliyor.{Colors.ENDC}")
                            # Hata durumunda konuyu kuyruktan SİLMEYEREK bir sonraki döngüde tekrar denenmesini sağlıyoruz.
                            # Bu, geçici ağ hataları veya araç sorunları gibi durumlarda görevin kaybolmasını önler.
                            # Eğer sürekli aynı hatayı alıyorsa, kuyruk dosyasını manuel olarak düzenlemek gerekebilir.
                            # Şimdilik, sadece işlemi atlayıp döngüye devam ediyoruz.
                            pass
    except Exception as e:
        print(f"\n{Colors.WARNING}[System]: Araştırma kuyruğu kontrol edilirken hata oluştu: {e}{Colors.ENDC}")

def handle_proactive_assistant(agent: Agent):
    """Kişisel notlardaki yeni konuları proaktif olarak araştırır veya ilgili araçları günceller."""
    try:

        strategies_path = os.path.join(os.path.dirname(__file__), '..', 'research_strategies.json')
        with open(strategies_path, 'r', encoding='utf-8') as f:
            research_strategies = json.load(f)

        research_queue_path = agent.personal_store.queue_path
        if not (os.path.exists(research_queue_path) and os.path.getsize(research_queue_path) > 0):
            return

        with open(research_queue_path, 'r+', encoding='utf-8') as f:
            lines = f.readlines()
            if not lines:
                return

            topic_to_research = lines[0].strip()
            if not topic_to_research:
                remaining_lines = lines[1:]
                f.seek(0)
                f.truncate()
                f.writelines(remaining_lines)
                return

            if not sys.stdout.isatty():
                print(f"\n{Colors.OKCYAN}[Proaktif Asistan]: Etkileşimli olmayan modda çalışıyor. Proaktif görev atlanıyor.{Colors.ENDC}")
                return

            clear_topic_from_queue = True
            topic_lower = topic_to_research.lower()


            is_owasp_topic = "owasp_tehlike" in topic_lower or "owasp_defense" in topic_lower
            sast_tool_exists = "python_sast_scanner" in agent.available_tools
            if is_owasp_topic and sast_tool_exists:
                return


            print(f"\n{Colors.HEADER}[Proaktif Asistan]:{Colors.ENDC} Kişisel notlarınızda yeni bir konu fark ettim: '{topic_to_research}'")
            user_consent = input(f"{Colors.BOLD}Bu konuda derinlemesine bir araştırma yapıp öğrenmemi ister misiniz? (e/h): {Colors.ENDC}").lower()

            if user_consent == 'e':

                strategy_key = "default"
                if any(kw in topic_lower for kw in ['bitcoin', 'ethereum', 'crypto', 'kripto']):
                    strategy_key = "crypto"
                elif any(kw in topic_lower for kw in ['hisse', 'fon', 'borsa', 'eregl', 'thyao']):
                    strategy_key = "stock"
                elif any(kw in topic_lower for kw in ['güvenlik', 'siber', 'zafiyet', 'hacker', 'owasp']):
                    strategy_key = "security"
                elif any(kw in topic_lower for kw in ['felsefe', 'stoacılık', 'varoluşçuluk']):
                    strategy_key = "philosophy"

                strategy = research_strategies[strategy_key]
                tools_to_run = strategy["tools"]
                synthesis_prompt_template = strategy["synthesis_prompt"]

                print(f"{Colors.OKCYAN}💡 Araştırma stratejisi olarak '{strategy_key}' seçildi. Kullanılacak araçlar: {', '.join(tools_to_run)}{Colors.ENDC}")


                tool_results = {}
                all_chunks = []

                def safe_tool_wrapper(tool_name: str, query: str):
                    """Araç fonksiyonunu güvenli bir şekilde çalıştırır."""
                    try:
                        if tool_name not in AVAILABLE_TOOLS:
                            error_msg = f"'{tool_name}' aracı bulunamadı."
                            print(f"{Colors.FAIL}[Araç Hatası]: {error_msg}{Colors.ENDC}")
                            return {"status": "error", "result": error_msg}

                        tool_func = AVAILABLE_TOOLS[tool_name]['func']
                        result = tool_func(query)

                        if isinstance(result, dict) and result.get('status') == 'error':
                            print(f"\n{Colors.FAIL}[Araç Hatası - {tool_name}]: {result.get('message', 'Bilinmeyen bir hata oluştu.')}{Colors.ENDC}")
                        return {tool_name: result}
                    except Exception as e:
                        error_msg = f"'{tool_name}' aracı çalıştırılırken beklenmedik bir istisna oluştu: {e}"
                        print(f"\n{Colors.FAIL}[Araç Çalıştırma Hatası]: {error_msg}{Colors.ENDC}")
                        traceback.print_exc()
                        return {"status": "error", "result": error_msg}

                print(f"{Colors.HEADER}--- Proaktif Çoklu Araç Araştırması Başlatılıyor ---{Colors.ENDC}")
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(tools_to_run)) as executor:
                    future_to_tool = {executor.submit(safe_tool_wrapper, name, topic_to_research): name for name in tools_to_run}
                    for future in concurrent.futures.as_completed(future_to_tool):
                        tool_name = future_to_tool[future]
                        try:
                            result_dict = future.result()
                            if result_dict and result_dict.get("status") != "error":
                                tool_results.update(result_dict)
                                print(f"{Colors.OKGREEN}✅ '{tool_name}' aracı başarıyla tamamlandı.{Colors.ENDC}")
                                # Sonuçtan 'chunks' ayıklama
                                if isinstance(result_dict.get(tool_name), dict):
                                    chunks = result_dict.get(tool_name, {}).get('chunks', [])
                                    if chunks:
                                        all_chunks.extend(chunks)
                            else:
                                print(f"{Colors.FAIL}❌ '{tool_name}' aracı bir hata ile sonuçlandı.{Colors.ENDC}")
                        except Exception as exc:
                            print(f"{Colors.FAIL}❌ '{tool_name}' aracı çalıştırılırken bir istisna oluştu: {exc}{Colors.ENDC}")

                if not tool_results:
                    print(f"\n{Colors.FAIL}[System]: Hiçbir araç başarıyla sonuç döndürmedi. Araştırma tamamlanamadı.{Colors.ENDC}")
                    clear_topic_from_queue = False
                else:
                    # 3. Sonuçları Sentezleme
                    print(f"{Colors.OKBLUE}🌀 Toplanan tüm bilgiler birleştirilip sentezleniyor...{Colors.ENDC}")
                    synthesis_context = ""
                    for tool_name, result_data in tool_results.items():
                        # Gelen verinin içindeki asıl sonucu (result) alalım
                        actual_result = result_data.get(tool_name, {})
                        if isinstance(actual_result, dict):
                             result_text = actual_result.get('result', json.dumps(actual_result, ensure_ascii=False, indent=2))
                        else:
                             result_text = str(actual_result)
                        synthesis_context += f"--- '{tool_name}' Aracından Gelen Bilgiler ---\n{result_text}\n\n"

                    final_synthesis_prompt = f"{synthesis_prompt_template}\n\nİşte araçlardan toplanan ham veriler:\n\n{synthesis_context}"

                    final_report = ask(final_synthesis_prompt, max_new_tokens=4096)

                    # 4. Hafızaya Kaydetme
                    print(f"\n{Colors.HEADER}--- Araştırma Raporu (Özet) ---{Colors.ENDC}")
                    print(final_report[:1000] + "\n... (raporun tamamı hafızaya kaydediliyor) ...")
                    print(f"{Colors.HEADER}---------------------------------{Colors.ENDC}")
                    try:
                        print(f"{Colors.OKBLUE}🌀 Araştırma raporu parçalara ayrılıyor ve hafızaya kaydediliyor...{Colors.ENDC}")
                        note_topic = topic_to_research.split(']')[0].strip('[') if ']' in topic_to_research else "Araştırma Sonucu"

                        report_chunks_text = []
                        if all_chunks:
                            print(f"  -> Araştırmacılardan {len(all_chunks)} adet hazır parça (chunk) alındı.")
                            report_chunks_text = [f"Araştırma Konusu: {item['sub_topic']}\nAraştırma Özeti: {item['summary']}" for item in all_chunks]
                        else:
                            print(f"{Colors.WARNING}[System]: Araştırmacılar 'chunks' döndürmedi. Raporun tamamı parçalanıyor...{Colors.ENDC}")
                            report_chunks_text = [chunk.strip() for chunk in final_report.split('\n\n') if chunk.strip()]

                        if not report_chunks_text:
                            print(f"{Colors.WARNING}[System]: Rapor içeriği boş veya parçalara ayrılamadı.{Colors.ENDC}")
                        else:
                            print(f"  -> Rapor, {len(report_chunks_text)} adet anlamlı anıya bölündü.")
                            for i, chunk in enumerate(report_chunks_text):
                                if len(chunk) > 50:
                                    agent.personal_store.add(text=chunk, metadata=metadata, _add_to_queue=False)
                                    agent.knowledge_store.add(chunk)
                                    print(f"  -> Anı {i+1}/{len(report_chunks_text)} '{note_topic}' konusuna ve Ana Beyne eklendi.")
                            print(f"{Colors.OKGREEN}[Hafıza]: Araştırma sonucu {len(report_chunks_text)} parça halinde başarıyla kaydedildi.{Colors.ENDC}")
                        clear_topic_from_queue = True
                    except Exception as e:
                        print(f"{Colors.FAIL}[System]: Araştırma sonucu hafızaya kaydedilirken bir hata oluştu: {e}{Colors.ENDC}")
                        clear_topic_from_queue = False

            elif user_consent == 'h':
                print(f"{Colors.OKCYAN}[System]: Araştırma isteği iptal edildi. Not hafızada kalmaya devam edecek.{Colors.ENDC}")
                clear_topic_from_queue = True
            else:
                print(f"{Colors.WARNING}[System]: Geçersiz giriş ('{user_consent}'). İşlem iptal edildi. Konu kuyrukta kalmaya devam edecek.{Colors.ENDC}")
                clear_topic_from_queue = False

            if clear_topic_from_queue:
                remaining_lines = lines[1:]
                f.seek(0)
                f.truncate()
                f.writelines(remaining_lines)


    except Exception as e:
        # Hatanın hangi aşamada olduğunu belirtmek daha faydalı olur.
        print(f"\n{Colors.WARNING}[System]: Proaktif asistan döngüsünde beklenmedik bir hata oluştu: {e}{Colors.ENDC}")
        traceback.print_exc()

def handle_note_file_changes(agent: Agent, last_meta_mtime: float) -> float:
    """Kişisel not dosyasındaki değişiklikleri kontrol eder ve indeksi günceller."""
    try:
        personal_notes_path = os.path.join(agent.personal_store.store_path, 'meta.json')
        if os.path.exists(personal_notes_path):
            current_meta_mtime = os.path.getmtime(personal_notes_path)
            if current_meta_mtime > last_meta_mtime:
                print(f"\n{Colors.OKCYAN}[System]: Kişisel not dosyasında değişiklik algılandı. İndeks yeniden oluşturuluyor...{Colors.ENDC}")
                agent.personal_store.rebuild_from_meta()
                last_meta_mtime = current_meta_mtime
                print(f"{Colors.OKGREEN}[System]: Kişisel notlar başarıyla güncellendi.{Colors.ENDC}")
    except Exception as e:
        print(f"\n{Colors.WARNING}[System]: Not dosyası kontrol edilirken hata oluştu: {e}{Colors.ENDC}")
    return last_meta_mtime

def read_text_chunks(file_path: str, chunk_size_chars: int = 4000):
    """
    Bir dosyayı karakter cinsinden belirtilen boyutta parçalar halinde okuyan bir generator.
    Dosyanın tamamını belleğe yüklemez.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            while True:
                chunk = f.read(chunk_size_chars)
                if not chunk:
                    break
                yield chunk
    except FileNotFoundError:
        # Bu hata, çağıran tarafından ele alınmalı.
        raise
    except Exception as e:
        print(f"{Colors.FAIL}Dosya okunurken bir hata oluştu: {e}{Colors.ENDC}")
        raise

def run_map_reduce_summary(file_path: str):
    """
    Büyük metin dosyalarını streaming yaparak ve MapReduce benzeri bir yöntemle özetler.
    Dosyayı tamamen belleğe yüklemez.
    """
    print(f"{Colors.HEADER}--- MapReduce Özetleme Başlatıldı ({os.path.basename(file_path)}) ---{Colors.ENDC}")


    CHUNK_SIZE_CHARS = 4000

    print(f"{Colors.OKBLUE}🌀 Map Aşaması: Dosya okunuyor ve parçalar özetlenmek üzere {config.SUMMARY_MAX_WORKERS} işçi ile havuza gönderiliyor...{Colors.ENDC}")

    def summarize_chunk(chunk_with_index):
        """Bir metin parçasını özetler."""
        index, chunk = chunk_with_index
        try:
            if len(chunk.strip()) < 50:
                return None
            prompt = f"Bu metni 2 paragraflık kısa bir özet haline getir:\n\n---\n{chunk}\n---"
            summary = ask(prompt, max_new_tokens=512)
            print(f"{Colors.OKGREEN}   - Parça {index + 1} özeti tamamlandı.{Colors.ENDC}")
            return summary.strip()
        except Exception as e:
            print(f"{Colors.FAIL}   - Parça {index + 1} işlenirken hata: {e}{Colors.ENDC}")
            return f"[Hata: Parça {index + 1} özetlenemedi]"

    try:

        chunk_generator = read_text_chunks(file_path, chunk_size_chars=CHUNK_SIZE_CHARS)
        indexed_chunk_generator = enumerate(chunk_generator)

        summaries = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.SUMMARY_MAX_WORKERS) as executor:

            results_iterator = executor.map(summarize_chunk, indexed_chunk_generator)
            summaries = [summary for summary in results_iterator if summary]

    except Exception as e:

        print(f"{Colors.FAIL}Özetleme işlemi sırasında bir hata oluştu: {e}{Colors.ENDC}")
        traceback.print_exc()
        return


    print(f"{Colors.OKBLUE}🌀 Reduce Aşaması: Nihai rapor oluşturuluyor...{Colors.ENDC}")

    valid_summaries = [s for s in summaries if "[Hata:" not in s]
    if not valid_summaries:
        print(f"{Colors.FAIL}Hiçbir parça başarıyla özetlenemediği için nihai rapor oluşturulamıyor.{Colors.ENDC}")
        return

    combined_summaries = "\n\n---\n\n".join([f"Bölüm Özeti:\n{s}" for s in valid_summaries])
    final_prompt = f"Elimde bir metnin farklı bölümlerinden alınmış şu özetler var:\n\n---\n{combined_summaries}\n---\n\nBu özetleri kullanarak, metnin tamamını kapsayan, ana başlık ve anlamlı alt başlıklar içeren, tutarlı ve akıcı bir final raporu yaz. Rapor profesyonel bir tonda olmalı."

    try:
        final_report = ask(final_prompt, max_new_tokens=4096)
        print(f"{Colors.HEADER}\n--- Nihai Özet Raporu ---{Colors.ENDC}")
        print(final_report)
        print(f"{Colors.HEADER}--------------------------{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.FAIL}Nihai rapor oluşturulurken bir hata oluştu: {e}{Colors.ENDC}")

def main():
    global PHILOSOPHY_MODE_ENABLED
    print(f"{Colors.HEADER}Yapay Zeka Asistanı başlatılıyor... Lütfen bekleyin.{Colors.ENDC}")


    print(f"{Colors.OKBLUE}🔹 Modeller ve altyapı yükleniyor (Bu işlem uzun sürebilir)...{Colors.ENDC}")
    from agent.models.llm import download_model, load_model
    download_model()
    load_model()
    print(f"{Colors.OKGREEN}✅ Modeller başarıyla yüklendi.{Colors.ENDC}")


    initial_tools, initial_non_cacheable = _load_all_tools()


    agent_instance = Agent(
        available_tools=initial_tools,
        non_cacheable_tools=initial_non_cacheable,
        reload_tools_func=get_tools_and_reload_function
    )

    print(f"\n{Colors.HEADER}Asistan hazır. Çıkmak için 'q', agent mantığını yeniden yüklemek için '/reload' yazın.{Colors.ENDC}")
    print(f"{Colors.OKCYAN}Diğer komutlar: /felsefe, /ozetle <dosya_yolu>, /konularim, /notlarim <konu>, /not <konu> <notunuz>{Colors.ENDC}")

    notes_file_path_str = os.path.join(agent_instance.personal_store.store_path, 'meta.json')
    print(f"{Colors.OKCYAN}Notlarınızı manuel olarak '{notes_file_path_str}' dosyasından düzenleyebilirsiniz. Değişiklikler otomatik algılanacaktır.{Colors.ENDC}")

    personal_notes_path = os.path.join(agent_instance.personal_store.store_path, 'meta.json')
    last_meta_mtime = os.path.getmtime(personal_notes_path) if os.path.exists(personal_notes_path) else 0

    try:
        while True:
            handle_proactive_assistant(agent_instance)
            last_meta_mtime = handle_note_file_changes(agent_instance, last_meta_mtime)

            user_input = input(f"{Colors.BOLD}\nSen: {Colors.ENDC}")

            if user_input.lower() in ['q', 'quit', 'exit']:
                print("Asistan kapatılıyor...")
                break

            if user_input.lower() == '/reload':
                NewAgentClass = reload_agent_logic()
                if NewAgentClass:
                    print(f"{Colors.OKCYAN}Agent örneği yeniden oluşturuluyor...{Colors.ENDC}")
                    # Agent örneğini yeniden oluştur
                    agent_instance = NewAgentClass(
                        available_tools=AVAILABLE_TOOLS,
                        non_cacheable_tools=NON_CACHEABLE_TOOLS,
                        reload_tools_func=get_tools_and_reload_function
                    )
                    # Araçları yeniden yükle ve yeni örneği güncelle
                    get_tools_and_reload_function(agent_instance)
                continue

            if user_input.lower().startswith('/ozetle'):
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    print(f"{Colors.WARNING}Kullanım: /ozetle <dosya_yolu>{Colors.ENDC}")
                else:
                    file_path = parts[1].strip().strip("'\"")
                    expanded_path = os.path.expanduser(file_path)

                    if not os.path.exists(expanded_path):
                        print(f"{Colors.FAIL}Hata: '{expanded_path}' dosyası bulunamadı.{Colors.ENDC}")
                    elif os.path.getsize(expanded_path) == 0:
                        print(f"{Colors.WARNING}Uyarı: '{expanded_path}' dosyası boş.{Colors.ENDC}")
                    else:
                        run_map_reduce_summary(expanded_path)
                continue

            if user_input.lower() == '/ozetle_hafiza':
                if 'memory_consolidator' in agent_instance.available_tools:
                    tool_func = agent_instance.available_tools['memory_consolidator']['func']
                    print(f"{Colors.OKCYAN}Hafıza birleştirme aracı manuel olarak tetikleniyor...{Colors.ENDC}")
                    tool_func(agent_instance=agent_instance)
                else:
                    print(f"{Colors.FAIL}Hata: 'memory_consolidator' aracı bulunamadı. Araçların doğru yüklendiğinden emin olun.{Colors.ENDC}")
                continue

            if user_input.lower().startswith('/notlarim'):
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    print(f"{Colors.WARNING}Kullanım: /notlarim <konu>. Kayıtlı tüm konuları görmek için /konularim yazabilirsiniz.{Colors.ENDC}")
                    all_topics = agent_instance.personal_store.get_all_topics()
                    if all_topics:
                        print(f"{Colors.OKCYAN}Mevcut Konular:{Colors.ENDC}")
                        for topic in all_topics:
                            print(f"- {topic}")
                else:
                    topic_to_show = parts[1].strip()
                    notes = agent_instance.personal_store.get_notes_by_topic(topic_to_show)
                    if not notes:
                        print(f"{Colors.OKCYAN}'{topic_to_show}' konusunda hiç not bulunamadı.{Colors.ENDC}")
                    else:
                        print(f"{Colors.OKCYAN}'{topic_to_show}' Konusundaki Notlar:{Colors.ENDC}")
                        for note in notes:
                            print(f"- {note.get('text')}")
                continue

            if user_input.lower() == '/konularim':
                all_topics = agent_instance.personal_store.get_all_topics()
                if not all_topics:
                    print(f"{Colors.OKCYAN}Henüz kayıtlı bir konu başlığı yok. '/not <konu> <notunuz>' komutuyla yeni bir not ekleyebilirsiniz.{Colors.ENDC}")
                else:
                    print(f"{Colors.OKCYAN}Kayıtlı Konular:{Colors.ENDC}")
                    for topic in all_topics:
                        print(f"- {topic}")
                continue

            if user_input.lower() == '/felsefe':
                PHILOSOPHY_MODE_ENABLED = not PHILOSOPHY_MODE_ENABLED
                status = "AKTİF" if PHILOSOPHY_MODE_ENABLED else "DEVRE DIŞI"
                print(f"{Colors.OKCYAN}Felsefe Modu şimdi {status}.{Colors.ENDC}")
                continue

            if user_input.startswith('/not'):
                command_parts = user_input.split(maxsplit=2)
                if len(command_parts) < 3:
                    print(f"{Colors.WARNING}Hatalı format. Kullanım: /not <konu> <kaydedilecek metin>{Colors.ENDC}")
                else:
                    topic = command_parts[1]
                    note_content = command_parts[2]

                    def add_new_note():
                        """Yeni notu ekler ve durumu günceller."""
                        agent_instance.personal_store.add(text=note_content, metadata={'topic': topic, 'type': 'user_note'})
                        print(f"{Colors.OKGREEN}[Kişisel Not]: '{note_content}' notu '{topic}' konusuna eklendi.{Colors.ENDC}")
                        return os.path.getmtime(personal_notes_path) if os.path.exists(personal_notes_path) else last_meta_mtime

                    # Mevcut notları kontrol et
                    if agent_instance.personal_store.get_notes_by_topic(topic):
                        choice = input(f"{Colors.BOLD}'{topic}' konusunda zaten notlar var. Üzerine yazmak için (g), yeni not olarak eklemek için (y), iptal için herhangi bir tuşa basın: {Colors.ENDC}").lower()
                        if choice == 'g':
                            agent_instance.personal_store.delete_by_topic(topic)
                            last_meta_mtime = add_new_note()
                        elif choice == 'y':
                            last_meta_mtime = add_new_note()
                        else:
                            print(f"{Colors.FAIL}İşlem iptal edildi.{Colors.ENDC}")
                    else:
                        last_meta_mtime = add_new_note()
                continue

            if PHILOSOPHY_MODE_ENABLED:
                run_philosophy_mode(user_input, agent_instance)
            else:
                agent_instance.run(user_input)

    except KeyboardInterrupt:
        print("\nAsistan kapatılıyor...")
    finally:
        if 'agent_instance' in locals() and hasattr(agent_instance, 'knowledge_store'):
            agent_instance.knowledge_store.close()

if __name__ == "__main__":
    main()
    