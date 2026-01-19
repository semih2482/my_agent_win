import json
import re
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from agent.core.agent import smart_truncate, extract_json # Merkezi fonksiyonları import et

def _extract_keywords_for_search(topic: str, llm_ask_func, colors) -> str:
    """Extracts concise search keywords from a long topic description."""
    print(f"{colors.OKBLUE}🌀 Adım 1.5: Arama için anahtar kelimeler çıkarılıyor...{colors.ENDC}")
    prompt = f"""Analyze the following user request and extract a concise search query of 3-5 keywords. This query will be used for a web search.

User Request: "{topic}"

Search Query:"""
    try:
        keywords = llm_ask_func(prompt, max_new_tokens=64)
        print(f"{colors.OKCYAN}   Çıkarılan Anahtar Kelimeler: {keywords}{colors.ENDC}")
        return keywords.strip().replace('"', '')
    except Exception as e:
        print(f"{colors.WARNING}   Anahtar kelime çıkarılamadı, orijinal konu kullanılıyor. Hata: {e}{colors.ENDC}")
        return topic

def deep_research_and_learn(topic: str, llm_ask_func, search_func, knowledge_store, colors):
    """
    Bir konu hakkında proaktif, çok adımlı araştırma yapar ve öğrenir.
    """
    print(f"{colors.HEADER}--- Proaktif Araştırmacı Başlatıldı: '{topic}' ---{colors.ENDC}")

    # 1. Adım: Mevcut bilgiyi kontrol et
    print(f"{colors.OKBLUE}🌀 1. Adım: Mevcut bilgi tabanı taranıyor...{colors.ENDC}")
    existing_knowledge = knowledge_store.search(topic, top_k=5)
    if existing_knowledge and any(item[1] < 0.2 for item in existing_knowledge): # Eşik değeri (similarity score) ayarlanabilir
        print(f"{colors.OKGREEN}✅ Bu konu hakkında zaten yeterli bilgi mevcut. Araştırma atlanıyor.{colors.ENDC}")
        return "\n".join([item[0] for item in existing_knowledge if item[1] < 0.2])

    # Adım 1.5: Aramayı iyileştirmek için anahtar kelimeleri çıkar
    search_topic = _extract_keywords_for_search(topic, llm_ask_func, colors)

    # 2. Adım: İlk Genel Araştırma
    print(f"{colors.OKBLUE}🌀 2. Adım: Konu hakkında ilk genel araştırma yapılıyor...{colors.ENDC}")
    initial_summary_raw = search_func(search_topic)

    # Hata durumunda (araç None döndürdüğünde) işlemi sonlandır.
    if initial_summary_raw is None:
        print(f"{colors.FAIL}İlk araştırma aracı bir sonuç döndürmedi veya bir hatayla karşılaştı. Görev sonlandırılıyor.{colors.ENDC}")
        return None

    # Araç sözlük döndürürse (örn: {'status': 'success', 'result': '...'}), asıl sonucu çıkar
    if isinstance(initial_summary_raw, dict):
        initial_summary = initial_summary_raw.get('result', str(initial_summary_raw))
    else:
        initial_summary = str(initial_summary_raw)

    print(f"{colors.OKCYAN}   İlk Özet: {initial_summary[:250]}...{colors.ENDC}")

    # 3. Adım: Alt Başlıkları ve Soruları Belirleme
    print(f"{colors.OKBLUE}🌀 3. Adım: Araştırmayı derinleştirmek için alt başlıklar belirleniyor...{colors.ENDC}")
    sub_query_prompt = f"""Kullanıcı İsteği: "{topic}"

Bu istekle doğrudan ilgili, konuyu derinlemesine anlamak için araştırılması gereken 3 ila 5 adet alt başlık, soru veya anahtar kavram belirle.
İsteğin konusu dışına kesinlikle çıkma. Yanıtın, başka hiçbir metin veya açıklama olmadan, SADECE JSON listesinin kendisi olmalıdır.

**ÖRNEK:**
```json
[
  "OWASP Top 10 nedir?",
  "Yaygın web uygulama zafiyetleri",
  "Etik hackerlığa nasıl başlanır?",
  "Siber güvenlik haberleri için en iyi bloglar"
]
```

İLK ARAŞTIRMA ÖZETİ:
"{initial_summary}"""

    try:
        response = llm_ask_func(sub_query_prompt, max_new_tokens=2048)

        # Merkezi ve daha gelişmiş JSON ayıklama fonksiyonunu kullan
        # Bu fonksiyon onarım ve LLM ile düzeltme yeteneklerine sahip.
        sub_queries = extract_json(response)

        if not sub_queries:
            print(f"{colors.WARNING}   Could not get a valid JSON list from LLM, attempting to clean the response...{colors.ENDC}")
            # Fallback: Clean the response and split by lines
            cleaned_response = response.replace('`', '').replace('json', '').strip()
            potential_queries = [line.strip(' -*,"[]') for line in cleaned_response.split('\n')]
            sub_queries = [q for q in potential_queries if len(q) > 5]


        if not sub_queries:
            raise ValueError(f"LLM yanıtından geçerli alt başlıklar çıkarılamadı. Yanıt: {response}")

        print(f"{colors.OKCYAN}   Belirlenen Alt Başlıklar: {', '.join(sub_queries)}{colors.ENDC}")
    except Exception as e:
        print(f"{colors.FAIL}Alt başlıklar belirlenirken hata oluştu: {e}. Görev sonlandırılıyor.{colors.ENDC}")
        return

    # 4. Adım: Derinlemesine Araştırma
    print(f"{colors.OKBLUE}🌀 4. Adım: Her bir alt başlık için derinlemesine araştırma yapılıyor...{colors.ENDC}")
    print(f"{colors.OKCYAN}   {len(sub_queries)} alt başlık için paralel arama başlatıldı...{colors.ENDC}")
    deep_dive_results = {}
    with ThreadPoolExecutor(max_workers=len(sub_queries) or 1) as executor:
        future_to_query = {executor.submit(search_func, query): query for query in sub_queries}
        for future in as_completed(future_to_query):
            query = future_to_query[future]
            try:
                result = future.result()
                # Araç sözlük döndürürse (örn: {'status': 'success', 'result': '...'}), asıl sonucu çıkar
                if isinstance(result, dict):
                    deep_dive_results[query] = result.get('result', str(result))
                else:
                    deep_dive_results[query] = str(result)
                print(f"   - ✅ '{query}' araştırması tamamlandı.")
            except Exception as exc:
                print(f"{colors.FAIL}   - ❌ '{query}' araştırması sırasında hata: {exc}{colors.ENDC}")
                deep_dive_results[query] = f"Bu alt başlık araştırılırken bir hata oluştu: {exc}"
    print(f"{colors.OKGREEN}   Paralel araştırma tamamlandı.{colors.ENDC}")

    # 5. Adım: Bilgiyi Sentezleme ve Eleştirel Özet Oluşturma
    print(f"{colors.OKBLUE}🌀 5. Adım: Toplanan tüm bilgiler birleştirilip sentezleniyor...{colors.ENDC}")

    # Token limitini aşmamak için toplanan bilgileri kontrol et ve gerekirse özetle
    # Kabaca bir limit belirleyelim (örn: 12000 karakter ~ 3000 token)
    # Bu, prompt'un geri kalanı için bolca yer bırakır. (MapReduce'un Map adımı)
    summarized_initial = smart_truncate(initial_summary, 10000, context_prompt=topic)
    summarized_deep_dives = {}
    for query, result in deep_dive_results.items():
        summarized_deep_dives[query] = smart_truncate(result, 10000, context_prompt=query)

    total_text_len = len(summarized_initial) + sum(len(v) for v in summarized_deep_dives.values())
    print(f"{colors.OKCYAN}   Sentezlenecek toplam metin boyutu (yaklaşık): {total_text_len} karakter.{colors.ENDC}")

    # MapReduce'un Reduce adımı
    synthesis_prompt = f"""Bir araştırma analisti olarak görev yapıyorsun. '{topic}' konusu hakkında aşağıdaki bilgileri topladın. Bu bilgileri birleştirerek kapsamlı, akıcı ve iyi yapılandırılmış bir final raporu oluştur. Varsa farklı bakış açılarını veya çelişkili bilgileri de belirt. Sadece oluşturduğun raporu yaz.

İLK ÖZET:
{summarized_initial}

DETAYLI ARAŞTIRMALAR:
"""
    for query, result in summarized_deep_dives.items():
        synthesis_prompt += f"- Alt Başlık '{query}':\n{result}\n\n"

    # max_new_tokens'ı 4096 olarak güncelleyerek daha kapsamlı raporlara izin verelim.
    final_report = llm_ask_func(synthesis_prompt, max_new_tokens=4096)
    print(f"{colors.OKGREEN}✅ Araştırma tamamlandı ve final raporu oluşturuldu.{colors.ENDC}")

    # 6. Adım: Öğrenme (Kalıcı Hafızaya Kaydetme)
 #   print(f"{colors.OKBLUE}🌀 6. Adım: Öğrenilen bilgiler kalıcı hafızaya kaydediliyor...{colors.ENDC}")
 #   knowledge_to_save = f"""Proaktif Araştırma Raporu: {topic}

#{final_report}
#"""
 ##   try:
   ##     knowledge_store.add(knowledge_to_save)
     ##   print(f"{colors.OKGREEN}✅ Bilgiler başarıyla hafızaya kaydedildi.{colors.ENDC}")
   # except Exception as e:
    #    print(f"{colors.FAIL}Hafızaya kaydetme sırasında hata: {e}{colors.ENDC}")

    return final_report
