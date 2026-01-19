# agent/tools/internet_search.py
from typing import Dict, Any, Union, List
from ddgs.ddgs import DDGS
from agent.models.llm import ask
import concurrent.futures


CRYPTO_KEYWORDS = [
    'kripto', 'crypto', 'bitcoin', 'ethereum', 'btc', 'eth', 'altcoin',
    'solana', 'xrp', 'doge', 'avax', 'ada', 'dot', 'matic', 'binance', 'coinbase'
]

CRYPTO_AUTHORITY_SITES = [
    "coindesk.com", "cointelegraph.com", "theblockcrypto.com", "decrypt.co",
    "messari.io", "glassnode.com", "defillama.com", "etherscan.io", "bscscan.com"
]

TOOL_INFO = {
    "name": "internet_search",
    "description": "İnternette bir konu hakkında genel bir araştırma yapmak, güncel bilgi bulmak veya bir soruyu cevaplamak için kullanılır. Sonuçları özetleyerek verir.",
    "cacheable": False,
    "args_schema": {"query": "string"}
}

def run(args: Union[dict, str], agent_instance=None) -> dict:
    """Wrapper function to call the appropriate search function."""
    if isinstance(args, str):
        query = args
    elif isinstance(args, dict):
        query = args.get('query')
    else:
        return {"status": "error", "message": f"Invalid input type for 'args': {type(args)}. Expected dict or str."}

    if not query:
        return {"status": "error", "message": "Missing 'query' in arguments."}

    # Agent'ın kendi LLM fonksiyonunu kullanmayı tercih et (varsa)
    llm_func = ask
    if agent_instance and hasattr(agent_instance, 'ask'):
        # Bu varsayımsal bir durum, agent'ın 'ask' metodu varsa onu kullanır.
        # Mevcut yapıda agent.run() var ama doğrudan agent.ask() yok. Bu geleceğe dönük bir iyileştirme.
        pass # llm_func zaten 'ask' olarak ayarlandı.
    return search_and_summarize(query, llm_ask_function=llm_func)

def search_for_snippets(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    DuckDuckGo kullanarak internette arama yapar ve LLM ile özetleme yapmadan
    doğrudan arama sonuçlarındaki snippet'leri (metin parçacıkları) döndürür.
    Bu araç, `search_and_summarize`'a göre çok daha hızlı bir alternatiftir.
    """
    print(f"🔎 DuckDuckGo ile snippet aranıyor: '{query}' (max {max_results} sonuç)")
    try:
        results = []
        with DDGS(timeout=20) as ddgs:
            search_results = ddgs.text(query, max_results=max_results)
            if search_results:
                results = list(search_results)

        if not results:
            return {"status": "empty", "message": "Arama sonucu bulunamadı."}

        # Sonuçları ve kaynakları topla
        sources = []
        content_snippets = []
        for item in results:
            snippet = item.get("body")
            if snippet:
                content_snippets.append(f"--- KAYNAK: {item.get('title')}\nURL: {item.get('href')}\n{snippet}")
                sources.append({"url": item.get('href'), "title": item.get('title'), "snippet": snippet})

        if not content_snippets:
             return {"status": "empty", "message": "Arama sonucu bulunamadı (içerik yok)."}

        # LLM çağırmadan, birleştirilmiş snippet metnini ve kaynakları döndür
        return {"status": "success", "result": "\n\n".join(content_snippets), "sources": sources}

    except Exception as e:
        return {"status": "error", "message": f"Snippet araması sırasında hata: {e}"}


def search_and_summarize(query: str, llm_ask_function=None, max_results: int = 5) -> Dict[str, Any]:
    """
    DuckDuckGo kullanarak internette arama yapar.
    Eğer sorgu kripto para ile ilgiliyse, hem otorite sitelerde hem de genel internette
    paralel arama yaparak sonuçları birleştirir ve daha sonra özetler.
    """
    llm_ask_function = llm_ask_function or ask
    try:
        # Kripto Sorgusu Tespiti ve İki Yönlü Arama
        is_crypto_query = any(keyword in query.lower() for keyword in CRYPTO_KEYWORDS)
        search_queries = []

        if is_crypto_query:
            print(f"💡 Kripto para sorgusu tespit edildi. İki yönlü arama başlatılıyor...")
            # Otorite siteler için arama sorgusu
            authority_site_string = " OR ".join([f"site:{site}" for site in CRYPTO_AUTHORITY_SITES])
            authority_query = f'{query} ({authority_site_string})'
            search_queries.append(("Otorite Arama", authority_query))
            # Genel görüşler için standart arama
            search_queries.append(("Genel Arama", query))
        else:
            print(f"🔎 Standart arama yapılıyor: '{query}'")
            search_queries.append(("Standart Arama", query))

        all_results = []
        seen_urls = set()

        def perform_search(search_type: str, q: str):
            """Paralel arama için yardımcı fonksiyon."""
            print(f"  -> {search_type} başlatılıyor: '{q}'")
            with DDGS(timeout=20) as ddgs:
                return list(ddgs.text(q, max_results=max_results))

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(search_queries)) as executor:
            future_to_query = {executor.submit(perform_search, stype, q): (stype, q) for stype, q in search_queries}
            for future in concurrent.futures.as_completed(future_to_query):
                try:
                    results_list = future.result()
                    if results_list:
                        for item in results_list:
                            url = item.get('href')
                            if url and url not in seen_urls:
                                all_results.append(item)
                                seen_urls.add(url)
                except Exception as exc:
                    stype, q = future_to_query[future]
                    print(f"'{stype}' araması sırasında hata oluştu: {exc}")

        if not all_results:
            return {"status": "empty", "message": "Arama sonucu bulunamadı."}

        print(f"🔗 {len(all_results)} adet birleştirilmiş arama özeti (snippet) bulundu. LLM ile analiz ediliyor...")


        sources = []
        content_snippets = []
        for item in all_results:
            snippet = item.get("body")
            if snippet:
                content_snippets.append(f"---\nKAYNAK: {item.get('title')}\nURL: {item.get('href')}\nÖZET: {snippet}\n---")
                sources.append({"url": item.get('href'), "title": item.get('title'), "snippet": snippet})

        if not content_snippets:
             return {"status": "empty", "message": "Arama sonucu bulunamadı (içerik yok)."}

        # Tüm snippet'leri tek bir metinde birleştir
        all_snippets = "\n\n".join(content_snippets)

        # Tek bir prompt ile LLM'e sor (Prompt'u değiştirmedik, çünkü hala aynı işi yapıyor)
        combine_prompt = (
            f"Kullanıcının orijinal sorusu şudur: '{query}'\n\n"
            "Aşağıda, bu soruyla ilgili farklı internet kaynaklarından toplanmış özetler bulunmaktadır. "
            "Bu kaynakları kullanarak, kullanıcının sorusuna nihai, tutarlı ve kapsamlı bir cevap oluştur.\n\n"
            "KURALLAR:\n"
            "1. Cevabın sadece sorulan soruyla ilgili olsun.\n"
            "2. Farklı özetlerdeki bilgileri birleştirerek tutarlı bir metin oluştur.\n"
            "3. 'Özet olarak', 'sonuç olarak' gibi ifadelerle başlama, doğrudan cevabı ver.\n"
            "4. Cevabını, sanki tüm bilgiyi kendin biliyormuşsun gibi akıcı bir dille yaz.\n"
            "5. Eğer metinlerde çelişkili bilgiler varsa, bu çelişkiyi belirt.\n\n"
            f"--- KAYNAKLAR ---\n{all_snippets}\n\n"
            "ÖNEMLİ: Cevabı oluşturmadan önce yukarıdaki TÜM kaynakları dikkate aldığından emin ol.\n"
            "KULLANICININ SORUSUNA YÖNELİK, KAPSAMLI CEVAP:"
        )

        combined = llm_ask_function(combine_prompt, max_new_tokens=1024)

        return {"status": "success", "result": combined, "sources": sources}

    except Exception as e:
        return {"status": "error", "message": f"Arama sırasında hata: {e}"}


def search_urls(query, max_results=5) -> Dict[str, Any]:
    """Belirtilen sorgu için URL'leri arar ve bir liste olarak döndürür."""
    try:
        with DDGS(timeout=20) as ddgs:
            # ddgs.text() bize 'title', 'href' ve 'body' (snippet) içeren bir dict listesi verir
            search_results = ddgs.text(query, max_results=max_results)
            if not search_results:
                 return {"status": "empty", "message": "Sonuç bulunamadı."}

            results = [
                {"index": i + 1, "url": item.get('href'), "title": item.get('title')}
                for i, item in enumerate(search_results)
            ]
        return {"status": "success", "result": results}
    except Exception as e:
        return {"status": "error", "message": f"Arama sırasında hata oluştu: {e}"}