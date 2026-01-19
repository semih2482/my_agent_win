import ast
import json
import time
from typing import Dict, Any
import re
from agent.models.llm import ask, config
import concurrent.futures
from agent.tools.internet_search import search_urls
from agent.core.agent import extract_json # Gelişmiş JSON ayıklayıcıyı import et
from agent.tools.web_reader import read_url

TOOL_INFO = {
    "name": "critical_web_researcher",
    "description": "Belirli bir konu hakkında, akademik ve güvenilir kaynaklara öncelik vererek derinlemesine ve eleştirel bir internet araştırması yapar. Sonuçları sentezler, çelişkileri belirtir ve güvenilir kaynaklara dayanarak bir özet sunar."
}

def _truncate_text(text: str, max_chars: int = 8000) -> str:
    """Truncates text to a maximum character count, trying to preserve whole sentences."""
    if len(text) <= max_chars:
        return text

    # Find the last period within the character limit
    truncated = text[:max_chars]
    last_period = truncated.rfind('.')
    if last_period != -1:
        return truncated[:last_period + 1]
    return truncated

def _get_authority_sites_for_topic(topic: str) -> list:
    """
    Determines the most reliable (authority) 3-5 websites for the given topic.
    This ensures the research focuses on authorities rather than irrelevant blogs.
    """
    print(f"  -> Determining authority sites for: '{topic}'")
    prompt = f"""
    You are an expert academic and web researcher. Your task is to identify the TOP 8 most authoritative and reliable websites (domains) for the topic: "{topic}".

    **Principles for Selection:**
    1.  **Primary Sources:** Prioritize government agencies, international standards bodies, and major academic institutions.
    2.  **Reputable News & Analysis:** Highly respected, globally recognized news organizations.
    3.  **Avoid Low-Quality Sources:** Explicitly exclude personal blogs, forums, and marketing-driven sites.

    **CRITICAL INSTRUCTIONS:**
    - Your response **MUST** be a single, valid JSON list of strings.
    - Your response **MUST NOT** contain any text other than the JSON list.
    - Do **NOT** use markdown like ```json.
    - The output should start with `[` and end with `]`.

    **Examples:**
    - Topic: "web security vulnerabilities" -> ["owasp.org", "mitre.org", "nist.gov", "sans.org", "bleepingcomputer.com"]
    - Topic: "climate change causes and effects" -> ["ipcc.ch", "nasa.gov", "noaa.gov", "royalsociety.org", "un.org"]

    Topic: "{topic}"
    VALID JSON RESPONSE:
    """
    try:
        response_str = ask(prompt, max_new_tokens=256)
        sites = extract_json(response_str) # Gelişmiş ayıklayıcıyı kullan
        if isinstance(sites, list) and all(isinstance(s, str) for s in sites):
            print(f"  -> Found Authority Sites: {sites}")
            return sites

    except (json.JSONDecodeError, Exception) as e:
        print(f"  -> Could not determine authority sites, will use default search. Error: {e}")

    # If there's an error, fall back to a robust default set
    return ["wikipedia.org", "reuters.com", "apnews.com", "google.scholar.com"]

def _research_sub_topic(sub_topic: str, authority_sites: list) -> Dict[str, Any]:
    """
    Researches a single sub-topic, summarizes it, and collects sources.
    This function is designed to be run in parallel.
    """
    print(f"\n🔎 Researching sub-topic: '{sub_topic}'")
    try:
        # Step 2a: Find URLs with an enhanced query on authority sites
        if not authority_sites:
            authority_sites = ["wikipedia.org", "reuters.com", "apnews.com"]

        site_query_string = " OR ".join([f"site:{site.strip()}" for site in authority_sites])
        enhanced_query = f'{sub_topic} ({site_query_string})'
        print(f"  -> Dynamic authority query: '{enhanced_query}'")
        search_results = search_urls(enhanced_query, max_results=2)

        # Fallback: If authority search yields no results, do a general search
        if search_results.get("status") != "success" or not search_results.get("result"):
            print(f"  -> Authority search failed for '{sub_topic}'. Falling back to general search.")
            search_results = search_urls(sub_topic, max_results=2)

        if search_results.get("status") != "success" or not search_results.get("result"):
            print(f"  -> No URLs found for '{sub_topic}' after fallback: {search_results.get('message')}")
            return {"summary": None, "sources": []}

        urls = [res["url"] for res in search_results.get("result", [])]
        page_summaries = []
        sub_topic_sources = []

        # Step 2b: Read and summarize each URL (with retry mechanism)
        for url in urls:
            content = None
            max_read_retries = 3
            for attempt in range(max_read_retries):
                print(f"  -> Reading and summarizing: {url} (Attempt {attempt + 1}/{max_read_retries})")
                content = read_url(url)
                if content and not content.startswith("URL read error"):
                    break
                print(f"  -> Failed to read content. Retrying in 2 seconds...")
                time.sleep(2)

            if content and not content.startswith("URL read error"):
                truncated_content = _truncate_text(content)
                summarize_prompt = f"""
                Analyze the following text in the context of '{sub_topic}'.
                Extract the key arguments, findings, and evidence.
                Provide a concise, analytical summary of 3-4 sentences.

                Text:
                {truncated_content}
                """
                summary = ask(summarize_prompt, max_new_tokens=512)
                page_summaries.append(summary)
                sub_topic_sources.append({"url": url, "title": sub_topic, "snippet": summary[:150]})
                print(f"  -> Summarized: {url}")
            else:
                print(f"  -> Could not read content: {url}")
                sub_topic_sources.append({"url": url, "title": sub_topic, "snippet": "[Content could not be read or was empty]"})

        # Step 2c: Combine page summaries into a sub-topic summary (Reduce)
        if page_summaries:
            combined_page_summaries = "\n\n---\n\n".join(page_summaries)
            sub_topic_summary_prompt = f"""
            Synthesize the following summaries into a single, coherent paragraph for the sub-topic '{sub_topic}'.
            Identify the main theme and integrate the key points smoothly.

            Summaries:
            {combined_page_summaries}

            Synthesized Sub-Topic Summary:
            """
            final_sub_topic_summary = ask(sub_topic_summary_prompt, max_new_tokens=1024)
            print(f"  -> Completed summary for sub-topic '{sub_topic}'.")
            return {"summary": {"sub_topic": sub_topic, "summary": final_sub_topic_summary}, "sources": sub_topic_sources}

    except Exception as e:
        print(f"  -> Error while researching '{sub_topic}': {str(e)}")

    return {"summary": None, "sources": []}

def run(args: str | dict, agent_instance=None) -> dict:
    """
    Performs an in-depth, hierarchical web investigation on a given topic by reading web pages.
    This function operates based on the 'Hierarchical Summarization' logic.
    """
    topic = ""
    if isinstance(args, str):
        topic = args
    elif isinstance(args, dict):
        # 'query' veya 'topic' anahtarlarını kontrol et, esnekliği artır
        topic = args.get("query", args.get("topic"))

    if not topic or not topic.strip():
        return {"status": "error", "message": "The 'topic' for research cannot be empty. Please provide a specific topic."}

    print(f"🔬 Critical Researcher: Initiating in-depth research for '{topic}'...")
    authority_sites = _get_authority_sites_for_topic(topic)
    # Step 1: Decompose the Topic into Sub-Topics (with Retry Mechanism)
    sub_topics = None
    max_retries = 3
    for attempt in range(max_retries):
        print(f"  -> Attempt {attempt + 1}/{max_retries} to determine sub-topics...")
        sub_topics_prompt = f"""
        You are an expert researcher. Your task is to break down the main topic, "{topic}", into 3-4 distinct, comprehensive, and non-overlapping sub-topics for a detailed research report.

        **CRITICAL INSTRUCTIONS:**
        1.  Cover the most critical aspects of the main topic.
        2.  Ensure each sub-topic is distinct and avoids overlap.
        3.  The response **MUST** be a single, valid JSON object.
        4.  The response **MUST NOT** contain any text other than the JSON object.
        5.  Do **NOT** use markdown like ```json.
        6.  The output must be in the format: `{{"sub_topics": ["topic1", "topic2", ...]}}`

        **Example for a different topic ("AI job market impact"):**
        `{{"sub_topics": ["Definition and types of AI impacting jobs", "Analysis of jobs being automated or augmented by AI", "New job roles created by the AI industry", "Economic and social consequences for the workforce"]}}`

        Main Topic: "{topic}"
        VALID JSON RESPONSE:
        """
        try:
            sub_topics_str = ask(sub_topics_prompt, max_new_tokens=1024)
            data = extract_json(sub_topics_str) # Gelişmiş ayıklayıcıyı kullan
            sub_topics = data.get("sub_topics") if isinstance(data, dict) else None

            if isinstance(sub_topics, list) and sub_topics:
                print(f"  -> Successfully identified sub-topics: {sub_topics}")
                break
            else:
                print(f"  -> Attempt {attempt + 1} failed: Could not parse the list. Response: {sub_topics_str}")
                sub_topics = None
        except (json.JSONDecodeError, Exception) as e:
            print(f"  -> Attempt {attempt + 1} failed with an exception: {e}. Response: {sub_topics_str}")
            sub_topics = None

    if not sub_topics:
        print(f"  -> Could not generate sub-topics after {max_retries} attempts. Falling back to using the main topic.")
        sub_topics = [topic]

    # Adım 2 & 3: Her alt konu için URL'leri bul, oku ve özetle (Paralel Map-Reduce Aşaması)
    sub_topic_summaries = []
    all_sources = []
    # ThreadPoolExecutor kullanarak alt konu araştırmalarını paralelleştir
    # Worker sayısı config dosyasından alınır.
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.RESEARCHER_MAX_WORKERS) as executor:
        # Her bir alt konu için _research_sub_topic fonksiyonunu çalıştıracak görevleri gönder
        future_to_subtopic = {executor.submit(_research_sub_topic, sub_topic, authority_sites): sub_topic for sub_topic in sub_topics}

        for future in concurrent.futures.as_completed(future_to_subtopic):
            result = future.result()
            if result and result["summary"]:
                sub_topic_summaries.append(result["summary"])
            if result and result["sources"]:
                all_sources.extend(result["sources"])

    if not sub_topic_summaries:
        return {"status": "error", "message": "No information could be gathered during the research."}

    # Step 4: Generate the Final Report (Map-Reduce Stage 2 - Final Reduce)
    print("\n📝 Consolidating summaries and generating the final report...")

    combined_summaries_text = ""
    for summary_item in sub_topic_summaries:
        combined_summaries_text += f"### {summary_item['sub_topic']}\n{summary_item['summary']}\n\n"

    # Context Window Check & Map-Reduce Summarization
    # The model's context window is ~4096 tokens. We use a safe character limit.
    SAFE_CHAR_LIMIT = 10000 # Approx. 2500 tokens to be safe.

    if len(combined_summaries_text) > SAFE_CHAR_LIMIT:
        print(f"  -> Combined text is too long ({len(combined_summaries_text)} chars). Starting map-reduce summarization...")

        # 1. Split the text into chunks
        chunks = [combined_summaries_text[i:i + SAFE_CHAR_LIMIT] for i in range(0, len(combined_summaries_text), SAFE_CHAR_LIMIT)]
        print(f"  -> Split text into {len(chunks)} chunks.")

        # 2. Summarize each chunk (Map)
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"  -> Summarizing chunk {i+1}/{len(chunks)}...")
            summarize_prompt = f"""
            The following is a piece of a larger research report. Summarize it concisely,
            retaining all key facts, names, and findings.

            Text Chunk:
            {chunk}

            Concise Summary of the Chunk:
            """
            summary = ask(summarize_prompt, max_new_tokens=1024) # Keep summaries shorter
            chunk_summaries.append(summary)

        # 3. Combine the summaries (Reduce)
        combined_summaries_text = "\n\n---\n\n".join(chunk_summaries)
        print(f"  -> Combined chunk summaries. New length is {len(combined_summaries_text)} chars.")

    # Final check: If the combined summaries are STILL too long, do one last summarization.
    if len(combined_summaries_text) > SAFE_CHAR_LIMIT:
        print(f"  -> Combined summaries are still too long. Performing a final consolidation...")
        final_consolidation_prompt = f"""
        The following text contains several summaries from a research report.
        Synthesize them into a single, final, coherent text. Retain all critical information
        but make it as concise as possible.

        Summaries to consolidate:
        {_truncate_text(combined_summaries_text, 15000)}
        """
        combined_summaries_text = ask(final_consolidation_prompt, max_new_tokens=2048)
        print(f"  -> Final consolidated text length is now {len(combined_summaries_text)} chars.")


    final_report_prompt = f"""
    As an expert research analyst, your task is to write a comprehensive, analytical report on '{topic}'.
    You have been provided with summaries for various sub-topics.
    Your report must be well-structured, insightful, and critical.

    Instructions:
    1.  **Title:** Start with a clear, descriptive title for the report.
    2.  **Executive Summary:** Write a brief overview (2-3 sentences) of the main findings.
    3.  **Detailed Analysis:** For each sub-topic, present a detailed analysis based on the provided summaries. Use the sub-topic as a heading.
    4.  **Synthesis and Contradictions:** After presenting the details, add a section named "Synthesis and Conclusion". In this section, synthesize the information, identify any contradictions or gaps in the information, and provide a concluding perspective.
    5.  **Tone:** Maintain a formal, objective, and analytical tone.

    **PROVIDED SUMMARIES:**
    {combined_summaries_text}

    **FINAL REPORT:**
    """
    try:
        final_report = ask(final_report_prompt, max_new_tokens=2048) # Reduced max_tokens
        # Deduplicate sources based on URL
        unique_sources = []
        seen_urls = set()
        for source in all_sources:
            if isinstance(source, dict) and "url" in source:
                if source["url"] not in seen_urls:
                    unique_sources.append(source)
                    seen_urls.add(source["url"])

        return {"status": "success", "result": final_report, "sources": unique_sources, "chunks": sub_topic_summaries}
    except Exception as e:
        error_message = f"An error occurred while generating the final report: {str(e)}"
        print(f"[ERROR] {error_message}")
        return {"status": "error", "message": error_message, "chunks": sub_topic_summaries}