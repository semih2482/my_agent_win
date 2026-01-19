# agent/tools/document_summarizer.py
from typing import Dict, Any, List
import re

# lazy imports
_tfidf_available = False
_try_tfidf = True
_vectorizer = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    _tfidf_available = True
except Exception:
    _tfidf_available = False

from agent.models.llm import ask

_sentence_split_re = re.compile(r'(?<=[.!?])\s+')

def _extractive_summary(text: str, max_sentences: int = 3) -> Dict[str, Any]:
    sentences = [s.strip() for s in _sentence_split_re.split(text) if s.strip()]
    if not sentences:
        return {"summary": "", "method": "extractive", "highlights": [], "keywords": []}

    # if TF-IDF available, score sentences by sum of tf-idf weights
    if _tfidf_available and len(sentences) > 1:
        try:
            global _vectorizer
            if _vectorizer is None:
                _vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
                _vectorizer.fit(sentences)
            mat = _vectorizer.transform(sentences)
            scores = mat.sum(axis=1).A1  # dense
            ranked_idx = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
            selected_idx = sorted(ranked_idx[:max_sentences])
            summary = " ".join([sentences[i] for i in selected_idx])
            # keywords: top features overall
            try:
                feat = _vectorizer.get_feature_names_out()
                # get mean tfidf per feature
                mean_tfidf = mat.mean(axis=0).A1
                kw_idx = mean_tfidf.argsort()[::-1][:10]
                keywords = [feat[i] for i in kw_idx]
            except Exception:
                keywords = []
            return {"summary": summary, "method": "extractive_tfidf", "highlights": [sentences[i] for i in selected_idx], "keywords": keywords}
        except Exception:
            pass

    # fallback: first-N sentences
    selected = sentences[:max_sentences]
    return {"summary": " ".join(selected), "method": "extractive_firstn", "highlights": selected, "keywords": []}


def summarize(text: str, max_sentences: int = 3, method: str = "auto", llm_ask_function=None) -> Dict[str, Any]:
    """
    method: "auto" (llm if available else extractive), "llm", "extractive"
    """
    llm_ask_function = llm_ask_function or ask
    if method == "llm" or (method == "auto" and llm_ask_function is not None):
        # ask the LLM for an abstractive JSON response. Use safe parsing fallback.
        prompt = (
            "Aşağıdaki metni Türkçe olarak kısa ve özet bir şekilde özetle. "
            "Cevabı JSON formatında ver: {\"summary\":\"...\",\"highlights\":[\"...\",\"...\"],\"keywords\":[\"...\",\"...\"]}\n\n"
            f"Metin:\n{text}\n\nJSON:"
        )
        try:
            resp = llm_ask_function(prompt, max_new_tokens=400)
            import json, re
            m = re.search(r'\{.*\}', resp, re.DOTALL)
            if m:
                parsed = json.loads(m.group(0))
                return {"method":"llm", **parsed}
        except Exception:
            # fallback to extractive
            pass

    # extractive fallback
    return _extractive_summary(text, max_sentences=max_sentences)
