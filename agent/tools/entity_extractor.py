# agent/tools/entity_extractor.py
from typing import Dict, Any, List
import re

# Lazy spaCy import
_nlp = None
_nlp_name_tried = None

def _load_spacy(preferred_model: str = None):
    global _nlp, _nlp_name_tried
    if _nlp is not None:
        return _nlp
    try:
        import spacy
        # tercihli model varsa dene, yoksa öncelikle çok-dilli, sonra en_core_web_sm
        candidates = []
        if preferred_model:
            candidates.append(preferred_model)
        candidates += ["xx_ent_wiki_sm", "en_core_web_sm"]
        for m in candidates:
            try:
                _nlp = spacy.load(m)
                _nlp_name_tried = m
                return _nlp
            except Exception:
                continue
    except Exception:
        _nlp = None
    return None


# Regex helpers
_RE_PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "phone": re.compile(r"\+?\d[\d\s\-\(\)]{6,}\d"),
    "url": re.compile(r"https?://[^\s]+"),
    "date": re.compile(r"\b(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4})\b"),
    "number": re.compile(r"\b\d+(?:[\.,]\d+)?\b"),
}


def _regex_extract(text: str) -> Dict[str, List[Dict[str, Any]]]:
    found = {"persons": [], "organizations": [], "locations": [], "dates": [], "numbers": [], "emails": [], "phones": [], "urls": []}
    for k, pat in _RE_PATTERNS.items():
        for m in pat.finditer(text):
            entry = {"text": m.group(0), "start": m.start(), "end": m.end(), "confidence": 0.9}
            if k == "email":
                found["emails"].append(entry)
            elif k == "phone":
                found["phones"].append(entry)
            elif k == "url":
                found["urls"].append(entry)
            elif k == "date":
                found["dates"].append(entry)
            elif k == "number":
                found["numbers"].append(entry)
    # crude proper-noun detection: Capitalized words not at sentence start
    caps = re.findall(r"\b([A-ZŞÇİÖÜ][a-zşçğıöüİ]+(?:\s+[A-ZŞÇİÖÜ][a-zşçğıöüİ]+)*)\b", text)
    for c in set(caps):
        # treat as person/org candidate with low confidence
        found["persons"].append({"text": c, "start": text.find(c), "end": text.find(c)+len(c), "confidence": 0.4})
    return found


def extract_entities(text: str, preferred_spacy_model: str = None) -> Dict[str, Any]:
    """
    Gelişmiş entity extractor:
      - Öncelikle spaCy NER kullanır (varsa)
      - Yoksa regex + basit heuristics ile fallback yapar
    Dönen yapı:
      {
        "status": "success",
        "source": "spacy"|"regex",
        "entities": {
            "persons": [...],
            "organizations": [...],
            "locations": [...],
            "dates": [...],
            "numbers": [...],
            "emails": [...],
            "phones": [...],
            "urls": [...]
        }
      }
    """
    nlp = _load_spacy(preferred_spacy_model)
    if nlp:
        try:
            doc = nlp(text)
            out = {"persons": [], "organizations": [], "locations": [], "dates": [], "numbers": [], "emails": [], "phones": [], "urls": []}
            for ent in doc.ents:
                label = ent.label_.lower()
                entry = {"text": ent.text, "start": ent.start_char, "end": ent.end_char, "label": ent.label_, "confidence": getattr(ent, "kb_id_", 1.0)}
                if label in ("person", "per", "persons", "name", "person_name"):
                    out["persons"].append(entry)
                elif label in ("org", "organization", "company"):
                    out["organizations"].append(entry)
                elif label in ("gpe", "loc", "location"):
                    out["locations"].append(entry)
                elif label in ("date",):
                    out["dates"].append(entry)
                elif label in ("money","percent","quantity","cardinal","number"):
                    out["numbers"].append(entry)
                else:
                    # unknown label -> try to categorize
                    if "@" in ent.text:
                        out["emails"].append(entry)
                    elif re.search(r"\d", ent.text):
                        out["numbers"].append(entry)
                    else:
                        out["organizations"].append(entry)
            # add regex extras (emails/phones/urls) if missing
            regex_extra = _regex_extract(text)
            for k in ("emails","phones","urls","dates","numbers"):
                for e in regex_extra[k]:
                    # avoid duplicates by text
                    if not any(x['text']==e['text'] for x in out.get(k,[])):
                        out[k].append(e)
            return {"status":"success","source": f"spacy:{_nlp_name_tried}", "entities": out}
        except Exception as e:
            # fallback to regex
            pass

    # fallback
    out = _regex_extract(text)
    return {"status":"success", "source":"regex", "entities": out}
