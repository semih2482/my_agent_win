# agent/tools/knowledge_graph.py
from typing import Dict, Any, List
from collections import defaultdict
import re

# lazy spaCy use
try:
    import spacy
    _SPACY_AVAILABLE = True
except Exception:
    spacy = None
    _SPACY_AVAILABLE = False

# helper for fallback simple triple extraction
_sentence_split_re = re.compile(r'(?<=[.!?])\s+')

def _simple_triples(text: str):
    triples = []
    sentences = [s.strip() for s in _sentence_split_re.split(text) if s.strip()]
    for s in sentences:
        words = s.split()
        if len(words) < 3:
            continue
        subj = words[0]
        verb = words[1]
        obj = " ".join(words[2:])
        triples.append({"subject": subj, "relation": verb, "object": obj, "confidence": 0.4, "sentence": s})
    return triples

def _spacy_triples(text: str, model_name: str = None):
    """
    Use spaCy dependency parse to extract SVO triples.
    """
    model_candidates = [model_name] if model_name else []
    # if model_name not given, try loaded default small multi-language or en model
    triples = []
    try:
        if model_name:
            nlp = spacy.load(model_name)
        else:
            # try common small English/multilingual
            try:
                nlp = spacy.load("xx_sent_ud_sm")
            except Exception:
                nlp = spacy.load("en_core_web_sm")
        docs = nlp(text).sents
        for sent in docs:
            sent_doc = sent.as_doc()
            subj = None
            obj = None
            rel = None
            # find nominal subject
            for tok in sent_doc:
                if tok.dep_.lower() in ("nsubj", "nsubjpass"):
                    subj = tok
                    # try to find verb head
                    if tok.head:
                        rel = tok.head.lemma_
                    # find direct object
                    for child in tok.head.children:
                        if child.dep_.lower() in ("dobj","obj","pobj"):
                            obj = child
                            break
                    break
            if subj and obj and rel:
                triples.append({
                    "subject": subj.text,
                    "relation": rel,
                    "object": obj.text,
                    "confidence": 0.95,
                    "sentence": sent.text
                })
        return triples
    except Exception:
        return []

def build_knowledge_graph(text: str, spacy_model: str = None) -> Dict[str, Any]:
    """
    Metinden (özne-ilişki-nesne) triple'ları çıkarır ve graph JSON döner:
    {
      "nodes": [{"id": "...", "label":"...", "type":"entity"/"concept"}],
      "edges": [{"source": id, "target": id, "relation": "...", "confidence": 0.9, "sentence": "..."}]
    }
    """
    triples = []
    if _SPACY_AVAILABLE:
        triples = _spacy_triples(text, model_name=spacy_model)
    if not triples:
        triples = _simple_triples(text)

    # build nodes/edges unique map
    node_id_map = {}
    nodes = []
    edges = []
    next_id = 1
    def _get_node_id(label):
        nonlocal next_id
        if label in node_id_map:
            return node_id_map[label]
        nid = f"N{next_id}"
        next_id += 1
        node_id_map[label] = nid
        nodes.append({"id": nid, "label": label})
        return nid

    for t in triples:
        s_id = _get_node_id(t["subject"])
        o_id = _get_node_id(t["object"])
        edges.append({
            "source": s_id,
            "target": o_id,
            "relation": t.get("relation", ""),
            "confidence": t.get("confidence", 0.5),
            "sentence": t.get("sentence", "")
        })

    graph = {"nodes": nodes, "edges": edges, "triples_count": len(triples)}
    return {"status":"success", "graph": graph}
