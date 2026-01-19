import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent.tools.entity_extractor import extract_entities
from agent.tools.knowledge_graph import build_knowledge_graph
from agent.tools.document_summarizer import summarize
from agent.tools.sentiment_analyzer import analyze_sentiment

# Test metni
text = "Albert Einstein discovered relativity in 1905. Microsoft HQ is in Redmond, Washington. Bu proje beni çok mutlu ediyor!"

# Entity Extractor
entities = extract_entities(text)
print("Entities:", entities)

# Knowledge Graph
graph = build_knowledge_graph(text)
print("Knowledge Graph:", graph)

# Summarizer
summary = summarize(text, method="extractive")
print("Summary:", summary)

# Sentiment
sentiment = analyze_sentiment(text)
print("Sentiment:", sentiment)
