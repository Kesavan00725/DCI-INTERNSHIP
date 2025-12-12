import json
import faiss
import numpy as np
from django.conf import settings
from sentence_transformers import SentenceTransformer

_index = None
_meta = None
_model = None

def load_all():
    global _index, _meta, _model
    if _index is None:
        _index = faiss.read_index(settings.FAISS_INDEX)
        with open(settings.META_JSON, "r", encoding="utf-8") as f:
            _meta = json.load(f)
        _model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve(query, top_k=3):
    load_all()
    q_emb = _model.encode([query]).astype("float32")
    _, I = _index.search(q_emb, top_k)
    return [_meta[i] for i in I[0]]
