import json
from pathlib import Path
import faiss
import nltk
import numpy as np
from sentence_transformers import SentenceTransformer

KNOWLEDGE_BASE_PATH = Path(__file__).resolve().parent / "knowledge_base.json"

def build_knowledge_base(knowledge_base_path):
    knowledge_base_path = Path(knowledge_base_path)

    with knowledge_base_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    kb_content = []
    for item in data:
        if item.get("doc_type") in {"resolution_guide", "sop"}:
            kb_content.append(str(item.get("content", "")).replace("\n", " "))

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    kb_embeddings = embedding_model.encode(kb_content, show_progress_bar=False)
    kb_embeddings = np.asarray(kb_embeddings, dtype="float32")
    dimension = kb_embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(kb_embeddings)

    return {
        "embedding_model": embedding_model,
        "kb_content": kb_content,
        "index": index,
    }


def query_knowledge_base(query_text):
    knowledge_bundle = build_knowledge_base(KNOWLEDGE_BASE_PATH)
    if not knowledge_bundle["kb_content"]:
        return ""

    k = 2
    query_embedding = knowledge_bundle["embedding_model"].encode([query_text], show_progress_bar=False).astype("float32")
    k = min(k, len(knowledge_bundle["kb_content"]))
    distances, indices = knowledge_bundle["index"].search(np.asarray(query_embedding).astype("float32"), k)
    retrieved = [knowledge_bundle["kb_content"][int(idx)] for idx in indices[0]]
    return "\n".join(retrieved)