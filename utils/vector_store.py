
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

FAISS_INDEX_FILE = "data_prep/faiss_index.index"
EMBEDDINGS_FILE = "data_prep/embeddings_data.pkl"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print("📌 Loading FAISS index and metadata...")
index = faiss.read_index(FAISS_INDEX_FILE)
with open(EMBEDDINGS_FILE, "rb") as f:
    data = pickle.load(f)
metadata = data["metadata"]
model = SentenceTransformer(EMBEDDING_MODEL_NAME)


def search_similar(query: str, top_k: int = 5):
    query_vec = model.encode([query], normalize_embeddings=True)
    faiss.normalize_L2(query_vec)
    distances, indices = index.search(query_vec, top_k)

    results = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
        item = metadata[idx]
        results.append({  # ✅ Move inside the loop
            "rank": rank,
            "id": item.get("id", f"record_{idx}"),
            "similarity_score": round(float(dist), 4),
            "source": item.get("source", "unknown"),
            "training_text": item.get("training_text", "")[:500],
            "assignment_group": item.get("Assignment group", "Not Provided"),
            "configuration_item": item.get("Configuration item", "Not Provided"),
            "category": item.get("Category", "Not Provided")  # ✅ NEW for KB articles           
        })



    return results

