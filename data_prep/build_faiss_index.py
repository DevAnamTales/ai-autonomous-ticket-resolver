
import pickle
import faiss
import numpy as np
import os

# ============================================================================
# CONFIGURATION
# ============================================================================
EMBEDDINGS_FILE = "embeddings_data.pkl"
FAISS_INDEX_FILE = "faiss_index.index"  # ✅ Updated extension

# ============================================================================
# LOAD EMBEDDINGS
# ============================================================================
print("="*70)
print("STEP 2: BUILD FAISS INDEX")
print("="*70)

print(f"\n📂 Loading embeddings from {EMBEDDINGS_FILE}...")
with open(EMBEDDINGS_FILE, 'rb') as f:
    data = pickle.load(f)

embeddings = data['embeddings']  # shape: (num_records, embedding_dim)
metadata = data['metadata']
embedding_dim = embeddings.shape[1]

print(f"✅ Loaded {embeddings.shape[0]} embeddings of dimension {embedding_dim}")

# ============================================================================
# CREATE FAISS INDEX
# ============================================================================
print("\n🚀 Creating FAISS index (cosine similarity)...")

# Normalize embeddings for cosine similarity
faiss.normalize_L2(embeddings)

# Create index
index = faiss.IndexFlatIP(embedding_dim)  # Inner Product for cosine similarity
index.add(embeddings)

print(f"✅ FAISS index created with {index.ntotal} vectors")

# ============================================================================
# SAVE INDEX
# ============================================================================
print(f"\n💾 Saving FAISS index to {FAISS_INDEX_FILE}...")
faiss.write_index(index, FAISS_INDEX_FILE)
file_size = os.path.getsize(FAISS_INDEX_FILE) / (1024**2)
print(f"✅ Saved FAISS index ({file_size:.2f} MB)")

# ============================================================================
# SEARCH FUNCTION
# ============================================================================
def search_faiss(query_text, model, index, metadata, top_k=5):
    """
    Search FAISS index for similar records.
    Args:
        query_text (str): The text to search for.
        model: SentenceTransformer model for encoding query.
        index: FAISS index object.
        metadata: List of metadata records.
        top_k (int): Number of results to return.
    Returns:
        List of dicts with id, source, score, and snippet.
    """
    # Encode query
    query_vector = model.encode([query_text], normalize_embeddings=True)
    faiss.normalize_L2(query_vector)

    # Search
    D, I = index.search(query_vector, top_k)
    results = []
    for rank, idx in enumerate(I[0]):
        results.append({
            "id": metadata[idx].get("id", ""),
            "source": metadata[idx].get("source", ""),
            "score": float(D[0][rank]),
            "snippet": metadata[idx].get("training_text", "")[:200]
        })
    return results

# ============================================================================
# TEST SEARCH
# ============================================================================
print("\n🔍 Testing search with sample query...")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(data['model_info']['model_name'])

sample_query = "Email service down"
results = search_faiss(sample_query, model, index, metadata, top_k=5)

print(f"\nQuery: {sample_query}")
print("Top 5 similar records:")
for r in results:
    print(f"- [{r['source']}] {r['id']} (score={r['score']:.3f})")
    print(f"  {r['snippet']}...\n")

