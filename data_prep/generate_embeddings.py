
import os
import pandas as pd
import numpy as np
import pickle
import time
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_FILE = "combined_training_data.csv"  # ✅ Updated file name
OUTPUT_FILE = "embeddings_data.pkl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 16
MAX_TEXT_LENGTH = 2000

# ============================================================================
# LOAD DATA
# ============================================================================
print("="*70)
print("STEP 1: GENERATE EMBEDDINGS")
print("="*70)

print(f"\n📂 Loading data from {INPUT_FILE}...")
try:
    df = pd.read_csv(INPUT_FILE)
    print(f"✅ Loaded {len(df)} records")
    print(f"   Columns: {list(df.columns)}")
except FileNotFoundError:
    print(f"❌ ERROR: {INPUT_FILE} not found!")
    exit(1)

# ============================================================================
# PREPARE TEXT
# ============================================================================
print(f"\n🔧 Preparing text data...")

# Use training_text column directly
if 'training_text' not in df.columns:
    print("❌ ERROR: 'training_text' column not found in input file!")
    exit(1)

# Clean and truncate
df['training_text'] = df['training_text'].fillna('').astype(str).str[:MAX_TEXT_LENGTH]

# Filter out empty entries
original_len = len(df)
df = df[df['training_text'].str.len() > 10].reset_index(drop=True)
print(f"✅ Prepared {len(df)} records (removed {original_len - len(df)} empty entries)")

# Show sample
print(f"\n📝 Sample text (first record):")
print(f"   {df['training_text'].iloc[0][:100]}...")

# ============================================================================
# LOAD MODEL
# ============================================================================
print(f"\n🚀 Loading model: {MODEL_NAME}")
start_time = time.time()
model = SentenceTransformer(MODEL_NAME)
model.max_seq_length = 256
load_time = time.time() - start_time
print(f"✅ Model loaded in {load_time:.2f}s")
print(f"   Embedding dimension: {model.get_sentence_embedding_dimension()}")

# ============================================================================
# GENERATE EMBEDDINGS
# ============================================================================
print(f"\n⚡ Generating embeddings for {len(df)} records...")
start_time = time.time()
embeddings = model.encode(
    df['training_text'].tolist(),
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)
elapsed = time.time() - start_time
print(f"\n✅ Generated {len(embeddings)} embeddings in {elapsed:.2f}s")

# ============================================================================
# PREPARE METADATA
# ============================================================================
metadata = df.to_dict('records')
print(f"✅ Prepared metadata for {len(metadata)} records")

# ============================================================================
# SAVE TO DISK
# ============================================================================
output_data = {
    'embeddings': embeddings.astype('float32'),
    'metadata': metadata,
    'model_info': {
        'model_name': MODEL_NAME,
        'embedding_dim': embeddings.shape[1],
        'num_records': len(embeddings),
        'date_created': time.strftime('%Y-%m-%d %H:%M:%S')
    }
}
with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(output_data, f)
file_size = os.path.getsize(OUTPUT_FILE) / (1024**2)
print(f"✅ Saved {file_size:.2f} MB to {OUTPUT_FILE}")

# ============================================================================
# TEST SIMILARITY
# ============================================================================
print(f"\n📊 Testing similarity search...")
sample_idx = 0
similarities = cosine_similarity([embeddings[sample_idx]], embeddings)[0]
top_5_indices = np.argsort(similarities)[-6:-1][::-1]
print(f"\n   Query (record {sample_idx}):")
print(f"   {df.iloc[sample_idx]['training_text'][:80]}...")
print(f"\n   Top 5 similar records:")
for rank, idx in enumerate(top_5_indices, 1):
    sim_score = similarities[idx]
    text = df.iloc[idx]['training_text'][:80]
    print(f"   {rank}. [similarity={sim_score:.3f}] {text}...")

print("\n✅ STEP 1 COMPLETE!")

