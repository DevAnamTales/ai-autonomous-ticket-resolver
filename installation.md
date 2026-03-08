# Installation Guide

Complete step-by-step instructions to set up the Agentic ITSM system from scratch.

---

## 📋 Prerequisites

### Required Accounts
- ✅ **ServiceNow PDI** - Personal Developer Instance (free)
  - Sign up: https://developer.servicenow.com/
- ✅ **Groq API** - For LLaMA 3.3 70B (free tier)
  - Sign up: https://console.groq.com/
- ✅ **Google Gemini** - For Oscillator LLM (paid)
  - Get API key: https://makersuite.google.com/app/apikey
- ✅ **Invoice Ninja** - For MCP integration (free trial)
  - Sign up: https://invoiceninja.com/

### System Requirements
- **OS**: Linux (Ubuntu 20.04+) or macOS
- **Python**: 3.9 or higher
- **RAM**: 4GB minimum (8GB recommended for FAISS)
- **Disk**: 2GB free space
- **Network**: Internet access for API calls

### Software Dependencies
```bash
# Check Python version
python3 --version  # Should be 3.9+

# Check pip
pip3 --version

# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev build-essential

# For macOS
brew install python@3.9
```

---

## 🚀 Step 1: Clone Repository

```bash
# Clone the repository

git clone <your-github-repository-url>
cd <your-repo-folder>


# Verify structure
ls -la
# You should see: README.md, requirements.txt, tool_server.py, etc.
```

---

## 📦 Step 2: Install Python Dependencies

```bash
# Install all required packages
pip3 install -r requirements.txt --break-system-packages

# Verify installations
python3 -c "import flask; print('Flask:', flask.__version__)"
python3 -c "import groq; print('Groq:', groq.__version__)"
python3 -c "import faiss; print('FAISS: OK')"
python3 -c "from sentence_transformers import SentenceTransformer; print('Sentence Transformers: OK')"
```

**Note**: The `--break-system-packages` flag is needed on some systems to install packages globally. If you prefer, use a virtual environment:

```bash
# Alternative: Use virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip3 install -r requirements.txt
```

---

## 🔧 Step 3: Configure Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit with your credentials
nano .env  # or use your preferred editor
```

### Required Configuration

Update `.env` with your actual credentials:

```bash
# ServiceNow Configuration
SERVICENOW_INSTANCE=https://your-instance.service-now.com # ← Your PDI instance
SERVICENOW_USERNAME=admin
SERVICENOW_PASSWORD=your_actual_password

# AI API Keys
GEMINI_API_KEY=your_gemini_api_key_here  # ← From Google AI Studio
GROQ_API_KEY=your_groq_api_key_here      # ← From Groq Console

# Invoice Ninja
NINJAINVOICE_API_KEY=...  # ← From Invoice Ninja settings
NINJA_URL=https://invoicing.co/api/v1
```

### Verify Environment Variables

```bash
# Test loading environment
python3 -c "
from dotenv import load_dotenv
import os
load_dotenv()
print('ServiceNow:', os.getenv('SERVICENOW_INSTANCE'))
print('Groq Key:', 'Loaded' if os.getenv('GROQ_API_KEY') else 'Missing')
"
```

---

## 📊 Step 4: Prepare Data & Build FAISS Index

### Option A: Use Existing Data (Provided)

If the repository includes pre-built data:

```bash
cd data_prep

# Verify files exist
ls -lh faiss_index.index embeddings_data.pkl combined_training_data.csv

# If files exist, you're done! Skip to Step 5.
```

### Option B: Build Index from Scratch

If you need to rebuild the FAISS index:

```bash
cd data_prep

# 1. Prepare dataset (combines incidents + KB articles)
python3 prepare_dataset_from_incidents_and_kb.py

# 2. Generate embeddings using sentence-transformers
python3 generate_embeddings.py

# 3. Build FAISS index
python3 build_faiss_index.py

# Verify index created
ls -lh faiss_index.index
# Should be several MB in size
```

**Expected Output:**
```
Building FAISS index...
Loaded 10,247 embeddings
FAISS index created: faiss_index.index
Index size: 8.3 MB
```

### Test FAISS Search

```bash
# Test vector search
python3 -c "
from utils.vector_store import search_similar_incidents
results = search_similar_incidents('invoice missing', top_k=3)
print(f'Found {len(results)} similar incidents')
for r in results:
    print(f'- {r[\"id\"]}: {r[\"short_description\"][:50]}...')
"
```

---

## 🖥️ Step 5: Configure ServiceNow

### Create Custom Field for AI Suggestions

1. Log in to your ServiceNow PDI instance
2. Navigate to: **System Definition → Tables**
3. Search for table: `incident`
4. Click on **Incident** table
5. Switch to **Columns** tab
6. Click **New** to create a new column:
   - **Type**: String (or Large Text)
   - **Column label**: AI Suggestion
   - **Column name**: `u_ai_suggestion`
   - **Max length**: 4000 (or use Text for unlimited)
7. Click **Submit**

### Verify Field Creation

```bash
# Test ServiceNow API access
python3 -c "
from utils.servicenow_api import test_connection
test_connection()
"
```

Expected output:
```
ServiceNow connection: OK
Instance: https://<your-instance>.service-now.com
```

---

## 🔥 Step 6: Start Flask Tool Server

```bash
# Return to project root
cd /path/to/agentic-itsm

# Start Flask server in background
python3 tool_server.py &

# Watch logs
tail -f nohup.out
```

**Expected Output:**
```
[LLAMA] Initializing Groq with key: gsk_abc123...
[LLAMA] Groq client initialized successfully
[FAISS] Loading index from data_prep/faiss_index.index
[FAISS] Index loaded: 10247 vectors
 * Running on http://0.0.0.0:5001
```

### Test Flask Endpoints

```bash
# Test health check
curl http://localhost:5001/

# Test search_incidents endpoint
curl -X POST http://localhost:5001/tool/search_incidents \
  -H "Content-Type: application/json" \
  -d '{"query": "invoice missing", "top_k": 3}'

# Should return JSON with similar incidents
```

**If Flask doesn't start**, check:
- Port 5001 is not already in use: `lsof -i :5001`
- Python path is correct
- .env file is loaded
- Check logs: `tail -100 nohup.out`

---

## 🌊 Step 7: Install & Configure Flowise

### Install Flowise

```bash
# Install Flowise globally
npm install -g flowise

# Start Flowise
npx flowise start

# Or with custom port
npx flowise start -p 3000
```

**Expected Output:**
```
Flowise started on port 3000
Open http://localhost:3000 in your browser
```

### Access Flowise UI

1. Open browser: `http://localhost:3000`
2. Create an account (local, no signup needed)
3. You'll see the Flowise dashboard

---

## 🔧 Step 8: Configure Flowise Flow

### Option A: Import Pre-Built Flow (Recommended)

If you have the exported flow JSON:

1. In Flowise UI, click **"Import Chatflow"**
2. Upload `flowise_flow_export.json`
3. Flow will be imported with all agents configured

### Option B: Build Flow Manually

---

## ✅ Step 9: Verify Installation

### Test 1: Flask Endpoints

```bash
# Test each endpoint
curl -X POST http://localhost:5001/tool/search_incidents \
  -H "Content-Type: application/json" \
  -d '{"query": "invoice missing", "top_k": 3}'
```

### Test 2: FAISS Search

```python
python3 << EOF
from utils.vector_store import search_similar_incidents
results = search_similar_incidents("invoice problem", top_k=3)
print(f"Found {len(results)} incidents")
for r in results:
    print(f"- {r['id']}: Similarity {r['similarity']:.2f}")
EOF
```

### Test 3: LLaMA Integration

```python
python3 << EOF
from chains.agent_chain import create_incident_agent, process_incident

agent = create_incident_agent()
result = process_incident(
    "Customer X missing invoice 100 USD",
    agent=agent,
    similar_incidents=[]
)
print("Root cause:", result['root_cause'][:100])
print("Confidence:", result['confidence'])
EOF
```

### Test 4: End-to-End Flow

In Flowise chat, type:
```
Customer K9b6XNmNeE missing invoice for 150 USD service charge
```

**Expected**:
1. Create Incident executes → Returns INC number
2. Troubleshoot executes → RAG + LLaMA analysis
3. Governance executes → Decision: auto_execute
4. Change executes → Invoice created, ticket closed

Check ServiceNow for the created and resolved incident.

---

## 🐛 Troubleshooting

### Flask Won't Start

**Error**: `Address already in use`
```bash
# Find process using port 5001
lsof -i :5001

# Kill it
kill -9 <PID>

# Restart Flask
python3 tool_server.py &
```

### FAISS Index Not Found

**Error**: `FileNotFoundError: faiss_index.index`
```bash
cd data_prep
python3 build_faiss_index.py
```

### Groq API Error

**Error**: `GROQ_API_KEY not found`
```bash
# Verify .env file
cat .env | grep GROQ

# Test loading
python3 -c "
from dotenv import load_dotenv
import os
load_dotenv()
print(os.getenv('GROQ_API_KEY'))
"
```

### ServiceNow Connection Failed

**Error**: `401 Unauthorized`
- Check username/password in `.env`
- Verify ServiceNow instance URL (no trailing slash)
- PDI instance must be awake (wake it at developer.servicenow.com)

### Flowise Tool Error

**Error**: `Cannot read properties of null`
- This is a known Flowise bug with state management
- Workaround: Use the validated happy path query
- For production: Consider migrating to LangGraph

---

## 🎉 Installation Complete!

You should now have:
- ✅ Flask server running on port 5001
- ✅ FAISS index with 10K+ incidents
- ✅ Flowise orchestrator configured
- ✅ ServiceNow API connected
- ✅ LLaMA integration working
- ✅ Invoice Ninja MCP ready

---

## 📞 Need Help?

**Common Issues**:
1. Check Flask logs: `tail -f nohup.out`
2. Verify environment: `python3 -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.environ)"`
3. Test API connectivity: `curl http://localhost:5001/`

**Still stuck?**
- Verify all prerequisites are met
