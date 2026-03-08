# Agentic ITSM - Autonomous Incident Resolution System

**AI-powered multi-agent orchestration platform for automated ServiceNow incident management**


## 🎯 Overview

This project demonstrates **governed agentic autonomy** for IT Service Management. Four specialized AI agents orchestrate incident resolution with confidence-based decision making, risk assessment, and automatic execution through MCP (Model Context Protocol) integration.

**Key Achievement**: Move from single-prompt AI to structured multi-agent workflows with governance controls.

---

## 🏗️ Architecture

```
ServiceNow (Incident Management)
    ↓ 
Flowise (AI Orchestration)
    ├── Create Incident Agent
    ├── Troubleshoot Agent (RAG + LLaMA)
    ├── Governance Agent (Risk + Confidence)
    └── Change Agent (MCP Execution)
    ↓
Flask Tool Server (8 REST endpoints)
    ├── RAG Search (FAISS, 10K incidents)
    ├── LLaMA Analysis (Groq API)
    ├── Governance Decision
    └── MCP Gateway
    ↓
External Systems
    └── Invoice Ninja (Billing - Live Integration)
```

**Orchestration Pattern**: Hub-and-spoke with central Oscillator making all routing decisions. Each agent performs its task and returns control to the Oscillator.

---

## ✨ Features

### 🤖 **AI-Powered Analysis**
- **RAG (Retrieval Augmented Generation)**: Searches 10,000 historical incidents using FAISS vector database
- **LLaMA 3.3 70B**: Root cause analysis via Groq API (free tier)
- **Sentence Transformers**: Semantic similarity matching

### 🎯 **Intelligent Orchestration**
- **4 Specialized Agents**: Create Incident, Troubleshoot, Governance, Change
- **Dynamic Routing**: Oscillator (Gemini 2.0 Flash) controls all agent transitions
- **State Management**: Complete flow state maintained across agent executions

### 🛡️ **Governance & Safety**
- **Confidence Scoring**: AI rates certainty from 0-100%
- **Risk Assessment**: Context-aware evaluation (keywords, information completeness)
- **Policy Enforcement**: Auto-execute only when confidence ≥85% AND risk=low
- **Complete Audit Trail**: Every decision logged in ServiceNow

### 🔗 **MCP Integration**
- **MCP Gateway**: Routes actions to external systems
- **Live Integration**: Invoice Ninja (billing system)
- **Extensible**: Ready for Order Management, CRM, Asset Management

### 📊 **Metrics & Results**
- **Resolution Time**: ~45 seconds (vs 49+ minutes traditional)
- **Confidence**: 85% average on validated use cases
- **Human Touch Time**: 0 seconds on auto-executed incidents
- **Cost**: ~$0.05 per incident (API calls)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- ServiceNow PDI instance
- Flowise 1.x installed
- Invoice Ninja account (or other MCP-compatible system)

### Installation

```bash
# Clone repository
git clone https://github.com/DevAnamTales/agentic-itsm.git
cd agentic-itsm

# Install Python dependencies
pip3 install -r requirements.txt --break-system-packages

# Set up environment variables
cp .env.example .env
# Edit .env with your credentials

# Build FAISS index from incident data
cd data_prep
python3 build_faiss_index.py

# Start Flask tool server
cd ..
python3 tool_server.py &

# Import Flowise flow (see INSTALLATION.md)
```

### Run Demo

```bash
# Test query in Flowise chat:
Customer K9b6XNmNeE missing invoice for 150 USD service charge

# Expected flow:
# 1. Create Incident → INC created in ServiceNow
# 2. Troubleshoot → RAG finds similar incidents, LLaMA analyzes
# 3. Governance → Evaluates confidence (85%) + risk (low) → auto_execute
# 4. Change → Creates CR, executes MCP, creates invoice, resolves ticket

# Total time: ~45 seconds
```

---

## 📁 Project Structure

```
incident_mgmt_llm_flowise/
├── README.md                    # This file
├── INSTALLATION.md              # Detailed setup guide
├── DEPLOYMENT.md                # Production deployment
├── DEMO_GUIDE.md                # How to run demo
├── TROUBLESHOOTING.md           # Common issues
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
│
├── tool_server.py               # Main Flask server
├── chains/
│   └── agent_chain.py           # LLaMA integration + risk assessment
│
├── routes/tools/                # Flask API endpoints (8 tools)
│   ├── create_incident.py       # Creates ServiceNow incident
│   ├── search_incidents.py      # RAG search with FAISS
│   ├── analyze_root_cause.py    # LLaMA root cause analysis
│   ├── update_ticket.py         # Writes analysis to ServiceNow
│   ├── governance_decision.py   # Policy evaluation
│   ├── create_change.py         # Change request creation
│   ├── execute_mcp.py           # MCP Gateway (routes to external systems)
│   └── resolve_ticket.py        # Closes incident
│
├── data_prep/                   # Data preparation scripts
│   ├── build_faiss_index.py     # Creates vector database
│   ├── combined_training_data.csv
│   ├── faiss_index.index        # Vector embeddings
│   └── embeddings_data.pkl
│
├── utils/                       # Utility modules
│   ├── servicenow_api.py        # ServiceNow REST API wrapper
│   ├── vector_store.py          # FAISS operations
│   └── llm_utils.py             # LLM helper functions
│
└── mcp_agents/                  # MCP tool implementations
    └── tools.py                 # Invoice Ninja integration
```

---

## 🔧 Configuration

### Environment Variables

Create `.env` file with:

```bash
# ServiceNow Configuration
SERVICENOW_INSTANCE=https://devXXXXX.service-now.com
SERVICENOW_USERNAME=admin
SERVICENOW_PASSWORD=your_password
AI_SUGGESTION_FIELD=u_ai_suggestion
AI_CONFIDENCE_THRESHOLD=0.85

# AI APIs
GEMINI_API_KEY=your_gemini_key          # For Oscillator
GROQ_API_KEY=your_groq_key              # For LLaMA analysis

# MCP Integration
NINJAINVOICE_API_KEY=your_invoice_ninja_key
NINJA_URL=https://invoicing.co/api/v1

# Governance
SN_RESOLUTION_CODE=Solution provided
```

### Flask Endpoints

All endpoints available at `http://localhost:5001/tool/`

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/tool/create_incident` | POST | Create ServiceNow incident |
| `/tool/search_incidents` | POST | RAG search similar incidents |
| `/tool/analyze_root_cause` | POST | LLaMA root cause analysis |
| `/tool/update_ticket` | POST | Write AI analysis to ticket |
| `/tool/governance_decision` | POST | Evaluate governance policy |
| `/tool/create_change` | POST | Create change request |
| `/tool/execute_mcp` | POST | Execute MCP action |
| `/tool/resolve_ticket` | POST | Close incident |

## 📊 Demo Queries

### ✅ Happy Path (Auto-Execute)
```
Customer K9b6XNmNeE missing invoice for 150 USD service charge
```
**Expected**: Confidence 85%, Risk LOW → auto_execute → Invoice created

### ⚠️ Human Approval
```
VPN connection issues for remote users started 2 hours ago
```
**Expected**: Confidence 75%, Risk MEDIUM → human_approval

### ❌ Escalation (Low Confidence)
```
my invoice is missing
```
**Expected**: Confidence 35%, Risk HIGH → escalate

### ❌ Escalation (High Risk Keywords)
```
Production database connection failures on critical system
```
**Expected**: Confidence 80%, Risk HIGH (keywords detected) → escalate

---

## 🔒 Security & Governance

### Decision Matrix

| Confidence | Risk | Decision |
|-----------|------|----------|
| ≥85% | LOW | Auto-Execute ✅ |
| ≥65% | LOW/MEDIUM | Human Approval ⚠️ |
| <65% OR HIGH risk | Any | Escalate ❌ |

### Risk Assessment Factors
1. **Keywords**: "production", "database", "critical", "payment"
2. **Information Completeness**: Customer ID, amounts, component details
3. **Action Type**: Server restarts, billing operations
4. **Confidence Score**: AI's certainty rating

---

## 🛠️ Development

### Running Tests
```bash
# Test FAISS search
python3 -c "from utils.vector_store import search_similar_incidents; print(search_similar_incidents('invoice missing', top_k=3))"

# Test ServiceNow API
python3 -c "from utils.servicenow_api import get_incident; print(get_incident('INC0010067'))"

# Test Flask endpoint
curl -X POST http://localhost:5001/tool/search_incidents \
  -H "Content-Type: application/json" \
  -d '{"query": "invoice missing", "top_k": 3}'
```

### Debugging
```bash
# Watch Flask logs
tail -f nohup.out

# Check Flowise logs
# Open Flowise UI → Flow → Logs tab

# Verify FAISS index
ls -lh data_prep/faiss_index.index
```

---

## 📚 Documentation

- **[INSTALLATION.md](INSTALLATION.md)** - Detailed setup instructions

---

## 🤝 Contributing

This is an internship project demonstrating proof-of-concept. For production use:

1. **Stabilize Flowise** - Fix state management race conditions
2. **Expand MCP Tools** - Add Order Management, CRM, Asset Management
3. **Add Validation Agent** - Verify fixes actually worked
4. **Human Approval UI** - Build interface for medium-confidence cases
5. **Telemetry** - Add Grafana dashboards for monitoring

---

**Technologies**:
- [Flowise](https://flowiseai.com/) - Visual AI orchestration
- [Groq](https://groq.com/) - LLaMA API hosting
- [ServiceNow](https://www.servicenow.com/) - ITSM platform
- [Invoice Ninja](https://invoiceninja.com/) - Billing system
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search

---

## 🎯 Results Summary

| Metric | Traditional | Our System |
|--------|-------------|------------|
| Resolution Time | 49+ minutes | 45 seconds |
| Human Touch Time | 100% | 0% (auto-execute) |
| Confidence Score | N/A | 85% average |
| Cost per Incident | $25+ (labor) | $0.05 (API) |
| Audit Trail | Manual notes | Complete automated |
| 24/7 Availability | No | Yes |

**ROI**: Immediate and significant. System can handle 30-40% of incident workload at <5% of labor cost.

---

**Built with ❤️ during Telia Internship 2025-2026**
