# agent_chain.py

import json
import re
from langchain.agents import Tool, initialize_agent
from langchain.prompts import PromptTemplate

# MCP tool imports
from mcp_agents.tools import (
    retry_order_mcp,
    sync_customer_data_mcp,
    fix_asset_mismatch_mcp,
    create_invoice_mcp
)

# LLaMA LangChain LLM wrapper
from utils.llama_wrapper import LlamaLangChainWrapper

# ============================================================
# 1️⃣ Initialize LLM (LLaMA)
# ============================================================

llm_model = LlamaLangChainWrapper()

# ============================================================
# 2️⃣ MCP Tools
# ============================================================

tools = [
    Tool(
        name="retry_order",
        description="Use for OSM / ROD-OSM / order fallout issues.",
        func=retry_order_mcp
    ),
    Tool(
        name="sync_customer_data",
        description="Use for Sie-CRM / OurTelco / CRM profile mismatch.",
        func=sync_customer_data_mcp
    ),
    Tool(
        name="fix_asset_mismatch",
        description="Use for ROD-BRM / BRM asset mismatch.",
        func=fix_asset_mismatch_mcp
    ),
    Tool(
        name="create_invoice",
        description="Creates invoice in Ninjainvoice. Only auto-execute for approved actions (create/update invoice).",
        func=create_invoice_mcp
    ),
]

# ============================================================
# 3️⃣ Instruction prompt
# ============================================================

prompt = PromptTemplate(
    input_variables=["query", "ci", "context"],
    template="""
You are an Autonomous Ticket Resolution Agent.
Follow ALL strict business rules below.

===========================================================
STRICT RULESET
===========================================================

1️⃣ **Deciding When to Use Automation (MCP Tools)**
Automation can ONLY be used when:
- The recommended action is in the AUTO-APPROVED LIST
- Confidence ≥ 0.90
- CI matches the allowed tool logic

2️⃣ **Auto-Approved Actions**
You may trigger automation ONLY for:
- create_order
- update_order
- create_invoice
- update_invoice
- restart_server
- sync_customer_data

If the customer issue does **not** imply one of these actions → *do NOT call any tool*.

3️⃣ **Network Issue Rule**
If query includes: *vpn, login, connectivity, internet, WiFi, SSO*
→ ALWAYS classify as **network issue**
→ NEVER call CRM/BRM/OSM tools
→ ONLY give manual troubleshooting steps

4️⃣ **CI mismatch rule**
If CI does not logically relate to the issue:
Respond EXACTLY in JSON:
{
  "classification": "unknown",
  "ci_relevance": "not_related",
  "confidence": 0.3,
  "reasoning": "Issue unrelated to CI.",
  "resolution_steps": []
}

5️⃣ **Invoice Action Payload Rule**
If action is **create_invoice** or **update_invoice**:
- You MUST extract a valid invoice payload in JSON containing:
{
  "client_id": "<client_id>",
  "line_items": [
    {
      "product_key": "<string>",
      "notes": "<string>",
      "cost": <number>,
      "quantity": <number>
    }
  ]
}
- If payload cannot be extracted → DO NOT execute automation

6️⃣ **Output Rules**
You MUST output ONE of the following:
A) Tool Call ONLY in ReAct format:
Thought: I need to call a tool
Action: <tool_name>
Action Input: <value>

B) JSON decision ONLY (for evaluation / non-auto actions)

===========================================================
INPUT:
Incident: {query}
Configuration Item: {ci}
Context: {context}
===========================================================
"""
)

# ============================================================
# 4️⃣ LangChain Agent
# ============================================================

def create_incident_agent():
    """Creates the LangChain agent."""
    return initialize_agent(
        tools=tools,
        llm=llm_model,
        agent="zero-shot-react-description",
        verbose=True,
        max_iterations=3,
        handle_parsing_errors=True
    )

# ============================================================
# 5️⃣ Auto-Approved Action Logic
# ============================================================

AUTO_APPROVED_ACTIONS = [
    "create_order",
    "update_order",
    "create_invoice",
    "update_invoice",
    "restart_server",
    "sync_customer_data"
]

ISSUE_CI_MAP = {
    "vpn": "network",
    "connectivity": "network",
    "sso": "network",
    "login": "network",
    "auth": "network",
    "crm": "crm",
    "profile": "crm",
    "order": "order_mgmt",
    "fallout": "order_mgmt",
    "billing": "billing",
    "invoice": "billing",
}

def classify_issue_type(text: str):
    text = text.lower()
    for keyword, cat in ISSUE_CI_MAP.items():
        if keyword in text:
            return cat
    return "unknown"

# ============================================================
# 6️⃣ Decision Engine — ensures safe automation
# ============================================================

def process_incident(query, ci_name, agent):
    """
    Process incident safely:
    - Returns JSON with 'action', 'confidence', 'reasoning'
    - Only triggers MCP tools if auto-approved and valid
    """
    issue_type = classify_issue_type(query)

    # Network issue rule
    if issue_type == "network" and ci_name.lower() in ["sie-crm", "rod-brm", "rod-osm"]:
        return {
            "automation_allowed": False,
            "approved_action": None,
            "confidence": 0.0,
            "reason": "Network issues cannot trigger CRM/BRM/OSM automation."
        }

    # Ask LLM to classify intended action (DIRECT LLM CALL)
    decision_prompt = f"""
You MUST reply in EXACT JSON only. No extra text before or after.

Extract:
- action: intended operation (one of: {', '.join(AUTO_APPROVED_ACTIONS)}, or "none")
- confidence: number between 0.0 and 1.0
- reasoning: short explanation

If unclear or query is too vague → action = "none", confidence = 0.0

QUERY: "{query}"
CI: "{ci_name}"

Output ONLY valid JSON:
"""

    # ✅ Safe LLaMA call
    try:
        llm_response = llm_model._call(decision_prompt)
        if isinstance(llm_response, list):
            ai_output = llm_response[0].content
        else:
            ai_output = str(llm_response)
    except Exception as e:
        return {
            "automation_allowed": False,
            "approved_action": None,
            "confidence": 0.0,
            "reason": f"LLM call failed: {str(e)}"
        }

    # Extract JSON safely
    try:
        cleaned_output = ai_output.strip()
        if cleaned_output.startswith("```json"):
            cleaned_output = cleaned_output.replace("```json", "").replace("```", "").strip()
        elif cleaned_output.startswith("```"):
            cleaned_output = cleaned_output.replace("```", "").strip()

        match = re.search(r'\{.*\}', cleaned_output, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON found in output: {ai_output[:200]}")
        decision = json.loads(match.group())
    except Exception as e:
        return {
            "automation_allowed": False,
            "approved_action": None,
            "confidence": 0.0,
            "reason": f"Invalid JSON returned by LLM. Error: {str(e)}. Output was: {ai_output[:200]}"
        }

    action = decision.get("action", "none")
    conf = float(decision.get("confidence", 0))

    # Only allow auto-approved + high confidence
    automation_allowed = action in AUTO_APPROVED_ACTIONS and conf >= 0.90

    payload = None
    # If invoice, validate payload
    if automation_allowed and action in ["create_invoice", "update_invoice"]:
        payload_prompt = f"""
You MUST return ONLY valid JSON payload for the invoice. No extra text.

Extract from this query:
QUERY: "{query}"
CI: "{ci_name}"

Required format:
{{
  "client_id": "extracted_client_id",
  "line_items": [
    {{
      "product_key": "item_name",
      "notes": "description",
      "cost": 100.0,
      "quantity": 1
    }}
  ]
}}

If you cannot extract client_id or line_items, return: {{"error": "insufficient_data"}}
"""
        try:
            payload_response = llm_model._call(payload_prompt)
            if isinstance(payload_response, list):
                payload_output = payload_response[0].content
            else:
                payload_output = str(payload_response)

            cleaned_payload = payload_output.strip()
            if cleaned_payload.startswith("```json"):
                cleaned_payload = cleaned_payload.replace("```json", "").replace("```", "").strip()
            elif cleaned_payload.startswith("```"):
                cleaned_payload = cleaned_payload.replace("```", "").strip()

            match = re.search(r'\{.*\}', cleaned_payload, re.DOTALL)
            if not match:
                raise ValueError("No invoice JSON found")
            payload = json.loads(match.group())

            if "error" in payload or "client_id" not in payload or "line_items" not in payload:
                automation_allowed = False
                payload = None
        except Exception:
            automation_allowed = False
            payload = None

    return {
        "automation_allowed": automation_allowed,
        "approved_action": action if automation_allowed else None,
        "confidence": conf,
        "reason": "Action auto-approved and payload validated." if automation_allowed else "Action not auto-approved or confidence too low.",
        "payload": payload,
        "llm_raw_output": ai_output[:500]  # Include for debugging
    }

