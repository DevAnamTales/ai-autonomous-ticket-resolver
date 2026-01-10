# app.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  
from flask import Flask, request, jsonify
from utils.servicenow_api import update_ticket_v2, set_fields_and_note
from chains.diagnose_chain import diagnose_issue
from chains.agent_chain import (
    create_incident_agent,
    process_incident,
    tools  # ✅ Import tools from agent_chain
)

# Import MCP functions directly
from mcp_agents.tools import (
    retry_order_mcp,
    sync_customer_data_mcp,
    fix_asset_mismatch_mcp,
    create_invoice_mcp
)

app = Flask(__name__)

# Initialize LangChain Agent
agent = create_incident_agent()

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "service": "ServiceNow RAG API"})

@app.route("/incident", methods=["POST"])
def search_incident():
    data = request.get_json()
    query = data.get("query", "")
    configuration_item = data.get("configuration_item", "")
    # Accept both keys so Postman can send either
    ticket_id = data.get("ticket_id") or data.get("sys_id")
    top_k = data.get("top_k", 5)

    if not query.strip():
        return jsonify({"status": "error", "message": "Query required"}), 400

    # --------------------------------------------------------------
    # STEP 1 — RAG Pipeline (retrieve similar incidents)
    # --------------------------------------------------------------
    rag_result = diagnose_issue(query, top_k, configuration_item)

    # --------------------------------------------------------------
    # STEP 2 — Decision Engine (safe automation)
    # --------------------------------------------------------------
    decision = process_incident(query, configuration_item, agent)

    # --------------------------------------------------------------
    # STEP 3 — If automation is allowed → run MCP tool
    # --------------------------------------------------------------
    mcp_result = None
    if decision.get("automation_allowed"):
        action = decision.get("approved_action")
        payload = decision.get("payload")

        # ✅ Map action names to actual MCP functions
        action_map = {
            "retry_order": retry_order_mcp,
            "update_order": retry_order_mcp,  # Same function handles both
            "sync_customer_data": sync_customer_data_mcp,
            "fix_asset_mismatch": fix_asset_mismatch_mcp,
            "create_invoice": create_invoice_mcp,
            "update_invoice": create_invoice_mcp  # Same function handles both
        }

        try:
            mcp_func = action_map.get(action)

            if not mcp_func:
                mcp_result = {
                    "status": "error",
                    "message": f"No MCP function mapped for action: {action}"
                }
            else:
                # For invoice actions, pass payload
                if action in ["create_invoice", "update_invoice"]:
                    if payload:
                        mcp_result = mcp_func(payload)
                    else:
                        mcp_result = {
                            "status": "error",
                            "message": "Invoice action requires payload"
                        }
                else:
                    # For other actions, pass the query or CI
                    mcp_result = mcp_func(query)

        except Exception as e:
            mcp_result = {"status": "error", "message": str(e)}

    # --------------------------------------------------------------
    # STEP 4 — Build Final Response
    # --------------------------------------------------------------
    final_output = {
        "query": query,
        "configuration_item": configuration_item,
        "similar_items": rag_result.get("similar_items", []),
        "ai_suggestion": rag_result.get("ai_suggestion", ""),
        "decision_engine": decision,
        "automation_triggered": decision.get("automation_allowed", False),
        "mcp_action_result": mcp_result,
    }

    # --------------------------------------------------------------
    # STEP 5 — Update ServiceNow Ticket (if ticket_id provided)
    # --------------------------------------------------------------
    AI_SUGGESTION_FIELD = os.getenv("AI_SUGGESTION_FIELD", "u_ai_suggestion")
    CONFIDENCE_THRESHOLD = float(os.getenv("AI_CONFIDENCE_THRESHOLD", "0.9"))

    if ticket_id:
        # Pull values from your final_output (as in your Postman response)
        ai_suggestion = final_output.get("ai_suggestion", "")
        decision      = (final_output.get("decision_engine") or {})
        mcp_result    = (final_output.get("mcp_action_result") or {})  # may be None → {}

        automation_allowed = bool(decision.get("automation_allowed", False))
        confidence         = float(decision.get("confidence") or 0.0)
        decision_reason    = decision.get("reason") or "No decision reason provided."
        llm_raw            = decision.get("llm_raw_output") or ""
        mcp_status         = mcp_result.get("status")
        mcp_message        = mcp_result.get("message", "No MCP message")
        approved_action    = decision.get("approved_action") or "N/A"
        payload_summary    = decision.get("payload")

        # 1) Store AI suggestion in custom field + add a context work note
        base_note = (
            f"AI suggestion saved to '{AI_SUGGESTION_FIELD}'.\n"
            f"Decision reason: {decision_reason}\n"
            f"MCP status: {mcp_status}\n"
            f"MCP message: {mcp_message}"
        )
        ok_set, sn_resp_set = set_fields_and_note(
            ticket_id=ticket_id,
            field_updates={AI_SUGGESTION_FIELD: ai_suggestion},
            note_text=base_note
        )
        final_output["ticket_ai_field_update_ok"] = ok_set
        final_output["ticket_ai_field_update_resp"] = sn_resp_set

        # 2) Auto-resolve if automation succeeded and confidence is high
        success_criteria = automation_allowed and (mcp_status == "success") and (confidence >= CONFIDENCE_THRESHOLD)

        if success_criteria:
            resolution_text = (
                "Resolved by AI automation.\n"
                f"Approved action: {approved_action}\n"
                f"Confidence: {confidence:.2f}\n"
                f"Payload: {payload_summary}\n"
                f"MCP Result: {mcp_result}\n"
            )
            ok_resolve, sn_resp_resolve = update_ticket_v2(ticket_id, "resolve", resolution_text)
            final_output["ticket_update_status"] = "resolved" if ok_resolve else "resolve_failed"
            final_output["ticket_update_response"] = sn_resp_resolve

        else:
            # 3) On fail/low confidence/not allowed → write detailed failure context to Work Notes and keep ticket in progress
            fail_bits = []
            if not automation_allowed: fail_bits.append("automation not allowed")
            if mcp_status != "success": fail_bits.append(f"MCP status: {mcp_status}")
            if confidence < CONFIDENCE_THRESHOLD: fail_bits.append(f"low confidence ({confidence:.2f} < {CONFIDENCE_THRESHOLD})")
            fail_reason = ", ".join(fail_bits) or "Unspecified"

            failure_note = (
                "AI could not auto-resolve.\n"
                f"Failure reason: {fail_reason}\n\n"
                f"AI Failure Message (from suggestion):\n{ai_suggestion}\n\n"
                f"Decision reason: {decision_reason}\n"
                f"LLM raw output: {llm_raw}\n"
                f"Payload: {payload_summary}\n"
                f"MCP Result: {mcp_result}\n"
            )
            ok_escalate, sn_resp_escalate = update_ticket_v2(ticket_id, "escalate", failure_note)
            final_output["ticket_update_status"] = "escalated" if ok_escalate else "escalate_failed"
            final_output["ticket_update_response"] = sn_resp_escalate

    # Always return the final output
    return jsonify(final_output), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

