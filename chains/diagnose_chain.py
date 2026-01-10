
from utils.vector_store import search_similar
from utils.llm_utils import generate_llm_response, llm_model
from mcp_agents.tools import retry_order_mcp, sync_customer_data_mcp, fix_asset_mismatch_mcp

def predict_assignment_group(query: str, similar_items: list) -> str:
    """
    Predict assignment group using metadata from the most similar incident.
    Fallback to LLM if metadata is missing.
    """
    
    # ✅ Check top match
    if similar_items:
        top_item = similar_items[0]
        if top_item.get("source") == "incident" and top_item.get("assignment_group") not in ["Not Provided", "Not Applicable"]:
            return top_item["assignment_group"]


    # ✅ Fallback: Use LLM if metadata missing or Use LLM if top match is KB or assignment group missing
    if llm_model:
        context = "\n".join([
            f"{item['source']} {i+1}: {item['training_text']}"
            for i, item in enumerate(similar_items[:3])
        ])
        prompt = f"""
        You are an IT support assistant.
        USER ISSUE: {query}
        SIMILAR CONTEXT:
        {context}

        Predict the most appropriate assignment group for handling this issue.
        Examples: Application Support, Network Support, Database Team, Service Desk.
        Return ONLY the group name.
        """
        try:
            response = llm_model.generate_content(prompt)
            return response.text.strip()
        except Exception:
            return "Service Desk"

    return "Service Desk"  # Final fallback


def diagnose_issue(query: str, top_k: int = 5, configuration_item: str = "") -> dict:
    """
    Main pipeline: search similar incidents, generate AI suggestion, predict assignment group.
    """
    similar_items = search_similar(query, top_k)
    ai_suggestion = generate_llm_response(query, similar_items)
    assignment_group = predict_assignment_group(query, similar_items)

    # ---------------------------------------------------
    # ✅ STEP 3: CALL MCP TOOL BASED ON THE CONFIGURATION ITEM
    # ---------------------------------------------------
    mcp_action_result = None

    from mcp_agents.tools import (
        retry_order_mcp,
        sync_customer_data_mcp,
        fix_asset_mismatch_mcp
    )

    if configuration_item.upper() in ["ROD-OSM", "OSM"]:
        mcp_action_result = retry_order_mcp(query)

    elif configuration_item.upper() in ["OURTELCO", "CRM", "SIE-CRM"]:
        mcp_action_result = sync_customer_data_mcp(query)

    elif configuration_item.upper() in ["ROD-BRM", "BRM"]:
        mcp_action_result = fix_asset_mismatch_mcp(query)

    else:
        mcp_action_result = "No automation available for this CI"

    # ---------------------------------------------------

    return {
        "query": query,
        "configuration_item": configuration_item,
        "ai_suggestion": ai_suggestion,
        "assignment_group": assignment_group,
        "similar_items": similar_items,
        "mcp_action_result": mcp_action_result   # <-- STEP 3 OUTPUT
    }


