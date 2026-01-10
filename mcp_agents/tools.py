# mcp_agents/tools.py

import requests

# MCP TOOL WRAPPERS (simple LangChain-callable functions)

def retry_order_mcp(input_text: str):
    """
    Expects: order_id inside input_text
    Example: "order_id=12345"
    """
    order_id = input_text.strip()
    resp = requests.post(
        "http://localhost:7001/retry-order",
        json={"order_id": order_id}
    )
    return resp.json()

def sync_customer_data_mcp(input_text: str):
    """
    Expects: customer_id inside input_text
    """
    customer_id = input_text.strip()
    resp = requests.post(
        "http://localhost:7002/sync-customer-data",
        json={"customer_id": customer_id}
    )
    return resp.json()

def fix_asset_mismatch_mcp(input_text: str):
    """
    Expects: asset_id inside input_text
    """
    asset_id = input_text.strip()
    resp = requests.post(
        "http://localhost:7003/fix-asset",
        json={"asset_id": asset_id}
    )
    return resp.json()

def create_invoice_mcp(invoice_data: dict):
    """
    Calls Ninjainvoice API to create invoice. 
    Can be triggered by your LangChain agent if approved.
    """
    import requests, os
    API_KEY = os.getenv("NINJAINVOICE_API_KEY")
    BASE_URL = os.getenv("NINJA_URL")

    headers = {"Content-Type": "application/json", 
        "X-Requested-With": "XMLHttpRequest", "X-API-TOKEN": API_KEY}
    response = requests.post(f"{BASE_URL}/invoices", json=invoice_data, headers=headers)

    if response.status_code == 200:
        return {"status": "success", "message": "Invoice created successfully", "invoice_id": response.json().get("id")}
    else:
        return {"status": "failure", "message": response.text}

