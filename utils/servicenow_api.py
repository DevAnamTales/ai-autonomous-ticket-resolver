
# utils/servicenow_api.py
import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env (if present)
load_dotenv()

# Read and normalize ServiceNow credentials/settings
SNOW_INSTANCE = (os.getenv("SERVICENOW_INSTANCE") or "").rstrip("/")   # e.g., https://yourinstance.service-now.com
SNOW_USER = os.getenv("SERVICENOW_USERNAME")
SNOW_PASS = os.getenv("SERVICENOW_PASSWORD")

# Common Incident state codes (adjust if your PDI differs)
# Typical defaults: 1=New, 2=In Progress, 3=On Hold, 6=Resolved, 7=Closed
STATE_IN_PROGRESS = 2
STATE_RESOLVED = 6

def _headers():
    """Default JSON headers for ServiceNow API."""
    return {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

def _patch_incident(sys_id: str, payload: dict, timeout: int = 15):
    """
    Low-level helper to PATCH an incident.
    Returns (ok_bool, response_json_or_text).
    """
    # Validate env config early
    if not (SNOW_INSTANCE and SNOW_USER and SNOW_PASS):
        return False, {
            "error": "Missing SERVICENOW_INSTANCE / SERVICENOW_USERNAME / SERVICENOW_PASSWORD env vars",
            "instance": SNOW_INSTANCE,
            "user_set": SNOW_USER is not None,
            "password_set": SNOW_PASS is not None
        }

    url = f"{SNOW_INSTANCE}/api/now/table/incident/{sys_id}"
    print(f"[SNOW] PATCH {url} payload={payload}")

    try:
        resp = requests.patch(
            url,
            auth=(SNOW_USER, SNOW_PASS),
            headers=_headers(),
            json=payload,
            timeout=timeout
        )
        print(f"[SNOW] -> status={resp.status_code}")
    except requests.exceptions.RequestException as e:
        # Network/timeouts/connection issues
        return False, {"error": f"Network error updating ServiceNow: {e}"}

    # Parse JSON if possible; otherwise return raw text
    try:
        data = resp.json()
    except ValueError:
        data = resp.text

    ok = 200 <= resp.status_code < 300
    if not ok and isinstance(data, dict):
        # Add status code to error payload for easier debugging
        data.setdefault("http_status", resp.status_code)

    return ok, data

def update_ticket_v2(ticket_id: str, update_type: str, message: str):
    """
    Flexible updater for Incident records.

    update_type:
      - "worknote": add work_notes (internal note).
      - "resolve" : set state=Resolved (6), add close_notes + work_notes.
      - "escalate": set state=In Progress (2) + work_notes (no assignment).

    Returns (ok_bool, response_json_or_text).
    """
    update_type = (update_type or "").strip().lower()
    message = message or ""

    if update_type == "worknote":
        return _patch_incident(ticket_id, {"work_notes": message})

    if update_type == "resolve":
        payload = {
            "state": STATE_RESOLVED,
            "close_notes": message,
            "work_notes": f"Resolution details:\n{message}",
        }
        return _patch_incident(ticket_id, payload)

    if update_type == "escalate":
        payload = {
            "state": STATE_IN_PROGRESS,
            "work_notes": f"Escalated to human: {message}",
        }
        return _patch_incident(ticket_id, payload)

    return False, {"error": f"Unknown update_type '{update_type}'"}

def set_fields_and_note(ticket_id: str, field_updates: dict, note_text: str):
    """
    Sets one or more fields (e.g., custom fields) and adds a work note
    in a single PATCH.

    Example:
        field_updates = {"u_ai_suggestion": "text from AI"}
        note_text = "AI suggestion stored and context logged."

    Returns (ok_bool, response_json_or_text).
    """
    payload = dict(field_updates or {})
    payload["work_notes"] = note_text or ""
    return _patch_incident(ticket_id, payload)

