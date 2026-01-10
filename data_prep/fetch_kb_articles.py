import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import requests
import os
# Load environment variables
load_dotenv()

SERVICENOW_INSTANCE = os.getenv("SN_INSTANCE_URL")
SERVICENOW_USER = os.getenv("SN_USERNAME")
SERVICENOW_PASSWORD = os.getenv("SN_PASSWORD")
GENAI_KEY = os.getenv("GEMINI_API_KEY")

API_URL = f"{SERVICENOW_INSTANCE}/api/now/table/kb_knowledge?sysparm_limit=200"

def html_to_text(html):
    """Convert HTML article body into clean readable text."""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n").strip()

def fetch_kb_articles():
    print("📥 Fetching KB Articles from ServiceNow...")

    response = requests.get(API_URL, auth=(SERVICENOW_USER, SERVICENOW_PASSWORD))

    if response.status_code != 200:
        print(f"❌ Error fetching KB articles: {response.status_code}")
        return None

    data = response.json().get("result", [])

    records = []
    for item in data:
        raw_html = item.get("article_body", "")
        cleaned_text = html_to_text(raw_html)

        records.append({
            "Title": item.get("short_description", ""),
            "Article Body (Cleaned)": cleaned_text,
            "Category": item.get("category", "")
        })

    df = pd.DataFrame(records)
    df.to_csv("kb_articles_cleaned.csv", index=False)

    print("✅ KB Articles saved locally as kb_articles_cleaned.csv")
    print(df.head())

if __name__ == "__main__":
    fetch_kb_articles()

