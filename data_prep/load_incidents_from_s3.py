
import boto3
import pandas as pd
from io import BytesIO

BUCKET_NAME = "ai-incident-record"
FILE_KEY = "LatestdemoData.xlsx"

def load_excel_from_s3(bucket, key):
    s3 = boto3.client("s3")
    print(f"Fetching file '{key}' from bucket '{bucket}'...")
    
    excel_obj = s3.get_object(Bucket=bucket, Key=key)
    file_content = excel_obj["Body"].read()
    
    df = pd.read_excel(BytesIO(file_content))
    return df

if __name__ == "__main__":
    # Load Excel from S3
    df = load_excel_from_s3(BUCKET_NAME, FILE_KEY)

    print("\n✅ File Loaded Successfully!")
    print(f"Rows Loaded: {len(df)}\n")
    print("📌 Preview of Data:")
    print(df.head())
    print("\n📌 Available Columns:")
    print(df.columns.tolist())

    # ---- PREPROCESS ----
    # Keep only rows where Short description exists
    df = df.dropna(subset=["Short description"])

    # Fill missing values in required columns
    df["Resolution notes"] = df["Resolution notes"].fillna("No resolution provided.")
    df["Assignment group"] = df["Assignment group"].fillna("Not Provided")
    df["Configuration item"] = df["Configuration item"].fillna("Not Provided")

    # Select required columns
    selected_columns = ["Number", "Short description", "Assignment group", "Configuration item", "Resolution notes"]
    df_selected = df[selected_columns]

    # Create training_text column for embeddings
    df_selected["training_text"] = (
        "Issue: " + df_selected["Short description"].astype(str)
        + "\nResolution: " + df_selected["Resolution notes"].astype(str)
        + "\nAssignment Group: " + df_selected["Assignment group"].astype(str)
        + "\nConfiguration Item: " + df_selected["Configuration item"].astype(str)
    )

    print("\n🧠 Sample Training Text:")
    print(df_selected["training_text"].head())

    # Save both metadata and training_text
    df_selected.to_csv("selected_incidents_with_training_text.csv", index=False)
    print("\n✅ File saved as selected_incidents_with_training_text.csv")

