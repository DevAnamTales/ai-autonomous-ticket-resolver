
import pandas as pd

# Load both files
incidents_file = "selected_incidents_with_training_text.csv"
kb_file = "kb_articles_cleaned.csv"

df_incidents = pd.read_csv(incidents_file)
df_kb = pd.read_csv(kb_file)

print(f"✅ Incidents loaded: {len(df_incidents)} rows")
print(f"✅ KB Articles loaded: {len(df_kb)} rows")

# Prepare incidents DataFrame (keep Number as id)
df_incidents_combined = pd.DataFrame({
    "id": df_incidents["Number"],  # Original incident ID
    "source": "incident",
    "training_text": df_incidents["training_text"],
    "Assignment group": df_incidents.get("Assignment group", "Not Provided"),
    "Configuration item": df_incidents.get("Configuration item", "Not Provided"),
    "Category": "Not Applicable"
})

# Prepare KB articles DataFrame (use Title or generate KB_ prefix)
df_kb_combined = pd.DataFrame({
    "id": "KB_" + (df_kb.index + 1).astype(str),  # Generate KB IDs like KB_1, KB_2
    "source": "kb_article",
    "training_text": (
        "Title: " + df_kb["Title"].astype(str) +
        "\nContent: " + df_kb["Article Body (Cleaned)"].astype(str)
    ),
    "Assignment group": "Not Applicable",
    "Configuration item": "Not Applicable",
    "Category": df_kb["Category"]
})

# Combine both
df_combined = pd.concat([df_incidents_combined, df_kb_combined], ignore_index=True)

print(f"\n✅ Combined dataset rows: {len(df_combined)}")
print("📌 Preview:")
print(df_combined.head())

# Save combined file
output_file = "combined_training_data.csv"
df_combined.to_csv(output_file, index=False)
print(f"\n✅ Combined file saved as {output_file}")

