# filename: main.py
import pandas as pd
from features import FeatureEngine
import numpy as np

print("--- STARTING PIPELINE (DIRECT TEXT MODE) ---")

# 1. Load the Dataset
# Make sure your new downloaded file is renamed to 'new_dataset.csv'
filename = "new_dataset.csv" 

try:
    df = pd.read_csv(filename)
    print(f"Loaded {len(df)} rows from {filename}.")
except FileNotFoundError:
    print(f"ERROR: Could not find '{filename}'. Did you download the new dataset?")
    exit()

# 2. Smart Column Detection
# We need to find which column holds the essay and which holds the label
possible_text_cols = ['text', 'essay', 'content', 'source_text']
possible_label_cols = ['label', 'generated', 'class', 'target']

text_col = next((col for col in possible_text_cols if col in df.columns), None)
label_col = next((col for col in possible_label_cols if col in df.columns), None)

if not text_col:
    print(f"CRITICAL ERROR: Could not find a text column. Your columns are: {list(df.columns)}")
    exit()
else:
    print(f"-> Found text column: '{text_col}'")

# 3. Extract Features
print("Extracting features... (This may take a moment)")

results = []
cleaned_texts = [] # We'll store the clean text to verify quality later

for index, row in df.iterrows():
    # Get the raw text directly from the cell
    raw_text = str(row[text_col])
    
    # Run the Engine
    engine = FeatureEngine(raw_text)
    features = engine.extract_all()
    
    results.append(features)
    
    # Progress Check
    if index % 1000 == 0 and index > 0:
        print(f"Processed {index} rows...")

# 4. Save Results
features_df = pd.DataFrame(results)

# Combine original label with new features (we don't need the massive text column anymore)
# If we found a label column, keep it.
if label_col:
    final_df = pd.concat([df[[label_col]], features_df], axis=1)
else:
    # If no label found, just save features (unsupervised)
    final_df = features_df

output_file = "final_high_quality_features.csv"
final_df.to_csv(output_file, index=False)

print(f"\n--- SUCCESS ---")
print(f"Saved features to '{output_file}'.")
print("Send THIS file to Person 3. It will train a much smarter model.")
