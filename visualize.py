import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
INPUT_FILE = "final_high_quality_features.csv"
LABEL_COLUMN = "label"  # Change this if your CSV uses 'generated' or 'is_ai'

# 1. Load Data
try:
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} rows.")
except FileNotFoundError:
    print("Error: Could not find processed_data.csv")
    exit()

# Check if label column exists
if LABEL_COLUMN not in df.columns:
    print(f"ERROR: Could not find column '{LABEL_COLUMN}' in your CSV.")
    print(f"Your columns are: {list(df.columns)}")
    print("Please update the LABEL_COLUMN variable in the script.")
    exit()

# 2. Set the style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})

# 3. Define the features we want to compare
features_to_plot = ['avg_sent_len', 'vocab_richness', 'burstiness', 'readability']

print("Generating graphs...")

# 4. Loop through features and create a graph for each
for feature in features_to_plot:
    plt.figure(figsize=(10, 6))
    
    # Create the histogram
    # Label 0 = Human (Blue), Label 1 = AI (Red)
    sns.histplot(data=df, x=feature, hue=LABEL_COLUMN, kde=True, stat="density", common_norm=True, alpha=0.5)    
    plt.title(f"Human (0) vs AI (1): {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    
    # Save the graph as an image file
    filename = f"graph_{feature}.png"
    plt.savefig(filename)
    print(f"Saved {filename}")
    plt.close()

print("\n--- DONE ---")
print("Open the PNG images in your folder to see the results.")
