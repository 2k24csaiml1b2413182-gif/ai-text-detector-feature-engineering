import pandas as pd

# 1. Load the processed data
try:
    df = pd.read_csv("processed_data.csv")
    print("--- DATA HEALTH REPORT ---")
except:
    print("Error: Could not find processed_data.csv")
    exit()

# 2. Check the Balance (Human vs AI)
print(f"\nTotal Rows: {len(df)}")
print("Count by Category:")
print(df['category'].value_counts())

# 3. Check for Zeros (Did the feature extraction work?)
# We look at 'vocab_richness'. If it's 0, the text was empty.
zeros = df[df['vocab_richness'] == 0]
print(f"\nRows with ALL ZEROS (Failed to process): {len(zeros)}")

# 4. Check the Averages
print("\n--- AVERAGE SCORES ---")
print(df.groupby('category')[['avg_sent_len', 'vocab_richness', 'burstiness']].mean())

print("\n----------------------")
