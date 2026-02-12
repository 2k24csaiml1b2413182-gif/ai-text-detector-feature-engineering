import pandas as pd
import os

# Load Data
df = pd.read_csv("input_data.csv")

# Filter for AI rows (assuming the label is 'ai' based on your graph)
ai_rows = df[df['category'] == 'ai']

if len(ai_rows) > 0:
    # Get the address of the VERY FIRST AI file
    target_path = ai_rows.iloc[0]['file_path']
    
    print("\n--- DETECTIVE REPORT ---")
    print(f"1. The CSV is looking for this path:  {target_path}")
    print(f"2. Is the file actually there?        {os.path.exists(target_path)}")
    
    # Check what is inside your 'ai' folder
    if os.path.exists("ai"):
        print(f"3. First file inside your 'ai' folder: {os.listdir('ai')[0]}")
    else:
        print("3. Python cannot find a folder named 'ai' next to main.py")
else:
    print("Could not find any rows labeled 'ai' in the CSV.")
