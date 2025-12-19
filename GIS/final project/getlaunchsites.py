import zipfile
import os
import pandas as pd

# 1. Download dataset from Kaggle
print("Downloading dataset from Kaggle...")
os.system("kaggle datasets download -d thedevastator/rocket-launch-sites-a-comprehensive-list")

# 2. Unzip
zip_file = "rocket-launch-sites-a-comprehensive-list.zip"
extract_dir = "rocket_launch_sites_data"

print("Extracting dataset...")
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# 3. Load all CSVs into DataFrames
dfs = []
print("Loading CSV files...")
for file in os.listdir(extract_dir):
    if file.lower().endswith(".csv"):
        path = os.path.join(extract_dir, file)
        print(f"  → Loading {file}")
        dfs.append(pd.read_csv(path))

# 4. Merge all tables (outer join to preserve all fields)
print("Merging tables...")
merged = pd.concat(dfs, ignore_index=True)

# Optional cleanup: drop 100% empty columns
merged = merged.dropna(axis=1, how='all')

# 5. Save final combined CSV
output_file = "rocket_launch_sites_merged.csv"
merged.to_csv(output_file, index=False)

print(f"\n✅ DONE! Merged CSV saved as: {output_file}")
print(f"Total rows: {len(merged)}")
print(f"Total columns: {len(merged.columns)}")
