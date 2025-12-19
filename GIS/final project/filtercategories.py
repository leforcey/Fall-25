import pandas as pd

input_file = "LEO_Debris_With_Pos_Alt.csv"  
df = pd.read_csv(input_file)

# Step 1: Clean the SATNAME field
df['SATNAME_CLEAN'] = (
    df['SATNAME']
    .astype(str)
    .str.encode('ascii', 'ignore')  # remove non-ASCII
    .str.decode('ascii')
    .str.strip()                     # remove leading/trailing spaces
)

# Step 2: Extract a "group" from the first word
df['SAT_GROUP'] = df['SATNAME_CLEAN'].str.split().str[0]

# Step 3: Optional: inspect how many groups
print(df['SAT_GROUP'].nunique())
print(df['SAT_GROUP'].value_counts())

# Step 4: Save back to CSV for ArcGIS
df.to_csv('leo_debris_sorted_cats.csv', index=False)
