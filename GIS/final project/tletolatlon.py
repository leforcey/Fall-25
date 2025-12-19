import pandas as pd
from skyfield.api import EarthSatellite, load

# ===============================
# 1. Load CSV with TLEs
# ===============================
input_file = "full-debris-track-tle.csv"  
df = pd.read_csv(input_file)

# Required columns:
required = ["TLE1", "TLE2", "SATNAME", "NORAD_CAT_ID","ApA","PeA"]
for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing column in input CSV: {col}")

# ===============================
# 2. Setup Skyfield
# ===============================
ts = load.timescale()
t = ts.now()   # current UTC time

# ===============================
# 3. Convert TLE → lat/lon/alt
# ===============================
lats, lons, alts = [], [], []

for idx, row in df.iterrows():
    try:
        sat = EarthSatellite(row["TLE1"], row["TLE2"], row["SATNAME"], ts)
        geo = sat.at(t)
        sub = geo.subpoint()

        lats.append(sub.latitude.degrees)
        lons.append(sub.longitude.degrees)
        alts.append(sub.elevation.km)    # altitude above Earth center, in km

    except Exception as e:
        print(f"Warning: failed TLE at row {idx}: {e}")
        lats.append(None)
        lons.append(None)
        alts.append(None)

# ===============================
# 4. Output GIS-ready CSV
# ===============================
output_file = "LEO_Debris_With_Pos_Alt.csv"

out_df = pd.DataFrame({
    "NORAD_CAT_ID": df["NORAD_CAT_ID"],
    "SATNAME": df["SATNAME"],
    "PER_ALT": df["PeA"],
    "APO_ALT": df["ApA"],
    "LAT": lats,
    "LON": lons,
    "ALT_KM": alts,
    "TLE_LINE1": df["TLE1"],
    "TLE_LINE2": df["TLE2"]
})

out_df.to_csv(output_file, index=False)

print("CSV export complete!")
print(f"Saved to: {output_file}")

