import pandas as pd
import re

def clean_coord_text(text):
    """Remove broken unicode characters like Â°, â€², â€³, ï»¿."""
    if pd.isna(text):
        return ""
    # Common artifacts from copy/paste or encoding issues:
    replacements = {
        "Â": "",
        "â€²": "'",
        "â€³": '"',
        "â€": "",
        "ï»¿": "",
        "°": "°",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text

def extract_decimal_latlon(coord_text):
    """
    The dataset usually has two coordinate formats separated by `/`:
    - DMS (degrees/minutes/seconds)
    - Decimal degrees (clean and easy to parse)
    
    We will always try to extract the decimal degrees after the slash.
    """
    if coord_text is None or len(coord_text) == 0:
        return None, None

    # Split at "/", decimal coordinates are usually after the slash
    parts = coord_text.split("/")
    if len(parts) > 1:
        decimal_part = parts[-1]
    else:
        decimal_part = coord_text

    # Regex to detect decimal degrees like: 30.95875°S 136.50366°E
    pattern = r"([\-0-9\.]+)\s*°?\s*([NS])\s+([\-0-9\.]+)\s*°?\s*([EW])"
    m = re.search(pattern, decimal_part.strip(), flags=re.IGNORECASE)

    if not m:
        return None, None

    lat = float(m.group(1))
    lat_hemi = m.group(2).upper()
    lon = float(m.group(3))
    lon_hemi = m.group(4).upper()

    # Apply hemisphere signs
    if lat_hemi == "S":
        lat = -abs(lat)
    if lat_hemi == "N":
        lat = abs(lat)

    if lon_hemi == "W":
        lon = -abs(lon)
    if lon_hemi == "E":
        lon = abs(lon)

    return lat, lon

# ---------------------------
# Load your dataset
# ---------------------------

df = pd.read_csv("rocket_launch_sites_merged.csv")

# Clean coordinate text
df["Coordinates_clean"] = df["Coordinates"].astype(str).apply(clean_coord_text)

# Extract lat/lon
lats = []
lons = []

for coord in df["Coordinates_clean"]:
    lat, lon = extract_decimal_latlon(coord)
    lats.append(lat)
    lons.append(lon)

df["Latitude"] = lats
df["Longitude"] = lons

# Optional: drop temporary cleaned column
# df = df.drop(columns=["Coordinates_clean"])

# Save clean version
df.to_csv("launch_sites_with_latlon.csv", index=False)

print("Done! Output saved as launch_sites_with_latlon.csv")
