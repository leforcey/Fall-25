import pandas as pd
from skyfield.api import EarthSatellite, load, wgs84
from shapely.geometry import LineString
import geopandas as gpd
import numpy as np

# Load CSV
csv_path = 'leo_debris_sorted_cats.csv'  # Change to your CSV filename


import pandas as pd
import numpy as np
import math
from shapely.geometry import LineString
import geopandas as gpd
from skyfield.api import EarthSatellite, load

# Load timescale once
ts = load.timescale()

def get_orbit_line(tle_line1, tle_line2, num_points=25):
    satellite = EarthSatellite(tle_line1, tle_line2)
    
    # Generate times over next 90 minutes at equal intervals
    minutes = np.linspace(0, 90, num_points)
    t0 = ts.now()
    
    times = ts.utc(
        t0.utc_datetime().year,
        t0.utc_datetime().month,
        t0.utc_datetime().day,
        t0.utc_datetime().hour,
        t0.utc_datetime().minute + minutes,
        0
    )
    
    points = []
    for t in times:
        subpoint = satellite.at(t).subpoint()
        lon = subpoint.longitude.degrees
        lat = subpoint.latitude.degrees
        
        # Skip invalid or NaN coordinates
        if any([math.isnan(lon), math.isnan(lat), math.isinf(lon), math.isinf(lat)]):
            continue
        
        points.append((lon, lat))
    
    # Need at least two points to form a line
    if len(points) < 2:
        raise ValueError("Not enough valid points to create LineString")
    
    return LineString(points)

def main():
    # Load your CSV file - replace with your actual file path
    input_csv = 'leo_debris_sorted_cats.csv'  # <- change this path accordingly
    df = pd.read_csv(input_csv)
    
    lines = []
    ids = []
    
    for idx, row in df.iterrows():
        tle1 = row['TLE_LINE1']
        tle2 = row['TLE_LINE2']
        cat_id = row['NORAD_CAT_ID']
        
        try:
            line = get_orbit_line(tle1, tle2, num_points=25)
            lines.append(line)
            ids.append(cat_id)
        except Exception as e:
            print(f"Skipping satellite {cat_id} due to error: {e}")
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame({'NORAD_CAT_ID': ids, 'geometry': lines})
    
    # Set CRS to WGS84 (lat/lon)
    gdf.set_crs(epsg=4326, inplace=True)
    
    # Save to shapefile
    output_shapefile = 'satellite_orbits.shp'  # Change filename/path as needed
    gdf.to_file(output_shapefile)
    
    print(f"Saved {len(gdf)} orbit lines to {output_shapefile}")

if __name__ == "__main__":
    main()

