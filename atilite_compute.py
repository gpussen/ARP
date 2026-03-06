import atlite
import geopandas as gpd
from shapely.geometry import shape
import pandas as pd
import os
import time
from requests.exceptions import SSLError, ConnectionError

# ========================
# CONFIG
# ========================
cutout_path = "nashik_era5_2023.nc"
turbine_name = "NREL_ReferenceTurbine_5MW_offshore"  # Verify with: print(list(atlite.windturbines.keys())) if error occurs
panel_name   = "CSi"

# Delete old cutout if exists (avoids metadata corruption)
if os.path.exists(cutout_path):
    print(f"Removing old incomplete cutout: {cutout_path}")
    os.remove(cutout_path)

# ========================
# 1. Define buffered bounds (essential for ERA5 0.25° grid)
buffer_deg = 0.35
lon_min, lat_min, lon_max, lat_max = 74.1025, 20.029218, 74.131218, 20.056198
bounds = (
    lon_min - buffer_deg,
    lat_min - buffer_deg,
    lon_max + buffer_deg,
    lat_max + buffer_deg
)
print(f"Using buffered bounds: lon {bounds[0]:.3f}–{bounds[2]:.3f}, lat {bounds[1]:.3f}–{bounds[3]:.3f}")

# ========================
# 2. Create fresh cutout
cutout = atlite.Cutout(
    path=cutout_path,
    module="era5",
    x=slice(bounds[0], bounds[2]),
    y=slice(bounds[1], bounds[3]),
    time=slice(pd.Timestamp("2023-01-01 00:00"), pd.Timestamp("2023-12-31 23:00"))
)

# ========================
# 3. Prepare / download features with retry on transient errors
print("Preparing cutout (downloads via CDS API – may take 30–120+ min depending on server)...")
max_retries = 5
for attempt in range(1, max_retries + 1):
    try:
        cutout.prepare(
            features=["height", "wind", "influx", "temperature", "runoff"],
            monthly_requests=True,          # splits into ~12 monthly chunks → safer
            concurrent_requests=False,      # avoid overwhelming connection
            show_progress=True
        )
        print("Cutout prepared successfully!")
        break
    except (SSLError, ConnectionError) as e:
        print(f"Attempt {attempt}/{max_retries} failed with network/SSL error: {e}")
        if attempt == max_retries:
            raise RuntimeError("Max retries reached. Check CDS status, network, or try later.")
        wait_seconds = 60 * (2 ** (attempt - 1))  # 60s → 120s → 240s → 480s → 960s
        print(f"Retrying in {wait_seconds // 60} minutes...")
        time.sleep(wait_seconds)
    except Exception as e:
        print(f"Unexpected error during prepare: {e}")
        raise

# Quick sanity check
print("Time range:", cutout.data.time[0].values, "→", cutout.data.time[-1].values)
print("Grid shape:", cutout.grid.shape)  # Expect something like (N_lon, N_lat) with N > 1

if cutout.grid.shape[0] < 2 or cutout.grid.shape[1] < 2:
    raise ValueError("Grid too small – increase buffer_deg and re-run.")

# ========================
# 4. Define exact target polygon
geojson_geom = {
    "type": "Polygon",
    "coordinates": [[
        [74.1025, 20.029218],
        [74.1025, 20.056198],
        [74.131218, 20.056198],
        [74.131218, 20.029218],
        [74.1025, 20.029218]
    ]]
}

gdf = gpd.GeoDataFrame(
    index=["target_area"],
    geometry=[shape(geojson_geom)],
    crs="EPSG:4326"
)

# ========================
# 5. Compute capacity factors
print("Computing wind capacity factors...")
wind_cf = cutout.wind(
    turbine=turbine_name,
    shapes=gdf,
    per_unit=True
).sel(dim_0="target_area").to_pandas().squeeze()

print("Computing solar capacity factors...")
solar_cf = cutout.pv(
    panel=panel_name,
    orientation="latitude_optimal",
    shapes=gdf,
    per_unit=True
).sel(dim_0="target_area").to_pandas().squeeze()

# ========================
# 6. Save results
wind_cf.to_csv("wind_atlite_2023.csv", header=["capacity_factor"])
solar_cf.to_csv("solar_atlite_2023.csv", header=["capacity_factor"])

print("Done!")
print(f"Files saved: wind_atlite_2023.csv & solar_atlite_2023.csv")
print(f"Timesteps: {len(wind_cf)} (should be ~8760 for full 2023 hourly)")