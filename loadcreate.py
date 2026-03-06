import pandas as pd
import numpy as np

# ────────────────────────────────────────────────
# Base 24-hour profiles (kW) - from demand table
# ────────────────────────────────────────────────
base_households_kw = [120, 110, 100, 100, 120, 150, 200, 250, 280, 250, 200, 180,
                      160, 180, 200, 220, 250, 300, 350, 400, 380, 350, 250, 180]

base_pumps_kw      = [  0,   0,   0, 50, 100, 150, 200, 250, 300, 300, 250, 200,
                      150, 200, 250, 300, 300, 250, 200, 100, 50,   0,   0,   0]

base_agro_kw       = [  0,   0,   0,   0,   0,  50, 100, 100, 100, 150, 200, 200,
                      200, 200, 200, 200, 20, 20, 10, 10,  5,   0,   0,   0]

base_ev_kw         = [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  20,  20,
                       20,  20,  20,  20,  40,  60,  80, 100, 100,  50,  20,   0]

base_street_kw     = [ 30,  30,  30,  30,  30,  30,  20,   0,   0,   0,   0,   0,
                        0,   0,   0,   0,   0,  10,  20,  30,  30,  30,  30,  30]

# ────────────────────────────────────────────────
# Reference base counts (for scaling)
# ────────────────────────────────────────────────
base_reference = {
    'households': 400,
    'pumps':      60,
    'motors':     25,
    'street_km':  10
}

# ────────────────────────────────────────────────
# Village definitions
# ────────────────────────────────────────────────
villages = {
    'village1': {'name': 'Village 1', 'households': 600, 'pumps': 40,  'motors': 25, 'street_km': 10},
    'village2': {'name': 'Village 2', 'households': 700, 'pumps': 55,  'motors': 35, 'street_km': 12},
    'village3': {'name': 'Village 3', 'households': 500, 'pumps': 30,  'motors': 20, 'street_km': 15}
}

# ────────────────────────────────────────────────
# Full-year hourly timestamps (2023)
# ────────────────────────────────────────────────
year = 2023
snapshots = pd.date_range(start=f"{year}-01-01 00:00",
                          end=f"{year}-12-31 23:00",
                          freq="H")
n_hours = len(snapshots)

# ────────────────────────────────────────────────
# Generate profiles for normal + rabi season
# ────────────────────────────────────────────────
for village_key, info in villages.items():
    print(f"\nGenerating profiles for {info['name']} ({village_key})...")

    # Scaling factors
    scale_hh     = info['households'] / base_reference['households']
    scale_pumps  = info['pumps']      / base_reference['pumps']
    scale_motors = info['motors']     / base_reference['motors']
    scale_street = info['street_km']  / base_reference['street_km']

    # Base scaled components (24h arrays)
    hh     = np.array(base_households_kw) * scale_hh
    pumps  = np.array(base_pumps_kw)      * scale_pumps
    agro   = np.array(base_agro_kw)       * scale_motors
    ev     = np.array(base_ev_kw)         # fixed
    street = np.array(base_street_kw)     * scale_street

    # Rabi variant: double pumps (agro unchanged unless you want to double it too)
    rabi_pumps = pumps * 2.0
    # Optional: rabi_agro = agro * 2.0   # uncomment if motors/agro also increase in Rabi

    # ── Normal season ──
    total_normal_kw_24 = hh + pumps + agro + ev + street
    total_normal_kw    = np.tile(total_normal_kw_24, n_hours // 24 + 1)[:n_hours]
    df_normal = pd.DataFrame({'load_MW': total_normal_kw / 1000.0}, index=snapshots)
    df_normal.to_csv(f"{village_key}_load_normal_2023.csv", float_format="%.4f")
    print(f"  Saved normal season: {village_key}_load_normal_2023.csv   "
          f"(peak: {df_normal['load_MW'].max():.2f} MW)")

    # ── Rabi season ──
    total_rabi_kw_24 = hh + rabi_pumps + agro + ev + street   # or + rabi_agro if doubling agro
    total_rabi_kw    = np.tile(total_rabi_kw_24, n_hours // 24 + 1)[:n_hours]
    df_rabi = pd.DataFrame({'load_MW': total_rabi_kw / 1000.0}, index=snapshots)
    df_rabi.to_csv(f"{village_key}_load_rabi_2023.csv", float_format="%.4f")
    print(f"  Saved Rabi season:   {village_key}_load_rabi_2023.csv   "
          f"(peak: {df_rabi['load_MW'].max():.2f} MW)")

print("\nAll 6 CSV files generated (3 villages × 2 seasons).")