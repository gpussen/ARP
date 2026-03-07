# packages used
import pypsa
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# CONFIG & PATHS

year = 2023
snapshots = pd.date_range(f"{year}-01-01 00:00", f"{year}-12-31 23:00", freq="h")

wind_csv = "wind_atlite_2023.csv"
solar_csv = "solar_atlite_2023.csv"
wind_conf_file = "wind_conf.txt"
solar_conf_file = "solar_conf.txt"

# Village load CSVs (from your generation script)
villages = ['village1', 'village2', 'village3']
normal_csvs = {v: f"{v}_load_normal_2023.csv" for v in villages}
rabi_csvs = {v: f"{v}_load_rabi_2023.csv" for v in villages}

exchange_rate = 83  # INR per USD, adjust as needed (as of March 2026 approx.)

# Line efficiency assumption for 5% losses (5% range; adjust to 0.75-0.80 if needed)
line_efficiency = 0.95

exchange_rate = 83  # INR per USD, adjust as needed (as of March 2026 approx.)

# 1. Load capacity factors (per-unit, hourly)

wind_cf = pd.read_csv(wind_csv, index_col=0, parse_dates=True).squeeze()   # assumes datetime index + one column
solar_cf = pd.read_csv(solar_csv, index_col=0, parse_dates=True).squeeze()

# Ensure correct length & index alignment
assert len(wind_cf) == len(snapshots), "Wind CF length mismatch"
assert len(solar_cf) == len(snapshots), "Solar CF length mismatch"

# Reindex to be safe
wind_cf = wind_cf.reindex(snapshots).ffill().bfill()
solar_cf = solar_cf.reindex(snapshots).ffill().bfill()

# 2. Load & prepare demand profiles (With added variance)

village_loads = {}

for v in villages:
    normal = pd.read_csv(normal_csvs[v], index_col=0, parse_dates=True)['load_MW']
    rabi = pd.read_csv(rabi_csvs[v], index_col=0, parse_dates=True)['load_MW']
    
    normal = normal.reindex(snapshots).ffill().bfill()
    rabi = rabi.reindex(snapshots).ffill().bfill()
    
    # --- ADDED VARIANCE HERE ---
    # 1. Added 5% random noise so they aren't identical flat lines
    noise = np.random.normal(1.0, 0.05, len(snapshots)) 
    # 2. Optional: Shift the load by a random hour (-1, 0, or 1) so peaks don't align perfectly
    shift_val = np.random.choice([-1, 0, 1])
    
    combined = normal.copy()
    rabi_mask = (combined.index.month.isin([11, 12, 1, 2]))
    combined.loc[rabi_mask] = rabi.loc[rabi_mask]
    
    # Apply noise and shift
    village_loads[v] = (combined * noise).shift(shift_val).ffill().bfill()


# 3. Sugar Mill Load Profile (Seasonal)

# 1. Initialize base maintenance/off-season load (0.5 MW)
sugar_mill_load = pd.Series(0.5, index=snapshots)

# 2. Identify the crushing season (Nov 1st to April 30th)
crushing_season_mask = (sugar_mill_load.index.month >= 11) | (sugar_mill_load.index.month <= 4)

# 3. Identify working hours (9 AM to 5 PM)
# Note: hour < 17 ensures the load drops back to 0.5 MW at exactly 17:00 (5 PM)
working_hours_mask = (sugar_mill_load.index.hour >= 9) & (sugar_mill_load.index.hour < 17)

# 4. Set crushing load only if BOTH season AND working hours match
sugar_mill_load[crushing_season_mask & working_hours_mask] = 2.0

# 5. Add small hourly noise/variance (±5%)
np.random.seed(42)
sugar_mill_load *= np.random.uniform(0.95, 1.05, size=len(snapshots))

# 6. Reindex and fill 
sugar_mill_load = sugar_mill_load.reindex(snapshots).fillna(0)

# 3. Read technology configs & compute annualized costs (convert to INR)

def read_tech_conf(filename):
    conf = {}
    with open(filename, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Remove inline comment if present
            if '#' in line:
                line = line.split('#', 1)[0].strip()
            if '=' in line:
                k, v = line.split("=", 1)
                try:
                    conf[k.strip()] = float(v.strip())
                except ValueError as e:
                    print(f"Warning: Could not parse value for {k.strip()}: {v.strip()} → skipping")
    return conf

wind_conf = read_tech_conf(wind_conf_file)
solar_conf = read_tech_conf(solar_conf_file)



# Assume original conf in USD – convert capital_cost, fom, vom to INR
#for conf in [wind_conf, solar_conf]:
    # conf["capital_cost"] *= exchange_rate
    # conf["fom"] = conf.get("fom", 0) * exchange_rate
    # conf["vom"] = conf.get("vom", 0) * exchange_rate

def annualized_capex(capital_cost, lifetime, wacc):
    crf = wacc / (1 - (1 + wacc) ** -lifetime)   # capital recovery factor
    return capital_cost * crf

wind_ann_capex = annualized_capex(**{k: wind_conf[k] for k in ["capital_cost", "lifetime", "wacc"]})
solar_ann_capex = annualized_capex(**{k: solar_conf[k] for k in ["capital_cost", "lifetime", "wacc"]})

# sanity check
print("Solar CF stats:", solar_cf.describe())
print("Wind CF stats:", wind_cf.describe())
print("Any CF >1 or <0?", (solar_cf > 1).any() or (wind_cf > 1).any())
print("Wind CF mean:", wind_cf.mean())

# Battery config (example values in INR – create battery_conf.txt or adjust here)
battery_conf = {
    "capital_cost_power": 10000,  # INR/kW (converter, ~600 USD/kW)
    "capital_cost_energy": 7000, # INR/kWh (cells, ~120 USD/kWh)
    "lifetime": 8,
    "fom": 1000,                 # INR/kW/year
    "vom": 0,
    "wacc": 0.07,
    "efficiency": 0.7,          # round-trip (store = dispatch = sqrt(round-trip))
    "max_hours": 8               # E/P ratio (hours at full power)
}

battery_ann_capex_power = annualized_capex(battery_conf["capital_cost_power"], battery_conf["lifetime"], battery_conf["wacc"])
battery_ann_capex_energy = annualized_capex(battery_conf["capital_cost_energy"], battery_conf["lifetime"], battery_conf["wacc"])
# Effective capital_cost for StorageUnit (annualized INR/kW)
battery_capital_cost = battery_ann_capex_power + (battery_ann_capex_energy / battery_conf["max_hours"]) + battery_conf["fom"]

# Coal config (Assumed values)
# coal_conf = {
#    "capital_cost": 100000 ,  # INR/kW
#    "fom": 3000,  # INR/kW/year #Fixed O & M
#    "vom": 300 ,  # INR/MWh   # Variable O & M
#    "marginal_cost": 3000,  # INR/MWh (fuel + other variable costs)
#    "lifetime": 30,   # Avg lifetime of a plant
#    "wacc": 0.1   # Weighted Average Cost of Capital (Interest rate)
#}
#coal_ann_capex = annualized_capex(coal_conf["capital_cost"], coal_conf["lifetime"], coal_conf["wacc"])

# Nuclear_SMR config (Assumed values)
# nsmr_conf = {
#    "capital_cost": 300000 ,  # INR/kW
#    "fom": 9000,  # INR/kW/year #Fixed O & M
#    "vom": 900 ,  # INR/MWh   # Variable O & M
#    "marginal_cost": 750,  # INR/MWh (fuel + other variable costs)
#    "lifetime": 85,   # Avg lifetime of a plant
#    "wacc": 0.1   # Weighted Average Cost of Capital (Interest rate)
#}
#nsmr_ann_capex = annualized_capex(nsmr_conf["capital_cost"], nsmr_conf["lifetime"], nsmr_conf["wacc"])

# 4. Build PyPSA network

n = pypsa.Network()
n.set_snapshots(snapshots)
# Central microgrid bus (generation here)
n.add("Bus", "microgrid_bus", carrier="AC")
# Define carriers (prevents undefined carrier warnings)
n.add("Carrier", "AC")
n.add("Carrier", "wind")
n.add("Carrier", "solar")
n.add("Carrier", "diesel")

# Or shorter: let PyPSA auto-add them when components are added
# But explicit is safer

# Village buses
for v in villages:
    n.add("Bus", f"{v}_bus", carrier="AC")

# Loads (one per village)
for v in villages:
    n.add("Load",
          f"{v}_load",
          bus=f"{v}_bus",
          p_set=village_loads[v])

# Connections: Links from microgrid to each village (with losses)
for v in villages:
    n.add("Link",
          f"link_to_{v}",
          bus0="microgrid_bus",  # from central
          bus1=f"{v}_bus",       # to village
          p_nom_extendable=True, # optimize capacity if needed
          efficiency=line_efficiency,  # 5% losses
          capital_cost=0,        # no cost assumed; add if you have line capex (INR/kW/km)
          length=1.0             # placeholder km; adjust for actual distances if modeling losses vary
    )

# Wind generator (on central bus)
n.add("Generator",
      "wind",
      bus="microgrid_bus",
      carrier="wind",
      p_nom_extendable=True,
      p_nom_max=8,
      p_max_pu=wind_cf.values,          
      marginal_cost=wind_conf.get("vom", 0),
      capital_cost=wind_ann_capex + wind_conf.get("fom", 0),  
      lifetime=wind_conf["lifetime"])

# n.add("Load",
#     "Sugar Mill",
#     bus="microgrid_bus",
#     p_set=sugar_mill_load)

# Solar generator (on central bus)
n.add("Generator",
      "solar",
      bus="microgrid_bus",
      carrier="solar",
      p_nom_extendable=True,
      p_nom_max=10,
      p_max_pu=solar_cf.values,
      marginal_cost=solar_conf.get("vom", 0),
      capital_cost=solar_ann_capex + solar_conf.get("fom", 0),
      lifetime=solar_conf["lifetime"])

# Battery storage (on central bus)
n.add("StorageUnit",
      "battery",
      bus="microgrid_bus",
      p_nom_min=0,                     # min power capacity can be zero
      p_nom_extendable=True,       # disable optimization if required
      capital_cost=battery_capital_cost,
      marginal_cost=battery_conf["vom"],
      efficiency_store=battery_conf["efficiency"],
      efficiency_dispatch=battery_conf["efficiency"],
      max_hours=battery_conf["max_hours"],
      cyclic_state_of_charge=True,
      lifetime=battery_conf["lifetime"])

# Backup diesel (on central bus)
n.add("Generator",
    "diesel_backup",
      bus="microgrid_bus",
      carrier="diesel",
      p_nom_extendable=True,            
      marginal_cost=30000,  # INR/MWh
      capital_cost=20000,   
      lifetime=10)

# Then, after adding all components (generators, links, loads, etc.):
n.sanitize()  # fixes undefined carriers, consistency issues
print("Network summary before optimization:")
print(n)

# 5. Optimize (minimize cost, subject to meeting demand)

print("\nOptimizing least-cost dispatch & investment...")
n.optimize(solver_name="highs")   # free/open-source; use "gurobi" or "highs" if available

# 6. Results & Reliability / Cost Analysis

print("\nOptimization status:", n.model.status)
print(n.storage_units.p_nom_opt)
# Installed capacities [MW]
print("\nOptimal capacities (Generators MW):")
print(n.generators.p_nom_opt.round(2))
print("\nBattery power [MW]:", n.storage_units.p_nom_opt.round(2))
print("Battery energy [MWh]:", (n.storage_units.p_nom_opt * n.storage_units.max_hours).round(2))

# Annual energy [MWh]
print("\nAnnual generation [MWh]:")
print("\nAnnual generation [MWh]:")
print(n.generators_t.p.sum().round(0))

# Total system cost [INR/year] (annualized)
total_cost = n.statistics.system_cost().sum().round(0)
print(f"\nTotal annualized system cost: ₹{total_cost:,.0f} / year")


# ==========================================================
# COST BREAKDOWN BY GENERATION TECHNOLOGY
# ==========================================================

print("\n--- Annual Cost Breakdown by Generator ---")

gen_energy = n.generators_t.p.sum()          # MWh per year
gen_capacity = n.generators.p_nom_opt        # MW installed
gen_cap_cost = n.generators.capital_cost     # INR/MW/year
gen_marg_cost = n.generators.marginal_cost   # INR/MWh

cost_table = []

for g in n.generators.index:

    energy = gen_energy[g]

    cap = gen_capacity[g]

    capex = cap * gen_cap_cost[g]

    opex = energy * gen_marg_cost[g]

    total = capex + opex

    cost_table.append({
        "Generator": g,
        "Installed MW": cap,
        "Annual Energy (MWh)": energy,
        "Capex (₹/yr)": capex,
        "Opex (₹/yr)": opex,
        "Total Cost (₹/yr)": total
    })

cost_df = pd.DataFrame(cost_table)

print(cost_df.round(2))


print("\n--- Cost Shares ---")

total_system_cost = cost_df["Total Cost (₹/yr)"].sum()

cost_df["Share (%)"] = 100 * cost_df["Total Cost (₹/yr)"] / total_system_cost

print(cost_df[["Generator","Total Cost (₹/yr)","Share (%)"]].round(2))

# Reliability: Loss of Load (should be 0 if feasible)
lol = (n.loads_t.p_set - n.loads_t.p).clip(lower=0).sum().sum()
print(f"Total Loss of Load energy: {lol:.2f} MWh ({lol / n.loads_t.p_set.sum().sum() * 100:.4f}% of demand)")

# Capacity factors
print("\nAchieved capacity factors:")
print(n.statistics()["Capacity Factor"].round(3))


# MICROGRID LOAD ANALYSIS + VISUALIZATION

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx


# a. LOAD CHARACTERISTICS (Rabi vs Normal)


load_data = n.loads_t.p_set.copy()
load_data["month"] = load_data.index.month
load_data["is_rabi"] = load_data["month"].isin([11,12,1,2])

print("\n--- Load Characteristics (MW) ---")

all_loads = list(n.loads_t.p_set.columns)

for load in all_loads:

    rabi_stats = load_data[load_data["is_rabi"]][load].describe()
    normal_stats = load_data[~load_data["is_rabi"]][load].describe()

    print(f"\nLoad: {load}")

    summary = pd.DataFrame({
        "Rabi (Nov-Feb)": rabi_stats,
        "Normal (Mar-Oct)": normal_stats
    })

    print(summary.loc[["mean","max","std"]])


# b. DAILY LOAD CURVES (INCLUDING SUGAR MILL)


plot_df = n.loads_t.p_set.copy()
plot_df["hour"] = plot_df.index.hour
plot_df["month"] = plot_df.index.month_name()

months_to_plot = ["January","May","August","November"]

fig, axes = plt.subplots(len(all_loads),1,
                         figsize=(10,4*len(all_loads)),
                         sharex=True)

if len(all_loads) == 1:
    axes=[axes]

for i, load in enumerate(all_loads):

    ax = axes[i]

    for month in months_to_plot:

        monthly_data = plot_df[plot_df["month"]==month]
        hourly_avg = monthly_data.groupby("hour")[load].mean()
        ax.plot(hourly_avg.index,
                hourly_avg.values,
                lw=2.5,
                label=month)

    ax.set_title(f"Seasonal Load Profile: {load}",
                 fontsize=12,
                 fontweight="bold")

    ax.set_ylabel("Demand (MW)")
    ax.grid(alpha=0.3)
    ax.legend()

plt.xlabel("Hour of Day")
plt.tight_layout()
plt.show()


# c. MONTHLY LOAD TREND COMPARISON

monthly_load = n.loads_t.p_set.groupby(
    n.loads_t.p_set.index.month
).mean()

plt.figure(figsize=(12,6))

for load in monthly_load.columns:

    plt.plot(monthly_load.index,
             monthly_load[load],
             marker="o",
             linewidth=2,
             label=load)

plt.title("Seasonal Load Variation (Villages + Sugar Mill)",
          fontsize=14,
          fontweight="bold")

plt.xlabel("Month")
plt.ylabel("Average Load (MW)")

month_names = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

plt.xticks(range(1,13),month_names)

# highlight Rabi period
plt.axvspan(1,2,color="gray",alpha=0.1)
plt.axvspan(11,12,color="gray",alpha=0.1)

plt.grid(axis="y",linestyle="--",alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()



# d. CAPACITY UTILISATION

gen_util = (n.generators_t.p.mean() /
            n.generators.p_nom_opt) * 100

bat_util = (n.storage_units_t.p.abs().mean() /
            n.storage_units.p_nom_opt) * 100

utilization = pd.concat([gen_util,bat_util])

plt.figure(figsize=(10,6))

ax = utilization.plot(
    kind="bar",
    edgecolor="black",
    alpha=0.85
)

plt.title("Capacity Utilisation of Power Sources",
          fontsize=14,
          fontweight="bold")

plt.ylabel("Utilization (%)")
plt.xlabel("Source")

plt.grid(axis="y",linestyle="--",alpha=0.6)

for i,v in enumerate(utilization):
    ax.text(i,v+1,f"{v:.1f}%",ha="center",fontweight="bold")

plt.tight_layout()
plt.show()

print("\n--- Capacity Utilisation Results ---")

for name,val in utilization.items():
    print(f"{name:15}: {val:.2f}%")



# e. MICROGRID SYSTEM DIAGRAM
# (Loads + generators + statistics)


print("\nGenerating Microgrid Diagram...")

G = nx.DiGraph()

G.add_node("Microgrid Bus")

# ---------- LOAD STATISTICS ----------
for load in n.loads_t.p_set.columns:

    avg = n.loads_t.p_set[load].mean()
    std = n.loads_t.p_set[load].std()

    label = f"{load}\n{avg:.2f} MW (±{std:.2f})"

    G.add_node(label)

    G.add_edge("Microgrid Bus",label)


# ---------- GENERATOR STATS ----------

gen_energy = n.generators_t.p.sum()

for g in n.generators.index:

    energy = gen_energy[g]

    cap = n.generators.p_nom_opt[g]
    cost = n.generators.capital_cost[g] * cap

    label = f"{g}\n{energy:,.0f} MWh\n(₹{cost:,.0f})"

    G.add_node(label)

    G.add_edge(label,"Microgrid Bus")


# ---------- BATTERY ----------

bat_power = float(n.storage_units.p_nom_opt)
bat_energy = bat_power * battery_conf["max_hours"]

bat_label = f"Battery\n{bat_power:.1f} MW\n{bat_energy:.0f} MWh"

G.add_node(bat_label)

G.add_edge(bat_label,"Microgrid Bus")


# ---------- DRAW NETWORK ----------

plt.figure(figsize=(12,8))

pos = nx.spring_layout(G,seed=42)

nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=4200,
    node_color="lightblue",
    font_size=9,
    arrows=True
)

plt.title("Optimized Microgrid Architecture")

plt.show()
# 4. Print Summary Stats
print("\n--- Capacity Utilisation Results ---")
for name, val in utilization.items():
    print(f"{name:15}: {val:.2f}%")
