# esg_engine.py
"""
TerraZ — ESG & Ethical Mineral Discovery Engine
Quantifies environmental sustainability, human rights/governance risk,
water depletion, and carbon intensity for global critical minerals.
Aligned with UN SDGs 7, 12, 13, and 15.
"""

import numpy as np
import pandas as pd

# Baseline extraction carbon intensity (tCO2e per tonne of refined metal)
# Sourced from IEA, S&P Global, and academic life-cycle assessments (LCA)
COMMODITY_CARBON_PROFILES = {
    "Lithium": {
        "Hard-Rock (Spodumene)": 15.0,  # High roasting/calcination energy
        "Continental Brine": 4.5,       # Solar evaporation (low direct emissions)
        "Default": 9.5
    },
    "Nickel": {
        "Laterite (HPAL)": 20.0,
        "Laterite (RKEF/NPI)": 55.0,    # Very high coal-based smelting (Indonesia)
        "Magmatic Sulfide": 12.0,       # Lower carbon, higher sulfur
        "Default": 28.0
    },
    "Cobalt": {
        "Byproduct Copper-Cobalt": 8.0,
        "Byproduct Nickel Laterite": 18.0,
        "Primary / Vein": 14.0,
        "Default": 10.5
    },
    "Copper": {
        "Porphyry Open Pit": 3.8,
        "Sediment-Hosted": 3.2,
        "Default": 3.5
    },
    "Aluminum": {
        "Bauxite to Alumina": 2.5,
        "Smelted Aluminum": 11.5,
        "Default": 7.0
    },
    "Graphite": {
        "Natural Flake": 2.2,
        "Synthetic Graphite": 16.0,
        "Default": 3.0
    },
    "Rare Earth Elements": {
        "Carbonatite (Monazite/Baste)": 18.0,
        "Ion-Adsorption Clay": 25.0,
        "Default": 21.0
    },
    "Platinum": {
        "Deep Underground Bushveld": 3500.0, # per tonne metal (high energy per kg)
        "Default": 2800.0
    },
    "Manganese": {
        "Sedimentary Ore": 1.4,
        "Default": 1.4
    },
    "Vanadium": {
        "VTM Slag/Roasting": 14.0,
        "Default": 14.0
    },
    "Zinc": {
        "Sulfide Flotation/Roasting": 3.0,
        "Default": 3.0
    },
    "Titanium": {
        "Ilmenite/Rutile Sands": 2.8,
        "Default": 2.8
    }
}

# Country ESG Risk Indices (0-100 scale, higher is higher risk/concern)
# Synthesizing Aqueduct Water Risk, World Bank Worldwide Governance Indicators (WGI),
# EITI transparency ratings, and Key Biodiversity Areas (KBA) proximity.
COUNTRY_ESG_DATA = {
    # Americas
    "Chile": {"water_stress": 88, "governance_risk": 22, "biodiversity_risk": 45, "carbon_grid_factor": 0.38, "conflict_risk": 15},
    "Argentina": {"water_stress": 78, "governance_risk": 48, "biodiversity_risk": 40, "carbon_grid_factor": 0.34, "conflict_risk": 25},
    "Bolivia": {"water_stress": 72, "governance_risk": 68, "biodiversity_risk": 60, "carbon_grid_factor": 0.45, "conflict_risk": 55},
    "Brazil": {"water_stress": 35, "governance_risk": 46, "biodiversity_risk": 85, "carbon_grid_factor": 0.12, "conflict_risk": 38},
    "Peru": {"water_stress": 75, "governance_risk": 54, "biodiversity_risk": 70, "carbon_grid_factor": 0.28, "conflict_risk": 48},
    "Canada": {"water_stress": 20, "governance_risk": 10, "biodiversity_risk": 25, "carbon_grid_factor": 0.14, "conflict_risk": 8},
    "United States": {"water_stress": 62, "governance_risk": 15, "biodiversity_risk": 35, "carbon_grid_factor": 0.39, "conflict_risk": 10},
    "Mexico": {"water_stress": 68, "governance_risk": 58, "biodiversity_risk": 65, "carbon_grid_factor": 0.44, "conflict_risk": 50},
    "Colombia": {"water_stress": 28, "governance_risk": 52, "biodiversity_risk": 88, "carbon_grid_factor": 0.18, "conflict_risk": 58},
    "Ecuador": {"water_stress": 30, "governance_risk": 60, "biodiversity_risk": 90, "carbon_grid_factor": 0.22, "conflict_risk": 52},

    # Africa
    "DR Congo": {"water_stress": 25, "governance_risk": 88, "biodiversity_risk": 82, "carbon_grid_factor": 0.05, "conflict_risk": 92},
    "South Africa": {"water_stress": 82, "governance_risk": 45, "biodiversity_risk": 55, "carbon_grid_factor": 0.88, "conflict_risk": 42},
    "Zimbabwe": {"water_stress": 65, "governance_risk": 78, "biodiversity_risk": 50, "carbon_grid_factor": 0.65, "conflict_risk": 72},
    "Zambia": {"water_stress": 45, "governance_risk": 55, "biodiversity_risk": 55, "carbon_grid_factor": 0.10, "conflict_risk": 40},
    "Guinea": {"water_stress": 30, "governance_risk": 75, "biodiversity_risk": 78, "carbon_grid_factor": 0.35, "conflict_risk": 68},
    "Mozambique": {"water_stress": 40, "governance_risk": 72, "biodiversity_risk": 70, "carbon_grid_factor": 0.20, "conflict_risk": 75},
    "Madagascar": {"water_stress": 38, "governance_risk": 74, "biodiversity_risk": 95, "carbon_grid_factor": 0.40, "conflict_risk": 62},
    "Morocco": {"water_stress": 85, "governance_risk": 42, "biodiversity_risk": 45, "carbon_grid_factor": 0.68, "conflict_risk": 35},
    "Ghana": {"water_stress": 35, "governance_risk": 40, "biodiversity_risk": 65, "carbon_grid_factor": 0.38, "conflict_risk": 32},
    "Namibia": {"water_stress": 92, "governance_risk": 30, "biodiversity_risk": 35, "carbon_grid_factor": 0.52, "conflict_risk": 18},
    "Tanzania": {"water_stress": 48, "governance_risk": 62, "biodiversity_risk": 75, "carbon_grid_factor": 0.42, "conflict_risk": 45},
    "Mali": {"water_stress": 75, "governance_risk": 85, "biodiversity_risk": 40, "carbon_grid_factor": 0.55, "conflict_risk": 88},

    # Asia
    "China": {"water_stress": 70, "governance_risk": 62, "biodiversity_risk": 60, "carbon_grid_factor": 0.62, "conflict_risk": 30},
    "Indonesia": {"water_stress": 30, "governance_risk": 50, "biodiversity_risk": 92, "carbon_grid_factor": 0.72, "conflict_risk": 35},
    "India": {"water_stress": 84, "governance_risk": 48, "biodiversity_risk": 65, "carbon_grid_factor": 0.78, "conflict_risk": 40},
    "Philippines": {"water_stress": 32, "governance_risk": 56, "biodiversity_risk": 90, "carbon_grid_factor": 0.68, "conflict_risk": 50},
    "Vietnam": {"water_stress": 42, "governance_risk": 54, "biodiversity_risk": 80, "carbon_grid_factor": 0.60, "conflict_risk": 28},
    "Kazakhstan": {"water_stress": 68, "governance_risk": 58, "biodiversity_risk": 30, "carbon_grid_factor": 0.85, "conflict_risk": 38},
    "Uzbekistan": {"water_stress": 80, "governance_risk": 68, "biodiversity_risk": 35, "carbon_grid_factor": 0.72, "conflict_risk": 42},
    "Turkey": {"water_stress": 65, "governance_risk": 58, "biodiversity_risk": 52, "carbon_grid_factor": 0.48, "conflict_risk": 45},
    "Mongolia": {"water_stress": 76, "governance_risk": 46, "biodiversity_risk": 38, "carbon_grid_factor": 0.82, "conflict_risk": 22},
    "Japan": {"water_stress": 35, "governance_risk": 12, "biodiversity_risk": 40, "carbon_grid_factor": 0.45, "conflict_risk": 5},
    "Saudi Arabia": {"water_stress": 98, "governance_risk": 52, "biodiversity_risk": 25, "carbon_grid_factor": 0.65, "conflict_risk": 30},
    "Malaysia": {"water_stress": 22, "governance_risk": 42, "biodiversity_risk": 86, "carbon_grid_factor": 0.58, "conflict_risk": 20},

    # Europe
    "Russia": {"water_stress": 25, "governance_risk": 82, "biodiversity_risk": 48, "carbon_grid_factor": 0.38, "conflict_risk": 85},
    "Finland": {"water_stress": 10, "governance_risk": 8, "biodiversity_risk": 20, "carbon_grid_factor": 0.08, "conflict_risk": 4},
    "Sweden": {"water_stress": 12, "governance_risk": 6, "biodiversity_risk": 18, "carbon_grid_factor": 0.04, "conflict_risk": 3},
    "Norway": {"water_stress": 10, "governance_risk": 6, "biodiversity_risk": 22, "carbon_grid_factor": 0.02, "conflict_risk": 3},
    "Germany": {"water_stress": 32, "governance_risk": 12, "biodiversity_risk": 25, "carbon_grid_factor": 0.35, "conflict_risk": 5},
    "Poland": {"water_stress": 45, "governance_risk": 26, "biodiversity_risk": 30, "carbon_grid_factor": 0.75, "conflict_risk": 12},
    "Spain": {"water_stress": 78, "governance_risk": 20, "biodiversity_risk": 40, "carbon_grid_factor": 0.22, "conflict_risk": 10},
    "Portugal": {"water_stress": 62, "governance_risk": 18, "biodiversity_risk": 35, "carbon_grid_factor": 0.20, "conflict_risk": 8},
    "France": {"water_stress": 38, "governance_risk": 16, "biodiversity_risk": 30, "carbon_grid_factor": 0.06, "conflict_risk": 6},
    "United Kingdom": {"water_stress": 35, "governance_risk": 14, "biodiversity_risk": 25, "carbon_grid_factor": 0.22, "conflict_risk": 6},
    "Serbia": {"water_stress": 42, "governance_risk": 55, "biodiversity_risk": 45, "carbon_grid_factor": 0.70, "conflict_risk": 35},
    "Austria": {"water_stress": 18, "governance_risk": 10, "biodiversity_risk": 22, "carbon_grid_factor": 0.12, "conflict_risk": 4},

    # Oceania
    "Australia": {"water_stress": 80, "governance_risk": 12, "biodiversity_risk": 55, "carbon_grid_factor": 0.52, "conflict_risk": 6},
    "Papua New Guinea": {"water_stress": 15, "governance_risk": 76, "biodiversity_risk": 94, "carbon_grid_factor": 0.40, "conflict_risk": 65},
    "New Caledonia": {"water_stress": 22, "governance_risk": 35, "biodiversity_risk": 96, "carbon_grid_factor": 0.75, "conflict_risk": 42},
    "New Zealand": {"water_stress": 15, "governance_risk": 5, "biodiversity_risk": 30, "carbon_grid_factor": 0.10, "conflict_risk": 2}
}

DEFAULT_ESG_PROFILE = {
    "water_stress": 50,
    "governance_risk": 50,
    "biodiversity_risk": 50,
    "carbon_grid_factor": 0.45,
    "conflict_risk": 50
}

def calculate_esg_scorecard(country: str, mineral: str, grade: str = "A") -> dict:
    """
    Computes a comprehensive, multi-pillar ESG evaluation for a mineral project.
    Returns composite score (0-100, where higher is more sustainable),
    letter grade rating (AAA to CCC), and pillar sub-scores.
    """
    country_profile = COUNTRY_ESG_DATA.get(country, DEFAULT_ESG_PROFILE)
    carbon_profile = COMMODITY_CARBON_PROFILES.get(mineral, {"Default": 5.0})
    base_carbon = carbon_profile.get("Default", 5.0)

    # Grid carbon adjustment: extraction in coal-heavy regions has higher Scope 2 impact
    adjusted_carbon = base_carbon * (1.0 + country_profile["carbon_grid_factor"])
    if grade == "B":
        # Lower ore grades require more rock moving and energy per tonne metal
        adjusted_carbon *= 1.35

    # Environmental Pillar (0-100, higher is better)
    # Combines Water Conservation and Carbon Efficiency
    water_score = max(5, 100 - country_profile["water_stress"])
    carbon_risk_component = min(100, (adjusted_carbon / 30.0) * 100) if mineral != "Platinum" else min(100, (adjusted_carbon / 3500.0) * 100)
    carbon_score = max(5, 100 - carbon_risk_component)
    biodiversity_score = max(5, 100 - country_profile["biodiversity_risk"])
    
    env_pillar = round((0.40 * water_score) + (0.35 * carbon_score) + (0.25 * biodiversity_score), 1)

    # Social & Governance Pillar (0-100, higher is better)
    # Combines Governance, Human Rights, and Conflict/Artisanal Mining Risk
    gov_score = max(5, 100 - country_profile["governance_risk"])
    conflict_score = max(5, 100 - country_profile["conflict_risk"])
    soc_gov_pillar = round((0.55 * gov_score) + (0.45 * conflict_score), 1)

    # Composite ESG & Ethical Discovery Index (EEDI) (0-100, higher is more sustainable)
    composite_eedi = round((0.50 * env_pillar) + (0.50 * soc_gov_pillar), 1)

    # ESG Letter Rating
    if composite_eedi >= 85:
        rating = "AAA"
        rating_color = "#00d2be"
        tier = "Pioneer (ESG Benchmark)"
    elif composite_eedi >= 75:
        rating = "AA"
        rating_color = "#20c997"
        tier = "Leader (Low Risk)"
    elif composite_eedi >= 65:
        rating = "A"
        rating_color = "#66cc33"
        tier = "Above Average"
    elif composite_eedi >= 55:
        rating = "BBB"
        rating_color = "#ffbb00"
        tier = "Moderate Risk"
    elif composite_eedi >= 45:
        rating = "BB"
        rating_color = "#ff8800"
        tier = "Watchlist (Elevated Risk)"
    elif composite_eedi >= 35:
        rating = "B"
        rating_color = "#ff4444"
        tier = "High ESG Concern"
    else:
        rating = "CCC"
        rating_color = "#cc0000"
        tier = "Severe Risk / Critical Intervention Required"

    return {
        "country": country,
        "mineral": mineral,
        "grade": grade,
        "composite_eedi": composite_eedi,
        "rating": rating,
        "rating_color": rating_color,
        "tier": tier,
        "environmental_pillar": env_pillar,
        "social_governance_pillar": soc_gov_pillar,
        "carbon_intensity_tco2e": round(adjusted_carbon, 2),
        "water_stress_index": country_profile["water_stress"],
        "biodiversity_risk_score": country_profile["biodiversity_risk"],
        "conflict_governance_risk": country_profile["conflict_risk"],
        "sdg_impact": [
            {"sdg": "SDG 7 (Clean Energy)", "alignment": "High — Essential transition feedstock"},
            {"sdg": "SDG 12 (Responsible Production)", "alignment": f"{rating} rated supply pathway"},
            {"sdg": "SDG 13 (Climate Action)", "alignment": f"{adjusted_carbon:.1f} tCO2e/t metal footprint"},
            {"sdg": "SDG 15 (Life on Land)", "alignment": f"Biodiversity risk level: {country_profile['biodiversity_risk']}/100"}
        ]
    }

def get_global_esg_matrix() -> pd.DataFrame:
    """Generates the global ESG matrix across all 50 countries for executive benchmarking."""
    rows = []
    for country, profile in COUNTRY_ESG_DATA.items():
        # Score on standard Copper/Lithium baseline
        scorecard = calculate_esg_scorecard(country, "Copper", "A")
        rows.append({
            "Country": country,
            "ESG Rating": scorecard["rating"],
            "EEDI Score": scorecard["composite_eedi"],
            "Environmental Score": scorecard["environmental_pillar"],
            "Social & Governance Score": scorecard["social_governance_pillar"],
            "Water Stress (0-100)": profile["water_stress"],
            "Conflict & Artisanal Risk (0-100)": profile["conflict_risk"],
            "Biodiversity Impact (0-100)": profile["biodiversity_risk"],
            "Grid Carbon Factor": profile["carbon_grid_factor"]
        })
    df = pd.DataFrame(rows).sort_values(by="EEDI Score", ascending=False).reset_index(drop=True)
    return df
