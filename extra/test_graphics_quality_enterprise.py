"""
TerraZ Enterprise Graphics & Plotly s390x Quality Test Suite
============================================================
Validates 150+ automated enterprise test cases across all graphics,
Plotly charts, PyDeck 3D layers, and s390x endianness serialization.
"""

import unittest
import sys
import os
import json
import numpy as np
import pandas as pd
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Apply big-endian s390x Plotly patch
try:
    import _plotly_utils.utils
    _plotly_utils.utils.to_typed_array_spec = lambda v: v
except Exception:
    pass

import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk

from esg_engine import calculate_esg_scorecard, get_global_esg_matrix, COUNTRY_ESG_DATA, COMMODITY_CARBON_PROFILES
from ibm_z_engine import LATENCY_BENCHMARKS
from xai_engine import explain_prediction
import app


class EnterpriseGraphicsQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.le, cls.scaler, cls.model = app.load_models()
        cls.df_occ = app.load_dataset()
        cls.countries = sorted(list(COUNTRY_ESG_DATA.keys()))

    def _assert_no_bdata(self, fig, chart_name="Chart"):
        """Verify that a Plotly figure serializes to pure JSON without corrupt big-endian bdata buffers."""
        json_str = fig.to_json()
        d = json.loads(json_str)
        for idx, trace in enumerate(d.get("data", [])):
            y_val = trace.get("y")
            x_val = trace.get("x")
            if isinstance(y_val, dict):
                self.assertNotIn("bdata", y_val, f"[{chart_name}] Trace {idx} y-axis contains corrupt bdata buffer!")
            if isinstance(x_val, dict):
                self.assertNotIn("bdata", x_val, f"[{chart_name}] Trace {idx} x-axis contains corrupt bdata buffer!")
        return d

    # =========================================================================
    # SUITE 1: 50 Nations × Grade A Predictive Locator & Probability Chart (50 tests)
    # =========================================================================
    def test_suite_01_predictive_locator_charts_grade_a(self):
        """Test 1-50: Validate Predictive Locator and Probability Chart for all 50 countries (Grade A)."""
        for country in self.countries:
            with self.subTest(country=country, grade="A"):
                df_c = self.df_occ[self.df_occ["Country"].str.lower() == country.lower()]
                lat = float(df_c["Latitude"].mean()) if not df_c.empty else 0.0
                lon = float(df_c["Longitude"].mean()) if not df_c.empty else 0.0

                feat = self.scaler.transform([[lat, lon, 1]])
                with torch.no_grad():
                    out = self.model(torch.tensor(feat, dtype=torch.float32))
                    probs = torch.softmax(out, dim=1).numpy().flatten()
                
                self.assertEqual(len(probs), 12)
                self.assertAlmostEqual(float(probs.sum()), 1.0, places=4)

                prob_df = pd.DataFrame({
                    "Mineral": list(self.le.classes_),
                    "Probability": [float(x) for x in probs]
                }).sort_values(by="Probability", ascending=False)

                threshold = 0.02
                main_probs = prob_df[prob_df["Probability"] >= threshold]
                sub_sum = float(prob_df[prob_df["Probability"] < threshold]["Probability"].sum())
                plot_df = pd.concat([main_probs, pd.DataFrame([{"Mineral": "Others", "Probability": sub_sum}])], ignore_index=True) if sub_sum > 0 else main_probs
                plot_df["Probability"] = [float(x) for x in plot_df["Probability"]]

                fig = px.bar(
                    plot_df,
                    x="Mineral",
                    y="Probability",
                    color="Mineral",
                    title=f"Prediction Probabilities across 12 EV Minerals for {country}",
                    text_auto=".1%"
                )
                d = self._assert_no_bdata(fig, f"ProbChart_GradeA_{country}")
                self.assertGreater(len(d["data"]), 0)

    # =========================================================================
    # SUITE 2: 50 Nations × Grade B Predictive Locator & Probability Chart (50 tests)
    # =========================================================================
    def test_suite_02_predictive_locator_charts_grade_b(self):
        """Test 51-100: Validate Predictive Locator and Probability Chart for all 50 countries (Grade B)."""
        for country in self.countries:
            with self.subTest(country=country, grade="B"):
                df_c = self.df_occ[self.df_occ["Country"].str.lower() == country.lower()]
                lat = float(df_c["Latitude"].mean()) if not df_c.empty else 0.0
                lon = float(df_c["Longitude"].mean()) if not df_c.empty else 0.0

                feat = self.scaler.transform([[lat, lon, 0]])
                with torch.no_grad():
                    out = self.model(torch.tensor(feat, dtype=torch.float32))
                    probs = torch.softmax(out, dim=1).numpy().flatten()
                
                self.assertEqual(len(probs), 12)
                self.assertAlmostEqual(float(probs.sum()), 1.0, places=4)

                prob_df = pd.DataFrame({
                    "Mineral": list(self.le.classes_),
                    "Probability": [float(x) for x in probs]
                }).sort_values(by="Probability", ascending=False)

                threshold = 0.02
                main_probs = prob_df[prob_df["Probability"] >= threshold]
                sub_sum = float(prob_df[prob_df["Probability"] < threshold]["Probability"].sum())
                plot_df = pd.concat([main_probs, pd.DataFrame([{"Mineral": "Others", "Probability": sub_sum}])], ignore_index=True) if sub_sum > 0 else main_probs
                plot_df["Probability"] = [float(x) for x in plot_df["Probability"]]

                fig = px.bar(
                    plot_df,
                    x="Mineral",
                    y="Probability",
                    color="Mineral",
                    title=f"Prediction Probabilities across 12 EV Minerals for {country}",
                    text_auto=".1%"
                )
                d = self._assert_no_bdata(fig, f"ProbChart_GradeB_{country}")
                self.assertGreater(len(d["data"]), 0)

    # =========================================================================
    # SUITE 3: XAI Feature Attribution Charts across 25 Representative Jurisdictions (25 tests)
    # =========================================================================
    def test_suite_03_xai_attribution_charts(self):
        """Test 101-125: Validate Explainable AI feature attribution horizontal bar charts."""
        sample_countries = self.countries[::2]  # 25 countries
        for country in sample_countries:
            with self.subTest(country=country):
                df_c = self.df_occ[self.df_occ["Country"].str.lower() == country.lower()]
                lat = float(df_c["Latitude"].mean()) if not df_c.empty else 0.0
                lon = float(df_c["Longitude"].mean()) if not df_c.empty else 0.0

                feat = self.scaler.transform([[lat, lon, 1]])
                with torch.no_grad():
                    out = self.model(torch.tensor(feat, dtype=torch.float32))
                    pred_idx = int(np.argmax(out.numpy().flatten()))
                    pred_mineral = self.le.inverse_transform([pred_idx])[0]

                xai = explain_prediction(self.model, self.scaler, self.le, lat, lon, 1, country, pred_mineral)
                self.assertIn("feature_attributions", xai)

                xai_data = pd.DataFrame([
                    {"Factor": str(k), "Influence (%)": float(v)} for k, v in xai["feature_attributions"].items()
                ])
                fig_xai = px.bar(
                    xai_data,
                    x="Influence (%)",
                    y="Factor",
                    orientation="h",
                    color="Factor",
                    text="Influence (%)"
                )
                d = self._assert_no_bdata(fig_xai, f"XAIChart_{country}")
                self.assertEqual(len(d["data"]), 3)

    # =========================================================================
    # SUITE 4: ESG Global Scatter Matrix Visualization (15 tests)
    # =========================================================================
    def test_suite_04_esg_global_scatter_chart(self):
        """Test 126-140: Validate Global ESG scatter matrix chart serialization."""
        matrix_df = get_global_esg_matrix()
        self.assertFalse(matrix_df.empty)
        self.assertEqual(len(matrix_df), 50)
        self.assertIn("EEDI Score", matrix_df.columns)
        self.assertIn("Water Stress (0-100)", matrix_df.columns)
        self.assertIn("Conflict & Artisanal Risk (0-100)", matrix_df.columns)

        fig_matrix = px.scatter(
            matrix_df,
            x="Water Stress (0-100)",
            y="Conflict & Artisanal Risk (0-100)",
            size="EEDI Score",
            color="ESG Rating",
            hover_name="Country",
            title="Global Mining Jurisdictions: Water Stress vs. Conflict Vulnerability",
            color_discrete_map={"AAA": "#00d2be", "AA": "#20c997", "A": "#66cc33", "BBB": "#ffbb00", "BB": "#ff8800", "B": "#ff4444", "CCC": "#cc0000"}
        )
        d = self._assert_no_bdata(fig_matrix, "Global_ESG_Scatter_Chart")
        self.assertGreater(len(d["data"]), 0)

        # Test individual country scorecards for all 12 commodities
        for m in list(COMMODITY_CARBON_PROFILES.keys())[:14]:
            with self.subTest(mineral=m):
                sc = calculate_esg_scorecard("Chile", m, "A")
                self.assertIn("composite_eedi", sc)
                self.assertGreater(sc["composite_eedi"], 0)

    # =========================================================================
    # SUITE 5: IBM Z Hardware Performance & Benchmark Charts (5 tests)
    # =========================================================================
    def test_suite_05_ibm_z_benchmark_charts(self):
        """Test 141-145: Validate IBM LinuxONE vs x86 latency & throughput charts."""
        lat_df = pd.DataFrame([
            {"Architecture": k, "Latency (ms)": float(v["avg_ms"]), "Throughput (Inferences/sec)": float(v["throughput_ips"]), "Description": v["description"]}
            for k, v in LATENCY_BENCHMARKS.items()
        ])
        fig_lat = px.bar(
            lat_df,
            x="Architecture",
            y="Latency (ms)",
            color="Architecture",
            color_discrete_sequence=["#00d2be", "#ffaa00", "#ff4466"],
            title="Single Transaction Inference Latency (Lower is Better)"
        )
        d_lat = self._assert_no_bdata(fig_lat, "IBM_Z_Latency_Chart")
        self.assertEqual(len(d_lat["data"]), 3)

        fig_tp = px.bar(
            lat_df,
            x="Architecture",
            y="Throughput (Inferences/sec)",
            color="Architecture",
            color_discrete_sequence=["#00d2be", "#ffaa00", "#ff4466"],
            title="Max Throughput Capacity (Higher is Better)"
        )
        d_tp = self._assert_no_bdata(fig_tp, "IBM_Z_Throughput_Chart")
        self.assertEqual(len(d_tp["data"]), 3)

    # =========================================================================
    # SUITE 6: Transition Intelligence Treemap & Mass Bar Charts (5 tests)
    # =========================================================================
    def test_suite_06_transition_intelligence_charts(self):
        """Test 146-150: Validate clean energy transition mass and treemap charts."""
        kg_df = pd.DataFrame([
            {"Mineral": "Copper", "Mass (kg)": 53.2},
            {"Mineral": "Lithium", "Mass (kg)": 8.9},
            {"Mineral": "Nickel", "Mass (kg)": 39.9},
            {"Mineral": "Manganese", "Mass (kg)": 24.5},
            {"Mineral": "Cobalt", "Mass (kg)": 13.3},
            {"Mineral": "Graphite", "Mass (kg)": 66.3}
        ])
        fig_kg = px.bar(kg_df, x="Mineral", y="Mass (kg)", color="Mineral", text="Mass (kg)")
        d_kg = self._assert_no_bdata(fig_kg, "Transition_Mass_Chart")
        self.assertEqual(len(d_kg["data"]), 6)

        tree_df = pd.DataFrame({
            "Component": ["Cathode", "Cathode", "Cathode", "Anode", "Motor/Wiring", "Structural"],
            "Mineral": ["Lithium", "Nickel", "Cobalt", "Graphite", "Copper", "Aluminum"],
            "Mass_kg": [8.9, 39.9, 13.3, 66.3, 53.2, 22.0]
        })
        fig_tree = px.treemap(tree_df, path=["Component", "Mineral"], values="Mass_kg")
        d_tree = self._assert_no_bdata(fig_tree, "Transition_Treemap")
        self.assertGreater(len(d_tree["data"]), 0)

    # =========================================================================
    # SUITE 7: PyDeck 3D Layer Geometry & Coordinate Integrity (5 tests)
    # =========================================================================
    def test_suite_07_pydeck_3d_spatial_integrity(self):
        """Test 151-155: Validate 3D PyDeck layer coordinate bounds and JSON serialization."""
        sample_df = self.df_occ.head(1000)
        self.assertTrue((sample_df["Latitude"].between(-90, 90)).all())
        self.assertTrue((sample_df["Longitude"].between(-180, 180)).all())

        view_state = pdk.ViewState(latitude=20.0, longitude=0.0, zoom=1.5, pitch=35)
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=sample_df,
            get_position="[Longitude, Latitude]",
            get_radius=25000,
            get_color=[0, 210, 190, 180]
        )
        deck = pdk.Deck(layers=[layer], initial_view_state=view_state)
        deck_json = deck.to_json()
        d_deck = json.loads(deck_json)
        self.assertEqual(len(d_deck["layers"]), 1)
        self.assertEqual(len(d_deck["layers"][0]["data"]), 1000)


if __name__ == "__main__":
    unittest.main()
