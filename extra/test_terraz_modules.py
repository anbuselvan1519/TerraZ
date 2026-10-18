# extra/test_terraz_modules.py
"""
Test suite validating TerraZ modules:
- ESG & Ethical Discovery Engine (esg_engine.py)
- IBM Z & LinuxONE Showcase Engine (ibm_z_engine.py)
- Explainable AI Engine (xai_engine.py)
"""

import unittest
import numpy as np
import pandas as pd
import torch
import joblib
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from esg_engine import calculate_esg_scorecard, get_global_esg_matrix, COUNTRY_ESG_DATA, COMMODITY_CARBON_PROFILES
from ibm_z_engine import calculate_linuxone_sustainability, generate_quantum_safe_mineral_passport, LATENCY_BENCHMARKS
from xai_engine import explain_prediction

class TestTerraZModules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        le_path = os.path.join(BASE_DIR, "models", "label_encoder.pkl")
        scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")
        model_path = os.path.join(BASE_DIR, "models", "mineral_dl_model.pth")
        
        cls.le = joblib.load(le_path)
        cls.scaler = joblib.load(scaler_path)
        
        import torch.nn as nn
        class MineralNet(nn.Module):
            def __init__(self, input_dim=3, hidden_dim=64, output_dim=12):
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
        cls.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        cls.model.eval()

    # --- ESG ENGINE TESTS ---
    def test_esg_scorecard_all_50_countries(self):
        for country in COUNTRY_ESG_DATA.keys():
            card = calculate_esg_scorecard(country, "Lithium", "A")
            self.assertIn("composite_eedi", card)
            self.assertTrue(0 <= card["composite_eedi"] <= 100, f"Invalid EEDI for {country}")
            self.assertIn(card["rating"], ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"])
            self.assertGreater(card["carbon_intensity_tco2e"], 0)

    def test_esg_scorecard_all_12_commodities(self):
        for mineral in COMMODITY_CARBON_PROFILES.keys():
            card = calculate_esg_scorecard("Chile", mineral, "A")
            self.assertFalse(np.isnan(card["composite_eedi"]))
            self.assertFalse(np.isnan(card["carbon_intensity_tco2e"]))

    def test_global_esg_matrix(self):
        df_matrix = get_global_esg_matrix()
        self.assertEqual(len(df_matrix), 50)
        self.assertIn("ESG Rating", df_matrix.columns)
        self.assertIn("EEDI Score", df_matrix.columns)
        self.assertTrue(df_matrix["EEDI Score"].is_monotonic_decreasing)

    # --- IBM Z ENGINE TESTS ---
    def test_linuxone_sustainability_calculations(self):
        for nodes in [10, 40, 100, 200]:
            res = calculate_linuxone_sustainability(nodes)
            self.assertGreater(res["kwh_saved_annually"], 0)
            self.assertGreater(res["co2_tonnes_avoided"], 0)
            self.assertGreater(res["annual_cost_savings_usd"], 0)
            self.assertTrue(60.0 <= res["reduction_percentage"] <= 85.0)

    def test_quantum_safe_mineral_passport(self):
        esg_card = calculate_esg_scorecard("Australia", "Lithium", "A")
        passport = generate_quantum_safe_mineral_passport("Pilgangoora Project", "Lithium", "Australia", "A", esg_card)
        self.assertTrue(passport["passport_id"].startswith("IBMZ-MP-"))
        self.assertIn("ML-DSA-LATTICE-SIG", passport["quantum_signature"])
        self.assertTrue(passport["us_ira_fta_eligible"])
        self.assertEqual(passport["tamper_proof_status"], "VERIFIED (IBM Z Crypto Express 8S)")

    def test_latency_benchmarks_structure(self):
        self.assertIn("IBM Telum On-Chip AI Accelerator", LATENCY_BENCHMARKS)
        telum = LATENCY_BENCHMARKS["IBM Telum On-Chip AI Accelerator"]
        self.assertLess(telum["avg_ms"], 1.0)
        self.assertGreater(telum["throughput_ips"], 50000)

    # --- XAI ENGINE TESTS ---
    def test_xai_feature_attributions(self):
        explanation = explain_prediction(
            self.model, self.scaler, self.le,
            lat=-23.5, lon=-68.3, grade_val=1,
            country="Chile", predicted_mineral="Lithium"
        )
        attribs = explanation["feature_attributions"]
        total_pct = sum(attribs.values())
        self.assertAlmostEqual(total_pct, 100.0, places=0)
        self.assertIsInstance(explanation["geological_rationale"], str)

    def test_xai_different_commodities(self):
        for mineral in ["Cobalt", "Nickel", "Copper", "Platinum"]:
            exp = explain_prediction(
                self.model, self.scaler, self.le,
                lat=0.0, lon=0.0, grade_val=1,
                country="Global", predicted_mineral=mineral
            )
            self.assertIsNotNone(exp["geological_rationale"])

if __name__ == "__main__":
    unittest.main(verbosity=2)
