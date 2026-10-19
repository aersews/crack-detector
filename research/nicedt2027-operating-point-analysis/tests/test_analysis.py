import unittest
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, brier_score_loss
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analyze_predictions import validate, metric_values, pairwise_surface_recall

class TestPredictionAnalysis(unittest.TestCase):
    def make_df(self):
        return pd.DataFrame({
          "relative_path":["D/0001-a.jpg","D/0001-b.jpg","P/0002-a.jpg","W/0003-a.jpg"],
          "label":[0,1,1,0],"group_id":[1,1,2,3],"surface":["Deck","Deck","Pavement","Wall"],
          "probability":[0.1,0.8,0.7,0.2],"prediction":[0,1,1,0],"threshold":[0.31]*4
        })
    def test_validation(self):
        d=self.make_df()
        self.assertEqual(validate(d)["groups"],3)
    def test_metrics_against_sklearn(self):
        d=self.make_df(); got=metric_values(d)
        self.assertAlmostEqual(got["accuracy"],accuracy_score(d.label,d.prediction))
        self.assertAlmostEqual(got["roc_auc"],roc_auc_score(d.label,d.probability))
        self.assertAlmostEqual(got["average_precision"],average_precision_score(d.label,d.probability))
        self.assertAlmostEqual(got["brier"],brier_score_loss(d.label,d.probability))
    def test_pairwise_surface_recall_contrasts(self):
        d = pd.DataFrame({
            "group_id": [1, 2, 3, 4, 5, 6],
            "surface": ["Deck", "Pavement", "Wall", "Deck", "Pavement", "Wall"],
            "label": [1, 1, 1, 1, 1, 1],
            "prediction": [1, 1, 0, 0, 1, 1],
        })
        got = pairwise_surface_recall(d, n=100, seed=44)
        self.assertEqual(len(got), 3)
        self.assertEqual(got.iloc[0]["contrast"], "Pavement recall - Deck recall")
        self.assertAlmostEqual(got.iloc[0]["difference_pp"], 50.0)

    def test_bad_threshold_detected(self):
        d=self.make_df(); d.loc[0,"prediction"]=1
        with self.assertRaises(ValueError): validate(d)

if __name__=="__main__": unittest.main()
