
import unittest
import torch
import numpy as np
from src.training_metric import OnlineAuSPRO

class TestTrainingMetric(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metric = OnlineAuSPRO(num_thresholds=50, saturation_threshold=0.5).to(self.device)

    def test_perfect_score(self):
        # Case 1: Perfect prediction
        # 2 images, 1 defect each
        B, H, W = 2, 100, 100
        preds = torch.zeros(B, H, W).to(self.device)
        target = torch.zeros(B, H, W).to(self.device)
        
        # Create a defect
        target[:, 40:60, 40:60] = 1.0
        # Perfect prediction matches target exactly
        preds[:, 40:60, 40:60] = 1.0
        
        self.metric.update(preds, target)
        auc = self.metric.compute()
        
        print(f"Perfect Score AUC: {auc}")
        # AUC should be close to 1.0
        self.assertAlmostEqual(auc, 1.0, delta=0.02)

    def test_partial_overlap(self):
        # Case 2: Partial overlap
        # With saturation_threshold=0.5, if we cover >50% of defect, sPRO=1.0
        
        preds = torch.zeros(1, 100, 100).to(self.device)
        target = torch.zeros(1, 100, 100).to(self.device)
        
        # Defect: 100 pixels (10x10)
        target[0, 40:50, 40:50] = 1.0
        
        # Prediction: Cover 60% (6x10) -> 60 pixels
        # 60 / (100 * 0.5) = 60 / 50 = 1.2 -> clipped to 1.0
        preds[0, 40:46, 40:50] = 1.0
        
        self.metric.update(preds, target)
        
        # Check internal stats for a low threshold (e.g. 0.1)
        # Threshold index for 0.1?
        # thresholds are 0..1 (50 steps). 0.1 is approx index 5.
        # Actually, let's just check compute() which integrates over all.
        # Since pred is binary 1.0, for all thresholds <= 1.0, we get the overlap.
        # FPR should be near 0 (tiny FP area if any).
        # sPRO should be 1.0 for all valid thresholds.
        
        auc = self.metric.compute()
        print(f"Saturated Overlap AUC: {auc}")
        self.assertAlmostEqual(auc, 1.0, delta=0.02)

    def test_no_overlap(self):
        preds = torch.zeros(1, 100, 100).to(self.device)
        target = torch.zeros(1, 100, 100).to(self.device)
        target[0, 50:60, 50:60] = 1.0 # defect
        preds[0, 10:20, 10:20] = 1.0 # miss (FP area = 100 pixels approx 1% of image)
        
        self.metric.update(preds, target)
        
        # The FP area is ~1% (0.01). 
        # If we look at max_fpr=0.001 (0.1%), we should have NO positives in the valid range 
        # (assuming threshold > 1.0 gives 0 FPR).
        # The curve is (0,0) to (0,0).
        auc = self.metric.compute(max_fpr=0.001)
        print(f"No Overlap AUC (max_fpr=0.001): {auc}")
        self.assertEqual(auc, 0.0)

if __name__ == '__main__':
    unittest.main()
