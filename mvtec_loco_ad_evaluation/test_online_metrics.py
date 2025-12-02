
import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.image import AnomalyMap, GroundTruthMap, GroundTruthChannel, DefectConfig, DefectsConfig
from src.aggregation import MetricsAggregator, ThresholdMetrics
from src.online_aggregation import OnlineMetricsAggregator

class TestOnlineMetrics(unittest.TestCase):

    def setUp(self):
        # Setup simple configs
        self.defect_config = DefectConfig(
            defect_name="scratch",
            pixel_value=255,
            saturation_threshold=0.5,
            relative_saturation=True
        )
        # shape 100x100
        self.shape = (100, 100)
        self.num_images = 5
        
        self.gt_maps = []
        self.anomaly_maps = []
        
        np.random.seed(42)

        for i in range(self.num_images):
            # Anomaly map: random floats 0..1
            am_arr = np.random.rand(*self.shape).astype(np.float32)
            self.anomaly_maps.append(AnomalyMap(am_arr, file_path=f"test_{i}.tiff"))
            
            # GT map: 50% chance of being good, else has defect
            if i % 2 == 0:
                self.gt_maps.append(None) # Good image
            else:
                # Create a random defect mask
                mask = np.random.rand(*self.shape) > 0.95 # Sparse defect
                channel = GroundTruthChannel(mask, self.defect_config)
                gt_map = GroundTruthMap(channels=[channel], file_path=f"test_{i}_gt")
                self.gt_maps.append(gt_map)

    def test_online_matches_offline(self):
        print("\nRunning Offline Aggregator...")
        # 1. Run Offline Aggregator
        offline_agg = MetricsAggregator(
            gt_maps=self.gt_maps,
            anomaly_maps=self.anomaly_maps
        )
        # Run with a small max_distance to generate some thresholds
        offline_metrics: ThresholdMetrics = offline_agg.run(curve_max_distance=0.1)
        
        thresholds = offline_metrics.anomaly_thresholds
        print(f"Generated {len(thresholds)} thresholds.")

        # 2. Run Online Aggregator with SAME thresholds
        print("Running Online Aggregator...")
        online_agg = OnlineMetricsAggregator(anomaly_thresholds=thresholds)
        
        for am, gt in zip(self.anomaly_maps, self.gt_maps):
            # using a dummy group 'test_group' for all
            online_agg.update(am, gt, group_id='test_group')
            
        # 3. Compare Results
        offline_fpr = offline_metrics.get_fp_rates()
        offline_spro = offline_metrics.get_mean_spros()
        
        online_results = online_agg.get_full_results(['test_group'])
        online_fpr = online_results['fprs']
        online_spro = online_results['mean_spros']
        
        # Check FPR equality
        np.testing.assert_allclose(online_fpr, offline_fpr, rtol=1e-6, atol=1e-8, 
                                   err_msg="FPR mismatch")
        print("FPRs match.")
        
        # Check sPRO equality
        np.testing.assert_allclose(online_spro, offline_spro, rtol=1e-6, atol=1e-8,
                                   err_msg="Mean sPRO mismatch")
        print("Mean sPROs match.")

        # Check AUC calculation
        # Offline AUC
        # Note: Offline metrics struct doesn't compute AUC directly, it's done in evaluate_experiment
        # We can use the util function directly
        from src.util import get_auc_for_max_fpr
        
        auc_offline = get_auc_for_max_fpr(offline_fpr, offline_spro, max_fpr=1.0, scale_to_one=True)
        auc_online = online_agg.get_auc_spro(['test_group'], max_fpr=1.0)
        
        self.assertAlmostEqual(auc_offline, auc_online, places=6, msg="AUC mismatch")
        print(f"AUC matches: {auc_online}")

if __name__ == '__main__':
    unittest.main()
