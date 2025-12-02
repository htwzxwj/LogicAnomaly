
from collections import defaultdict
from typing import Sequence, Optional, Dict, List, Iterable

import numpy as np

from src.image import AnomalyMap, GroundTruthMap
from src.metrics import get_spros_for_thresholds, get_fp_areas_for_thresholds, get_tn_areas_for_thresholds
from src.util import get_auc_for_max_fpr


class OnlineMetricsAggregator:
    """
    Computes sPRO and FPR values incrementally (online) for given ground truth
    and anomaly maps, without loading the entire dataset into memory.

    Unlike MetricsAggregator (offline), this class requires a fixed set of
    anomaly thresholds to be defined in advance.
    """

    def __init__(self, anomaly_thresholds: Sequence[float]):
        """
        Args:
            anomaly_thresholds: Fixed thresholds for obtaining binary anomaly maps.
                Must be sorted in descending order (high confidence to low).
        """
        # Ensure thresholds are sorted descending
        self.anomaly_thresholds = np.array(sorted(anomaly_thresholds, reverse=True))
        self.num_thresholds = len(self.anomaly_thresholds)

        # Storage for accumulated stats.
        # Structure: group_id -> {
        #   'fp_areas': np.array (sum of FP areas per threshold),
        #   'tn_areas': np.array (sum of TN areas per threshold),
        #   'spro_sum': np.array (sum of sPRO values per threshold),
        #   'defect_count': int (total number of defects seen),
        # }
        self.stats: Dict[str, Dict] = defaultdict(self._create_empty_stats)

    def _create_empty_stats(self):
        return {
            'fp_areas': np.zeros(self.num_thresholds, dtype=np.float64),
            'tn_areas': np.zeros(self.num_thresholds, dtype=np.float64),
            'spro_sum': np.zeros(self.num_thresholds, dtype=np.float64),
            'defect_count': 0
        }

    def update(self, anomaly_map: AnomalyMap, gt_map: Optional[GroundTruthMap], group_id: str = 'default'):
        """
        Update the running statistics with a single image pair.

        Args:
            anomaly_map: The anomaly map.
            gt_map: The corresponding ground truth map (or None for good images).
            group_id: Identifier for the group this image belongs to (e.g., "good",
                      "structural_anomalies", "bottle"). Used for later aggregation.
        """
        # 1. Compute FP areas
        fp_areas = get_fp_areas_for_thresholds(
            gt_map=gt_map,
            anomaly_map=anomaly_map,
            anomaly_thresholds=self.anomaly_thresholds
        )
        
        # 2. Compute TN areas
        # Note: get_tn_areas_for_thresholds can use the computed fp_areas for speedup
        tn_areas = get_tn_areas_for_thresholds(
            gt_map=gt_map,
            anomaly_map=anomaly_map,
            anomaly_thresholds=self.anomaly_thresholds,
            fp_areas=fp_areas
        )

        # 3. Compute sPROs (if applicable)
        spro_sum = np.zeros(self.num_thresholds, dtype=np.float64)
        defect_count = 0

        if gt_map is not None:
            for channel in gt_map.channels:
                spros = get_spros_for_thresholds(
                    gt_channel=channel,
                    anomaly_map=anomaly_map,
                    anomaly_thresholds=self.anomaly_thresholds
                )
                spro_sum += spros
                defect_count += 1

        # 4. Update running stats
        stats = self.stats[group_id]
        stats['fp_areas'] += fp_areas
        stats['tn_areas'] += tn_areas
        stats['spro_sum'] += spro_sum
        stats['defect_count'] += defect_count

    def get_auc_spro(self, group_ids: Iterable[str], max_fpr: float = 1.0) -> float:
        """
        Compute the AU sPRO for the combined statistics of the specified groups.

        Args:
            group_ids: List of group IDs to aggregate (e.g. ['good', 'structural_anomalies']).
            max_fpr: The integration limit for the AUC.

        Returns:
            The computed AUC sPRO value. Returns 0.0 if data is insufficient.
        """
        total_fp = np.zeros(self.num_thresholds)
        total_tn = np.zeros(self.num_thresholds)
        total_spro = np.zeros(self.num_thresholds)
        total_defects = 0

        found_any = False
        for gid in group_ids:
            if gid in self.stats:
                found_any = True
                s = self.stats[gid]
                total_fp += s['fp_areas']
                total_tn += s['tn_areas']
                total_spro += s['spro_sum']
                total_defects += s['defect_count']

        if not found_any:
            return 0.0

        # Avoid division by zero for FPR
        denominator = total_tn + total_fp
        # If denominator is 0, it means we have no valid pixels?? 
        # Or rather, if for a threshold TN+FP is 0 (unlikely unless image is empty), FPR is undefined.
        # We handle this safely.
        with np.errstate(invalid='ignore'):
            fp_rates = total_fp / denominator
            fp_rates = np.nan_to_num(fp_rates) # Replace NaNs with 0

        if total_defects == 0:
            mean_spros = np.zeros(self.num_thresholds)
        else:
            mean_spros = total_spro / total_defects

        auc = get_auc_for_max_fpr(
            fprs=fp_rates,
            y_values=mean_spros,
            max_fpr=max_fpr,
            scale_to_one=True
        )
        return auc

    def get_full_results(self, group_ids: Iterable[str]) -> Dict[str, np.ndarray]:
        """
        Return the raw curves (FPR and Mean sPRO) for inspection.
        """
        total_fp = np.zeros(self.num_thresholds)
        total_tn = np.zeros(self.num_thresholds)
        total_spro = np.zeros(self.num_thresholds)
        total_defects = 0

        for gid in group_ids:
            if gid in self.stats:
                s = self.stats[gid]
                total_fp += s['fp_areas']
                total_tn += s['tn_areas']
                total_spro += s['spro_sum']
                total_defects += s['defect_count']

        denominator = total_tn + total_fp
        with np.errstate(invalid='ignore'):
            fp_rates = total_fp / denominator
            fp_rates = np.nan_to_num(fp_rates)

        if total_defects == 0:
            mean_spros = np.zeros(self.num_thresholds)
        else:
            mean_spros = total_spro / total_defects

        return {
            'thresholds': self.anomaly_thresholds,
            'fprs': fp_rates,
            'mean_spros': mean_spros
        }
