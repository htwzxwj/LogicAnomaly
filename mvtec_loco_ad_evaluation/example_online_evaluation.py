
"""
Example script demonstrating how to use the OnlineMetricsAggregator.
This script mimics the logic of evaluate_experiment.py but uses the online implementation,
drastically reducing memory usage by processing images one by one.
"""

import os
import json
import numpy as np
from tqdm import tqdm

from src.image import AnomalyMap, GroundTruthMap, DefectsConfig
from src.online_aggregation import OnlineMetricsAggregator
from src.util import get_auc_for_max_fpr

# Configuration
DATASET_BASE_DIR = '/path/to/mvtec_loco_ad' # Replace with actual path
ANOMALY_MAPS_DIR = '/path/to/anomaly_maps'  # Replace with actual path
OBJECT_NAME = 'breakfast_box'               # Replace with actual object
NUM_THRESHOLDS = 100                        # Number of fixed thresholds

def main():
    # 1. Setup: Read defects config
    defects_config_path = os.path.join(
        DATASET_BASE_DIR, OBJECT_NAME, 'defects_config.json')
    
    # Note: In a real run, ensure this file exists.
    if not os.path.exists(defects_config_path):
        print(f"Config not found at {defects_config_path}. Please set paths correctly.")
        return

    with open(defects_config_path) as defects_config_file:
        defects_list = json.load(defects_config_file)
    defects_config = DefectsConfig.create_from_list(defects_list)

    # 2. Initialize Online Aggregator
    # For online evaluation, we need fixed thresholds.
    # A simple approach is linear spacing between 0 and 1 (if scores are normalized).
    # Or, if you know the range of your scores, use that.
    thresholds = np.linspace(0, 1, NUM_THRESHOLDS)
    online_agg = OnlineMetricsAggregator(anomaly_thresholds=thresholds)

    # 3. Iterate through data (Simulation of streaming)
    gt_dir = os.path.join(DATASET_BASE_DIR, OBJECT_NAME, 'ground_truth')
    anomaly_test_dir = os.path.join(ANOMALY_MAPS_DIR, OBJECT_NAME, 'test')

    # Get list of files (this part is still "offline" in terms of file listing, 
    # but we load images one by one)
    # ... (File listing logic omitted for brevity, reusing standard logic) ...
    # For this example, let's assume we have a list of (anomaly_path, gt_path_or_None, image_type)
    
    # Mocking the stream for demonstration
    image_stream = get_image_stream(gt_dir, anomaly_test_dir, defects_config)

    print("Starting online evaluation...")
    for anomaly_map, gt_map, image_type in tqdm(image_stream):
        # Update stats
        # image_type is e.g. 'good', 'logical_anomalies', 'structural_anomalies'
        online_agg.update(anomaly_map, gt_map, group_id=image_type)

    # 4. Compute Results
    print("\nComputing final metrics...")
    
    # Define max FPR limits we care about
    max_fprs = [0.01, 0.05, 0.1, 0.3, 1.0]
    
    results = {'auc_spro': {}}
    
    # Calculate for each category
    for category in ['structural_anomalies', 'logical_anomalies']:
        results['auc_spro'][category] = {}
        # Note: We usually include 'good' images when evaluating a category
        groups_to_include = [category, 'good']
        
        for max_fpr in max_fprs:
            auc = online_agg.get_auc_spro(groups_to_include, max_fpr=max_fpr)
            results['auc_spro'][category][max_fpr] = auc
            
    # Calculate Mean
    results['auc_spro']['mean'] = {}
    for max_fpr in max_fprs:
        mean_auc = 0.5 * (results['auc_spro']['structural_anomalies'][max_fpr] + 
                          results['auc_spro']['logical_anomalies'][max_fpr])
        results['auc_spro']['mean'][max_fpr] = mean_auc

    print(json.dumps(results, indent=4))


def get_image_stream(gt_dir, anomaly_test_dir, defects_config):
    """
    Yields (AnomalyMap, GroundTruthMap | None, image_type)
    This helper mimics the file reading logic of evaluate_experiment.py
    but yields one by one.
    """
    from src.util import listdir
    
    # Iterate over subdirectories in test dir (good, logical..., structural...)
    for subdir_name in listdir(anomaly_test_dir):
        subdir_path = os.path.join(anomaly_test_dir, subdir_name)
        if not os.path.isdir(subdir_path):
            continue
            
        for file_name in listdir(subdir_path):
            if not file_name.lower().endswith(('.tif', '.tiff')):
                continue
                
            # Read Anomaly Map
            anomaly_path = os.path.join(subdir_path, file_name)
            anomaly_map = AnomalyMap.read_from_tiff(anomaly_path)
            
            # Read GT Map if not 'good'
            gt_map = None
            if subdir_name != 'good':
                # GT maps are stored as folders of PNGs
                file_stem = os.path.splitext(file_name)[0]
                gt_path = os.path.join(gt_dir, subdir_name, file_stem)
                if os.path.isdir(gt_path):
                     gt_map = GroundTruthMap.read_from_png_dir(gt_path, defects_config)
            
            yield anomaly_map, gt_map, subdir_name

if __name__ == "__main__":
    main()
