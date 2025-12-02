import sys
from unittest.mock import MagicMock
import numpy as np

# Mocks
sys.modules["cv2"] = MagicMock()
sys.modules["sklearn"] = MagicMock()
sys.modules["sklearn.metrics"] = MagicMock()

# Mock skimage
skimage_mock = MagicMock()
sys.modules["skimage"] = skimage_mock
measure_mock = MagicMock()
sys.modules["skimage.measure"] = measure_mock
skimage_mock.measure = measure_mock

# Simple Connected Components implementation for testing
def simple_label(mask, return_num=False):
    # Assumes mask is 2D binary array or similar
    mask = np.array(mask)
    if mask.ndim > 2: mask = mask.squeeze()
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    current_label = 1
    visited = np.zeros((h, w), dtype=bool)
    
    for r in range(h):
        for c in range(w):
            if mask[r, c] > 0 and not visited[r, c]:
                # BFS
                q = [(r, c)]
                visited[r, c] = True
                labels[r, c] = current_label
                while q:
                    curr_r, curr_c = q.pop(0)
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if mask[nr, nc] > 0 and not visited[nr, nc]:
                                visited[nr, nc] = True
                                labels[nr, nc] = current_label
                                q.append((nr, nc))
                current_label += 1
    
    if return_num:
        return labels, current_label - 1
    return labels

measure_mock.label = simple_label

class RegionProp:
    def __init__(self, label_val, label_img):
        self.coords = np.argwhere(label_img == label_val)
        self.area = len(self.coords)
        self.label = label_val

def simple_regionprops(label_image):
    props = []
    unique_labels = np.unique(label_image)
    for l in unique_labels:
        if l == 0: continue
        props.append(RegionProp(l, label_image))
    return props

measure_mock.regionprops = simple_regionprops

import torch
from metrics import OnlineAuSPRO
import unittest

class TestOnlineAuSPRO(unittest.TestCase):
    def setUp(self):
        self.metric = OnlineAuSPRO(num_thresholds=50, saturation_threshold=0.5)

    def test_perfect_score(self):
        # 2 images, 1 defect each
        B, H, W = 2, 100, 100
        preds = torch.zeros(B, H, W)
        target = torch.zeros(B, H, W)
        
        # Create a defect
        target[:, 40:60, 40:60] = 1.0
        # Perfect prediction
        preds[:, 40:60, 40:60] = 1.0
        
        self.metric.update(preds, target)
        auc = self.metric.compute()
        
        print(f"Perfect Score AUC: {auc}")
        self.assertAlmostEqual(auc, 1.0, delta=0.02)

    def test_partial_overlap(self):
        # With saturation_threshold=0.5
        
        preds = torch.zeros(1, 100, 100)
        target = torch.zeros(1, 100, 100)
        
        # Defect: 100 pixels
        target[0, 40:50, 40:50] = 1.0
        
        # Prediction: Cover 60% (6x10)
        preds[0, 40:46, 40:50] = 1.0
        
        self.metric.update(preds, target)
        auc = self.metric.compute()
        print(f"Saturated Overlap AUC: {auc}")
        self.assertAlmostEqual(auc, 1.0, delta=0.02)

if __name__ == '__main__':
    unittest.main()