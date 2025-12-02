
import torch
import numpy as np
from scipy.ndimage import label
from typing import Optional, List, Union

class OnlineAuSPRO(torch.nn.Module):
    """
    PyTorch-compatible metric for calculating AU-sPRO (Area Under the Saturated Per-Region Overlap)
    during training or validation.
    
    This implementation approximates the offline metric by:
    1. Using fixed, pre-defined thresholds.
    2. Accumulating pixel-wise statistics for FPR.
    3. Identifying defects on-the-fly using Connected Components on the binary ground truth.
    """
    def __init__(self, 
                 num_thresholds: int = 100, 
                 range_min: float = 0.0, 
                 range_max: float = 1.0,
                 saturation_threshold: float = 1.0,
                 relative_saturation: bool = True):
        """
        Args:
            num_thresholds: Number of thresholds to use for the curve.
            range_min: Minimum value of anomaly scores (usually 0.0).
            range_max: Maximum value of anomaly scores (usually 1.0).
            saturation_threshold: The fraction of the defect area that needs to be covered 
                                  to achieve a score of 1.0. (Default: 1.0 = standard PRO).
            relative_saturation: If True, saturation_threshold is a ratio. If False, it's a pixel count.
        """
        super().__init__()
        self.num_thresholds = num_thresholds
        # Create thresholds tensor. 
        # We use buffers so they are saved with state_dict but not trained.
        self.register_buffer('thresholds', torch.linspace(range_min, range_max, num_thresholds))
        
        self.saturation_threshold = saturation_threshold
        self.relative_saturation = relative_saturation

        # Accumulators
        self.register_buffer('fp_areas', torch.zeros(num_thresholds, dtype=torch.float64))
        self.register_buffer('tn_areas', torch.zeros(num_thresholds, dtype=torch.float64))
        self.register_buffer('spro_sum', torch.zeros(num_thresholds, dtype=torch.float64))
        self.register_buffer('defect_count', torch.tensor(0, dtype=torch.long))
        
    def reset(self):
        """Resets the internal state."""
        self.fp_areas.zero_()
        self.tn_areas.zero_()
        self.spro_sum.zero_()
        self.defect_count.zero_()

    @torch.no_grad()
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update metrics with a batch of predictions and targets.

        Args:
            preds: Anomaly map tensor (B, H, W) or (B, 1, H, W). Range [0, 1].
            target: Ground truth binary mask (B, H, W) or (B, 1, H, W). Values 0 or 1.
        """
        # Ensure correct shapes (remove channel dim if 1)
        if preds.ndim == 4 and preds.shape[1] == 1:
            preds = preds.squeeze(1)
        if target.ndim == 4 and target.shape[1] == 1:
            target = target.squeeze(1)
            
        assert preds.shape == target.shape, f"Shape mismatch: {preds.shape} vs {target.shape}"
        
        # Ensure inputs are on the same device as thresholds
        preds = preds.to(self.thresholds.device)
        target = target.to(self.thresholds.device)
        
        # 1. Global FPR Statistics (Vectorized on GPU)
        # Broadcast comparison: (B, T, H, W)
        # This can be memory intensive. We iterate if thresholds are large?
        # For 100 thresholds and batch size 8, it's manageable (800 * H * W * bool).
        
        # Thresholding: preds >= t
        # shape: (T, B, H, W) -> permute to (B, T, H, W)
        binary_preds = preds.unsqueeze(1) >= self.thresholds.view(1, -1, 1, 1)
        
        # Identify GT Background (0) and Defect (1)
        # target shape (B, H, W) -> broadcast to (B, 1, H, W)
        gt_mask = target.unsqueeze(1).bool()
        
        # FP: Pred=1, GT=0
        fp_pixels = (binary_preds & (~gt_mask))
        self.fp_areas += fp_pixels.sum(dim=(0, 2, 3)).double() # Sum over B, H, W
        
        # TN: Pred=0, GT=0
        # Pred=0 is ~binary_preds
        tn_pixels = ((~binary_preds) & (~gt_mask))
        self.tn_areas += tn_pixels.sum(dim=(0, 2, 3)).double()
        
        # 2. sPRO Calculation (Per-Region)
        # We need to identify connected components in the target.
        # This is easier on CPU with scipy.ndimage.label
        
        preds_cpu = preds.detach().cpu().numpy()
        target_cpu = target.detach().cpu().numpy()
        
        # We can process the thresholding for sPRO on GPU or CPU.
        # CPU might be faster for the loop if we just use numpy, 
        # avoiding 100s of small GPU kernel launches.
        # But let's try to use the pre-computed binary_preds if memory allows, 
        # OR just recompute on CPU to avoid GPU sync overheads of large tensors.
        
        thresholds_np = self.thresholds.cpu().numpy()
        
        for i in range(preds.shape[0]):
            img_target = target_cpu[i]
            img_preds = preds_cpu[i]
            
            # Skip if good image (no defects)
            if img_target.sum() == 0:
                continue
                
            # Label connected components
            labeled_mask, num_features = label(img_target)
            
            for feature_id in range(1, num_features + 1):
                region_mask = (labeled_mask == feature_id)
                defect_area = region_mask.sum()
                
                # Saturation Area
                if self.relative_saturation:
                    sat_area = defect_area * self.saturation_threshold
                    # Minimum 1 pixel? usually not strictly enforced but good for stability
                    sat_area = max(sat_area, 1.0)
                else:
                    sat_area = min(defect_area, self.saturation_threshold)
                
                # Calculate TP area for this specific region across ALL thresholds
                # Optimization: extract only the region pixels
                region_scores = img_preds[region_mask] # 1D array of scores in the region
                
                # Vectorized thresholding on CPU
                # shape (T, PixelsInRegion)
                # We want to count how many pixels are >= threshold
                # Sort scores? No, just broadcasting is fine for 100 thresholds.
                # region_scores: (N,)
                # thresholds: (T,)
                # (T, 1) <= (1, N) -> (T, N) sum -> (T,)
                tp_area = (region_scores[None, :] >= thresholds_np[:, None]).sum(axis=1)
                
                spro = np.minimum(tp_area / sat_area, 1.0)
                
                self.spro_sum += torch.from_numpy(spro).to(self.spro_sum.device)
                self.defect_count += 1

    def compute(self, max_fpr: float = 1.0) -> float:
        """
        Compute the final AU-sPRO score.
        
        Args:
            max_fpr: The maximum FPR integration limit (default 1.0). 
                     The AUC will be normalized (scaled to 1) by this value.
        """
        if self.defect_count == 0:
            return 0.0
            
        # 1. Mean sPRO
        mean_spros = self.spro_sum / self.defect_count
        
        # 2. FPR
        # Handle division by zero
        denominator = self.tn_areas + self.fp_areas
        # If denominator is 0, FPR is 0 (no background pixels?)
        fprs = torch.zeros_like(self.fp_areas)
        mask = denominator > 0
        fprs[mask] = self.fp_areas[mask] / denominator[mask]
        
        # 3. AUC Integration (Trapezoidal rule)
        # Convert to CPU for numpy integration
        fprs_np = fprs.cpu().numpy()
        spros_np = mean_spros.cpu().numpy()
        
        # Sort by FPR
        sorted_indices = np.argsort(fprs_np)
        fprs_sorted = fprs_np[sorted_indices]
        spros_sorted = spros_np[sorted_indices]
        
        # Clip to max_fpr
        if max_fpr < 1.0:
            # Find indices where fpr <= max_fpr
            # We might need to interpolate the point exactly at max_fpr
            # for precise area, but for monitoring, simple clipping is often enough.
            # However, strictly we should add a point (max_fpr, interpolated_spro).
            
            mask = fprs_sorted <= max_fpr
            fprs_clipped = fprs_sorted[mask]
            spros_clipped = spros_sorted[mask]
            
            # If we have points beyond max_fpr, interpolate the value at max_fpr
            if np.any(~mask):
                # Get the first point beyond max_fpr
                idx_next = np.argmax(fprs_sorted > max_fpr)
                idx_prev = idx_next - 1
                
                if idx_prev >= 0:
                    x0, y0 = fprs_sorted[idx_prev], spros_sorted[idx_prev]
                    x1, y1 = fprs_sorted[idx_next], spros_sorted[idx_next]
                    
                    # Linear interpolation
                    y_interp = y0 + (y1 - y0) * (max_fpr - x0) / (x1 - x0)
                    
                    fprs_clipped = np.append(fprs_clipped, max_fpr)
                    spros_clipped = np.append(spros_clipped, y_interp)
            
            fprs_sorted = fprs_clipped
            spros_sorted = spros_clipped
            
        # Integration
        # Check if we have enough points
        if len(fprs_sorted) < 2:
            return 0.0
            
        auc = np.trapz(y=spros_sorted, x=fprs_sorted)
        
        # Scale to 1 (normalize by max_fpr)
        auc = auc / max_fpr
        
        return float(auc)

