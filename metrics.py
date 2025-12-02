"""Anomaly metrics."""
import cv2
import numpy as np
import os
import torch
from sklearn import metrics
import json


def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights, anomaly_ground_truth_labels
):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).

    Args:
        anomaly_prediction_weights: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    """
    fpr, tpr, thresholds = metrics.roc_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    auroc = metrics.roc_auc_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    
    precision, recall, _ = metrics.precision_recall_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    auc_pr = metrics.auc(recall, precision)
    
    return {"auroc": auroc, "fpr": fpr, "tpr": tpr, "threshold": thresholds}


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.

    Args:
        anomaly_segmentations: [list of np.arrays or np.array] [NxHxW] Contains
                                generated segmentation masks.
        ground_truth_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    fpr, tpr, thresholds = metrics.roc_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )

    precision, recall, thresholds = metrics.precision_recall_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    optimal_threshold = thresholds[np.argmax(F1_scores)]
    predictions = (flat_anomaly_segmentations >= optimal_threshold).astype(int)
    fpr_optim = np.mean(predictions > flat_ground_truth_masks)
    fnr_optim = np.mean(predictions < flat_ground_truth_masks)

    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
    }


from skimage import measure
from typing import Optional, Dict, List

def _collect_fpr_spro_curve(
    masks,
    amaps,
    num_thresholds: int = 200,
    structuring_element_size: int = 1,
    defect_id_to_config: Optional[Dict[int, Dict]] = None,
    saturation_threshold: Optional[float] = None,
    relative_saturation: bool = True,
):
    """
    Collect false positive rates and mean (saturated) PRO values over thresholds.

    This function can emulate the LOCO saturated PRO when a per-defect configuration
    is provided via `defect_id_to_config`. Each positive integer label in `masks`
    is treated as a "defect type/channel" and gets its own saturation area:

        - If cfg["relative"] is True:  saturation_area = round(cfg["saturation_threshold"] * defect_area)
        - Else (absolute):              saturation_area = min(cfg["saturation_threshold"], defect_area)

    When `defect_id_to_config` is None, it falls back to region-wise PRO using
    connected components. If `saturation_threshold` is provided, it applies saturation
    to these connected components (simple sPRO).

    Args:
        masks: [np.array or list] [NxHxW] Integer/binary masks (0=background). Positive values
               can encode different defect types if `defect_id_to_config` is used.
        amaps: [np.array or list] [NxHxW] Float anomaly maps.
        num_thresholds: Number of thresholds across [min(amap), max(amap)].
        structuring_element_size: If >1, apply dilation with a square kernel of this size.
                                  Default is 1 (no dilation) to match LOCO's exact pixel overlap.
        defect_id_to_config: Optional dict[int, dict(relative: bool, saturation_threshold: float|int)].
        saturation_threshold: Global saturation threshold for simple sPRO (used if defect_id_to_config is None).
        relative_saturation: Whether global saturation_threshold is relative (fraction) or absolute (pixels).

    Returns:
        (fpr_sorted, spro_sorted, thresholds_sorted)
    """
    masks_array = np.asarray(masks)
    amaps_array = np.asarray(amaps)

    # Normalize input shapes to (N, H, W)
    if masks_array.ndim == 4 and masks_array.shape[1] == 1:
        masks_array = masks_array[:, 0]
    if amaps_array.ndim == 4 and amaps_array.shape[1] == 1:
        amaps_array = amaps_array[:, 0]
    if masks_array.ndim == 2:
        masks_array = masks_array[np.newaxis, ...]
    if amaps_array.ndim == 2:
        amaps_array = amaps_array[np.newaxis, ...]

    if masks_array.shape != amaps_array.shape:
        raise ValueError("Masks and anomaly maps must have the same shape.")

    if num_thresholds < 2:
        num_thresholds = 2

    min_th = float(np.min(amaps_array))
    max_th = float(np.max(amaps_array))
    if max_th <= min_th:
        thresholds = np.array([max_th], dtype=np.float64)
    else:
        thresholds = np.linspace(min_th, max_th, num_thresholds, dtype=np.float64)

    # Background (defect-free) mask across all images
    inverse_masks = np.logical_not(masks_array.astype(bool))
    inverse_masks_sum = int(inverse_masks.sum())

    # Structuring element (only used if size > 1)
    if structuring_element_size and structuring_element_size > 1:
        struct_elem = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (structuring_element_size, structuring_element_size),
        )
    else:
        struct_elem = None

    fpr_values: List[float] = []
    spro_values: List[float] = []

    for threshold in thresholds:
        # Binary predictions
        binary_maps = (amaps_array > threshold)
        if struct_elem is not None:
            # apply dilation per image
            binary_maps = np.stack(
                [cv2.dilate(b.astype(np.uint8), struct_elem).astype(bool) for b in binary_maps],
                axis=0,
            )

        per_threshold_pros: List[float] = []

        for binary_map, mask in zip(binary_maps, masks_array):
            if defect_id_to_config is not None:
                # LOCO-like: iterate defect types (positive labels)
                labels = np.unique(mask)
                for lab in labels:
                    if lab <= 0:
                        continue
                    channel = (mask == lab)
                    defect_area = int(channel.sum())
                    if defect_area == 0:
                        continue

                    cfg = defect_id_to_config.get(int(lab)) if defect_id_to_config else None
                    if cfg is not None and cfg.get("relative", False):
                        sat = int(round(float(cfg.get("saturation_threshold", 1.0)) * defect_area))
                    else:
                        # absolute saturation (or missing cfg -> treat as full area)
                        abs_thr = cfg.get("saturation_threshold", defect_area) if cfg else defect_area
                        sat = int(min(int(abs_thr), defect_area))
                    sat = max(sat, 1)

                    tp_pixels = int(np.logical_and(binary_map, channel).sum())
                    pro = min(tp_pixels / float(sat), 1.0)
                    per_threshold_pros.append(pro)
            else:
                # Fallback: region-wise PRO by connected components (binary mask)
                label_image = measure.label(mask.astype(np.uint8))
                for region in measure.regionprops(label_image):
                    coords = region.coords
                    if coords.size == 0:
                        continue
                    
                    defect_area = region.area
                    sat = defect_area
                    if saturation_threshold is not None:
                        if relative_saturation:
                            sat = int(round(saturation_threshold * defect_area))
                        else:
                            sat = int(min(saturation_threshold, defect_area))
                        sat = max(sat, 1)

                    tp_pixels = int(binary_map[coords[:, 0], coords[:, 1]].sum())
                    pro = min(tp_pixels / float(sat), 1.0)
                    per_threshold_pros.append(pro)

        pro_value = float(np.mean(per_threshold_pros)) if per_threshold_pros else 0.0
        spro_values.append(pro_value)

        # FPR across all images
        if inverse_masks_sum == 0:
            fpr_value = 0.0  # no defect-free pixels at all; consistent with LOCO raising error upstream
        else:
            fp_pixels = int(np.logical_and(inverse_masks, binary_maps).sum())
            fpr_value = float(fp_pixels) / float(inverse_masks_sum)
        fpr_values.append(fpr_value)

    fpr_array = np.array(fpr_values, dtype=np.float64)
    spro_array = np.array(spro_values, dtype=np.float64)

    sort_idx = np.argsort(fpr_array)
    fpr_sorted = fpr_array[sort_idx]
    spro_sorted = spro_array[sort_idx]
    thresholds_sorted = thresholds[sort_idx]

    return fpr_sorted, spro_sorted, thresholds_sorted


def compute_pro(masks, amaps, num_th=200):
    """
    Computes the area under the PRO curve after normalizing the FPR to [0, 0.3].

    Args:
        masks: [np.array or list] [NxHxW] Binary ground-truth masks.
        amaps: [np.array or list] [NxHxW] Severity/anomaly maps.
        num_th: Number of thresholds used for curve construction.
    Returns:
        float: Normalized area under the PRO curve.
    """
    fprs, spro_values, _ = _collect_fpr_spro_curve(
        masks, amaps, num_thresholds=num_th
    )
    mask = fprs < 0.3
    if not np.any(mask):
        return 0.0

    trimmed_fprs = fprs[mask]
    trimmed_pros = spro_values[mask]
    max_fpr = trimmed_fprs.max()
    if max_fpr > 0:
        normalized_fprs = trimmed_fprs / max_fpr
    else:
        normalized_fprs = trimmed_fprs

    return metrics.auc(normalized_fprs, trimmed_pros)


def compute_auc_spro_curve(
    masks,
    amaps,
    max_fprs=None,
    num_thresholds: int = 200,
    structuring_element_size: int = 1,
    defect_id_to_config: Optional[Dict[int, Dict]] = None,
):
    """
    Computes the area under the (saturated) PRO curves up to several FPR cutoffs.

    Args:
         masks: [np.array or list] [NxHxW] Integer/binary ground-truth masks (0=background).
             If `defect_id_to_config` is provided, positive labels map to defect types.
         amaps: [np.array or list] [NxHxW] Severity/anomaly maps.
        max_fprs: Sequence of max FPR boundaries to integrate up to.
         num_thresholds: Number of thresholds used when building the curve.
         structuring_element_size: Kernel size for dilation; default 1 (no dilation) like LOCO.
         defect_id_to_config: Optional per-defect saturation config to emulate LOCO sPRO exactly.

    Returns:
        dict: {
            "auc_spro": {limit: auc},
            "fpr_curve": np.array,
            "spro_curve": np.array,
            "thresholds": np.array,
        }
    """
    if max_fprs is None:
        max_fprs = [0.01, 0.05, 0.1, 0.3, 1.0]
    max_fprs = sorted(set(max_fprs))

    fprs, spro_values, thresholds = _collect_fpr_spro_curve(
        masks,
        amaps,
        num_thresholds=num_thresholds,
        structuring_element_size=structuring_element_size,
        defect_id_to_config=defect_id_to_config,
    )

    auc_spros = {}
    for limit in max_fprs:
        auc = get_auc_for_max_fpr(
            fprs=fprs,
            y_values=spro_values,
            max_fpr=limit,
            scale_to_one=True,
        )
        auc_spros[limit] = auc

    return {
        "auc_spro": auc_spros,
        "fpr_curve": fprs,
        "spro_curve": spro_values,
        "thresholds": thresholds,
    }


def build_defect_id_to_config_from_json(defects_config_path: str) -> Dict[int, Dict]:
    """
    Build the defect-id -> saturation-config mapping from a LOCO-style JSON file.

    The JSON is expected to be a list of entries with fields:
      - defect_name: str (ignored here but useful for logging)
      - pixel_value: int (label value used in masks)
      - saturation_threshold: float|int
      - relative_saturation: bool

    Returns:
        Dict[int, Dict]: { pixel_value: {"relative": bool, "saturation_threshold": number } }
    """
    with open(defects_config_path, "r") as f:
        entries = json.load(f)

    mapping: Dict[int, Dict] = {}
    for e in entries:
        pixel_value = int(e["pixel_value"])
        mapping[pixel_value] = {
            "relative": bool(e.get("relative_saturation", False)),
            "saturation_threshold": e.get("saturation_threshold", 1.0),
        }
    return mapping


def compute_auc_spro_curve_from_defects_config(
    masks,
    amaps,
    defects_config_path: str,
    max_fprs=None,
    num_thresholds: int = 200,
    structuring_element_size: int = 1,
):
    """
    Convenience wrapper to compute LOCO-style AUC-sPRO using a defects_config.json file.

    Args:
        masks: (N,H,W) integer masks where background=0 and each defect type uses its pixel_value.
        amaps: (N,H,W) float anomaly maps.
        defects_config_path: Path to `<subclass>/defects_config.json`.
        max_fprs, num_thresholds, structuring_element_size: forwarded to compute_auc_spro_curve.

    Returns:
        Same dict as compute_auc_spro_curve.
    """
    defect_id_to_config = build_defect_id_to_config_from_json(defects_config_path)
    return compute_auc_spro_curve(
        masks=masks,
        amaps=amaps,
        max_fprs=max_fprs,
        num_thresholds=num_thresholds,
        structuring_element_size=structuring_element_size,
        defect_id_to_config=defect_id_to_config,
    )


def compute_simple_auc_spro(
    masks,
    amaps,
    saturation_threshold: float = 0.5,
    relative_saturation: bool = True,
    max_fprs=None,
    num_thresholds: int = 200,
):
    """
    Computes AUC sPRO with a simple global saturation threshold.
    
    This handles Connected Components sPRO where every defect region is saturated
    by the same global rule.

    Args:
        masks: [np.array or list] [NxHxW] Binary/Integer ground-truth masks.
        amaps: [np.array or list] [NxHxW] Anomaly maps.
        saturation_threshold: Global saturation threshold.
        relative_saturation: If True, threshold is a fraction of region area.
        max_fprs: List of FPR integration limits.

    Returns:
         dict: {
            "auc_spro": {limit: auc},
            "fpr_curve": np.array,
            "spro_curve": np.array,
            "thresholds": np.array,
        }
    """
    if max_fprs is None:
        max_fprs = [0.01, 0.05, 0.1, 0.3, 1.0]
    max_fprs = sorted(set(max_fprs))

    fprs, spro_values, thresholds = _collect_fpr_spro_curve(
        masks,
        amaps,
        num_thresholds=num_thresholds,
        # No defect_id_to_config -> triggers region-wise fallback
        defect_id_to_config=None,
        saturation_threshold=saturation_threshold,
        relative_saturation=relative_saturation,
    )

    auc_spros = {}
    for limit in max_fprs:
        auc = get_auc_for_max_fpr(
            fprs=fprs,
            y_values=spro_values,
            max_fpr=limit,
            scale_to_one=True,
        )
        auc_spros[limit] = auc

    return {
        "auc_spro": auc_spros,
        "fpr_curve": fprs,
        "spro_curve": spro_values,
        "thresholds": thresholds,
    }


def get_auc_for_max_fpr(fprs, y_values, max_fpr, scale_to_one=False):
    """
    Compute the area under a curve for a given maximum false positive rate.

    The area under the curve is computed by taking the area under the
    piecewise linear function that is defined by the given points of the
    curve.

    Args:
        fprs: The false positive rates of the curve.
        y_values: The corresponding y-values of the curve.
        max_fpr: The maximum false positive rate up to which the area under the
            curve is computed.
        scale_to_one: If True, the AUC is scaled to a range of [0, 1] by
            dividing through max_fpr.
    """
    # If max_fpr is smaller than the first FPR, the AUC is zero.
    if max_fpr < fprs[0]:
        return 0.

    # Sort the arrays by ascending FPRs.
    sorted_indices = np.argsort(fprs)
    fprs = fprs[sorted_indices]
    y_values = y_values[sorted_indices]

    # Add a point at max_fpr.
    if max_fpr not in fprs:
        # Get the y-value at max_fpr by linear interpolation.
        y_at_max_fpr = np.interp(max_fpr, fprs, y_values)
        # Add the point (max_fpr, y_at_max_fpr) to the arrays.
        fprs = np.append(fprs, max_fpr)
        y_values = np.append(y_values, y_at_max_fpr)
        # Sort again.
        sorted_indices = np.argsort(fprs)
        fprs = fprs[sorted_indices]
        y_values = y_values[sorted_indices]

    # Get all points up to max_fpr.
    mask = fprs <= max_fpr
    fprs_up_to_max_fpr = fprs[mask]
    y_values_up_to_max_fpr = y_values[mask]

    # Add a point at (0, y_value_at_0).
    if 0. not in fprs_up_to_max_fpr:
        # Prepend a point at (0, y_value_at_0).
        y_at_0 = np.interp(0., fprs, y_values)
        fprs_up_to_max_fpr = np.insert(fprs_up_to_max_fpr, 0, 0.)
        y_values_up_to_max_fpr = np.insert(y_values_up_to_max_fpr, 0, y_at_0)

    # Compute the area under the curve using the trapezoidal rule.
    auc = np.trapz(y_values_up_to_max_fpr, fprs_up_to_max_fpr)

    if scale_to_one:
        auc /= max_fpr

    return auc


def get_auc_spros_for_metrics(
        metrics,
        max_fprs,
        filter_defect_names_for_spro=None):
    """Compute AUC sPRO values for a given ThresholdMetrics instance.

    Args:
        metrics: The ThresholdMetrics instance.
        max_fprs: List of max fpr values to compute the AUC-sPRO for.
        filter_defect_names_for_spro: If not None, only the sPRO values from
            defect names in this sequence will be used. Does not affect the
            computation of FPRs!
    """
    auc_spros = {}
    for max_fpr in max_fprs:
        try:
            fp_rates = metrics.get_fp_rates()
        except ZeroDivisionError:
            auc = None
        else:
            mean_spros = metrics.get_mean_spros(
                filter_defect_names=filter_defect_names_for_spro)
            auc = get_auc_for_max_fpr(fprs=fp_rates,
                                      y_values=mean_spros,
                                      max_fpr=max_fpr,
                                      scale_to_one=True)
        auc_spros[max_fpr] = auc
    return auc_spros


def get_auc_spros_per_subdir(metrics,
                             anomaly_maps_test_dir,
                             max_fprs,
                             add_good_images):
    """Compute the AUC sPRO for images in subdirectories (usually "good",
    "structural_anomalies" and "logical_anomalies").

    If add_good_images is True, the images in the "good" subdirectory will be
    added to the images of each subdirectory for computing the corresponding
    AUC sPRO value. Hence, the result dict will not contain a "good" key.
    """
    aucs_per_subdir = {}
    subdir_names = os.listdir(anomaly_maps_test_dir)

    good_images = []
    if add_good_images:
        # Include the good images for each subdir.
        if 'good' in subdir_names:
            good_subdir = os.path.join(anomaly_maps_test_dir, 'good')
            good_subdir = os.path.realpath(good_subdir)
            good_images = [
                a for a in metrics.anomaly_maps
                if os.path.realpath(a.file_path).startswith(good_subdir)]
    # Regardless of add_good_images, we cannot compute an AUC sPRO value only
    # for the good images.
    if 'good' in subdir_names:
        subdir_names.remove('good')

    for subdir_name in subdir_names:
        subdir = os.path.join(anomaly_maps_test_dir, subdir_name)
        subdir = os.path.realpath(subdir)
        # Get all anomaly maps in here.
        subdir_anomaly_maps = [
            a for a in metrics.anomaly_maps
            if os.path.realpath(a.file_path).startswith(subdir)]
        if add_good_images:
            subdir_anomaly_maps += good_images

        subdir_metrics = metrics.reduce_to_images(subdir_anomaly_maps)

        aucs_per_subdir[subdir_name] = get_auc_spros_for_metrics(
            subdir_metrics, max_fprs)
    return aucs_per_subdir


def get_auc_spro_results(metrics,
                         anomaly_maps_test_dir,
                         max_fprs=[0.01, 0.05, 0.1, 0.3, 1.]):
    """Compute AUC sPRO values for all images, images in subdirectories and
    defect names.
    """
    # Compute the AUC sPRO for logical and structural anomalies.
    auc_spro = get_auc_spros_per_subdir(
        metrics=metrics,
        anomaly_maps_test_dir=anomaly_maps_test_dir,
        max_fprs=max_fprs,
        add_good_images=True)

    # Compute the mean performance over logical and structural anomalies.
    mean_spros = dict()
    if 'structural_anomalies' in auc_spro and 'logical_anomalies' in auc_spro:
        for limit in auc_spro['structural_anomalies'].keys():
            auc_spro_structural = auc_spro['structural_anomalies'][limit]
            auc_spro_logical = auc_spro['logical_anomalies'][limit]
            if auc_spro_structural is not None and auc_spro_logical is not None:
                mean = 0.5 * (auc_spro_structural + auc_spro_logical)
                mean_spros[limit] = mean
            else:
                mean_spros[limit] = None
        auc_spro['mean'] = mean_spros

    return {'auc_spro': auc_spro}


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
        # This is easier on CPU with skimage
        
        preds_cpu = preds.detach().cpu().numpy()
        target_cpu = target.detach().cpu().numpy()
        
        thresholds_np = self.thresholds.cpu().numpy()
        
        for i in range(preds.shape[0]):
            img_target = target_cpu[i]
            img_preds = preds_cpu[i]
            
            # Skip if good image (no defects)
            if img_target.sum() == 0:
                continue
                
            # Label connected components
            labeled_mask = measure.label(img_target)
            for region in measure.regionprops(labeled_mask):
                defect_area = region.area
                
                # Saturation Area
                if self.relative_saturation:
                    sat_area = defect_area * self.saturation_threshold
                    # Minimum 1 pixel? usually not strictly enforced but good for stability
                    sat_area = max(sat_area, 1.0)
                else:
                    sat_area = min(defect_area, self.saturation_threshold)
                
                # Calculate TP area for this specific region across ALL thresholds
                # Optimization: extract only the region pixels
                coords = region.coords
                region_scores = img_preds[coords[:, 0], coords[:, 1]]
                
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
