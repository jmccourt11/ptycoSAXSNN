import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from typing import List, Dict, Tuple, Optional, Union
import pandas as pd
import sys
import os
from matplotlib import colors
from skimage.metrics import structural_similarity as ssim
from scipy.signal import correlate2d
from scipy.ndimage import gaussian_filter
import importlib
from tqdm import tqdm
import h5py
import re
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../../NN/ptychosaxsNN/')))
import utils.ptychosaxsNN_utils as ptNN_U
importlib.reload(ptNN_U)
# Add the models directory to the path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../src/models/')))

class ModelComparer:
    """
    A comprehensive class for comparing neural network models with various metrics
    including PSNR, SSIM, cross-correlation, and peak detection analysis.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the ModelComparer.
        
        Args:
            device (str, optional): Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(self.device)
        
        # Default peak detection parameters
        self.default_peak_sigma = 0.714
        self.default_peak_threshold = 0.265
        self.peak_distance_threshold = 5.0
        self.ideal_peak_smoothing_sigma = None  # Additional smoothing for ideal images (None = no extra smoothing)
        
        # Adaptive thresholding parameters
        self.use_percentile_threshold = True  # If True, use percentile-based threshold instead of max-based
        self.percentile_threshold_value = 95.0  # Percentile to use for ideal images (e.g., 95.0 = 95th percentile)
        self.output_percentile_threshold_value = 90.0  # Percentile to use for output/deconvolved images (typically lower than ideal for better sensitivity)
        
        # Storage for cached data
        self._cached_ideal_peaks = {}
        self._cached_models = {}
    
    def ensure_tensor_format(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Ensure tensor is on the correct device and has the correct dtype.
        
        Args:
            tensor (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Tensor on correct device with float dtype
        """
        return tensor.to(device=self.device, dtype=torch.float)
    
    def normalize_image(self, img: torch.Tensor, apply_smoothing: bool = False, smoothing_sigma: float = None) -> torch.Tensor:
        """
        Normalize image by subtracting background (minimum value) and scaling to [0, 1].
        Optionally apply additional Gaussian smoothing (useful for ideal images before peak detection).
        
        Args:
            img (torch.Tensor): Input image
            apply_smoothing (bool): Whether to apply additional Gaussian smoothing
            smoothing_sigma (float): Sigma for Gaussian smoothing. If None and apply_smoothing=True, 
                                   uses self.ideal_peak_smoothing_sigma
            
        Returns:
            torch.Tensor: Normalized (and optionally smoothed) image
        """
        img_np = img.squeeze().cpu().numpy()
        img_np = img_np - np.min(img_np)
        max_val = np.max(img_np)-np.min(img_np)
        if max_val > 0:
            img_np = img_np / max_val
        
        # Apply additional smoothing if requested (typically for ideal images)
        if apply_smoothing:
            if smoothing_sigma is None:
                smoothing_sigma = self.ideal_peak_smoothing_sigma
            if smoothing_sigma is not None and smoothing_sigma > 0:
                img_np = gaussian_filter(img_np, sigma=smoothing_sigma)
                # Renormalize after smoothing to ensure [0, 1] range
                img_np = img_np - np.min(img_np)
                max_val = np.max(img_np)
                if max_val > 0:
                    img_np = img_np / max_val
        
        return torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
    
    def calculate_normalized_cross_correlation(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate 2D normalized cross-correlation between two images.
        
        Args:
            img1 (np.ndarray): First image
            img2 (np.ndarray): Second image
            
        Returns:
            float: Maximum normalized cross-correlation value
        """
        img1_norm = (img1 - np.mean(img1)) / (np.std(img1) * len(img1.ravel()))
        img2_norm = (img2 - np.mean(img2)) / np.std(img2)
        corr = correlate2d(img1_norm, img2_norm, mode='same')
        return np.max(corr)
    
    def find_peaks_and_fwhm(self, image: np.ndarray, 
                           threshold: float = None, 
                           sigma: float = None,
                           use_percentile: bool = None,
                           percentile: float = None) -> Tuple[List[Tuple[float, float]], List[float]]:
        """
        Find peaks and their FWHM in a 2D image, including peaks at edges.
        
        Args:
            image (np.ndarray): Input image
            threshold (float): Threshold for peak detection (relative to max if not using percentile)
            sigma (float): Sigma for Gaussian smoothing
            use_percentile (bool): If True, use percentile-based threshold. If None, uses self.use_percentile_threshold
            percentile (float): Percentile to use (e.g., 95.0 for 95th percentile). If None, uses self.percentile_threshold_value
            
        Returns:
            Tuple[List[Tuple[float, float]], List[float]]: Peak positions and FWHM values
        """
        if threshold is None:
            threshold = self.default_peak_threshold
        if sigma is None:
            sigma = self.default_peak_sigma
        if use_percentile is None:
            use_percentile = self.use_percentile_threshold
        if percentile is None:
            percentile = self.percentile_threshold_value
            
        smoothed = gaussian_filter(image, sigma=sigma)
        peaks = []
        fwhm_values = []
        
        # Calculate threshold based on method
        if use_percentile:
            # Percentile-based adaptive threshold
            threshold_val = np.percentile(smoothed, percentile)
        else:
            # Original max-based threshold
            max_val = np.max(smoothed)
            threshold_val = max_val * threshold
        
        height, width = smoothed.shape
        
        def is_local_max(i: int, j: int) -> bool:
            val = smoothed[i,j]
            i_start = max(0, i-1)
            i_end = min(height, i+2)
            j_start = max(0, j-1)
            j_end = min(width, j+2)
            neighborhood = smoothed[i_start:i_end, j_start:j_end]
            return val >= np.max(neighborhood)
        
        for i in range(height):
            for j in range(width):
                if smoothed[i,j] > threshold_val and is_local_max(i, j):
                    peaks.append((i, j))
                    
                    x_profile = smoothed[i,:]
                    y_profile = smoothed[:,j]
                    center_val = smoothed[i,j]
                    half_max = center_val / 2
                    
                    try:
                        x_above = x_profile > half_max
                        if j == 0 or j == width-1:
                            x_fwhm = 2 * np.sum(x_above)
                        else:
                            x_transitions = np.where(x_above[:-1] != x_above[1:])[0]
                            if len(x_transitions) >= 2:
                                x_fwhm = x_transitions[-1] - x_transitions[0]
                            else:
                                x_fwhm = np.sum(x_above)
                        
                        y_above = y_profile > half_max
                        if i == 0 or i == height-1:
                            y_fwhm = 2 * np.sum(y_above)
                        else:
                            y_transitions = np.where(y_above[:-1] != y_above[1:])[0]
                            if len(y_transitions) >= 2:
                                y_fwhm = y_transitions[-1] - y_transitions[0]
                            else:
                                y_fwhm = np.sum(y_above)
                        
                        fwhm_values.append((x_fwhm + y_fwhm) / 2)
                        
                    except Exception:
                        x_fwhm = np.sum(x_profile > half_max)
                        y_fwhm = np.sum(y_profile > half_max)
                        fwhm_values.append((x_fwhm + y_fwhm) / 2)
        
        return peaks, fwhm_values
    
    def _find_local_max_near(self, image: np.ndarray, center_row: float, center_col: float, 
                             sigma: float, search_radius: int = 5) -> Optional[Tuple[int, int]]:
        """
        Find local maximum in a small region around the center point.
        
        Args:
            image: Input image (should be normalized)
            center_row: Row coordinate of center point
            center_col: Column coordinate of center point
            sigma: Sigma for Gaussian smoothing
            search_radius: Radius to search around center point
            
        Returns:
            Tuple of (row, col) if local maximum found, None otherwise
        """
        smoothed = gaussian_filter(image, sigma=sigma)
        h, w = smoothed.shape
        row_start = max(0, int(center_row) - search_radius)
        row_end = min(h, int(center_row) + search_radius + 1)
        col_start = max(0, int(center_col) - search_radius)
        col_end = min(w, int(center_col) + search_radius + 1)
        
        region = smoothed[row_start:row_end, col_start:col_end]
        if region.size == 0:
            return None
        
        max_idx = np.unravel_index(np.argmax(region), region.shape)
        max_row = row_start + max_idx[0]
        max_col = col_start + max_idx[1]
        
        # Check if it's a local maximum
        val = smoothed[max_row, max_col]
        i_start = max(0, max_row - 1)
        i_end = min(h, max_row + 2)
        j_start = max(0, max_col - 1)
        j_end = min(w, max_col + 2)
        neighborhood = smoothed[i_start:i_end, j_start:j_end]
        
        if val >= np.max(neighborhood):
            return (max_row, max_col)
        return None
    
    def calculate_peak_confusion_matrix(self, output_peaks: List[Tuple[float, float]], 
                                        ideal_peaks: List[Tuple[float, float]],
                                        distance_threshold: float = None) -> Dict[str, Union[int, float]]:
        """
        Calculate True Positives, False Positives, False Negatives for peak detection.
        Uses one-to-one matching (greedy algorithm: match closest pairs first).
        
        Args:
            output_peaks: List of peak positions from output image [(row, col), ...]
            ideal_peaks: List of peak positions from ideal/ground truth image [(row, col), ...]
            distance_threshold: Maximum distance for considering peaks as matched. 
                               If None, uses self.peak_distance_threshold
        
        Returns:
            Dictionary with keys: 'tp', 'fp', 'fn', 'tn', 'precision', 'recall', 'f1_score', 'matches'
            Note: TN (true negatives) is not meaningful for peak detection and is set to 0
            'matches' is a list of tuples (out_idx, ideal_idx) for matched peaks
        """
        if distance_threshold is None:
            distance_threshold = self.peak_distance_threshold
        
        if not output_peaks and not ideal_peaks:
            return {
                'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0,
                'precision': 1.0, 'recall': 1.0, 'f1_score': 1.0,
                'matches': []
            }
        
        if not ideal_peaks:
            # All output peaks are false positives
            return {
                'tp': 0, 'fp': len(output_peaks), 'fn': 0, 'tn': 0,
                'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0,
                'matches': []
            }
        
        if not output_peaks:
            # All ideal peaks are false negatives
            return {
                'tp': 0, 'fp': 0, 'fn': len(ideal_peaks), 'tn': 0,
                'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0,
                'matches': []
            }
        
        # Create one-to-one matching using greedy algorithm
        # Sort all possible pairs by distance and match closest pairs first
        matches = []
        output_matched = set()
        ideal_matched = set()
        
        # Create list of all possible pairs with distances
        pairs = []
        for i, out_peak in enumerate(output_peaks):
            for j, ideal_peak in enumerate(ideal_peaks):
                dist = np.sqrt((out_peak[0] - ideal_peak[0])**2 + (out_peak[1] - ideal_peak[1])**2)
                if dist <= distance_threshold:
                    pairs.append((dist, i, j))
        
        # Sort by distance (closest first)
        pairs.sort(key=lambda x: x[0])
        
        # Greedy matching: take closest pairs first, ensuring one-to-one matching
        for dist, out_idx, ideal_idx in pairs:
            if out_idx not in output_matched and ideal_idx not in ideal_matched:
                matches.append((out_idx, ideal_idx))
                output_matched.add(out_idx)
                ideal_matched.add(ideal_idx)
        
        tp = len(matches)
        fp = len(output_peaks) - tp  # Output peaks not matched
        fn = len(ideal_peaks) - tp   # Ideal peaks not matched
        
        # Calculate precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': 0,  # Not meaningful for peak detection
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'num_output_peaks': len(output_peaks),
            'num_ideal_peaks': len(ideal_peaks),
            'matches': matches,
            'output_matched': output_matched,
            'ideal_matched': ideal_matched
        }
    
    def _detect_peaks_with_adaptive_threshold(self, output_tensor: torch.Tensor, 
                                             ideal_peaks: List[Tuple[float, float]],
                                             used_sigma: float, 
                                             used_threshold: float,
                                             verbose: bool = False) -> Tuple[List[Tuple[float, float]], Dict[str, Union[int, float]]]:
        """
        Detect peaks in output image with adaptive thresholding if needed.
        Returns output peaks and confusion matrix metrics.
        
        Args:
            output_tensor: Output tensor to detect peaks in
            ideal_peaks: List of ideal peak positions
            used_sigma: Sigma value for peak detection
            used_threshold: Threshold value for peak detection
            verbose: Whether to print peak counts
            
        Returns:
            Tuple of (output_peaks, confusion_metrics)
        """
        # Normalize output before peak detection
        output_norm = self.normalize_image(output_tensor)
        output_np = output_norm.squeeze().cpu().numpy()
        
        # Use output-specific percentile if using percentile threshold
        output_percentile = self.output_percentile_threshold_value if self.use_percentile_threshold else None
        output_peaks, _ = self.find_peaks_and_fwhm(output_np, 
                                                   sigma=used_sigma, 
                                                   threshold=used_threshold,
                                                   percentile=output_percentile)
        
        # If we're missing many peaks, try with a lower threshold/percentile
        if len(output_peaks) < len(ideal_peaks) * 0.7:
            if self.use_percentile_threshold:
                # Try with progressively lower percentiles
                for lower_percentile in [self.output_percentile_threshold_value * 0.95, 
                                         self.output_percentile_threshold_value * 0.90,
                                         self.output_percentile_threshold_value * 0.85,
                                         self.output_percentile_threshold_value * 0.80]:
                    output_peaks_lower, _ = self.find_peaks_and_fwhm(output_np, 
                                                                     sigma=used_sigma, 
                                                                     threshold=used_threshold,
                                                                     percentile=lower_percentile)
                    if len(output_peaks_lower) >= len(output_peaks):
                        output_peaks = output_peaks_lower
                        if len(output_peaks) >= len(ideal_peaks) * 0.8:
                            break  # Good enough, stop trying
            else:
                # Try with progressively lower thresholds
                for lower_thresh in [used_threshold * 0.8, used_threshold * 0.6, used_threshold * 0.4]:
                    output_peaks_lower, _ = self.find_peaks_and_fwhm(output_np, 
                                                                     sigma=used_sigma, 
                                                                     threshold=lower_thresh)
                    if len(output_peaks_lower) >= len(output_peaks):
                        output_peaks = output_peaks_lower
                        if len(output_peaks) >= len(ideal_peaks) * 0.8:
                            break  # Good enough, stop trying
        
        if verbose:
            print(f"Found {len(output_peaks)} output peaks vs {len(ideal_peaks)} ideal peaks")
        
        # Calculate confusion matrix
        confusion_metrics = self.calculate_peak_confusion_matrix(output_peaks, ideal_peaks)
        
        return output_peaks, confusion_metrics
    
    def _plot_peaks_on_axis(self, ax, output_peaks: List[Tuple[float, float]], 
                           ideal_peaks: List[Tuple[float, float]],
                           confusion_metrics: Dict[str, Union[int, float]],
                           show_peak_classification: bool = False):
        """
        Plot peaks on an axis with optional classification markers.
        
        Args:
            ax: Matplotlib axis to plot on
            output_peaks: List of output peak positions
            ideal_peaks: List of ideal peak positions
            confusion_metrics: Dictionary from calculate_peak_confusion_matrix
            show_peak_classification: If True, shows TP/FP/FN markers
        """
        matches = confusion_metrics['matches']
        output_matched = confusion_metrics['output_matched']
        ideal_matched = confusion_metrics['ideal_matched']
        
        if show_peak_classification:
            # Show TP/FP/FN classification with different markers
            # Plot TP (True Positives - matched peaks)
            for out_idx, ideal_idx in matches:
                tp_peak = output_peaks[out_idx]
                ax.plot(tp_peak[1], tp_peak[0], 'c+', markersize=10, markeredgewidth=2, alpha=0.7)
            
            # Plot FP (False Positives - output peaks not matched)
            for i, out_peak in enumerate(output_peaks):
                if i not in output_matched:
                    ax.plot(out_peak[1], out_peak[0], 'ro', markersize=10, markeredgewidth=2, alpha=0.7, fillstyle='none')
            
            # Plot FN (False Negatives - ideal peaks not matched)
            for j, ideal_peak in enumerate(ideal_peaks):
                if j not in ideal_matched:
                    ax.plot(ideal_peak[1], ideal_peak[0], 'y*', markersize=10, markeredgewidth=2, alpha=0.7)
            
            # Also plot ideal peaks that were matched (for reference)
            for out_idx, ideal_idx in matches:
                ideal_peak = ideal_peaks[ideal_idx]
                ax.plot(ideal_peak[1], ideal_peak[0], 'kx', markersize=10, markeredgewidth=1.5, alpha=0.7)
        else:
            # Original behavior: just show ideal peaks and matched output peaks
            for peak in ideal_peaks:
                ax.plot(peak[1], peak[0], 'g+', markersize=10, markeredgewidth=2, alpha=0.7)
            
            # Plot matched output peaks using one-to-one matches from confusion matrix
            for out_idx, ideal_idx in matches:
                tp_peak = output_peaks[out_idx]
                ax.plot(tp_peak[1], tp_peak[0], 'rx', markersize=8, markeredgewidth=2, alpha=0.7)
    
    def _build_metrics_text(self, metrics: Dict[str, float], 
                           calculate_psnr: bool, calculate_ssim: bool, calculate_xcorr: bool,
                           calculate_peaks: bool, ideal_peaks: Optional[List[Tuple[float, float]]],
                           confusion_metrics: Optional[Dict[str, Union[int, float]]] = None,
                           show_peak_classification: bool = False) -> List[str]:
        """
        Build metrics text list for display.
        
        Args:
            metrics: Dictionary of calculated metrics
            calculate_psnr: Whether PSNR was calculated
            calculate_ssim: Whether SSIM was calculated
            calculate_xcorr: Whether cross-correlation was calculated
            calculate_peaks: Whether peaks were calculated
            ideal_peaks: List of ideal peaks (for peak metrics)
            confusion_metrics: Dictionary from calculate_peak_confusion_matrix (optional)
            show_peak_classification: Whether to show TP/FP/FN breakdown
            
        Returns:
            List of metric strings
        """
        metrics_text = []
        
        if calculate_psnr and 'psnr' in metrics:
            metrics_text.append(f'PSNR: {metrics["psnr"]:.2f} dB')
        if calculate_ssim and 'ssim' in metrics:
            metrics_text.append(f'SSIM: {metrics["ssim"]:.4f}')
        if calculate_xcorr and 'xcorr' in metrics:
            metrics_text.append(f'XCORR: {metrics["xcorr"]:.4f}')
        
        if calculate_peaks and 'avg_peak_dist' in metrics and ideal_peaks:
            metrics_text.append(f'Peak Dist: {metrics["avg_peak_dist"]:.2f}')
            metrics_text.append(f'FWHM Diff: {metrics["fwhm_diff"]:.2f}')
            
            if confusion_metrics:
                tp_count = confusion_metrics['tp']
                fp_count = confusion_metrics['fp']
                fn_count = confusion_metrics['fn']
                matched_peaks = confusion_metrics['tp']
                
                if show_peak_classification:
                    metrics_text.append(f'TP: {tp_count}, FP: {fp_count}, FN: {fn_count}')
                    if tp_count + fp_count > 0:
                        precision = tp_count / (tp_count + fp_count)
                        metrics_text.append(f'Precision: {precision:.3f}')
                    if tp_count + fn_count > 0:
                        recall = tp_count / (tp_count + fn_count)
                        metrics_text.append(f'Recall: {recall:.3f}')
                else:
                    metrics_text.append(f'Matched: {matched_peaks}/{len(ideal_peaks)}')
        
        return metrics_text
    
    def _format_model_label(self, loss_type: str, use_unet: bool) -> str:
        """
        Format model label for display.
        
        Args:
            loss_type: Loss function type
            use_unet: Whether model uses Unet
            
        Returns:
            Formatted label string
        """
        model_type = "w/ skip connections" if use_unet else "w/o skip connections"
        loss_label = loss_type
        if loss_label == 'pearson_loss' or loss_label == 'pearson':
            loss_label = 'NPCC'
        return f"{loss_label}\n{model_type}"
    
    def _create_peak_legend_elements(self, show_peak_classification: bool) -> List:
        """
        Create legend elements for peak markers.
        
        Args:
            show_peak_classification: Whether classification mode is enabled
            
        Returns:
            List of matplotlib Line2D objects for legend
        """
        if show_peak_classification:
            return [
                plt.Line2D([0], [0], marker='+', color='c', label='TP (True Positive)', 
                          markersize=10, linestyle='None', markeredgewidth=2),
                plt.Line2D([0], [0], marker='o', color='r', label='FP (False Positive)', 
                          markersize=10, linestyle='None', markeredgewidth=2, fillstyle='none'),
                plt.Line2D([0], [0], marker='*', color='y', label='FN (False Negative)', 
                          markersize=10, linestyle='None', markeredgewidth=2),
                plt.Line2D([0], [0], marker='x', color='k', label='Ideal (matched)', 
                          markersize=10, linestyle='None', markeredgewidth=1.5)
            ]
        else:
            return [
                plt.Line2D([0], [0], marker='+', color='g', label='Ideal Peaks', 
                          markersize=8, linestyle='None'),
                plt.Line2D([0], [0], marker='x', color='r', label='Matched Model Peaks', 
                          markersize=8, linestyle='None')
            ]
    
    def calculate_peak_sensitivity_metrics(self, img1: np.ndarray, 
                                         img2: np.ndarray,
                                         sigma_range: List[float] = None,
                                         threshold_range: List[float] = None) -> Dict[str, float]:
        """
        Calculate comprehensive peak detection metrics across different peak finder parameters.
        
        Args:
            img1 (np.ndarray): First image (model output)
            img2 (np.ndarray): Second image (ground truth)
            sigma_range (List[float]): Range of sigma values for Gaussian smoothing
            threshold_range (List[float]): Range of threshold values for peak detection
            
        Returns:
            Dict[str, float]: Dictionary of peak sensitivity metrics
        """
        if sigma_range is None:
            sigma_range = [0.5, 0.714, 1.0, 1.5, 2.0]
        if threshold_range is None:
            threshold_range = [0.1, 0.2, 0.265, 0.3, 0.4]
            
        metrics = {
            'optimal_sigma': 0.0,
            'optimal_threshold': 0.0,
            'max_f1_score': 0.0,
            'peak_position_stability': 0.0,
            'peak_count_stability': 0.0,
            'parameter_sensitivity': 0.0,
            'false_positive_rate': 0.0,
            'false_negative_rate': 0.0,
            'peak_intensity_correlation': 0.0,
            'peak_shape_consistency': 0.0
        }
        
        results = []
        peak_positions_all = []
        peak_counts = []
        
        # Normalize images to 0-1 range before peak detection
        img1_norm = (img1 - img1.min()) / (img1.max() - img1.min() + 1e-10)
        img2_norm = (img2 - img2.min()) / (img2.max() - img2.min() + 1e-10)
        
        # Apply additional smoothing to ideal image (img2) if configured
        if self.ideal_peak_smoothing_sigma is not None and self.ideal_peak_smoothing_sigma > 0:
            img2_norm = gaussian_filter(img2_norm, sigma=self.ideal_peak_smoothing_sigma)
            # Renormalize after smoothing
            img2_norm = img2_norm - np.min(img2_norm)
            max_val = np.max(img2_norm)
            if max_val > 0:
                img2_norm = img2_norm / max_val
        
        mid_sigma = np.median(sigma_range)
        mid_threshold = np.median(threshold_range)
        gt_peaks, gt_fwhm = self.find_peaks_and_fwhm(img2_norm, threshold=mid_threshold, sigma=mid_sigma)
        
        for sigma in sigma_range:
            for threshold in threshold_range:
                peaks, fwhm = self.find_peaks_and_fwhm(img1_norm, threshold=threshold, sigma=sigma)
                peak_positions_all.extend(peaks)
                peak_counts.append(len(peaks))
                
                matched = 0
                false_positives = 0
                peak_intensities1 = []
                peak_intensities2 = []
                
                for peak in peaks:
                    min_dist = float('inf')
                    closest_gt_peak = None
                    
                    for gt_peak in gt_peaks:
                        dist = np.sqrt((peak[0]-gt_peak[0])**2 + (peak[1]-gt_peak[1])**2)
                        if dist < min_dist:
                            min_dist = dist
                            closest_gt_peak = gt_peak
                    
                    if min_dist <= self.peak_distance_threshold:
                        matched += 1
                        peak_intensities1.append(img1[peak[0], peak[1]])
                        peak_intensities2.append(img2[closest_gt_peak[0], closest_gt_peak[1]])
                    else:
                        false_positives += 1
                
                precision = matched / len(peaks) if peaks else 0
                recall = matched / len(gt_peaks) if gt_peaks else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                results.append({
                    'sigma': sigma,
                    'threshold': threshold,
                    'f1_score': f1,
                    'matched': matched,
                    'false_positives': false_positives,
                    'false_negatives': len(gt_peaks) - matched,
                    'peak_count': len(peaks)
                })
                
                if f1 > metrics['max_f1_score']:
                    metrics['max_f1_score'] = f1
                    metrics['optimal_sigma'] = sigma
                    metrics['optimal_threshold'] = threshold
        
        if peak_positions_all:
            peak_positions_array = np.array(peak_positions_all)
            metrics['peak_position_stability'] = 1.0 / (np.std(peak_positions_array[:, 0]) + 
                                                      np.std(peak_positions_array[:, 1]) + 1e-6)
            metrics['peak_count_stability'] = 1.0 / (np.std(peak_counts) / np.mean(peak_counts) + 1e-6)
        
        f1_scores = [r['f1_score'] for r in results]
        metrics['parameter_sensitivity'] = np.std(f1_scores) / (np.mean(f1_scores) + 1e-6)
        
        metrics['false_positive_rate'] = np.mean([r['false_positives'] / r['peak_count'] 
                                                if r['peak_count'] > 0 else 0 for r in results])
        metrics['false_negative_rate'] = np.mean([r['false_negatives'] / len(gt_peaks) 
                                                if gt_peaks else 0 for r in results])
        
        return metrics
    
    def calculate_metrics(self, img1: torch.Tensor, 
                         img2: torch.Tensor,
                         calculate_psnr: bool = True,
                         calculate_ssim: bool = True,
                         calculate_xcorr: bool = False,
                         calculate_peaks: bool = True,
                         calculate_peak_sensitivity: bool = False,
                         peak_sigma: float = None,
                         peak_threshold: float = None) -> Dict[str, float]:
        """
        Calculate selected metrics between two images.
        Images are normalized before metric calculation.
        
        Args:
            img1 (torch.Tensor): First image (model output)
            img2 (torch.Tensor): Second image (ground truth)
            calculate_psnr (bool): Whether to calculate PSNR
            calculate_ssim (bool): Whether to calculate SSIM
            calculate_xcorr (bool): Whether to calculate cross-correlation
            calculate_peaks (bool): Whether to calculate peak metrics
            calculate_peak_sensitivity (bool): Whether to optimize peak finding parameters first
            peak_sigma (float): Sigma for Gaussian smoothing in peak detection
            peak_threshold (float): Threshold for peak detection
            
        Returns:
            Dict[str, float]: Dictionary of calculated metrics
        """
        if peak_sigma is None:
            peak_sigma = self.default_peak_sigma
        if peak_threshold is None:
            peak_threshold = self.default_peak_threshold
            
        img1_norm = self.normalize_image(img1)
        img2_norm = self.normalize_image(img2)
        
        img1_np = img1_norm.squeeze().cpu().numpy()
        img2_np = img2_norm.squeeze().cpu().numpy()
        
        metrics = {}
        
        if calculate_psnr:
            mse = np.mean((img1_np - img2_np) ** 2)
            if mse == 0:
                metrics['psnr'] = float('inf')
            else:
                max_pixel = 1.0
                metrics['psnr'] = 20 * np.log10(max_pixel / np.sqrt(mse))
        
        if calculate_ssim:
            metrics['ssim'] = ssim(img1_np, img2_np, data_range=1.0)
        
        if calculate_xcorr:
            metrics['xcorr'] = self.calculate_normalized_cross_correlation(img1_np, img2_np)
        
        optimal_sigma = peak_sigma
        optimal_threshold = peak_threshold
        
        if calculate_peak_sensitivity:
            sensitivity_metrics = self.calculate_peak_sensitivity_metrics(
                img1_np, img2_np,
                sigma_range=[0.5, 0.714, 1.0, 1.5, 2.0],
                threshold_range=[0.1, 0.2, 0.265, 0.3, 0.4]
            )
            
            optimal_sigma = sensitivity_metrics['optimal_sigma']
            optimal_threshold = sensitivity_metrics['optimal_threshold']
            
            metrics.update({
                'optimal_sigma': sensitivity_metrics['optimal_sigma'],
                'optimal_threshold': sensitivity_metrics['optimal_threshold'],
                'max_f1_score': sensitivity_metrics['max_f1_score'],
                'peak_position_stability': sensitivity_metrics['peak_position_stability'],
                'peak_count_stability': sensitivity_metrics['peak_count_stability'],
                'parameter_sensitivity': sensitivity_metrics['parameter_sensitivity'],
                'false_positive_rate': sensitivity_metrics['false_positive_rate'],
                'false_negative_rate': sensitivity_metrics['false_negative_rate']
            })
        
        if calculate_peaks:
            # For peak detection, apply additional smoothing to ideal image (img2) if configured
            img2_np_for_peaks = img2_np.copy()
            if self.ideal_peak_smoothing_sigma is not None and self.ideal_peak_smoothing_sigma > 0:
                img2_np_for_peaks = gaussian_filter(img2_np_for_peaks, sigma=self.ideal_peak_smoothing_sigma)
                # Renormalize after smoothing
                img2_np_for_peaks = img2_np_for_peaks - np.min(img2_np_for_peaks)
                max_val = np.max(img2_np_for_peaks)
                if max_val > 0:
                    img2_np_for_peaks = img2_np_for_peaks / max_val
            
            # Use output-specific percentile for img1 (output), ideal percentile for img2 (ideal)
            output_percentile = self.output_percentile_threshold_value if self.use_percentile_threshold else None
            ideal_percentile = self.percentile_threshold_value if self.use_percentile_threshold else None
            peaks1, fwhm1 = self.find_peaks_and_fwhm(img1_np, threshold=optimal_threshold, sigma=optimal_sigma, percentile=output_percentile)
            peaks2, fwhm2 = self.find_peaks_and_fwhm(img2_np_for_peaks, threshold=optimal_threshold, sigma=optimal_sigma, percentile=ideal_percentile)
            
            metrics['peak_sigma_used'] = optimal_sigma
            metrics['peak_threshold_used'] = optimal_threshold
            
            if peaks1 and peaks2:
                peak_diffs = []
                fwhm_diffs = []
                
                for p1, f1 in zip(peaks1, fwhm1):
                    min_dist = float('inf')
                    closest_f2 = None
                    for p2, f2 in zip(peaks2, fwhm2):
                        dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                        if dist < min_dist:
                            min_dist = dist
                            closest_f2 = f2
                    peak_diffs.append(min_dist)
                    if closest_f2 is not None:
                        fwhm_diffs.append(abs(f1 - closest_f2))
                
                metrics['avg_peak_dist'] = np.mean(peak_diffs)
                metrics['max_peak_dist'] = np.max(peak_diffs)
                metrics['num_peaks1'] = len(peaks1)
                metrics['num_peaks2'] = len(peaks2)
                if fwhm_diffs:
                    metrics['fwhm_diff'] = np.mean(fwhm_diffs)
        
        return metrics
    
    def load_model(self, model_path: str, use_unet: bool = True) -> torch.nn.Module:
        """
        Load a trained model from the specified path.
        
        Args:
            model_path (str): Path to the saved model
            use_unet (bool): Whether to use Unet architecture
            
        Returns:
            torch.nn.Module: Loaded model
        """
        if model_path in self._cached_models:
            return self._cached_models[model_path]
            
        if use_unet:
            #from encoder1_old import recon_model
            #model = recon_model()
            from encoder1 import recon_model
            model = recon_model()
        else:
            #from encoder1_no_Unet_old import recon_model
            #model = recon_model()
            from encoder1_no_Unet import recon_model
            model = recon_model()
        
        model.load_state_dict(torch.load(model_path))
        model.to(self.device)
        model.eval()
        
        self._cached_models[model_path] = model
        return model
    
    def get_model_paths_from_config(self, model_configs: Dict, base_path: str = "") -> List[Dict]:
        """
        Generate model paths and metadata from model_configs dictionary.
        
        Args:
            model_configs (Dict): Dictionary containing model configurations
            base_path (str): Base path where models are stored
            
        Returns:
            List[Dict]: List of dictionaries containing model paths and metadata
        """
        model_info = []
        
        for model_name, model_path in model_configs['models'].items():
            parts = model_name.split('_')
            
            if 'Lattice' in model_name:
                lattice_type = next(p for p in parts if 'Lattice' in p).replace('Lattice','')
                probe_size = next(p for p in parts if 'Probe' in p).split('x')[0].replace('Probe','')
                noise_status = next(p for p in parts if 'Noise' in p)
                use_unet = 'no_Unet' not in model_name
                loss_type = parts[-1]
            else:
                loss_type = parts[0]
                use_unet = 'no_Unet' not in model_name
                lattice_type = None
                probe_size = None
                noise_status = None
            
            for iteration in model_configs['iterations']:
                full_path = Path(base_path) / model_path.format(iteration)
                    
                if full_path.exists():
                    info = {
                        'path': str(full_path),
                        'loss_type': loss_type,
                        'use_unet': use_unet,
                        'iterations': iteration
                    }
                    
                    if lattice_type:
                        info.update({
                            'lattice_type': lattice_type,
                            'probe_size': probe_size,
                            'noise_status': noise_status
                        })
                        
                    model_info.append(info)
        
        return model_info
    
    def create_comparison_grid(self, model_configs: Dict,
                             input_data: torch.Tensor,
                             ideal_data: torch.Tensor,
                             base_path: str = "",
                             figsize: Tuple[int, int] = (20, 15),
                             calculate_psnr: bool = True,
                             calculate_ssim: bool = True,
                             calculate_xcorr: bool = False,
                             calculate_peaks: bool = True,
                             calculate_peak_sensitivity: bool = False,
                             peak_sigma: float = None,
                             show_peak_classification: bool = False) -> plt.Figure:
        """
        Create a grid plot comparing outputs from different models using model_configs.
        Shows input and ideal images above the model comparison grid.
        Includes selected metrics for each model output compared to the ideal image.
        
        Args:
            model_configs: Dictionary containing model configurations
            input_data: Input tensor data
            ideal_data: Ideal/ground truth tensor data
            base_path: Base path for model files
            figsize: Figure size tuple
            calculate_psnr: Whether to calculate PSNR
            calculate_ssim: Whether to calculate SSIM
            calculate_xcorr: Whether to calculate cross-correlation
            calculate_peaks: Whether to calculate peak metrics
            calculate_peak_sensitivity: Whether to calculate peak sensitivity metrics
            peak_sigma: Sigma value for peak detection
            show_peak_classification: If True, shows TP/FP/FN peak classification with different markers:
                - Cyan '+' for TP (True Positives - correctly detected peaks)
                - Red 'o' for FP (False Positives - output peaks that shouldn't be there)
                - Yellow '*' for FN (False Negatives - ideal peaks that were missed)
                - Green '+' for ideal peaks that were matched (for reference)
        """
        if peak_sigma is None:
            peak_sigma = self.default_peak_sigma
            
        # Ensure input data is on the correct device and dtype
        input_data = self.ensure_tensor_format(input_data)
        ideal_data = self.ensure_tensor_format(ideal_data)
            
        model_info = self.get_model_paths_from_config(model_configs, base_path)
        
        if not model_info:
            raise ValueError("No valid model paths found in the configuration")
        
        df = pd.DataFrame(model_info)
        nrows = len(model_configs['iterations'])
        ncols = len(model_configs['models'])
        
        width_per_col = 6
        height_per_row = 5
        figsize = (width_per_col * ncols, height_per_row * (nrows + 1))
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(nrows + 1, ncols, 
                             height_ratios=[1.2] + [1]*nrows,
                             hspace=0.3, wspace=0.3)
        
        # Plot input and ideal images
        ax_input = fig.add_subplot(gs[0, 0])
        ax_ideal = fig.add_subplot(gs[0, 1])
        
        im_input = ax_input.imshow(input_data.squeeze().cpu().numpy(), cmap='jet')
        ax_input.set_title('Input Image', fontsize=24)
        plt.colorbar(im_input, ax=ax_input, fraction=0.046, pad=0.04)
        ax_input.tick_params(axis='both', labelsize=20)
       
        im_ideal = ax_ideal.imshow(ideal_data.squeeze().cpu().numpy(), cmap='jet')
        ax_ideal.set_title('Ideal Image', fontsize=24)
        plt.colorbar(im_ideal, ax=ax_ideal, fraction=0.046, pad=0.04)
        ax_ideal.tick_params(axis='both', labelsize=20)
        
        # Find peaks in ideal image if peak calculation is enabled
        ideal_peaks = None
        if calculate_peaks:
            # Normalize ideal data to 0-1 before peak detection for consistency
            # Apply additional smoothing to ideal image if configured
            ideal_data_norm = self.normalize_image(ideal_data, apply_smoothing=True, smoothing_sigma=self.ideal_peak_smoothing_sigma)
            ideal_peaks, ideal_fwhm = self.find_peaks_and_fwhm(ideal_data_norm.squeeze().cpu().numpy(), 
                                                               sigma=peak_sigma, 
                                                               threshold=self.default_peak_threshold)
            for peak in ideal_peaks:
                ax_ideal.plot(peak[1], peak[0], 'kx', markersize=10, markeredgewidth=2,alpha=0.7)
        
        # Create axes for model outputs
        axes = np.zeros((nrows, ncols), dtype=object)
        for i in range(nrows):
            for j in range(ncols):
                axes[i, j] = fig.add_subplot(gs[i+1, j])
        
        df = df.sort_values(['iterations', 'loss_type'])
        
        # Plot each model's output
        for idx, (_, row) in enumerate(df.iterrows()):
            row_idx = model_configs['iterations'].index(row['iterations'])
            col_idx = list(model_configs['models'].keys()).index(
                f"{row['loss_type']}_{'Unet' if row['use_unet'] else 'no_Unet'}")
            
            #print('\nModel path: ' +row['path'])
            
            model = self.load_model(row['path'], row['use_unet'])
            with torch.no_grad():
                output = model(input_data)
            
            metrics = self.calculate_metrics(output, ideal_data,
                                          calculate_psnr=calculate_psnr,
                                          calculate_ssim=calculate_ssim,
                                          calculate_xcorr=calculate_xcorr,
                                          calculate_peaks=calculate_peaks,
                                          calculate_peak_sensitivity=calculate_peak_sensitivity,
                                          peak_sigma=peak_sigma)
            #print('Metrics: ', metrics)
            
            used_sigma = metrics.get('peak_sigma_used', peak_sigma)
            used_threshold = metrics.get('peak_threshold_used', self.default_peak_threshold)
            
            ax = axes[row_idx, col_idx]
            im = ax.imshow(output.squeeze().cpu().numpy(), cmap='jet')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.tick_params(axis='both', labelsize=16)
            
            # Process peaks if enabled
            confusion_metrics = None
            if calculate_peaks and ideal_peaks:
                output_peaks, confusion_metrics = self._detect_peaks_with_adaptive_threshold(
                    output, ideal_peaks, used_sigma, used_threshold, verbose=True)
                self._plot_peaks_on_axis(ax, output_peaks, ideal_peaks, confusion_metrics, 
                                        show_peak_classification)
            
            # Build and display metrics text
            metrics_text = self._build_metrics_text(
                metrics, calculate_psnr, calculate_ssim, calculate_xcorr,
                calculate_peaks, ideal_peaks, confusion_metrics, show_peak_classification)
            
            ax.text(0.02, 0.98, '\n'.join(metrics_text),
                    transform=ax.transAxes, 
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            if col_idx == 0:
                ax.set_ylabel(f"{row['iterations']} epochs", fontsize=20)
            
            if row_idx == 0:
                ax.set_title(self._format_model_label(row['loss_type'], row['use_unet']), fontsize=20)
        
        # Add legend for peak markers
        if calculate_peaks:
            legend_elements = self._create_peak_legend_elements(show_peak_classification)
            if len(model_configs['models']) > 2:
                ax_legend = fig.add_subplot(gs[0, 2:])
                ax_legend.axis('off')
                ax_legend.legend(handles=legend_elements, loc='center left', frameon=False, fontsize=36, markerscale=3)
        
        plt.tight_layout()
        return fig
    
    def plot_metrics_vs_iteration(self, model_configs: Dict,
                                 input_data: torch.Tensor,
                                 ideal_data: torch.Tensor,
                                 loss_type: str,
                                 base_path: str = "",
                                 figsize: Tuple[int, int] = (15, 10),
                                 calculate_psnr: bool = True,
                                 calculate_ssim: bool = True,
                                 calculate_peaks: bool = True,
                                 peak_sigma: float = None,
                                 compare_architectures: bool = True) -> plt.Figure:
        """
        Plot performance metrics vs iteration for a given loss function.
        Compares Unet and no_Unet architectures.
        """
        if peak_sigma is None:
            peak_sigma = self.default_peak_sigma
            
        # Ensure input data is on the correct device and dtype
        input_data = self.ensure_tensor_format(input_data)
        ideal_data = self.ensure_tensor_format(ideal_data)
            
        iterations = model_configs['iterations']
        metrics_data = {
            'Unet': {
                'psnr': [], 'ssim': [], 'avg_peak_dist': [], 'fwhm_diff': [],
                'matched_peaks': [], 'total_peaks': []
            },
            'no_Unet': {
                'psnr': [], 'ssim': [], 'avg_peak_dist': [], 'fwhm_diff': [],
                'matched_peaks': [], 'total_peaks': []
            }
        }
        
        ideal_peaks = None
        if calculate_peaks:
            # Normalize ideal data before peak detection
            # Apply additional smoothing to ideal image if configured
            ideal_data_norm = self.normalize_image(ideal_data, apply_smoothing=True, smoothing_sigma=self.ideal_peak_smoothing_sigma)
            ideal_peaks, ideal_fwhm = self.find_peaks_and_fwhm(ideal_data_norm.squeeze().cpu().numpy(), sigma=peak_sigma)
        
        architectures = ['Unet', 'no_Unet'] if compare_architectures else ['Unet']
        
        for arch in architectures:
            model_key = f"{loss_type}_{arch}"
            
            if model_key not in model_configs['models']:
                print(f"Warning: {model_key} not found in model configs")
                continue
            
            for iteration in iterations:
                model_path = Path(base_path) / model_configs['models'][model_key].format(iteration)
                
                if not model_path.exists():
                    print(f"Warning: Model not found: {model_path}")
                    for metric in ['psnr', 'ssim', 'avg_peak_dist', 'fwhm_diff', 'matched_peaks', 'total_peaks']:
                        metrics_data[arch][metric].append(np.nan)
                    continue
                
                use_unet = (arch == 'Unet')
                model = self.load_model(str(model_path), use_unet)
                
                with torch.no_grad():
                    output = model(input_data)
                
                metrics = self.calculate_metrics(output, ideal_data,
                                              calculate_psnr=calculate_psnr,
                                              calculate_ssim=calculate_ssim,
                                              calculate_xcorr=False,
                                              calculate_peaks=calculate_peaks,
                                              calculate_peak_sensitivity=False,
                                              peak_sigma=peak_sigma)
                
                metrics_data[arch]['psnr'].append(metrics.get('psnr', np.nan))
                metrics_data[arch]['ssim'].append(metrics.get('ssim', np.nan))
                metrics_data[arch]['avg_peak_dist'].append(metrics.get('avg_peak_dist', np.nan))
                metrics_data[arch]['fwhm_diff'].append(metrics.get('fwhm_diff', np.nan))
                
                if calculate_peaks and ideal_peaks:
                    used_sigma = metrics.get('peak_sigma_used', peak_sigma)
                    used_threshold = metrics.get('peak_threshold_used', self.default_peak_threshold)
                    
                    # Normalize output before peak detection
                    output_norm = self.normalize_image(output)
                    # Use output-specific percentile if using percentile threshold
                    output_percentile = self.output_percentile_threshold_value if self.use_percentile_threshold else None
                    output_peaks, _ = self.find_peaks_and_fwhm(output_norm.squeeze().cpu().numpy(), 
                                                               sigma=used_sigma, 
                                                               threshold=used_threshold,
                                                               percentile=output_percentile)
                    matched = 0
                    for ideal_peak in ideal_peaks:
                        min_dist = float('inf')
                        for output_peak in output_peaks:
                            dist = np.sqrt((ideal_peak[0]-output_peak[0])**2 + (ideal_peak[1]-output_peak[1])**2)
                            min_dist = min(min_dist, dist)
                        if min_dist < self.peak_distance_threshold:
                            matched += 1
                    metrics_data[arch]['matched_peaks'].append(matched)
                    metrics_data[arch]['total_peaks'].append(len(ideal_peaks))
                else:
                    metrics_data[arch]['matched_peaks'].append(np.nan)
                    metrics_data[arch]['total_peaks'].append(np.nan)
        
        # Create subplots
        num_plots = sum([calculate_psnr, calculate_ssim])
        if calculate_peaks and ideal_peaks:
            num_plots += 3
        
        fig, axes = plt.subplots(num_plots, 1, figsize=figsize, sharex=True)
        if num_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        legend_added = False
        
        # Plot PSNR
        if calculate_psnr:
            ax = axes[plot_idx]
            display_loss = 'NPCC' if loss_type == 'pearson' else loss_type
            for arch in architectures:
                ax.plot(iterations, metrics_data[arch]['psnr'], 'o-', label=arch, linewidth=2, markersize=8)
            ax.set_ylabel('PSNR (dB)', fontsize=18, fontweight='bold')
            ax.set_title(f'PSNR vs Epochs ({display_loss} Loss)', fontsize=20, fontweight='bold')
            if not legend_added:
                ax.legend(fontsize=20, loc='center right', framealpha=0.9)
                legend_added = True
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=16)
            plot_idx += 1
        
        # Plot SSIM
        if calculate_ssim:
            ax = axes[plot_idx]
            display_loss = 'NPCC' if loss_type == 'pearson' else loss_type
            for arch in architectures:
                ax.plot(iterations, metrics_data[arch]['ssim'], 'o-', label=arch, linewidth=2, markersize=8)
            ax.set_ylabel('SSIM', fontsize=18, fontweight='bold')
            ax.set_title(f'SSIM vs Epochs ({display_loss} Loss)', fontsize=20, fontweight='bold')
            if not legend_added:
                ax.legend(fontsize=20, loc='center right', framealpha=0.9)
                legend_added = True
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=16)
            plot_idx += 1
        
        # Plot peak metrics
        if calculate_peaks and ideal_peaks:
            # Average peak distance
            ax = axes[plot_idx]
            display_loss = 'NPCC' if loss_type == 'pearson' else loss_type
            for arch in architectures:
                ax.plot(iterations, metrics_data[arch]['avg_peak_dist'], 'o-', label=arch, linewidth=2, markersize=8)
            ax.set_ylabel('Average Peak Distance (pixels)', fontsize=18, fontweight='bold')
            ax.set_title(f'Peak Distance vs Epochs ({display_loss} Loss)', fontsize=20, fontweight='bold')
            if not legend_added:
                ax.legend(fontsize=20, loc='center right', framealpha=0.9)
                legend_added = True
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=16)
            plot_idx += 1
            
            # FWHM difference
            ax = axes[plot_idx]
            for arch in architectures:
                ax.plot(iterations, metrics_data[arch]['fwhm_diff'], 'o-', label=arch, linewidth=2, markersize=8)
            ax.set_ylabel('FWHM Difference (pixels)', fontsize=18, fontweight='bold')
            ax.set_title(f'FWHM Difference vs Epochs ({display_loss} Loss)', fontsize=20, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=16)
            plot_idx += 1
            
            # Matched peaks (as percentage)
            ax = axes[plot_idx]
            for arch in architectures:
                matched_ratio = [m/t*100 if not np.isnan(m) and t > 0 else np.nan 
                               for m, t in zip(metrics_data[arch]['matched_peaks'], 
                                             metrics_data[arch]['total_peaks'])]
                ax.plot(iterations, matched_ratio, 'o-', label=arch, linewidth=2, markersize=8)
            ax.set_ylabel('Matched Peaks (%)', fontsize=18, fontweight='bold')
            ax.set_title(f'Peak Detection Rate vs Epochs ({display_loss} Loss)', fontsize=20, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=16)
            ax.set_ylim([0, 105])
            plot_idx += 1
        
        axes[-1].set_xlabel('Epochs', fontsize=18, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_metrics_vs_iteration_compare_loss(self, model_configs: Dict,
                                              input_data: torch.Tensor,
                                              ideal_data: torch.Tensor,
                                              loss_types: List[str],
                                              base_path: str = "",
                                              figsize: Tuple[int, int] = (15, 10),
                                              calculate_psnr: bool = True,
                                              calculate_ssim: bool = True,
                                              calculate_peaks: bool = True,
                                              calculate_peak_sensitivity: bool = False,
                                              peak_sigma: float = None,
                                              compare_architectures: bool = False) -> plt.Figure:
        """
        Plot performance metrics vs iteration comparing different loss functions.
        """
        if peak_sigma is None:
            peak_sigma = self.default_peak_sigma
            
        # Ensure input data is on the correct device and dtype
        input_data = self.ensure_tensor_format(input_data)
        ideal_data = self.ensure_tensor_format(ideal_data)
            
        iterations = model_configs['iterations']
        metrics_data = {}
        
        architectures = ['Unet', 'no_Unet'] if compare_architectures else ['Unet']
        
        for loss_type in loss_types:
            for arch in architectures:
                key = f"{loss_type}_{arch}"
                metrics_data[key] = {
                    'psnr': [], 'ssim': [], 'avg_peak_dist': [], 'fwhm_diff': [],
                    'matched_peaks': [], 'total_peaks': []
                }
        
        ideal_peaks = None
        if calculate_peaks:
            # Normalize ideal data before peak detection
            # Apply additional smoothing to ideal image if configured
            ideal_data_norm = self.normalize_image(ideal_data, apply_smoothing=True, smoothing_sigma=self.ideal_peak_smoothing_sigma)
            ideal_peaks, ideal_fwhm = self.find_peaks_and_fwhm(ideal_data_norm.squeeze().cpu().numpy(), sigma=peak_sigma)
        
        for loss_type in loss_types:
            for arch in architectures:
                model_key = f"{loss_type}_{arch}"
                data_key = f"{loss_type}_{arch}"
                
                if model_key not in model_configs['models']:
                    print(f"Warning: {model_key} not found in model configs")
                    continue
                
                use_unet = (arch == 'Unet')
                
                for iteration in iterations:
                    model_path = Path(base_path) / model_configs['models'][model_key].format(iteration)
                    
                    if not model_path.exists():
                        print(f"Warning: Model not found: {model_path}")
                        for metric in ['psnr', 'ssim', 'avg_peak_dist', 'fwhm_diff', 'matched_peaks', 'total_peaks']:
                            metrics_data[data_key][metric].append(np.nan)
                        continue
                    
                    model = self.load_model(str(model_path), use_unet)
                    
                    with torch.no_grad():
                        output = model(input_data)
                    
                    metrics = self.calculate_metrics(output, ideal_data,
                                                  calculate_psnr=calculate_psnr,
                                                  calculate_ssim=calculate_ssim,
                                                  calculate_xcorr=False,
                                                  calculate_peaks=calculate_peaks,
                                                  calculate_peak_sensitivity=calculate_peak_sensitivity,
                                                  peak_sigma=peak_sigma)
                    
                    metrics_data[data_key]['psnr'].append(metrics.get('psnr', np.nan))
                    metrics_data[data_key]['ssim'].append(metrics.get('ssim', np.nan))
                    metrics_data[data_key]['avg_peak_dist'].append(metrics.get('avg_peak_dist', np.nan))
                    metrics_data[data_key]['fwhm_diff'].append(metrics.get('fwhm_diff', np.nan))
                    
                    if calculate_peaks and ideal_peaks:
                        used_sigma = metrics.get('peak_sigma_used', peak_sigma)
                        used_threshold = metrics.get('peak_threshold_used', self.default_peak_threshold)
                        
                        # Use output-specific percentile if using percentile threshold
                        output_percentile = self.output_percentile_threshold_value if self.use_percentile_threshold else None
                        output_peaks, _ = self.find_peaks_and_fwhm(output.squeeze().cpu().numpy(), 
                                                                   sigma=used_sigma, 
                                                                   threshold=used_threshold,
                                                                   percentile=output_percentile)
                        matched = 0
                        for ideal_peak in ideal_peaks:
                            min_dist = float('inf')
                            for output_peak in output_peaks:
                                dist = np.sqrt((ideal_peak[0]-output_peak[0])**2 + (ideal_peak[1]-output_peak[1])**2)
                                min_dist = min(min_dist, dist)
                            if min_dist < self.peak_distance_threshold:
                                matched += 1
                        metrics_data[data_key]['matched_peaks'].append(matched)
                        metrics_data[data_key]['total_peaks'].append(len(ideal_peaks))
                    else:
                        metrics_data[data_key]['matched_peaks'].append(np.nan)
                        metrics_data[data_key]['total_peaks'].append(np.nan)
        
        # Create subplots
        num_plots = sum([calculate_psnr, calculate_ssim])
        if calculate_peaks and ideal_peaks:
            num_plots += 3
        
        fig, axes = plt.subplots(num_plots, 1, figsize=figsize, sharex=True)
        if num_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        legend_added = False
        
        # Plot PSNR
        if calculate_psnr:
            ax = axes[plot_idx]
            for loss_type in loss_types:
                data_key_unet = f"{loss_type}_Unet"
                display_name = 'NPCC' if loss_type == 'pearson' else loss_type
                line, = ax.plot(iterations, metrics_data[data_key_unet]['psnr'], 'o-', 
                              label=f'{display_name} (Unet)', linewidth=2, markersize=8)
                
                if compare_architectures:
                    data_key_no_unet = f"{loss_type}_no_Unet"
                    ax.plot(iterations, metrics_data[data_key_no_unet]['psnr'], 's--', 
                           label=f'{display_name} (no Unet)', linewidth=2, markersize=7, 
                           color=line.get_color())
            
            ax.set_ylabel('PSNR (dB)', fontsize=18, fontweight='bold')
            ax.set_title(f'PSNR vs Epochs', fontsize=20, fontweight='bold')
            if not legend_added:
                ax.legend(fontsize=20, loc='center right', framealpha=0.9)
                legend_added = True
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=16)
            plot_idx += 1
        
        # Plot SSIM
        if calculate_ssim:
            ax = axes[plot_idx]
            for loss_type in loss_types:
                data_key_unet = f"{loss_type}_Unet"
                display_name = 'NPCC' if loss_type == 'pearson' else loss_type
                line, = ax.plot(iterations, metrics_data[data_key_unet]['ssim'], 'o-', 
                              label=f'{display_name} (Unet)', linewidth=2, markersize=8)
                
                if compare_architectures:
                    data_key_no_unet = f"{loss_type}_no_Unet"
                    ax.plot(iterations, metrics_data[data_key_no_unet]['ssim'], 's--', 
                           label=f'{display_name} (no Unet)', linewidth=2, markersize=7, 
                           color=line.get_color())
            
            ax.set_ylabel('SSIM', fontsize=18, fontweight='bold')
            ax.set_title(f'SSIM vs Epochs', fontsize=20, fontweight='bold')
            if not legend_added:
                ax.legend(fontsize=20, loc='center right', framealpha=0.9)
                legend_added = True
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=16)
            plot_idx += 1
        
        # Plot peak metrics
        if calculate_peaks and ideal_peaks:
            # Average peak distance
            ax = axes[plot_idx]
            for loss_type in loss_types:
                data_key_unet = f"{loss_type}_Unet"
                display_name = 'NPCC' if loss_type == 'pearson' else loss_type
                line, = ax.plot(iterations, metrics_data[data_key_unet]['avg_peak_dist'], 'o-', 
                              label=f'{display_name} (Unet)', linewidth=2, markersize=8)
                
                if compare_architectures:
                    data_key_no_unet = f"{loss_type}_no_Unet"
                    ax.plot(iterations, metrics_data[data_key_no_unet]['avg_peak_dist'], 's--', 
                           label=f'{display_name} (no Unet)', linewidth=2, markersize=7, 
                           color=line.get_color())
            
            ax.set_ylabel('Average Peak Distance (pixels)', fontsize=18, fontweight='bold')
            ax.set_title(f'Peak Distance vs Epochs', fontsize=20, fontweight='bold')
            if not legend_added:
                ax.legend(fontsize=20, loc='center right', framealpha=0.9)
                legend_added = True
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=16)
            plot_idx += 1
            
            # FWHM difference
            ax = axes[plot_idx]
            for loss_type in loss_types:
                data_key_unet = f"{loss_type}_Unet"
                display_name = 'NPCC' if loss_type == 'pearson' else loss_type
                line, = ax.plot(iterations, metrics_data[data_key_unet]['fwhm_diff'], 'o-', 
                              label=f'{display_name} (Unet)', linewidth=2, markersize=8)
                
                if compare_architectures:
                    data_key_no_unet = f"{loss_type}_no_Unet"
                    ax.plot(iterations, metrics_data[data_key_no_unet]['fwhm_diff'], 's--', 
                           label=f'{display_name} (no Unet)', linewidth=2, markersize=7, 
                           color=line.get_color())
            
            ax.set_ylabel('FWHM Difference (pixels)', fontsize=18, fontweight='bold')
            ax.set_title(f'FWHM Difference vs Epochs', fontsize=20, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=16)
            plot_idx += 1
            
            # Matched peaks (as percentage)
            ax = axes[plot_idx]
            for loss_type in loss_types:
                data_key_unet = f"{loss_type}_Unet"
                display_name = 'NPCC' if loss_type == 'pearson' else loss_type
                matched_ratio_unet = [m/t*100 if not np.isnan(m) and t > 0 else np.nan 
                                     for m, t in zip(metrics_data[data_key_unet]['matched_peaks'], 
                                                   metrics_data[data_key_unet]['total_peaks'])]
                line, = ax.plot(iterations, matched_ratio_unet, 'o-', 
                              label=f'{display_name} (Unet)', linewidth=2, markersize=8)
                
                if compare_architectures:
                    data_key_no_unet = f"{loss_type}_no_Unet"
                    matched_ratio_no_unet = [m/t*100 if not np.isnan(m) and t > 0 else np.nan 
                                            for m, t in zip(metrics_data[data_key_no_unet]['matched_peaks'], 
                                                          metrics_data[data_key_no_unet]['total_peaks'])]
                    ax.plot(iterations, matched_ratio_no_unet, 's--', 
                           label=f'{display_name} (no Unet)', linewidth=2, markersize=7, 
                           color=line.get_color())
            
            ax.set_ylabel('Matched Peaks (%)', fontsize=18, fontweight='bold')
            ax.set_title(f'Peak Detection Rate vs Epochs', fontsize=20, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=16)
            ax.set_ylim([0, 105])
            plot_idx += 1
        
        axes[-1].set_xlabel('Epochs', fontsize=18, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def calculate_cumulative_stats(self, model_configs: Dict,
                                 indices_list: List[Tuple[int, int, int]],
                                 base_path: str = "",
                                 mask_path: str = '/home/beams/PTYCHOSAXS/deconvolutionNN/data/mask/mask_ZCB_9_3D.npy',
                                 data_path: str = '/net/micdata/data2/12IDC/ptychosaxs/data/diff_sim/lattice_ls400_gs1024_lsp6_r3.0_typeSC',
                                 calculate_psnr: bool = True,
                                 calculate_ssim: bool = True,
                                 calculate_xcorr: bool = False,
                                 calculate_peaks: bool = True,
                                 calculate_peak_sensitivity: bool = False,
                                 peak_sigma: float = None,
                                 central_only: bool = True,
                                 show_peak_classification: bool = False) -> Dict[str, Dict[str, float]]:
        """
        Calculate cumulative statistics across multiple input patterns.
        """
        if peak_sigma is None:
            peak_sigma = self.default_peak_sigma
            
        mask = np.load(mask_path)
        stats = {}
        
        model_info = self.get_model_paths_from_config(model_configs, base_path)
        if not model_info:
            raise ValueError("No valid model paths found in the configuration")
        
        df = pd.DataFrame(model_info)
        df = df.sort_values(['iterations', 'loss_type'])
        
        for _, row in df.iterrows():
            model_key = f"{row['loss_type']}_{'Unet' if row['use_unet'] else 'no_Unet'}_{row['iterations']}"
            stats[model_key] = {
                'psnr_sum': 0.0, 'ssim_sum': 0.0, 'xcorr_sum': 0.0,
                'peak_dist_sum': 0.0, 'fwhm_diff_sum': 0.0,
                'total_peaks_ideal': 0, 'total_peaks_matched': 0, 'pattern_count': 0,
                'optimal_sigma_sum': 0.0, 'optimal_threshold_sum': 0.0,
                'max_f1_score_sum': 0.0, 'peak_position_stability_sum': 0.0,
                'peak_count_stability_sum': 0.0, 'parameter_sensitivity_sum': 0.0,
                'false_positive_rate_sum': 0.0, 'false_negative_rate_sum': 0.0,
                'peak_tp_sum': 0, 'peak_fp_sum': 0, 'peak_fn_sum': 0,
                'peak_precision_sum': 0.0, 'peak_recall_sum': 0.0, 'peak_f1_sum': 0.0
            }
        
        for hr, kr, lr in tqdm(indices_list):
            #pattern_nums = [5] if central_only else range(1, 10)
            pattern_nums = [0] if central_only else range(1, 16)
            
            for num in pattern_nums:
                pattern_file = f'output_hanning_conv_{hr}_{kr}_{lr}_0000{num}.npz'
                try:
                    data = np.load(f'{data_path}/{pattern_file}')
                except FileNotFoundError:
                    print(f"Warning: File not found: {pattern_file}")
                    continue
                
                dp_pp, _, _ = ptNN_U.preprocess_ZCB_9(data['convDP'], mask)
                dp_pp_IDEAL, _, _ = ptNN_U.preprocess_ZCB_9(data['pinholeDP_extra_conv'], 
                                                             mask=np.ones(dp_pp[0][0].shape))
                
                dp_pp = dp_pp.to(device=self.device, dtype=torch.float)
                dp_pp_IDEAL = dp_pp_IDEAL.to(device=self.device, dtype=torch.float)
                
                for _, row in df.iterrows():
                    model_key = f"{row['loss_type']}_{'Unet' if row['use_unet'] else 'no_Unet'}_{row['iterations']}"
                    
                    model = self.load_model(row['path'], row['use_unet'])
                    with torch.no_grad():
                        output = model(dp_pp)
                    
                    metrics = self.calculate_metrics(output, dp_pp_IDEAL,
                                                 calculate_psnr=calculate_psnr,
                                                 calculate_ssim=calculate_ssim,
                                                 calculate_xcorr=calculate_xcorr,
                                                 calculate_peaks=calculate_peaks,
                                                 calculate_peak_sensitivity=calculate_peak_sensitivity,
                                                 peak_sigma=peak_sigma)
                    
                    if calculate_psnr and 'psnr' in metrics:
                        stats[model_key]['psnr_sum'] += metrics['psnr']
                    if calculate_ssim and 'ssim' in metrics:
                        stats[model_key]['ssim_sum'] += metrics['ssim']
                    if calculate_xcorr and 'xcorr' in metrics:
                        stats[model_key]['xcorr_sum'] += metrics['xcorr']
                    if calculate_peaks:
                        if 'avg_peak_dist' in metrics:
                            stats[model_key]['peak_dist_sum'] += metrics['avg_peak_dist']
                        if 'fwhm_diff' in metrics:
                            stats[model_key]['fwhm_diff_sum'] += metrics['fwhm_diff']
                        if 'num_peaks2' in metrics:
                            stats[model_key]['total_peaks_ideal'] += metrics['num_peaks2']
                            
                            used_sigma = metrics.get('peak_sigma_used', peak_sigma)
                            used_threshold = metrics.get('peak_threshold_used', self.default_peak_threshold)
                            
                            # Normalize images before peak detection
                            # Apply additional smoothing to ideal image if configured
                            ideal_norm = self.normalize_image(dp_pp_IDEAL, apply_smoothing=True, smoothing_sigma=self.ideal_peak_smoothing_sigma)
                            output_norm = self.normalize_image(output)
                            ideal_peaks, _ = self.find_peaks_and_fwhm(ideal_norm.squeeze().cpu().numpy(), 
                                                                      sigma=used_sigma, 
                                                                      threshold=used_threshold)
                            # Use output-specific percentile if using percentile threshold
                            output_percentile = self.output_percentile_threshold_value if self.use_percentile_threshold else None
                            output_peaks, _ = self.find_peaks_and_fwhm(output_norm.squeeze().cpu().numpy(), 
                                                                       sigma=used_sigma, 
                                                                       threshold=used_threshold,
                                                                       percentile=output_percentile)
                            
                            # Always use confusion matrix for consistent one-to-one matching
                            confusion_metrics = self.calculate_peak_confusion_matrix(output_peaks, ideal_peaks)
                            matched = confusion_metrics['tp']
                            stats[model_key]['total_peaks_matched'] += matched
                            
                            # Always track TP/FP/FN metrics (we calculate them anyway)
                            stats[model_key]['peak_tp_sum'] += confusion_metrics['tp']
                            stats[model_key]['peak_fp_sum'] += confusion_metrics['fp']
                            stats[model_key]['peak_fn_sum'] += confusion_metrics['fn']
                            stats[model_key]['peak_precision_sum'] += confusion_metrics['precision']
                            stats[model_key]['peak_recall_sum'] += confusion_metrics['recall']
                            stats[model_key]['peak_f1_sum'] += confusion_metrics['f1_score']
                    
                    if calculate_peak_sensitivity:
                        stats[model_key]['optimal_sigma_sum'] += metrics['optimal_sigma']
                        stats[model_key]['optimal_threshold_sum'] += metrics['optimal_threshold']
                        stats[model_key]['max_f1_score_sum'] += metrics['max_f1_score']
                        stats[model_key]['peak_position_stability_sum'] += metrics['peak_position_stability']
                        stats[model_key]['peak_count_stability_sum'] += metrics['peak_count_stability']
                        stats[model_key]['parameter_sensitivity_sum'] += metrics['parameter_sensitivity']
                        stats[model_key]['false_positive_rate_sum'] += metrics['false_positive_rate']
                        stats[model_key]['false_negative_rate_sum'] += metrics['false_negative_rate']
                    
                    stats[model_key]['pattern_count'] += 1
        
        # Calculate averages and create final statistics
        final_stats = {}
        for model_key, model_stats in stats.items():
            count = model_stats['pattern_count']
            if count > 0:
                final_stats[model_key] = {
                    'avg_psnr': model_stats['psnr_sum'] / count,
                    'avg_ssim': model_stats['ssim_sum'] / count,
                    'avg_xcorr': model_stats['xcorr_sum'] / count,
                    'avg_peak_dist': model_stats['peak_dist_sum'] / count,
                    'avg_fwhm_diff': model_stats['fwhm_diff_sum'] / count,
                    'peak_detection_rate': model_stats['total_peaks_matched'] / model_stats['total_peaks_ideal'] if model_stats['total_peaks_ideal'] > 0 else 0,
                    'total_peaks_matched': model_stats['total_peaks_matched'],
                    'total_peaks_ideal': model_stats['total_peaks_ideal'],
                    'patterns_processed': count,
                    # Always calculate precision/recall/F1 (more robust metrics)
                    'total_peak_tp': model_stats['peak_tp_sum'],
                    'total_peak_fp': model_stats['peak_fp_sum'],
                    'total_peak_fn': model_stats['peak_fn_sum'],
                    'avg_peak_precision': model_stats['peak_precision_sum'] / count if count > 0 else 0,
                    'avg_peak_recall': model_stats['peak_recall_sum'] / count if count > 0 else 0,
                    'avg_peak_f1': model_stats['peak_f1_sum'] / count if count > 0 else 0
                }
                
                if calculate_peak_sensitivity:
                    final_stats[model_key].update({
                        'avg_optimal_sigma': model_stats['optimal_sigma_sum'] / count,
                        'avg_optimal_threshold': model_stats['optimal_threshold_sum'] / count,
                        'avg_max_f1_score': model_stats['max_f1_score_sum'] / count,
                        'avg_peak_position_stability': model_stats['peak_position_stability_sum'] / count,
                        'avg_peak_count_stability': model_stats['peak_count_stability_sum'] / count,
                        'avg_parameter_sensitivity': model_stats['parameter_sensitivity_sum'] / count,
                        'avg_false_positive_rate': model_stats['false_positive_rate_sum'] / count,
                        'avg_false_negative_rate': model_stats['false_negative_rate_sum'] / count
                    })
        
        return final_stats
    
    def calculate_cumulative_stats_from_h5(self, model_configs: Dict,
                                          h5_file_path: str,
                                          base_path: str = "",
                                          mask_path: str = '/home/beams/PTYCHOSAXS/deconvolutionNN/data/mask/mask_ZCB_9_3D.npy',
                                          calculate_psnr: bool = True,
                                          calculate_ssim: bool = True,
                                          calculate_xcorr: bool = False,
                                          calculate_peaks: bool = True,
                                          calculate_peak_sensitivity: bool = False,
                                          peak_sigma: float = None,
                                          angle_indices: Optional[List[int]] = None,
                                          show_peak_classification: bool = False) -> Dict[str, Dict[str, float]]:
        """
        Calculate cumulative statistics from tomographic h5 file data.
        
        Args:
            model_configs: Dictionary of model configurations
            h5_file_path: Path to h5 file containing convDP and pinholeDP_raw_FFT
            base_path: Base path for model files
            mask_path: Path to mask file
            calculate_psnr: Whether to calculate PSNR
            calculate_ssim: Whether to calculate SSIM
            calculate_xcorr: Whether to calculate cross-correlation
            calculate_peaks: Whether to calculate peak metrics
            calculate_peak_sensitivity: Whether to calculate peak sensitivity metrics
            peak_sigma: Sigma value for peak detection
            angle_indices: Optional list of rotation angle indices to process (None = all)
        
        Returns:
            Dictionary of statistics for each model
        """
        if peak_sigma is None:
            peak_sigma = self.default_peak_sigma
            
        mask = np.load(mask_path)
        stats = {}
        
        model_info = self.get_model_paths_from_config(model_configs, base_path)
        if not model_info:
            raise ValueError("No valid model paths found in the configuration")
        
        df = pd.DataFrame(model_info)
        df = df.sort_values(['iterations', 'loss_type'])
        
        # Initialize statistics for each model
        for _, row in df.iterrows():
            model_key = f"{row['loss_type']}_{'Unet' if row['use_unet'] else 'no_Unet'}_{row['iterations']}"
            stats[model_key] = {
                'psnr_sum': 0.0, 'ssim_sum': 0.0, 'xcorr_sum': 0.0,
                'peak_dist_sum': 0.0, 'fwhm_diff_sum': 0.0,
                'total_peaks_ideal': 0, 'total_peaks_matched': 0, 'pattern_count': 0,
                'optimal_sigma_sum': 0.0, 'optimal_threshold_sum': 0.0,
                'max_f1_score_sum': 0.0, 'peak_position_stability_sum': 0.0,
                'peak_count_stability_sum': 0.0, 'parameter_sensitivity_sum': 0.0,
                'false_positive_rate_sum': 0.0, 'false_negative_rate_sum': 0.0,
                'peak_tp_sum': 0, 'peak_fp_sum': 0, 'peak_fn_sum': 0,
                'peak_precision_sum': 0.0, 'peak_recall_sum': 0.0, 'peak_f1_sum': 0.0
            }
        
        # Load h5 file
        try:
            with h5py.File(h5_file_path, 'r') as h5file:
                if 'convDP' not in h5file:
                    raise ValueError(f"Key 'convDP' not found in h5 file: {h5_file_path}")
                if 'pinholeDP_raw_FFT' not in h5file:
                    raise ValueError(f"Key 'pinholeDP_raw_FFT' not found in h5 file: {h5_file_path}")
                
                convDP = h5file['convDP'][:]
                pinholeDP_raw_FFT = h5file['pinholeDP_raw_FFT'][:]
                rotation_angles = h5file['rotation_angles'][:] if 'rotation_angles' in h5file else None
                
                num_patterns = convDP.shape[0]
                num_angles = rotation_angles.shape[0] if rotation_angles is not None else None
                
                # Determine which patterns to process
                if angle_indices is None:
                    # Process all patterns
                    pattern_indices = range(num_patterns)
                else:
                    # Process patterns for specific angles
                    # Assuming patterns are organized by angle (e.g., 16 patterns per angle)
                    patterns_per_angle = num_patterns // num_angles if num_angles else num_patterns
                    pattern_indices = []
                    for angle_idx in angle_indices:
                        start_idx = angle_idx * patterns_per_angle
                        end_idx = start_idx + patterns_per_angle
                        pattern_indices.extend(range(start_idx, min(end_idx, num_patterns)))
                
                # Process each pattern
                for pattern_idx in tqdm(pattern_indices, desc="Processing patterns"):
                    try:
                        # Extract single pattern
                        convDP_pattern = convDP[pattern_idx]
                        pinholeDP_pattern = pinholeDP_raw_FFT[pattern_idx]
                        
                        # Preprocess data
                        dp_pp, _, _ = ptNN_U.preprocess_ZCB_9(convDP_pattern, mask)
                        dp_pp_IDEAL, _, _ = ptNN_U.preprocess_ZCB_9(pinholeDP_pattern, 
                                                                     mask=np.ones(dp_pp[0][0].shape))
                        
                        dp_pp = dp_pp.to(device=self.device, dtype=torch.float)
                        dp_pp_IDEAL = dp_pp_IDEAL.to(device=self.device, dtype=torch.float)
                        
                        # Process with each model
                        for _, row in df.iterrows():
                            model_key = f"{row['loss_type']}_{'Unet' if row['use_unet'] else 'no_Unet'}_{row['iterations']}"
                            
                            model = self.load_model(row['path'], row['use_unet'])
                            with torch.no_grad():
                                output = model(dp_pp)
                            
                            metrics = self.calculate_metrics(output, dp_pp_IDEAL,
                                                         calculate_psnr=calculate_psnr,
                                                         calculate_ssim=calculate_ssim,
                                                         calculate_xcorr=calculate_xcorr,
                                                         calculate_peaks=calculate_peaks,
                                                         calculate_peak_sensitivity=calculate_peak_sensitivity,
                                                         peak_sigma=peak_sigma)
                            #print('Metrics: ', metrics)
                            
                            if calculate_psnr and 'psnr' in metrics:
                                stats[model_key]['psnr_sum'] += metrics['psnr']
                            if calculate_ssim and 'ssim' in metrics:
                                stats[model_key]['ssim_sum'] += metrics['ssim']
                            if calculate_xcorr and 'xcorr' in metrics:
                                stats[model_key]['xcorr_sum'] += metrics['xcorr']
                            if calculate_peaks:
                                if 'avg_peak_dist' in metrics:
                                    stats[model_key]['peak_dist_sum'] += metrics['avg_peak_dist']
                                if 'fwhm_diff' in metrics:
                                    stats[model_key]['fwhm_diff_sum'] += metrics['fwhm_diff']
                                if 'num_peaks2' in metrics:
                                    stats[model_key]['total_peaks_ideal'] += metrics['num_peaks2']
                                    
                                    used_sigma = metrics.get('peak_sigma_used', peak_sigma)
                                    used_threshold = metrics.get('peak_threshold_used', self.default_peak_threshold)
                                    
                                    # Normalize ideal image before peak detection
                                    ideal_norm = self.normalize_image(dp_pp_IDEAL, apply_smoothing=True, smoothing_sigma=self.ideal_peak_smoothing_sigma)
                                    ideal_percentile = self.percentile_threshold_value if self.use_percentile_threshold else None
                                    ideal_peaks, _ = self.find_peaks_and_fwhm(ideal_norm.squeeze().cpu().numpy(), 
                                                                              sigma=used_sigma, 
                                                                              threshold=used_threshold,
                                                                              percentile=ideal_percentile)
                                    
                                    # Use helper method for output peak detection with adaptive thresholding
                                    output_peaks, confusion_metrics = self._detect_peaks_with_adaptive_threshold(
                                        output, ideal_peaks, used_sigma, used_threshold, verbose=False)
                                    
                                    matched = confusion_metrics['tp']
                                    stats[model_key]['total_peaks_matched'] += matched
                                    
                                    if show_peak_classification:
                                        # Track TP/FP/FN metrics when flag is enabled
                                        stats[model_key]['peak_tp_sum'] += confusion_metrics['tp']
                                        stats[model_key]['peak_fp_sum'] += confusion_metrics['fp']
                                        stats[model_key]['peak_fn_sum'] += confusion_metrics['fn']
                                        stats[model_key]['peak_precision_sum'] += confusion_metrics['precision']
                                        stats[model_key]['peak_recall_sum'] += confusion_metrics['recall']
                                        stats[model_key]['peak_f1_sum'] += confusion_metrics['f1_score']
                            stats[model_key]['pattern_count'] += 1
                    
                    except Exception as e:
                        print(f"Warning: Error processing pattern {pattern_idx}: {e}")
                        continue
        except Exception as e:
            print(f"Error loading h5 file {h5_file_path}: {e}")
            raise
        
        # Calculate averages and create final statistics
        final_stats = {}
        for model_key, model_stats in stats.items():
            count = model_stats['pattern_count']
            if count > 0:
                final_stats[model_key] = {
                    'avg_psnr': model_stats['psnr_sum'] / count,
                    'avg_ssim': model_stats['ssim_sum'] / count,
                    'avg_xcorr': model_stats['xcorr_sum'] / count,
                    'avg_peak_dist': model_stats['peak_dist_sum'] / count,
                    'avg_fwhm_diff': model_stats['fwhm_diff_sum'] / count,
                    'peak_detection_rate': model_stats['total_peaks_matched'] / model_stats['total_peaks_ideal'] if model_stats['total_peaks_ideal'] > 0 else 0,
                    'total_peaks_matched': model_stats['total_peaks_matched'],
                    'total_peaks_ideal': model_stats['total_peaks_ideal'],
                    'patterns_processed': count,
                    # Always calculate precision/recall/F1 (more robust metrics)
                    'total_peak_tp': model_stats['peak_tp_sum'],
                    'total_peak_fp': model_stats['peak_fp_sum'],
                    'total_peak_fn': model_stats['peak_fn_sum'],
                    'avg_peak_precision': model_stats['peak_precision_sum'] / count if count > 0 else 0,
                    'avg_peak_recall': model_stats['peak_recall_sum'] / count if count > 0 else 0,
                    'avg_peak_f1': model_stats['peak_f1_sum'] / count if count > 0 else 0
                }
                
                if calculate_peak_sensitivity:
                    final_stats[model_key].update({
                        'avg_optimal_sigma': model_stats['optimal_sigma_sum'] / count,
                        'avg_optimal_threshold': model_stats['optimal_threshold_sum'] / count,
                        'avg_max_f1_score': model_stats['max_f1_score_sum'] / count,
                        'avg_peak_position_stability': model_stats['peak_position_stability_sum'] / count,
                        'avg_peak_count_stability': model_stats['peak_count_stability_sum'] / count,
                        'avg_parameter_sensitivity': model_stats['parameter_sensitivity_sum'] / count,
                        'avg_false_positive_rate': model_stats['false_positive_rate_sum'] / count,
                        'avg_false_negative_rate': model_stats['false_negative_rate_sum'] / count
                    })
        
        return final_stats
    
    def calculate_cumulative_stats_from_h5_summed_by_angle(self, model_configs: Dict,
                                                          h5_file_path: str,
                                                          base_path: str = "",
                                                          mask_path: str = '/home/beams/PTYCHOSAXS/deconvolutionNN/data/mask/mask_ZCB_9_3D.npy',
                                                          patterns_per_angle: int = 16,
                                                          calculate_psnr: bool = True,
                                                          calculate_ssim: bool = True,
                                                          calculate_xcorr: bool = False,
                                                          calculate_peaks: bool = True,
                                                          calculate_peak_sensitivity: bool = False,
                                                          peak_sigma: float = None,
                                                          angle_indices: Optional[List[int]] = None,
                                                          show_peak_classification: bool = False) -> Dict[str, Dict[str, float]]:
        """
        Calculate cumulative statistics from tomographic h5 file data by summing patterns per angle.
        
        This function processes patterns in groups by projection angle:
        1. Deconvolves each pattern individually (16 patterns per angle)
        2. Sums the deconvolved outputs for each angle
        3. Sums the ideal patterns for each angle
        4. Performs peak analysis on the summed patterns (180 summed patterns for 180 angles)
        
        Args:
            model_configs: Dictionary of model configurations
            h5_file_path: Path to h5 file containing convDP and pinholeDP_raw_FFT
            base_path: Base path for model files
            mask_path: Path to mask file
            patterns_per_angle: Number of patterns per projection angle (default: 16)
            calculate_psnr: Whether to calculate PSNR
            calculate_ssim: Whether to calculate SSIM
            calculate_xcorr: Whether to calculate cross-correlation
            calculate_peaks: Whether to calculate peak metrics
            calculate_peak_sensitivity: Whether to calculate peak sensitivity metrics
            peak_sigma: Sigma value for peak detection
            angle_indices: Optional list of rotation angle indices to process (None = all)
            show_peak_classification: Whether to track detailed peak classification metrics
        
        Returns:
            Dictionary of statistics for each model
        """
        if peak_sigma is None:
            peak_sigma = self.default_peak_sigma
            
        mask = np.load(mask_path)
        stats = {}
        
        model_info = self.get_model_paths_from_config(model_configs, base_path)
        if not model_info:
            raise ValueError("No valid model paths found in the configuration")
        
        df = pd.DataFrame(model_info)
        df = df.sort_values(['iterations', 'loss_type'])
        
        # Initialize statistics for each model
        for _, row in df.iterrows():
            model_key = f"{row['loss_type']}_{'Unet' if row['use_unet'] else 'no_Unet'}_{row['iterations']}"
            stats[model_key] = {
                'psnr_sum': 0.0, 'ssim_sum': 0.0, 'xcorr_sum': 0.0,
                'peak_dist_sum': 0.0, 'fwhm_diff_sum': 0.0,
                'total_peaks_ideal': 0, 'total_peaks_matched': 0, 'angle_count': 0,
                'optimal_sigma_sum': 0.0, 'optimal_threshold_sum': 0.0,
                'max_f1_score_sum': 0.0, 'peak_position_stability_sum': 0.0,
                'peak_count_stability_sum': 0.0, 'parameter_sensitivity_sum': 0.0,
                'false_positive_rate_sum': 0.0, 'false_negative_rate_sum': 0.0,
                'peak_tp_sum': 0, 'peak_fp_sum': 0, 'peak_fn_sum': 0,
                'peak_precision_sum': 0.0, 'peak_recall_sum': 0.0, 'peak_f1_sum': 0.0
            }
        
        # Load h5 file
        try:
            with h5py.File(h5_file_path, 'r') as h5file:
                if 'convDP' not in h5file:
                    raise ValueError(f"Key 'convDP' not found in h5 file: {h5_file_path}")
                if 'pinholeDP_raw_FFT' not in h5file:
                    raise ValueError(f"Key 'pinholeDP_raw_FFT' not found in h5 file: {h5_file_path}")
                
                convDP = h5file['convDP'][:]
                pinholeDP_raw_FFT = h5file['pinholeDP_raw_FFT'][:]
                rotation_angles = h5file['rotation_angles'][:] if 'rotation_angles' in h5file else None
                
                num_patterns = convDP.shape[0]
                num_angles = num_patterns // patterns_per_angle
                
                if num_angles * patterns_per_angle != num_patterns:
                    print(f"Warning: Total patterns ({num_patterns}) is not divisible by patterns_per_angle ({patterns_per_angle}). "
                          f"Using {num_angles} angles with {num_angles * patterns_per_angle} patterns.")
                
                # Determine which angles to process
                if angle_indices is None:
                    angle_indices = range(num_angles)
                else:
                    # Validate angle indices
                    angle_indices = [idx for idx in angle_indices if 0 <= idx < num_angles]
                
                # Process each angle
                for angle_idx in tqdm(angle_indices, desc="Processing angles"):
                    try:
                        # Get pattern indices for this angle
                        start_idx = angle_idx * patterns_per_angle
                        end_idx = start_idx + patterns_per_angle
                        
                        # Initialize summed arrays for this angle
                        # We'll sum them after processing each pattern individually
                        summed_output = None
                        summed_ideal = None
                        summed_convoluted = None
                        
                        # Process each pattern in this angle individually
                        for pattern_offset in range(patterns_per_angle):
                            pattern_idx = start_idx + pattern_offset
                            if pattern_idx >= num_patterns:
                                break
                            
                            # Extract single pattern
                            convDP_pattern = convDP[pattern_idx]
                            pinholeDP_pattern = pinholeDP_raw_FFT[pattern_idx]
                            
                            # Preprocess data
                            dp_pp, _, _ = ptNN_U.preprocess_ZCB_9(convDP_pattern, mask)
                            dp_pp_IDEAL, _, _ = ptNN_U.preprocess_ZCB_9(pinholeDP_pattern, 
                                                                         mask=np.ones(dp_pp[0][0].shape))
                            
                            dp_pp = dp_pp.to(device=self.device, dtype=torch.float)
                            dp_pp_IDEAL = dp_pp_IDEAL.to(device=self.device, dtype=torch.float)
                            
                            # Initialize summed arrays on first pattern
                            if summed_ideal is None:
                                summed_ideal = torch.zeros_like(dp_pp_IDEAL)
                                summed_convoluted = torch.zeros_like(dp_pp)
                            
                            # Accumulate ideal and convoluted patterns
                            summed_ideal += dp_pp_IDEAL
                            summed_convoluted += dp_pp
                        

                        
                        # Process with each model
                        for _, row in df.iterrows():
                            model_key = f"{row['loss_type']}_{'Unet' if row['use_unet'] else 'no_Unet'}_{row['iterations']}"
                            
                            # Load model once per model per angle
                            model = self.load_model(row['path'], row['use_unet'])
                            
                            # Reset summed output for this model
                            summed_output = None
                            
                            # Deconvolve each pattern individually and sum the outputs
                            for pattern_offset in range(patterns_per_angle):
                                pattern_idx = start_idx + pattern_offset
                                if pattern_idx >= num_patterns:
                                    break
                                
                                convDP_pattern = convDP[pattern_idx]
                                dp_pp, _, _ = ptNN_U.preprocess_ZCB_9(convDP_pattern, mask)
                                dp_pp = dp_pp.to(device=self.device, dtype=torch.float)
                                
                                with torch.no_grad():
                                    output = model(dp_pp)
                                
                                # Initialize summed output on first pattern
                                if summed_output is None:
                                    summed_output = torch.zeros_like(output)
                                
                                # Accumulate deconvolved output
                                summed_output += output
                            # Now calculate metrics on the summed patterns
                            metrics = self.calculate_metrics(summed_output, summed_ideal,
                                                             calculate_psnr=calculate_psnr,
                                                             calculate_ssim=calculate_ssim,
                                                             calculate_xcorr=calculate_xcorr,
                                                             calculate_peaks=calculate_peaks,
                                                             calculate_peak_sensitivity=calculate_peak_sensitivity,
                                                             peak_sigma=peak_sigma)
                            
                            if calculate_psnr and 'psnr' in metrics:
                                stats[model_key]['psnr_sum'] += metrics['psnr']
                            if calculate_ssim and 'ssim' in metrics:
                                stats[model_key]['ssim_sum'] += metrics['ssim']
                            if calculate_xcorr and 'xcorr' in metrics:
                                stats[model_key]['xcorr_sum'] += metrics['xcorr']
                            if calculate_peaks:
                                if 'avg_peak_dist' in metrics:
                                    stats[model_key]['peak_dist_sum'] += metrics['avg_peak_dist']
                                if 'fwhm_diff' in metrics:
                                    stats[model_key]['fwhm_diff_sum'] += metrics['fwhm_diff']
                                if 'num_peaks2' in metrics:
                                    stats[model_key]['total_peaks_ideal'] += metrics['num_peaks2']
                                    
                                    used_sigma = metrics.get('peak_sigma_used', peak_sigma)
                                    used_threshold = metrics.get('peak_threshold_used', self.default_peak_threshold)
                                    
                                    # Normalize ideal image before peak detection (done once per angle)
                                    ideal_norm = self.normalize_image(summed_ideal, apply_smoothing=True, smoothing_sigma=self.ideal_peak_smoothing_sigma)
                                    ideal_percentile = self.percentile_threshold_value if self.use_percentile_threshold else None
                                    ideal_peaks, _ = self.find_peaks_and_fwhm(ideal_norm.squeeze().cpu().numpy(), 
                                                                              sigma=used_sigma, 
                                                                              threshold=used_threshold,
                                                                              percentile=ideal_percentile)
                                    
                                    # Use helper method for output peak detection with adaptive thresholding
                                    output_peaks, confusion_metrics = self._detect_peaks_with_adaptive_threshold(
                                        summed_output, ideal_peaks, used_sigma, used_threshold, verbose=False)
                                    
                                    matched = confusion_metrics['tp']
                                    stats[model_key]['total_peaks_matched'] += matched
                                    
                                    if show_peak_classification:
                                        # Track TP/FP/FN metrics when flag is enabled
                                        stats[model_key]['peak_tp_sum'] += confusion_metrics['tp']
                                        stats[model_key]['peak_fp_sum'] += confusion_metrics['fp']
                                        stats[model_key]['peak_fn_sum'] += confusion_metrics['fn']
                                        stats[model_key]['peak_precision_sum'] += confusion_metrics['precision']
                                        stats[model_key]['peak_recall_sum'] += confusion_metrics['recall']
                                        stats[model_key]['peak_f1_sum'] += confusion_metrics['f1_score']
                            stats[model_key]['angle_count'] += 1
                                                    
                        # Visualize the summed patterns
                        visualize_summed_patterns = False
                        if visualize_summed_patterns:
                            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
                            axs[0, 0].imshow(summed_ideal.squeeze().cpu().numpy(), cmap='jet')
                            axs[0, 0].set_title('Ideal')
                            axs[0, 1].imshow(summed_convoluted.squeeze().cpu().numpy(), cmap='jet')
                            axs[0, 1].set_title('Convoluted')
                            axs[1, 0].imshow(summed_output.squeeze().cpu().numpy(), cmap='jet')
                            axs[1, 0].set_title('Output')
                            axs[1, 1].imshow(summed_ideal.squeeze().cpu().numpy() - summed_output.squeeze().cpu().numpy(), cmap='RdBu')
                            axs[1, 1].set_title('Difference')
                            plt.show() 
                    except Exception as e:
                        print(f"Warning: Error processing angle {angle_idx}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
        except Exception as e:
            print(f"Error loading h5 file {h5_file_path}: {e}")
            raise
        
        # Calculate averages and create final statistics
        final_stats = {}
        for model_key, model_stats in stats.items():
            count = model_stats['angle_count']
            if count > 0:
                final_stats[model_key] = {
                    'avg_psnr': model_stats['psnr_sum'] / count,
                    'avg_ssim': model_stats['ssim_sum'] / count,
                    'avg_xcorr': model_stats['xcorr_sum'] / count,
                    'avg_peak_dist': model_stats['peak_dist_sum'] / count,
                    'avg_fwhm_diff': model_stats['fwhm_diff_sum'] / count,
                    'peak_detection_rate': model_stats['total_peaks_matched'] / model_stats['total_peaks_ideal'] if model_stats['total_peaks_ideal'] > 0 else 0,
                    'total_peaks_matched': model_stats['total_peaks_matched'],
                    'total_peaks_ideal': model_stats['total_peaks_ideal'],
                    'angles_processed': count,
                    # Always calculate precision/recall/F1 (more robust metrics)
                    'total_peak_tp': model_stats['peak_tp_sum'],
                    'total_peak_fp': model_stats['peak_fp_sum'],
                    'total_peak_fn': model_stats['peak_fn_sum'],
                    'avg_peak_precision': model_stats['peak_precision_sum'] / count if count > 0 else 0,
                    'avg_peak_recall': model_stats['peak_recall_sum'] / count if count > 0 else 0,
                    'avg_peak_f1': model_stats['peak_f1_sum'] / count if count > 0 else 0
                }
                
                if calculate_peak_sensitivity:
                    final_stats[model_key].update({
                        'avg_optimal_sigma': model_stats['optimal_sigma_sum'] / count,
                        'avg_optimal_threshold': model_stats['optimal_threshold_sum'] / count,
                        'avg_max_f1_score': model_stats['max_f1_score_sum'] / count,
                        'avg_peak_position_stability': model_stats['peak_position_stability_sum'] / count,
                        'avg_peak_count_stability': model_stats['peak_count_stability_sum'] / count,
                        'avg_parameter_sensitivity': model_stats['parameter_sensitivity_sum'] / count,
                        'avg_false_positive_rate': model_stats['false_positive_rate_sum'] / count,
                        'avg_false_negative_rate': model_stats['false_negative_rate_sum'] / count
                    })
        
        return final_stats

    def create_comparison_grid_from_h5_summed_by_angle(self, model_configs: Dict,
                                                       h5_file_path: str,
                                                       base_path: str = "",
                                                       mask_path: str = '/home/beams/PTYCHOSAXS/deconvolutionNN/data/mask/mask_ZCB_9_3D.npy',
                                                       patterns_per_angle: int = 16,
                                                       angle_index: int = 0,
                                                       figsize: Tuple[int, int] = (20, 15),
                                                       calculate_psnr: bool = True,
                                                       calculate_ssim: bool = True,
                                                       calculate_xcorr: bool = False,
                                                       calculate_peaks: bool = True,
                                                       calculate_peak_sensitivity: bool = False,
                                                       peak_sigma: float = None,
                                                       show_peak_classification: bool = False) -> plt.Figure:
        """
        Create a comparison grid similar to `create_comparison_grid`, but using
        a single projection angle from an H5 tomographic dataset where patterns
        are processed individually and summed by angle.

        The top row shows the summed convoluted input and the summed ideal.
        The grid shows, for each model configuration, the summed deconvolved output
        with metrics and optional peak overlays against the summed ideal.
        """
        if peak_sigma is None:
            peak_sigma = self.default_peak_sigma

        # Resolve model paths/order just like in create_comparison_grid
        model_info = self.get_model_paths_from_config(model_configs, base_path)
        if not model_info:
            raise ValueError("No valid model paths found in the configuration")
        df = pd.DataFrame(model_info)
        df = df.sort_values(['iterations', 'loss_type'])

        nrows = len(model_configs['iterations'])
        ncols = len(model_configs['models'])

        width_per_col = 6
        height_per_row = 5
        figsize = (width_per_col * ncols, height_per_row * (nrows + 1))

        # Load data and build summed inputs for the requested angle
        mask = np.load(mask_path)
        with h5py.File(h5_file_path, 'r') as h5file:
            if 'convDP' not in h5file:
                raise ValueError(f"Key 'convDP' not found in h5 file: {h5_file_path}")
            if 'pinholeDP_raw_FFT' not in h5file:
                raise ValueError(f"Key 'pinholeDP_raw_FFT' not found in h5 file: {h5_file_path}")

            convDP = h5file['convDP'][:]
            pinholeDP_raw_FFT = h5file['pinholeDP_raw_FFT'][:]
            rotation_angles = h5file['rotation_angles'][:] if 'rotation_angles' in h5file else None

        num_patterns = convDP.shape[0]
        num_angles = num_patterns // patterns_per_angle
        if not (0 <= angle_index < num_angles):
            raise ValueError(f"angle_index {angle_index} out of range [0, {num_angles - 1}]")

        start_idx = angle_index * patterns_per_angle
        end_idx = min(start_idx + patterns_per_angle, num_patterns)

        summed_ideal = None
        summed_convoluted = None

        for pattern_idx in range(start_idx, end_idx):
            convDP_pattern = convDP[pattern_idx]
            pinholeDP_pattern = pinholeDP_raw_FFT[pattern_idx]

            dp_pp, _, _ = ptNN_U.preprocess_ZCB_9(convDP_pattern, mask)
            dp_pp_IDEAL, _, _ = ptNN_U.preprocess_ZCB_9(pinholeDP_pattern, mask=np.ones(dp_pp[0][0].shape))

            dp_pp = dp_pp.to(device=self.device, dtype=torch.float)
            dp_pp_IDEAL = dp_pp_IDEAL.to(device=self.device, dtype=torch.float)

            if summed_ideal is None:
                summed_ideal = torch.zeros_like(dp_pp_IDEAL)
                summed_convoluted = torch.zeros_like(dp_pp)

            summed_ideal += dp_pp_IDEAL
            summed_convoluted += dp_pp

        # Prepare figure and axes
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(nrows + 1, ncols,
                              height_ratios=[1.2] + [1] * nrows,
                              hspace=0.3, wspace=0.3)

        # Top row: input (summed convoluted) and ideal (summed ideal)
        ax_input = fig.add_subplot(gs[0, 0])
        ax_ideal = fig.add_subplot(gs[0, 1])
        im_input = ax_input.imshow(summed_convoluted.squeeze().cpu().numpy(), cmap='jet')
        angle_title = f"Angle {angle_index}"
        if rotation_angles is not None and 0 <= angle_index < len(rotation_angles):
            angle_title += f" ({rotation_angles[angle_index]}°)"
        ax_input.set_title(f'Input (summed)\n{angle_title}', fontsize=24)
        plt.colorbar(im_input, ax=ax_input, fraction=0.046, pad=0.04)
        ax_input.tick_params(axis='both', labelsize=20)

        im_ideal = ax_ideal.imshow(summed_ideal.squeeze().cpu().numpy(), cmap='jet')
        ax_ideal.set_title('Ideal (summed)', fontsize=24)
        plt.colorbar(im_ideal, ax=ax_ideal, fraction=0.046, pad=0.04)
        ax_ideal.tick_params(axis='both', labelsize=20)

        # Peak detection on ideal, once
        ideal_peaks = None
        if calculate_peaks:
            ideal_data_norm = self.normalize_image(summed_ideal, apply_smoothing=True, smoothing_sigma=self.ideal_peak_smoothing_sigma)
            ideal_peaks, ideal_fwhm = self.find_peaks_and_fwhm(ideal_data_norm.squeeze().cpu().numpy(),
                                                               sigma=peak_sigma,
                                                               threshold=self.default_peak_threshold)
            for peak in ideal_peaks:
                ax_ideal.plot(peak[1], peak[0], 'g+', markersize=10, markeredgewidth=2, alpha=0.3)

        # Create model axes matrix
        axes = np.zeros((nrows, ncols), dtype=object)
        for i in range(nrows):
            for j in range(ncols):
                axes[i, j] = fig.add_subplot(gs[i + 1, j])

        # Iterate models in consistent order
        for _, row in df.iterrows():
            row_idx = model_configs['iterations'].index(row['iterations'])
            col_idx = list(model_configs['models'].keys()).index(
                f"{row['loss_type']}_{'Unet' if row['use_unet'] else 'no_Unet'}")

            model = self.load_model(row['path'], row['use_unet'])

            # Sum outputs for this angle
            summed_output = None
            with torch.no_grad():
                for pattern_idx in range(start_idx, end_idx):
                    convDP_pattern = convDP[pattern_idx]
                    dp_pp, _, _ = ptNN_U.preprocess_ZCB_9(convDP_pattern, mask)
                    dp_pp = dp_pp.to(device=self.device, dtype=torch.float)
                    output = model(dp_pp)
                    if summed_output is None:
                        summed_output = torch.zeros_like(output)
                    summed_output += output

            # Metrics
            metrics = self.calculate_metrics(summed_output, summed_ideal,
                                             calculate_psnr=calculate_psnr,
                                             calculate_ssim=calculate_ssim,
                                             calculate_xcorr=calculate_xcorr,
                                             calculate_peaks=calculate_peaks,
                                             calculate_peak_sensitivity=calculate_peak_sensitivity,
                                             peak_sigma=peak_sigma)

            used_sigma = metrics.get('peak_sigma_used', peak_sigma)
            used_threshold = metrics.get('peak_threshold_used', self.default_peak_threshold)

            ax = axes[row_idx, col_idx]
            im = ax.imshow(summed_output.squeeze().cpu().numpy(), cmap='jet')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.tick_params(axis='both', labelsize=16)

            # Process peaks if enabled
            confusion_metrics = None
            if calculate_peaks and ideal_peaks:
                output_peaks, confusion_metrics = self._detect_peaks_with_adaptive_threshold(
                    summed_output, ideal_peaks, used_sigma, used_threshold, verbose=True)
                self._plot_peaks_on_axis(ax, output_peaks, ideal_peaks, confusion_metrics, 
                                        show_peak_classification)

            # Build and display metrics text
            metrics_text = self._build_metrics_text(
                metrics, calculate_psnr, calculate_ssim, calculate_xcorr,
                calculate_peaks, ideal_peaks, confusion_metrics, show_peak_classification)

            ax.text(0.02, 0.98, '\n'.join(metrics_text),
                    transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            if col_idx == 0:
                ax.set_ylabel(f"{row['iterations']} epochs", fontsize=20)

            if row_idx == 0:
                ax.set_title(self._format_model_label(row['loss_type'], row['use_unet']), fontsize=20)

        # Legend for peak markers
        if calculate_peaks:
            legend_elements = self._create_peak_legend_elements(show_peak_classification)
            if len(model_configs['models']) > 2:
                ax_legend = fig.add_subplot(gs[0, 2:])
                ax_legend.axis('off')
                ax_legend.legend(handles=legend_elements, loc='center left', frameon=False, fontsize=36, markerscale=3)

        plt.tight_layout()
        return fig
    
    def group_stats_by_model_type(self, stats: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Group statistics by model type (L1/L2/pearson, Unet/no_Unet), combining iterations.
        """
        grouped_stats = {}
        
        for model_key, metrics in stats.items():
            model_type = '_'.join(model_key.split('_')[:-1])
            
            if model_type not in grouped_stats:
                grouped_stats[model_type] = {
                    'avg_psnr': [], 'avg_ssim': [], 'avg_xcorr': [], 'avg_peak_dist': [],
                    'avg_fwhm_diff': [], 'peak_detection_rate': [], 'total_peaks_matched': 0,
                    'total_peaks_ideal': 0, 'total_patterns': 0, 'avg_optimal_sigma': [],
                    'avg_optimal_threshold': [], 'avg_max_f1_score': [], 'avg_peak_position_stability': [],
                    'avg_peak_count_stability': [], 'avg_parameter_sensitivity': [],
                    'avg_false_positive_rate': [], 'avg_false_negative_rate': [],
                    'total_peak_tp': 0, 'total_peak_fp': 0, 'total_peak_fn': 0,
                    'avg_peak_precision': [], 'avg_peak_recall': [], 'avg_peak_f1': []
                }
            
            for metric in ['avg_psnr', 'avg_ssim', 'avg_xcorr', 'avg_peak_dist', 'avg_fwhm_diff', 'peak_detection_rate',
                          'avg_optimal_sigma', 'avg_optimal_threshold', 'avg_max_f1_score',
                          'avg_peak_position_stability', 'avg_peak_count_stability', 'avg_parameter_sensitivity',
                          'avg_false_positive_rate', 'avg_false_negative_rate',
                          'avg_peak_precision', 'avg_peak_recall', 'avg_peak_f1']:
                if metric in metrics:
                    grouped_stats[model_type][metric].append(metrics[metric])
            
            grouped_stats[model_type]['total_peaks_matched'] += metrics['total_peaks_matched']
            grouped_stats[model_type]['total_peaks_ideal'] += metrics['total_peaks_ideal']
            grouped_stats[model_type]['total_patterns'] += metrics['patterns_processed']
            if 'total_peak_tp' in metrics:
                grouped_stats[model_type]['total_peak_tp'] += metrics['total_peak_tp']
            if 'total_peak_fp' in metrics:
                grouped_stats[model_type]['total_peak_fp'] += metrics['total_peak_fp']
            if 'total_peak_fn' in metrics:
                grouped_stats[model_type]['total_peak_fn'] += metrics['total_peak_fn']
        
        # Calculate averages for each model type
        final_grouped_stats = {}
        for model_type, metrics in grouped_stats.items():
            final_grouped_stats[model_type] = {
                'avg_psnr': np.mean(metrics['avg_psnr']) if metrics['avg_psnr'] else 0,
                'avg_ssim': np.mean(metrics['avg_ssim']) if metrics['avg_ssim'] else 0,
                'avg_xcorr': np.mean(metrics['avg_xcorr']) if metrics['avg_xcorr'] else 0,
                'avg_peak_dist': np.mean(metrics['avg_peak_dist']) if metrics['avg_peak_dist'] else 0,
                'avg_fwhm_diff': np.mean(metrics['avg_fwhm_diff']) if metrics['avg_fwhm_diff'] else 0,
                'peak_detection_rate': metrics['total_peaks_matched'] / metrics['total_peaks_ideal'] if metrics['total_peaks_ideal'] > 0 else 0,
                'total_peaks_matched': metrics['total_peaks_matched'],
                'total_peaks_ideal': metrics['total_peaks_ideal'],
                'total_patterns': metrics['total_patterns'],
                'total_peak_tp': metrics['total_peak_tp'],
                'total_peak_fp': metrics['total_peak_fp'],
                'total_peak_fn': metrics['total_peak_fn'],
                'avg_peak_precision': np.mean(metrics['avg_peak_precision']) if metrics['avg_peak_precision'] else 0,
                'avg_peak_recall': np.mean(metrics['avg_peak_recall']) if metrics['avg_peak_recall'] else 0,
                'avg_peak_f1': np.mean(metrics['avg_peak_f1']) if metrics['avg_peak_f1'] else 0
            }
            
            if metrics['avg_optimal_sigma']:
                final_grouped_stats[model_type].update({
                    'avg_optimal_sigma': np.mean(metrics['avg_optimal_sigma']),
                    'avg_optimal_threshold': np.mean(metrics['avg_optimal_threshold']),
                    'avg_max_f1_score': np.mean(metrics['avg_max_f1_score']),
                    'avg_peak_position_stability': np.mean(metrics['avg_peak_position_stability']),
                    'avg_peak_count_stability': np.mean(metrics['avg_peak_count_stability']),
                    'avg_parameter_sensitivity': np.mean(metrics['avg_parameter_sensitivity']),
                    'avg_false_positive_rate': np.mean(metrics['avg_false_positive_rate']),
                    'avg_false_negative_rate': np.mean(metrics['avg_false_negative_rate'])
                })
        
        return final_grouped_stats
    
    def print_cumulative_stats(self, stats: Dict[str, Dict[str, float]], 
                              sort_by: str = 'avg_ssim', 
                              group_by_model: bool = True):
        """
        Print cumulative statistics in a formatted table, sorted by a specified metric.
        Can group statistics by model type.
        
        Args:
            stats (Dict[str, Dict[str, float]]): Dictionary of statistics per model
            sort_by (str): Metric to sort by
            group_by_model (bool): Whether to group statistics by model type
        """
        if group_by_model:
            stats = self.group_stats_by_model_type(stats)
        
        rows = []
        for model, metrics in stats.items():
            row = {'Model': model}
            row.update(metrics)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df = df.sort_values(by=sort_by, ascending=False)
        
        # Reorder columns to show key metrics first (precision/recall/F1 are most important)
        priority_columns = ['Model', 'avg_peak_precision', 'avg_peak_recall', 'avg_peak_f1', 
                           'peak_detection_rate', 'avg_psnr', 'avg_ssim', 'avg_peak_dist', 
                           'avg_fwhm_diff', 'total_peak_tp', 'total_peak_fp', 'total_peak_fn',
                           'total_peaks_matched', 'total_peaks_ideal', 'patterns_processed']
        
        # Get remaining columns not in priority list
        remaining_cols = [col for col in df.columns if col not in priority_columns]
        # Reorder: priority columns first, then remaining columns
        column_order = [col for col in priority_columns if col in df.columns] + remaining_cols
        df = df[column_order]
        
        formatted_df = df.copy()
        for col in df.columns:
            if col == 'Model':
                continue
            if col in ['total_peaks_matched', 'total_peaks_ideal', 'total_patterns', 'patterns_processed',
                      'total_peak_tp', 'total_peak_fp', 'total_peak_fn']:
                formatted_df[col] = df[col].map(lambda x: f"{int(x):,}")
            else:
                formatted_df[col] = df[col].map(lambda x: f"{x:.4f}")
        
        print("\nCumulative Statistics {} (sorted by {}):\n".format(
            "Grouped by Model Type" if group_by_model else "Per Model and Iteration",
            sort_by
        ))
        print(formatted_df.to_string())
    
    def create_stats_table_figure(self, stats: Dict[str, Dict[str, float]], 
                                sort_by: str = 'avg_ssim',
                                group_by_model: bool = True,
                                figsize: Tuple[float, float] = (15, 5)) -> plt.Figure:
        """
        Create a formatted table figure from the statistics.
        """
        if group_by_model:
            stats = self.group_stats_by_model_type(stats)
        
        rows = []
        for model, metrics in stats.items():
            row = {'Model': model}
            row.update(metrics)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df = df.sort_values(by=sort_by, ascending=False)
        
        columns_to_show = [
            'Model', 'avg_psnr', 'avg_ssim', 'peak_detection_rate',
            'avg_peak_dist', 'avg_fwhm_diff', 'total_peaks_matched',
            'total_peaks_ideal', 'pattern_count' if 'pattern_count' in df.columns else 'total_patterns'
        ]
        
        if 'avg_optimal_sigma' in df.columns:
            columns_to_show.extend([
                'avg_optimal_sigma', 'avg_optimal_threshold', 'avg_max_f1_score',
                'avg_false_positive_rate', 'avg_false_negative_rate'
            ])
        
        column_labels = {
            'Model': 'Model Type', 'avg_psnr': 'PSNR (dB)', 'avg_ssim': 'SSIM',
            'peak_detection_rate': 'Peak Detection Rate', 'avg_peak_dist': 'Avg Peak Distance',
            'avg_fwhm_diff': 'Avg FWHM Diff', 'total_peaks_matched': 'Peaks Matched',
            'total_peaks_ideal': 'Total Peaks', 'pattern_count': 'Patterns',
            'total_patterns': 'Patterns', 'avg_optimal_sigma': 'Optimal Sigma',
            'avg_optimal_threshold': 'Optimal Threshold', 'avg_max_f1_score': 'Peak F1 Score',
            'avg_false_positive_rate': 'False Pos Rate', 'avg_false_negative_rate': 'False Neg Rate'
        }
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('tight')
        ax.axis('off')
        
        display_df = df[columns_to_show].copy()
        
        for col in display_df.columns:
            if col == 'Model':
                continue
            if col in ['total_peaks_matched', 'total_peaks_ideal', 'pattern_count', 'total_patterns']:
                display_df[col] = display_df[col].map(lambda x: f"{int(x):,}")
            elif col in ['avg_psnr', 'avg_peak_dist', 'avg_fwhm_diff']:
                display_df[col] = display_df[col].map(lambda x: f"{x:.2f}")
            else:
                display_df[col] = display_df[col].map(lambda x: f"{x:.4f}")
        
        table = ax.table(
            cellText=display_df.values,
            colLabels=[column_labels[col] for col in columns_to_show],
            cellLoc='center',
            loc='center',
            colColours=['#E6E6E6'] * len(columns_to_show)
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.8)
        
        for (row, col), cell in table.get_celld().items():
            if col == 0:
                cell.set_width(0.2)
            else:
                cell.set_width(0.1)
        
        plt.tight_layout()
        return fig
    
    def save_figure(self, fig: plt.Figure, save_path: str, dpi: int = 300):
        """
        Save a figure to file.
        
        Args:
            fig (plt.Figure): Figure to save
            save_path (str): Path where to save the figure
            dpi (int): DPI for saving
        """
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    
    def cross_validate_models(self, model_base_path: str,
                            data_base_path: str,
                            mask_path: str,
                            hkl_index: Tuple[int, int, int],
                            lattice_types: List[str] = ['SC', 'ClathII'],
                            noise_statuses: List[str] = ['Noise', 'noNoise'],
                            probe_sizes: List[int] = [128, 256],
                            loss_function: str = 'pearson_loss',
                            unet_status: str = 'Unet',
                            epoch: int = 25,
                            pattern_num: int = 5,
                            figsize: Tuple[int, int] = (20, 20),
                            cmap: str = 'jet') -> plt.Figure:
        """
        Cross-validate models by showing outputs from all trained model conditions
        applied to all test data conditions in a confusion matrix style grid.
        """
        mask = np.load(mask_path)
        hr, kr, lr = hkl_index
        
        conditions = []
        for lattice in lattice_types:
            for noise in noise_statuses:
                for probe in probe_sizes:
                    conditions.append({
                        'lattice': lattice,
                        'noise': noise,
                        'probe': probe
                    })
        
        n_conditions = len(conditions)
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(n_conditions + 1, n_conditions + 3,
                             hspace=0.4, wspace=0.4,
                             left=0.08, right=0.98, top=0.95, bottom=0.05)
        
        loss_display = 'NPCC' if 'pearson' in loss_function else loss_function.replace('_loss', '')
        fig.suptitle(f'Network ({epoch} epochs, {loss_display})',
                    fontsize=24, fontweight='bold')
        
        # Top-left corner
        ax_corner = fig.add_subplot(gs[0, 0])
        ax_corner.axis('off')
        ax_corner.text(0.5, 0.5, 'Lattice\n(hkl)', 
                      ha='center', va='center', fontsize=14, fontweight='bold')
        
        # Column headers
        for col_idx, cond in enumerate(conditions):
            ax = fig.add_subplot(gs[0, col_idx + 1])
            ax.axis('off')
            noise_label = 'Noise' if cond['noise'] == 'Noise' else 'NoNoise'
            label_text = f"{noise_label}\n{cond['lattice']}"
            ax.text(0.5, 0.8, label_text, 
                   ha='center', va='top', fontsize=12, fontweight='bold')
            ax.text(0.5, 0.2, f"{cond['probe']}x{cond['probe']}", 
                   ha='center', va='bottom', fontsize=10)
        
        # Ground truth column headers
        ax_gt_conv = fig.add_subplot(gs[0, n_conditions + 1])
        ax_gt_conv.axis('off')
        ax_gt_conv.text(0.5, 0.5, 'Convolved\n(Input)', 
                       ha='center', va='center', fontsize=12, fontweight='bold')
        
        ax_gt_ideal = fig.add_subplot(gs[0, n_conditions + 2])
        ax_gt_ideal.axis('off')
        ax_gt_ideal.text(0.5, 0.5, 'Ground truth\n(Ideal)', 
                        ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Load all models
        print("Loading models...")
        models = {}
        for col_idx, model_cond in enumerate(tqdm(conditions, desc="Loading models")):
            model_path = (f"{model_base_path}best_model_Lattice{model_cond['lattice']}_"
                         f"Probe{model_cond['probe']}x{model_cond['probe']}_ZCB_9_3D__{model_cond['noise']}_"
                         f"sim_ZCB_9_3D_S5065_N600_steps4_dp256_{unet_status}_epoch_{epoch}_"
                         f"{loss_function}_symmetry_0.0.pth")
            
            if os.path.exists(model_path):
                try:
                    use_unet = (unet_status == 'Unet')
                    model = self.load_model(model_path, use_unet)
                    model.eval()
                    models[col_idx] = model
                except Exception as e:
                    print(f"Error loading model {model_cond['lattice']}/{model_cond['noise']}: {e}")
                    models[col_idx] = None
            else:
                print(f"Warning: Model not found: {model_path}")
                models[col_idx] = None
        
        # Process each row (test condition)
        print("\nGenerating outputs...")
        for row_idx, test_cond in enumerate(tqdm(conditions, desc="Processing test data")):
            # Row header
            ax = fig.add_subplot(gs[row_idx + 1, 0])
            ax.axis('off')
            noise_label = 'Noise' if test_cond['noise'] == 'Noise' else 'NoNoise'
            label_text = f"{noise_label}\n{test_cond['lattice']}"
            ax.text(0.8, 0.5, label_text, 
                   ha='right', va='center', fontsize=12, fontweight='bold')
            ax.text(0.8, 0.2, f"{test_cond['probe']}x{test_cond['probe']}", 
                   ha='right', va='bottom', fontsize=10)
            
            # Load test data for this row
            data_dir = f'Lattice{test_cond["lattice"]}_Probe{test_cond["probe"]}x{test_cond["probe"]}_ZCB_9_3D__{test_cond["noise"]}_hkl{hr}{kr}{lr}'
            data_file = f'{data_base_path}{data_dir}/sim_ZCB_9_3D_S5065_N1_steps4_dp256.h5'
            
            if os.path.exists(data_file):
                try:
                    with h5py.File(data_file, 'r') as h5f:
                        if 'convDP' in h5f.keys():
                            conv_dp_data = h5f['convDP'][pattern_num - 1]
                            if 'pinholeDP_raw_FFT' in h5f.keys():
                                ideal_dp_data = h5f['pinholeDP_raw_FFT'][pattern_num - 1]
                            elif 'pinholeDP_extra_conv' in h5f.keys():
                                ideal_dp_data = h5f['pinholeDP_extra_conv'][pattern_num - 1]
                            elif 'pinholeDP' in h5f.keys():
                                ideal_dp_data = h5f['pinholeDP'][pattern_num - 1]
                            else:
                                ideal_dp_data = None
                        else:
                            conv_dp_data = h5f[f'convDP_{pattern_num}'][:]
                            ideal_dp_data = h5f[f'pinholeDP_{pattern_num}'][:] if f'pinholeDP_{pattern_num}' in h5f.keys() else None
                        
                    dp_pp, _, _ = ptNN_U.preprocess_ZCB_9(conv_dp_data, mask)
                    dp_pp = dp_pp.to(device=self.device, dtype=torch.float)
                    
                    if ideal_dp_data is not None:
                        dp_pp_IDEAL, _, _ = ptNN_U.preprocess_ZCB_9(ideal_dp_data, 
                                                                     mask=np.ones(dp_pp[0][0].shape))
                        dp_pp_IDEAL = dp_pp_IDEAL.to(device=self.device, dtype=torch.float)
                    else:
                        dp_pp_IDEAL = None
                    
                    # Apply each model to this test data
                    for col_idx, model_cond in enumerate(conditions):
                        ax = fig.add_subplot(gs[row_idx + 1, col_idx + 1])
                        
                        if models[col_idx] is not None:
                            try:
                                with torch.no_grad():
                                    output = models[col_idx](dp_pp)
                                
                                output_np = output.squeeze().cpu().numpy()
                                im = ax.imshow(output_np, cmap=cmap)
                                
                                from mpl_toolkits.axes_grid1 import make_axes_locatable
                                divider = make_axes_locatable(ax)
                                cax = divider.append_axes("right", size="5%", pad=0.05)
                                plt.colorbar(im, cax=cax, format='%.1e')
                                
                            except Exception as e:
                                ax.text(0.5, 0.5, f'Error:\n{str(e)[:20]}', 
                                       ha='center', va='center', fontsize=8)
                        else:
                            ax.text(0.5, 0.5, 'Model\nNot Found', 
                                   ha='center', va='center', fontsize=8)
                        
                        ax.set_xticks([])
                        ax.set_yticks([])
                        
                        # Highlight diagonal
                        if row_idx == col_idx:
                            for spine in ax.spines.values():
                                spine.set_edgecolor('red')
                                spine.set_linewidth(3)
                    
                    # Add ground truth images
                    ax_conv = fig.add_subplot(gs[row_idx + 1, n_conditions + 1])
                    conv_input_np = dp_pp.squeeze().cpu().numpy()
                    im_conv = ax_conv.imshow(conv_input_np, cmap=cmap)
                    from mpl_toolkits.axes_grid1 import make_axes_locatable
                    divider = make_axes_locatable(ax_conv)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(im_conv, cax=cax, format='%.1e')
                    ax_conv.set_xticks([])
                    ax_conv.set_yticks([])
                    
                    ax_ideal = fig.add_subplot(gs[row_idx + 1, n_conditions + 2])
                    if dp_pp_IDEAL is not None:
                        ideal_np = dp_pp_IDEAL.squeeze().cpu().numpy()
                        im_ideal = ax_ideal.imshow(ideal_np, cmap=cmap)
                        divider = make_axes_locatable(ax_ideal)
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        plt.colorbar(im_ideal, cax=cax, format='%.1e')
                    else:
                        ax_ideal.text(0.5, 0.5, 'Ideal\nNot Available', 
                                     ha='center', va='center', fontsize=8)
                    ax_ideal.set_xticks([])
                    ax_ideal.set_yticks([])
                    
                except Exception as e:
                    print(f"Error loading test data {test_cond['lattice']}/{test_cond['noise']}: {e}")
                    for col_idx in range(n_conditions + 2):
                        ax = fig.add_subplot(gs[row_idx + 1, col_idx + 1])
                        ax.text(0.5, 0.5, 'Test Data\nNot Found', 
                               ha='center', va='center', fontsize=8)
                        ax.set_xticks([])
                        ax.set_yticks([])
            else:
                print(f"Warning: Test data not found: {data_file}")
                for col_idx in range(n_conditions + 2):
                    ax = fig.add_subplot(gs[row_idx + 1, col_idx + 1])
                    ax.text(0.5, 0.5, 'Test Data\nNot Found', 
                           ha='center', va='center', fontsize=8)
                    ax.set_xticks([])
                    ax.set_yticks([])
        
        return fig
    
    def read_loss_value(self, filepath: str) -> Optional[float]:
        """
        Read loss value from a text file.
        
        Args:
            filepath (str): Path to the loss file
            
        Returns:
            Optional[float]: Loss value or None if not found
        """
        try:
            with open(filepath, 'r') as f:
                content = f.read().strip()
                return float(content)
        except (ValueError, FileNotFoundError):
            return None
    
    def parse_loss_filename(self, filename: str) -> Tuple[Optional[int], Optional[str], Optional[str]]:
        """
        Parse filename to extract epoch, loss function, and configuration.
        Handles multiple patterns:
        1. Original: best_loss_epoch_LatticeSC_Probe256x256_ZCB_9_3D__Noise_sim_ZCB_9_3D_S5065_N600_steps4_dp256_Unet_{epoch}_{loss}_symmetry_0.0.txt
        2. L1: best_loss_epoch_{unet_status}_{epoch}.txt
        3. L2: best_loss_epoch_32_{unet_status}_{epoch}_L2.txt
        4. Pearson: best_loss_epoch_{unet_status}_{epoch}_pearson_loss.txt
        
        Args:
            filename (str): Filename to parse
            
        Returns:
            Tuple[Optional[int], Optional[str], Optional[str]]: (epoch, loss_func, config)
        """
        # Try L2 pattern first: best_loss_epoch_32_{unet_status}_{epoch}_L2.txt
        l2_match = re.search(r'best_loss_epoch_32_(no_Unet|Unet)_(\d+)_L2\.txt', filename)
        if l2_match:
            unet_status = l2_match.group(1)
            epoch = int(l2_match.group(2))
            return epoch, 'L2', unet_status
        
        # Try L1 pattern with Unet: best_loss_epoch_Unet_{epoch}.txt
        l1_unet_match = re.search(r'best_loss_epoch_Unet_(\d+)\.txt$', filename)
        if l1_unet_match:
            epoch = int(l1_unet_match.group(1))
            return epoch, 'L1', 'Unet'
        
        # Try L1 pattern without Unet: best_loss_epoch_{epoch}.txt
        l1_no_unet_match = re.search(r'best_loss_epoch_(\d+)\.txt$', filename)
        if l1_no_unet_match:
            epoch = int(l1_no_unet_match.group(1))
            return epoch, 'L1', 'no_Unet'
        
        # Try Pearson pattern: best_loss_epoch_{unet_status}_{epoch}_pearson_loss.txt
        pearson_match = re.search(r'best_loss_epoch_(no_Unet|Unet)_(\d+)_pearson_loss\.txt', filename)
        if pearson_match:
            unet_status = pearson_match.group(1)
            epoch = int(pearson_match.group(2))
            return epoch, 'pearson_loss', unet_status
        
        # Try original pattern: best_loss_epoch_LatticeSC_Probe256x256_ZCB_9_3D__Noise_sim_ZCB_9_3D_S5065_N600_steps4_dp256_Unet_{epoch}_{loss}_symmetry_0.0.txt
        epoch_match = re.search(r'_Unet_(\d+)_', filename)
        epoch = int(epoch_match.group(1)) if epoch_match else None
        
        loss_match = re.search(r'_Unet_\d+_(L1|L2|pearson_loss)_', filename)
        loss_func = loss_match.group(1) if loss_match else None
        
        lattice_match = re.search(r'best_loss_epoch_(LatticeSC|LatticeClathII)_', filename)
        lattice_type = lattice_match.group(1) if lattice_match else None
        
        return epoch, loss_func, lattice_type
    
    def load_loss_data(self, data_dir: str, 
                      expected_epochs: List[int] = None,
                      expected_losses: List[str] = None) -> Dict[str, Dict[str, Dict[int, float]]]:
        """
        Load loss data from a directory containing loss files.
        
        Args:
            data_dir (str): Directory containing loss files
            expected_epochs (List[int]): List of expected epochs to look for
            expected_losses (List[str]): List of expected loss functions
            
        Returns:
            Dict[str, Dict[str, Dict[int, float]]]: Nested dict {config: {loss_func: {epoch: loss_value}}}
        """
        if expected_epochs is None:
            expected_epochs = [2, 5, 10, 25, 50, 100, 150, 200, 250, 300, 400, 500]
        if expected_losses is None:
            expected_losses = ['L1', 'L2', 'pearson_loss']
            
        data = defaultdict(lambda: defaultdict(dict))
        
        print("Scanning directory for loss files...")
        files_found = 0
        
        for filename in os.listdir(data_dir):
            if filename.endswith('.txt') and 'best_loss_epoch' in filename:
                epoch, loss_func, config = self.parse_loss_filename(filename)
                
                if epoch and loss_func and config:
                    # Process files matching the new patterns or original pattern with symmetry_0.0
                    should_process = False
                    if ('_symmetry_0.0.txt' in filename or  # Original pattern
                        re.match(r'best_loss_epoch_Unet_\d+\.txt$', filename) or  # L1 with Unet pattern
                        re.match(r'best_loss_epoch_\d+\.txt$', filename) or  # L1 without Unet pattern
                        re.match(r'best_loss_epoch_32_(no_Unet|Unet)_\d+_L2\.txt$', filename) or  # L2 pattern
                        re.match(r'best_loss_epoch_(no_Unet|Unet)_\d+_pearson_loss\.txt$', filename)):  # Pearson pattern
                        should_process = True
                    
                    if should_process:
                        filepath = os.path.join(data_dir, filename)
                        loss_value = self.read_loss_value(filepath)
                        
                        if loss_value is not None:
                            data[config][loss_func][epoch] = loss_value
                            files_found += 1
                            print(f"Found: {config}, {loss_func}, epoch {epoch}, loss = {loss_value:.6f}")
        
        print(f"\nTotal files processed: {files_found}")
        return dict(data)
    
    def normalize_losses(self, data: Dict[str, Dict[str, Dict[int, float]]]) -> Dict[str, Dict[str, Dict[int, float]]]:
        """
        Normalize losses using min-max normalization for each loss function within each configuration.
        Formula: (x - min) / (max - min)
        
        Args:
            data (Dict): Loss data dictionary
            
        Returns:
            Dict: Normalized loss data
        """
        normalized_data = defaultdict(lambda: defaultdict(dict))
        
        for config in data.keys():
            for loss_func in data[config].keys():
                losses = list(data[config][loss_func].values())
                if losses:
                    min_loss = min(losses)
                    max_loss = max(losses)
                    
                    # Avoid division by zero
                    if max_loss == min_loss:
                        normalized_data[config][loss_func] = data[config][loss_func].copy()
                    else:
                        for epoch, loss_value in data[config][loss_func].items():
                            normalized_value = (loss_value - min_loss) / (max_loss - min_loss)
                            normalized_data[config][loss_func][epoch] = normalized_value
        
        return dict(normalized_data)
    
    def print_model_loss_mapping(self, data_dir: str):
        """
        Print all matching models with their corresponding loss txt files to help clarify relationships.
        
        Args:
            data_dir (str): Directory containing model and loss files
        """
        print(f"\nModel-Loss File Mapping for directory: {data_dir}")
        print("=" * 80)
        
        # Get all files in the directory
        all_files = os.listdir(data_dir)
        
        # Separate model files (.pth) and loss files (.txt)
        model_files = [f for f in all_files if f.endswith('.pth')]
        loss_files = [f for f in all_files if f.endswith('.txt') and 'best_loss_epoch' in f]
        
        print(f"\nFound {len(model_files)} model files (.pth)")
        print(f"Found {len(loss_files)} loss files (.txt)")
        
        # Show all model files to understand the naming pattern
        print(f"\nAll model files found:")
        for model_file in sorted(model_files):
            print(f"  {model_file}")
        
        print(f"\nAll loss files found:")
        for loss_file in sorted(loss_files):
            print(f"  {loss_file}")
        
        # Parse loss files to extract information
        loss_info = {}
        for loss_file in loss_files:
            epoch, loss_func, config = self.parse_loss_filename(loss_file)
            if epoch and loss_func and config:
                key = (epoch, loss_func, config)
                loss_info[key] = loss_file
        
        # Group by configuration and loss function
        configs = {}
        for (epoch, loss_func, config), loss_file in loss_info.items():
            if config not in configs:
                configs[config] = {}
            if loss_func not in configs[config]:
                configs[config][loss_func] = []
            configs[config][loss_func].append((epoch, loss_file))
        
        # Print organized mapping
        for config in sorted(configs.keys()):
            print(f"\n{config} Configuration:")
            print("-" * 40)
            
            for loss_func in sorted(configs[config].keys()):
                print(f"\n  {loss_func} Loss:")
                epochs_and_files = sorted(configs[config][loss_func])
                
                for epoch, loss_file in epochs_and_files:
                    # Try to find corresponding model file
                    model_patterns = []
                    
                    if loss_func == 'L1':
                        if config == 'Unet':
                            model_patterns = [f"best_model_ZCB_9_Unet_epoch_{epoch}.pth"]
                        else:  # no_Unet
                            model_patterns = [f"best_model_ZCB_9_epoch_{epoch}.pth"]
                    elif loss_func == 'L2':
                        if config == 'Unet':
                            model_patterns = [
                                f"best_model_ZCB_9_32_Unet_epoch_{epoch}_L2.pth",
                                f"best_model_ZCB_9_Unet_epoch_{epoch}.pth"
                            ]
                        else:  # no_Unet
                            model_patterns = [
                                f"best_model_ZCB_9_32_epoch_{epoch}_L2.pth",
                                f"best_model_ZCB_9_epoch_{epoch}.pth"
                            ]
                    elif loss_func == 'pearson_loss':
                        if config == 'Unet':
                            model_patterns = [
                                f"best_model_ZCB_9_Unet_epoch_{epoch}_pearson_loss.pth",
                                f"best_model_ZCB_9_Unet_epoch_{epoch}.pth"
                            ]
                        else:  # no_Unet
                            model_patterns = [
                                f"best_model_ZCB_9_epoch_{epoch}_pearson_loss.pth",
                                f"best_model_ZCB_9_epoch_{epoch}.pth"
                            ]
                    
                    # Check which model files actually exist
                    existing_models = []
                    for pattern in model_patterns:
                        if pattern in model_files:
                            existing_models.append(pattern)
                    
                    print(f"    Epoch {epoch:3d}: {loss_file}")
                    if existing_models:
                        for model in existing_models:
                            print(f"             -> {model}")
                    else:
                        print(f"             -> No matching model file found")
                        print(f"             -> Looking for: {', '.join(model_patterns)}")
        
        print(f"\n" + "=" * 80)
    
    def plot_loss_comparison(self, data_dir: str,
                           expected_epochs: List[int] = None,
                           expected_losses: List[str] = None,
                           show_normalized: bool = True,
                           show_combined: bool = True,
                           figsize: Tuple[int, int] = (16, 12)) -> List[plt.Figure]:
        """
        Create comprehensive loss comparison plots.
        
        Args:
            data_dir (str): Directory containing loss files
            expected_epochs (List[int]): List of expected epochs
            expected_losses (List[str]): List of expected loss functions
            show_normalized (bool): Whether to show normalized plots
            show_combined (bool): Whether to show combined plots
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            List[plt.Figure]: List of created figures
        """
        if expected_epochs is None:
            expected_epochs = [2, 5, 10, 25, 50, 100, 150, 200, 250, 300, 400, 500]
        if expected_losses is None:
            expected_losses = ['L1', 'L2', 'pearson_loss']
        
        # Load loss data
        data = self.load_loss_data(data_dir, expected_epochs, expected_losses)
        
        if not data:
            print("No loss data found!")
            return []
        
        # Print model-loss file mapping
        self.print_model_loss_mapping(data_dir)
        
        figures = []
        
        # Create plots for each configuration (original data)
        for config in data.keys():
            print(f"\nCreating plot for {config}...")
            
            fig = plt.figure(figsize=figsize)
            
            # Define colors and markers based on configuration
            config_colors = {'Unet': 'blue', 'no_Unet': 'red', 'LatticeSC': 'blue', 'LatticeClathII': 'orange'}
            loss_markers = {'L1': 'o', 'L2': 's', 'pearson_loss': '^'}
            
            for loss_func in expected_losses:
                if loss_func in data[config]:
                    epochs = []
                    losses = []
                    
                    for epoch in expected_epochs:
                        if epoch in data[config][loss_func]:
                            epochs.append(epoch)
                            losses.append(data[config][loss_func][epoch])
                    
                    if epochs:  # Only plot if we have data
                        plt.plot(epochs, losses, 
                               color=config_colors.get(config, 'black'), 
                               marker=loss_markers[loss_func],
                               linewidth=2, 
                               markersize=12,
                               label=f'{loss_func} Loss',
                               alpha=0.8)
            
            plt.xlabel('Epoch', fontsize=20)
            plt.ylabel('Loss Value', fontsize=20)
            plt.title(f'Loss Comparison Across Different Loss Functions - {config}', fontsize=22)
            plt.legend(fontsize=18)
            plt.grid(True, alpha=0.3)
            plt.xscale('log')  # Log scale for epochs for better visualization
            
            # Set x-axis ticks to show actual epoch values
            plt.xticks(expected_epochs, expected_epochs, fontsize=18)
            plt.yticks(fontsize=18)
            
            plt.tight_layout()
            figures.append(fig)
        
        # Create normalized plots if requested
        if show_normalized:
            print("\nNormalizing losses using min-max normalization...")
            normalized_data = self.normalize_losses(data)
            
            for config in normalized_data.keys():
                print(f"\nCreating normalized plot for {config}...")
                
                fig = plt.figure(figsize=figsize)
                
                # Define colors and markers based on configuration
                config_colors = {'Unet': 'blue', 'no_Unet': 'red', 'LatticeSC': 'blue', 'LatticeClathII': 'orange'}
                loss_markers = {'L1': 'o', 'L2': 's', 'pearson_loss': '^'}
                
                for loss_func in expected_losses:
                    if loss_func in normalized_data[config]:
                        epochs = []
                        losses = []
                        
                        for epoch in expected_epochs:
                            if epoch in normalized_data[config][loss_func]:
                                epochs.append(epoch)
                                losses.append(normalized_data[config][loss_func][epoch])
                        
                        if epochs:  # Only plot if we have data
                            if loss_func == 'pearson_loss':
                                plt.plot(epochs, losses, 
                                        color=config_colors.get(config, 'black'), 
                                        marker=loss_markers[loss_func],
                                        linewidth=2, 
                                        markersize=12,
                                        label=f'NPCC Loss (Normalized)',
                                        alpha=0.8)
                            else:
                                plt.plot(epochs, losses, 
                                    color=config_colors.get(config, 'black'), 
                                    marker=loss_markers[loss_func],
                                    linewidth=2, 
                                    markersize=12,
                                    label=f'{loss_func} Loss (Normalized)',
                                    alpha=0.8)
                
                plt.xlabel('Epoch', fontsize=20)
                plt.ylabel('Normalized Loss Value (0-1)', fontsize=20)
                plt.title(f'Normalized Loss Comparison - {config}', fontsize=22)
                plt.legend(fontsize=18)
                plt.grid(True, alpha=0.3)
                plt.xscale('log')  # Log scale for epochs for better visualization
                plt.ylim(-0.05, 1.05)  # Set y-axis with buffer room for normalized data
                
                # Set x-axis ticks to show actual epoch values
                plt.xticks(expected_epochs, expected_epochs, fontsize=18)
                plt.yticks(fontsize=18)
                
                plt.tight_layout()
                figures.append(fig)
        
        # Create combined plots if requested and multiple configurations exist
        if show_combined and len(data) > 1:
            print(f"\nCreating combined plot for all configurations...")
            fig = plt.figure(figsize=(20, 14))
            
            # Define colors and markers for different configurations and loss functions
            config_colors = {
                'Unet': 'blue', 'no_Unet': 'red',  # For unet statuses
                'LatticeSC': 'blue', 'LatticeClathII': 'orange'  # For lattice types
            }
            loss_markers = {'L1': 'o', 'L2': 's', 'pearson_loss': '^'}
            
            for config in data.keys():
                for loss_func in expected_losses:
                    if loss_func in data[config]:
                        epochs = []
                        losses = []
                        
                        for epoch in expected_epochs:
                            if epoch in data[config][loss_func]:
                                epochs.append(epoch)
                                losses.append(data[config][loss_func][epoch])
                        
                        if epochs:  # Only plot if we have data
                            plt.plot(epochs, losses, 
                                   color=config_colors.get(config, 'black'), 
                                   marker=loss_markers[loss_func],
                                   linewidth=2, 
                                   markersize=12,
                                   label=f'{config} - {loss_func}',
                                   alpha=0.8)
            
            plt.xlabel('Epoch', fontsize=20)
            plt.ylabel('Loss Value', fontsize=20)
            plt.title('Combined Loss Comparison: All Configurations and Loss Functions', fontsize=22)
            plt.legend(fontsize=16, ncol=2)
            plt.grid(True, alpha=0.3)
            plt.xscale('log')  # Log scale for epochs for better visualization
            
            # Set x-axis ticks to show actual epoch values
            plt.xticks(expected_epochs, expected_epochs, fontsize=18)
            plt.yticks(fontsize=18)
            
            plt.tight_layout()
            figures.append(fig)
            
            # Create combined normalized plot
            if show_normalized:
                print(f"\nCreating combined normalized plot for all configurations...")
                fig = plt.figure(figsize=(20, 14))
                
                # Define colors and markers for different configurations and loss functions
                config_colors = {
                    'Unet': 'blue', 'no_Unet': 'red',  # For unet statuses
                    'LatticeSC': 'blue', 'LatticeClathII': 'orange'  # For lattice types
                }
                loss_markers = {'L1': 'o', 'L2': 's', 'pearson_loss': '^'}
                
                for config in normalized_data.keys():
                    for loss_func in expected_losses:
                        if loss_func in normalized_data[config]:
                            epochs = []
                            losses = []
                            
                            for epoch in expected_epochs:
                                if epoch in normalized_data[config][loss_func]:
                                    epochs.append(epoch)
                                    losses.append(normalized_data[config][loss_func][epoch])
                            
                            if epochs:  # Only plot if we have data
                                plt.plot(epochs, losses, 
                                       color=config_colors.get(config, 'black'), 
                                       marker=loss_markers[loss_func],
                                       linewidth=2, 
                                       markersize=12,
                                       label=f'{config} - {loss_func} (Normalized)',
                                       alpha=0.8)
                
                plt.xlabel('Epoch', fontsize=20)
                plt.ylabel('Normalized Loss Value (0-1)', fontsize=20)
                plt.title('Combined Normalized Loss Comparison: All Configurations and Loss Functions', fontsize=22)
                plt.legend(fontsize=16, ncol=2)
                plt.grid(True, alpha=0.3)
                plt.xscale('log')  # Log scale for epochs for better visualization
                plt.ylim(-0.05, 1.05)  # Set y-axis with buffer room for normalized data
                
                # Set x-axis ticks to show actual epoch values
                plt.xticks(expected_epochs, expected_epochs, fontsize=18)
                plt.yticks(fontsize=18)
                
                plt.tight_layout()
                figures.append(fig)
        
        # Print summary statistics
        print(f"\nSummary Statistics:")
        print("=" * 50)
        for config in data.keys():
            print(f"\n{config}:")
            for loss_func in expected_losses:
                if loss_func in data[config]:
                    losses = list(data[config][loss_func].values())
                    if losses:
                        print(f"  {loss_func} (Original): min={min(losses):.6f}, max={max(losses):.6f}, mean={np.mean(losses):.6f}")
                        
                if show_normalized and loss_func in normalized_data[config]:
                    norm_losses = list(normalized_data[config][loss_func].values())
                    if norm_losses:
                        print(f"  {loss_func} (Normalized): min={min(norm_losses):.6f}, max={max(norm_losses):.6f}, mean={np.mean(norm_losses):.6f}")
        
        return figures
