#%%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from typing import List, Dict, Tuple, Optional
import pandas as pd
import sys
import os
from matplotlib import colors
from skimage.metrics import structural_similarity as ssim
from scipy.signal import correlate2d, find_peaks
from scipy.ndimage import gaussian_filter
import random
import importlib
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../../NN/ptychosaxsNN/')))
import utils.ptychosaxsNN_utils as ptNN_U
importlib.reload(ptNN_U)
# Add the models directory to the path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../src/models/')))


def normalize_image(img: torch.Tensor) -> torch.Tensor:
    """
    Normalize image by subtracting background (minimum value) and scaling to [0, 1].
    
    Args:
        img (torch.Tensor): Input image
        
    Returns:
        torch.Tensor: Normalized image
    """
    # Convert to numpy for easier manipulation
    img_np = img.squeeze().cpu().numpy()
    
    # Subtract background (minimum value)
    img_np = img_np - np.min(img_np)
    
    # Scale to [0, 1]
    max_val = np.max(img_np)
    if max_val > 0:  # Avoid division by zero
        img_np = img_np / max_val
    
    return torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)

def calculate_normalized_cross_correlation(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate 2D normalized cross-correlation between two images.
    
    Args:
        img1 (np.ndarray): First image
        img2 (np.ndarray): Second image
        
    Returns:
        float: Maximum normalized cross-correlation value
    """
    # Normalize images
    img1_norm = (img1 - np.mean(img1)) / (np.std(img1) * len(img1.ravel()))
    img2_norm = (img2 - np.mean(img2)) / np.std(img2)
    
    # Calculate cross-correlation
    corr = correlate2d(img1_norm, img2_norm, mode='same')
    
    # Return maximum correlation value
    return np.max(corr)

# def find_peaks_and_fwhm(image: np.ndarray, threshold: float = 0.265, sigma: float = 0.714, distance: float = 2.0) -> Tuple[List[Tuple[float, float]], List[float]]:
#     """
#     Find peaks and their FWHM in a 2D image, including peaks at edges.
    
#     Args:
#         image (np.ndarray): Input image
#         threshold (float): Threshold for peak detection (relative to max)
#         sigma (float): Sigma for Gaussian smoothing
#         distance (float): Minimum distance between peaks (in pixels)
        
#     Returns:
#         Tuple[List[Tuple[float, float]], List[float]]: Peak positions and FWHM values
#     """
#     # Smooth the image
#     smoothed = gaussian_filter(image, sigma=sigma)
    
#     # Find peaks
#     peaks = []
#     fwhm_values = []
    
#     # Find local maxima
#     max_val = np.max(smoothed)
#     threshold_val = max_val * threshold
    
#     height, width = smoothed.shape
    
#     # Function to check if a point is a local maximum in its available neighborhood
#     def is_local_max(i: int, j: int) -> bool:
#         val = smoothed[i,j]
        
#         # Define the neighborhood bounds, accounting for edges
#         i_start = max(0, i-1)
#         i_end = min(height, i+2)
#         j_start = max(0, j-1)
#         j_end = min(width, j+2)
        
#         # Get the neighborhood
#         neighborhood = smoothed[i_start:i_end, j_start:j_end]
        
#         # For edge pixels, we only require them to be maximum in their partial neighborhood
#         return val >= np.max(neighborhood)
    
#     # Find peaks above threshold, including at edges
#     for i in range(height):
#         for j in range(width):
#             if smoothed[i,j] > threshold_val and is_local_max(i, j):
#                 peaks.append((i, j))
                
#                 # Calculate FWHM in x and y directions
#                 x_profile = smoothed[i,:]
#                 y_profile = smoothed[:,j]
#                 center_val = smoothed[i,j]
#                 half_max = center_val / 2
                
#                 try:
#                     # X direction FWHM
#                     x_above = x_profile > half_max
                    
#                     # Handle edge cases for X direction
#                     if j == 0 or j == width-1:
#                         # If peak is at edge, measure FWHM from the edge
#                         x_fwhm = 2 * np.sum(x_above)  # Double to account for assumed symmetry
#                     else:
#                         x_transitions = np.where(x_above[:-1] != x_above[1:])[0]
#                         if len(x_transitions) >= 2:
#                             x_fwhm = x_transitions[-1] - x_transitions[0]
#                         else:
#                             x_fwhm = np.sum(x_above)
                    
#                     # Y direction FWHM
#                     y_above = y_profile > half_max
                    
#                     # Handle edge cases for Y direction
#                     if i == 0 or i == height-1:
#                         # If peak is at edge, measure FWHM from the edge
#                         y_fwhm = 2 * np.sum(y_above)  # Double to account for assumed symmetry
#                     else:
#                         y_transitions = np.where(y_above[:-1] != y_above[1:])[0]
#                         if len(y_transitions) >= 2:
#                             y_fwhm = y_transitions[-1] - y_transitions[0]
#                         else:
#                             y_fwhm = np.sum(y_above)
                    
#                     # Use average of x and y FWHM
#                     fwhm_values.append((x_fwhm + y_fwhm) / 2)
                    
#                 except Exception:
#                     # Fallback method for FWHM calculation
#                     x_fwhm = np.sum(x_profile > half_max)
#                     y_fwhm = np.sum(y_profile > half_max)
#                     fwhm_values.append((x_fwhm + y_fwhm) / 2)
    
#     # # Filter peaks based on distance parameter
#     # if distance > 0 and len(peaks) > 1:
#     #     # Create list of (peak, fwhm, intensity) tuples
#     #     peak_data = [(peaks[i], fwhm_values[i], smoothed[peaks[i][0], peaks[i][1]]) 
#     #                  for i in range(len(peaks))]
        
#     #     # Sort by intensity (descending)
#     #     peak_data.sort(key=lambda x: x[2], reverse=True)
        
#     #     # Filter peaks based on distance
#     #     filtered_peaks = []
#     #     filtered_fwhm = []
        
#     #     for peak, fwhm, intensity in peak_data:
#     #         # Check if this peak is far enough from all already accepted peaks
#     #         if all(np.sqrt((peak[0] - p[0])**2 + (peak[1] - p[1])**2) >= distance 
#     #                for p in filtered_peaks):
#     #             filtered_peaks.append(peak)
#     #             filtered_fwhm.append(fwhm)
        
#     #     return filtered_peaks, filtered_fwhm
    
#     return peaks, fwhm_values


def find_peaks_and_fwhm(image: np.ndarray, threshold: float = 0.265, sigma: float = 0.714) -> Tuple[List[Tuple[float, float]], List[float]]:
    """
    Find peaks and their FWHM in a 2D image, including peaks at edges.
    
    Args:
        image (np.ndarray): Input image
        threshold (float): Threshold for peak detection (relative to max)
        sigma (float): Sigma for Gaussian smoothing
        
    Returns:
        Tuple[List[Tuple[float, float]], List[float]]: Peak positions and FWHM values
    """
    # Smooth the image
    smoothed = gaussian_filter(image, sigma=sigma)
    
    # Find peaks
    peaks = []
    fwhm_values = []
    
    # Find local maxima
    max_val = np.max(smoothed)
    threshold_val = max_val * threshold
    
    height, width = smoothed.shape
    
    # Function to check if a point is a local maximum in its available neighborhood
    def is_local_max(i: int, j: int) -> bool:
        val = smoothed[i,j]
        
        # Define the neighborhood bounds, accounting for edges
        i_start = max(0, i-1)
        i_end = min(height, i+2)
        j_start = max(0, j-1)
        j_end = min(width, j+2)
        
        # Get the neighborhood
        neighborhood = smoothed[i_start:i_end, j_start:j_end]
        
        # For edge pixels, we only require them to be maximum in their partial neighborhood
        return val >= np.max(neighborhood)
    
    # Find peaks above threshold, including at edges
    for i in range(height):
        for j in range(width):
            if smoothed[i,j] > threshold_val and is_local_max(i, j):
                peaks.append((i, j))
                
                # Calculate FWHM in x and y directions
                x_profile = smoothed[i,:]
                y_profile = smoothed[:,j]
                center_val = smoothed[i,j]
                half_max = center_val / 2
                
                try:
                    # X direction FWHM
                    x_above = x_profile > half_max
                    
                    # Handle edge cases for X direction
                    if j == 0 or j == width-1:
                        # If peak is at edge, measure FWHM from the edge
                        x_fwhm = 2 * np.sum(x_above)  # Double to account for assumed symmetry
                    else:
                        x_transitions = np.where(x_above[:-1] != x_above[1:])[0]
                        if len(x_transitions) >= 2:
                            x_fwhm = x_transitions[-1] - x_transitions[0]
                        else:
                            x_fwhm = np.sum(x_above)
                    
                    # Y direction FWHM
                    y_above = y_profile > half_max
                    
                    # Handle edge cases for Y direction
                    if i == 0 or i == height-1:
                        # If peak is at edge, measure FWHM from the edge
                        y_fwhm = 2 * np.sum(y_above)  # Double to account for assumed symmetry
                    else:
                        y_transitions = np.where(y_above[:-1] != y_above[1:])[0]
                        if len(y_transitions) >= 2:
                            y_fwhm = y_transitions[-1] - y_transitions[0]
                        else:
                            y_fwhm = np.sum(y_above)
                    
                    # Use average of x and y FWHM
                    fwhm_values.append((x_fwhm + y_fwhm) / 2)
                    
                except Exception:
                    # Fallback method for FWHM calculation
                    x_fwhm = np.sum(x_profile > half_max)
                    y_fwhm = np.sum(y_profile > half_max)
                    fwhm_values.append((x_fwhm + y_fwhm) / 2)
    
    return peaks, fwhm_values



def calculate_peak_sensitivity_metrics(img1: np.ndarray, 
                                  img2: np.ndarray,
                                  sigma_range: List[float] = [0.5, 1.0, 1.5, 2.0],
                                  threshold_range: List[float] = [0.1, 0.2, 0.3, 0.4],
                                  distance_threshold: float = 5.0) -> Dict[str, float]:
    """
    Calculate comprehensive peak detection metrics across different peak finder parameters.
    
    Args:
        img1 (np.ndarray): First image (model output)
        img2 (np.ndarray): Second image (ground truth)
        sigma_range (List[float]): Range of sigma values for Gaussian smoothing
        threshold_range (List[float]): Range of threshold values for peak detection
        distance_threshold (float): Maximum distance for peaks to be considered matched
        
    Returns:
        Dict[str, float]: Dictionary of peak sensitivity metrics
    """
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
    
    # Store results for each parameter combination
    results = []
    peak_positions_all = []
    peak_counts = []
    
    # Calculate ground truth peaks with middle parameters
    mid_sigma = np.median(sigma_range)
    mid_threshold = np.median(threshold_range)
    gt_peaks, gt_fwhm = find_peaks_and_fwhm(img2, threshold=mid_threshold, sigma=mid_sigma)
    
    # Test all parameter combinations
    for sigma in sigma_range:
        for threshold in threshold_range:
            # Find peaks with current parameters
            peaks, fwhm = find_peaks_and_fwhm(img1, threshold=threshold, sigma=sigma)
            peak_positions_all.extend(peaks)
            peak_counts.append(len(peaks))
            
            # Calculate matching metrics
            matched = 0
            false_positives = 0
            peak_intensities1 = []
            peak_intensities2 = []
            
            for peak in peaks:
                # Find closest ground truth peak
                min_dist = float('inf')
                closest_gt_peak = None
                
                for gt_peak in gt_peaks:
                    dist = np.sqrt((peak[0]-gt_peak[0])**2 + (peak[1]-gt_peak[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_gt_peak = gt_peak
                
                if min_dist <= distance_threshold:
                    matched += 1
                    peak_intensities1.append(img1[peak[0], peak[1]])
                    peak_intensities2.append(img2[closest_gt_peak[0], closest_gt_peak[1]])
                else:
                    false_positives += 1
            
            # Calculate F1 score
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
            
            # Update best parameters if this combination gives better F1 score
            if f1 > metrics['max_f1_score']:
                metrics['max_f1_score'] = f1
                metrics['optimal_sigma'] = sigma
                metrics['optimal_threshold'] = threshold
    
    # Calculate stability metrics
    if peak_positions_all:
        # Peak position stability (standard deviation of peak positions across parameters)
        peak_positions_array = np.array(peak_positions_all)
        metrics['peak_position_stability'] = 1.0 / (np.std(peak_positions_array[:, 0]) + 
                                                  np.std(peak_positions_array[:, 1]) + 1e-6)
        
        # Peak count stability (coefficient of variation of peak counts)
        metrics['peak_count_stability'] = 1.0 / (np.std(peak_counts) / np.mean(peak_counts) + 1e-6)
    
    # Calculate parameter sensitivity
    f1_scores = [r['f1_score'] for r in results]
    metrics['parameter_sensitivity'] = np.std(f1_scores) / (np.mean(f1_scores) + 1e-6)
    
    # Calculate average false positive and negative rates
    metrics['false_positive_rate'] = np.mean([r['false_positives'] / r['peak_count'] 
                                            if r['peak_count'] > 0 else 0 for r in results])
    metrics['false_negative_rate'] = np.mean([r['false_negatives'] / len(gt_peaks) 
                                            if gt_peaks else 0 for r in results])
    
    return metrics

def calculate_metrics(img1: torch.Tensor, 
                     img2: torch.Tensor,
                     calculate_psnr: bool = True,
                     calculate_ssim: bool = True,
                     calculate_xcorr: bool = False,
                     calculate_peaks: bool = True,
                     calculate_peak_sensitivity: bool = False,
                     peak_sigma: float = 0.714,
                     peak_threshold: float = 0.265) -> Dict[str, float]:
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
        peak_sigma (float): Sigma for Gaussian smoothing in peak detection (used if not optimizing)
        peak_threshold (float): Threshold for peak detection (used if not optimizing)
        
    Returns:
        Dict[str, float]: Dictionary of calculated metrics
    """
    # Normalize both images
    img1_norm = normalize_image(img1)
    img2_norm = normalize_image(img2)
    
    # Convert to numpy arrays
    img1_np = img1_norm.squeeze().cpu().numpy()
    img2_np = img2_norm.squeeze().cpu().numpy()
    
    metrics = {}
    
    # Calculate selected metrics
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
        metrics['xcorr'] = calculate_normalized_cross_correlation(img1_np, img2_np)
    
    # Determine optimal peak finding parameters if requested
    optimal_sigma = peak_sigma
    optimal_threshold = peak_threshold
    
    if calculate_peak_sensitivity:
        # Calculate peak sensitivity metrics to find optimal parameters
        sensitivity_metrics = calculate_peak_sensitivity_metrics(
            img1_np, 
            img2_np,
            sigma_range=[0.5, 0.714, 1.0, 1.5, 2.0],
            threshold_range=[0.1, 0.2, 0.265, 0.3, 0.4]
        )
        
        # Use the optimized parameters
        optimal_sigma = sensitivity_metrics['optimal_sigma']
        optimal_threshold = sensitivity_metrics['optimal_threshold']
        
        # Add sensitivity metrics to the output
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
        
        print('--------------------------------')
        print('Optimal sigma:' +str(metrics['optimal_sigma']))
        print('Optimal threshold: ' +str(metrics['optimal_threshold']))
        print('Max F1 score: ' +str(metrics['max_f1_score']))
        print('Peak position stability: ' +str(metrics['peak_position_stability']))
        print('Peak count stability: ' +str(metrics['peak_count_stability']))
        print('Parameter sensitivity: ' +str(metrics['parameter_sensitivity']))
        print('False positive rate: ' +str(metrics['false_positive_rate']))
        print('False negative rate: ' +str(metrics['false_negative_rate']))
    
    if calculate_peaks:
        # Find peaks in both images using optimal/specified parameters
        peaks1, fwhm1 = find_peaks_and_fwhm(img1_np, threshold=optimal_threshold, sigma=optimal_sigma)
        peaks2, fwhm2 = find_peaks_and_fwhm(img2_np, threshold=optimal_threshold, sigma=optimal_sigma)
        
        # Store the parameters actually used
        metrics['peak_sigma_used'] = optimal_sigma
        metrics['peak_threshold_used'] = optimal_threshold
        
        # Calculate peak position differences
        if peaks1 and peaks2:
            # Find closest matching peaks
            peak_diffs = []
            fwhm_diffs = []
            
            # For each peak in image 1, find closest peak in image 2
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
            if fwhm_diffs:  # Only calculate if we found matching peaks
                metrics['fwhm_diff'] = np.mean(fwhm_diffs)
    
    return metrics

def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio between two images.
    
    Args:
        img1 (torch.Tensor): First image
        img2 (torch.Tensor): Second image
        
    Returns:
        float: PSNR value in dB
    """
    # Convert to numpy arrays
    img1 = img1.squeeze().cpu().numpy()
    img2 = img2.squeeze().cpu().numpy()
    
    # Calculate MSE
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    # Calculate PSNR
    max_pixel = 1.0  # Assuming normalized images
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return psnr

def load_model(model_path: str, use_unet: bool = True) -> torch.nn.Module:
    """
    Load a trained model from the specified path.
    
    Args:
        model_path (str): Path to the saved model
        use_unet (bool): Whether to use Unet architecture
        
    Returns:
        torch.nn.Module: Loaded model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize appropriate model class
    if use_unet:
        from encoder1_old import recon_model
        model = recon_model()
        #from encoder1 import recon_model
        #model = recon_model()
    else:
        from encoder1_no_Unet_old import recon_model
        model = recon_model()
        #from encoder1_no_Unet import recon_model
        #model = recon_model()
    
    # Load model weights
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    return model


def get_model_paths_from_config(model_configs: Dict, base_path: str = "") -> List[Dict]:
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
        # Parse model name to get architecture and loss type
        parts = model_name.split('_')
        
        # Handle both old and new model path formats
        if 'Lattice' in model_name:
            # New format
            lattice_type = next(p for p in parts if 'Lattice' in p).replace('Lattice','')
            probe_size = next(p for p in parts if 'Probe' in p).split('x')[0].replace('Probe','')
            noise_status = next(p for p in parts if 'Noise' in p)
            use_unet = 'no_Unet' not in model_name
            loss_type = parts[-1]  # Last part contains loss type
        else:
            # Old format
            loss_type = parts[0]  # L1, L2 or pearson
            use_unet = 'no_Unet' not in model_name
            lattice_type = None
            probe_size = None
            noise_status = None
        
        for iteration in model_configs['iterations']:
            # Format path with iteration number
            full_path = Path(base_path) / model_path.format(iteration)
                
            if full_path.exists():
                info = {
                    'path': str(full_path),
                    'loss_type': loss_type,
                    'use_unet': use_unet,
                    'iterations': iteration
                }
                
                # Add additional metadata for new format
                if lattice_type:
                    info.update({
                        'lattice_type': lattice_type,
                        'probe_size': probe_size,
                        'noise_status': noise_status
                    })
                    
                model_info.append(info)
    
    return model_info

def create_comparison_grid_from_config(model_configs: Dict,
                                     input_data: torch.Tensor,
                                     ideal_data: torch.Tensor,
                                     base_path: str = "",
                                     figsize: Tuple[int, int] = (20, 15),
                                     calculate_psnr: bool = True,
                                     calculate_ssim: bool = True,
                                     calculate_xcorr: bool = False,
                                     calculate_peaks: bool = True,
                                     calculate_peak_sensitivity: bool = False,
                                     peak_sigma: float = 1.0) -> plt.Figure:
    """
    Create a grid plot comparing outputs from different models using model_configs.
    Shows input and ideal images above the model comparison grid.
    Includes selected metrics for each model output compared to the ideal image.
    
    Args:
        model_configs (Dict): Dictionary containing model configurations
        input_data (torch.Tensor): Input data to run through the models
        ideal_data (torch.Tensor): Ideal/ground truth data for comparison
        base_path (str): Base path where models are stored
        figsize (Tuple[int, int]): Figure size
        calculate_psnr (bool): Whether to calculate PSNR
        calculate_ssim (bool): Whether to calculate SSIM
        calculate_xcorr (bool): Whether to calculate cross-correlation
        calculate_peaks (bool): Whether to calculate peak metrics
        peak_sigma (float): Sigma for Gaussian smoothing in peak detection
        
    Returns:
        plt.Figure: The created figure
    """
    # Get model paths and metadata
    model_info = get_model_paths_from_config(model_configs, base_path)
    
    if not model_info:
        raise ValueError("No valid model paths found in the configuration")
    
    # Create DataFrame for easier organization
    df = pd.DataFrame(model_info)
    
    # Determine grid dimensions
    nrows = len(model_configs['iterations'])
    ncols = len(model_configs['models'])
    
    # Adjust figure size based on number of columns
    width_per_col = 6  # Increased from 5 to 6 inches per column
    height_per_row = 5  # Increased from 4 to 5 inches per row
    figsize = (width_per_col * ncols, height_per_row * (nrows + 1))
    
    # Create figure with extra row for input and ideal images
    fig = plt.figure(figsize=figsize)
    
    # Create a grid for the entire figure with adjusted spacing and height ratios
    gs = fig.add_gridspec(nrows + 1, ncols, 
                         height_ratios=[1.2] + [1]*nrows,  # Increased from 0.7 to 1.2 to make top row larger
                         hspace=0.3, wspace=0.3)
    
    # Plot input and ideal images in the first two columns of the top row
    ax_input = fig.add_subplot(gs[0, 0])
    ax_ideal = fig.add_subplot(gs[0, 1])
    
    # Plot input image
    im_input = ax_input.imshow(input_data.squeeze().cpu().numpy(), cmap='jet')
    ax_input.set_title('Input Image', fontsize=24)
    plt.colorbar(im_input, ax=ax_input, fraction=0.046, pad=0.04)
    ax_input.tick_params(axis='both', labelsize=20)
   
    # Plot ideal image
    im_ideal = ax_ideal.imshow(ideal_data.squeeze().cpu().numpy(), cmap='jet')
    ax_ideal.set_title('Ideal Image', fontsize=24)
    plt.colorbar(im_ideal, ax=ax_ideal, fraction=0.046, pad=0.04)
    ax_ideal.tick_params(axis='both', labelsize=20)
    
    # Find peaks in ideal image if peak calculation is enabled
    # These are calculated ONCE with consistent parameters and used for all models
    ideal_peaks = None
    
    if calculate_peaks:
        ideal_peaks, ideal_fwhm = find_peaks_and_fwhm(ideal_data.squeeze().cpu().numpy(), 
                                                       sigma=peak_sigma, 
                                                       threshold=0.265)
        # Plot peaks on ideal image
        for peak in ideal_peaks:
            ax_ideal.plot(peak[1], peak[0], 'g+', markersize=10, markeredgewidth=2)
    
    # Create axes for model outputs
    axes = np.zeros((nrows, ncols), dtype=object)
    for i in range(nrows):
        for j in range(ncols):
            axes[i, j] = fig.add_subplot(gs[i+1, j])
    
    # Sort by iterations and model name for consistent plotting
    df = df.sort_values(['iterations', 'loss_type'])
    
    # Plot each model's output
    for idx, (_, row) in enumerate(df.iterrows()):
        row_idx = model_configs['iterations'].index(row['iterations'])
        col_idx = list(model_configs['models'].keys()).index(
            f"{row['loss_type']}_{'Unet' if row['use_unet'] else 'no_Unet'}")
        
        print('\nModel path: ' +row['path'])
        
        # Load and run model
        model = load_model(row['path'], row['use_unet'])
        with torch.no_grad():
            output = model(input_data)
        
        # Calculate selected metrics
        metrics = calculate_metrics(output, ideal_data,
                                  calculate_psnr=calculate_psnr,
                                  calculate_ssim=calculate_ssim,
                                  calculate_xcorr=calculate_xcorr,
                                  calculate_peaks=calculate_peaks,
                                  calculate_peak_sensitivity=calculate_peak_sensitivity,
                                  peak_sigma=peak_sigma)
        
        # Extract parameters used for peak finding (will be optimized if sensitivity analysis was run)
        used_sigma = metrics.get('peak_sigma_used', peak_sigma)
        used_threshold = metrics.get('peak_threshold_used', 0.265)
        
        # Plot
        ax = axes[row_idx, col_idx]
        im = ax.imshow(output.squeeze().cpu().numpy(), cmap='jet')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.tick_params(axis='both', labelsize=16)
        
        # Plot peaks if peak calculation is enabled
        matched_peaks = 0
        if calculate_peaks and ideal_peaks:
            # Plot the SAME ideal peaks for all models (calculated once above)
            for peak in ideal_peaks:
                ax.plot(peak[1], peak[0], 'g+', markersize=10, markeredgewidth=2)
            
            # Find peaks in model output using the optimized parameters from calculate_metrics
            output_peaks, _ = find_peaks_and_fwhm(output.squeeze().cpu().numpy(), 
                                                   sigma=used_sigma, 
                                                   threshold=used_threshold)
            
            # Only plot output peaks that are close to ideal peaks
            for ideal_peak in ideal_peaks:
                min_dist = float('inf')
                closest_output_peak = None
                
                for output_peak in output_peaks:
                    dist = np.sqrt((ideal_peak[0]-output_peak[0])**2 + (ideal_peak[1]-output_peak[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_output_peak = output_peak
                
                # If we found a close peak (within 5 pixels), plot it
                if closest_output_peak is not None and min_dist < 5:
                    ax.plot(closest_output_peak[1], closest_output_peak[0], 'rx', markersize=8, markeredgewidth=2)
                    matched_peaks += 1
        
        # Build metrics text
        metrics_text = []
        if calculate_psnr:
            metrics_text.append(f'PSNR: {metrics["psnr"]:.2f} dB')
        if calculate_ssim:
            metrics_text.append(f'SSIM: {metrics["ssim"]:.4f}')
        if calculate_xcorr:
            metrics_text.append(f'XCORR: {metrics["xcorr"]:.4f}')
        if calculate_peaks and 'avg_peak_dist' in metrics:
            metrics_text.append(f'Peak Dist: {metrics["avg_peak_dist"]:.2f}')
            metrics_text.append(f'FWHM Diff: {metrics["fwhm_diff"]:.2f}')
            if ideal_peaks:
                metrics_text.append(f'Matched: {matched_peaks}/{len(ideal_peaks)}')
                #metrics_text.append(f'Sigma: {peak_sigma:.1f}')
        
        # Add metrics text in the corner
        ax.text(0.02, 0.98, '\n'.join(metrics_text),
                transform=ax.transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add row labels for iterations
        if col_idx == 0:
            ax.set_ylabel(f"{row['iterations']} epochs",fontsize=20)
        
        # Add column labels for model types
        if row_idx == 0:
            model_type = "Unet" if row['use_unet'] else "No Unet"
            ax.set_title(f"{row['loss_type']}\n{model_type}", fontsize=20)
    
    # Add legend for peak markers in a better position
    if calculate_peaks:
        legend_elements = [
            plt.Line2D([0], [0], marker='+', color='g', label='Ideal Peaks', markersize=8, linestyle='None'),
            plt.Line2D([0], [0], marker='x', color='r', label='Matched Model Peaks', markersize=8, linestyle='None')
        ]
        # Place legend in the empty space in the top row
        if len(model_configs['models']) > 2:
            ax_legend = fig.add_subplot(gs[0, 2:])
            ax_legend.axis('off')
            ax_legend.legend(handles=legend_elements, loc='center left', frameon=False, fontsize=36,markerscale=3)  # Increased fontsize from default to 14
    
    plt.tight_layout()
    return fig

def save_comparison_grid(fig: plt.Figure, save_path: str):
    """
    Save the comparison grid plot.
    
    Args:
        fig (plt.Figure): Figure to save
        save_path (str): Path where to save the figure
    """
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_metrics_vs_iteration(model_configs: Dict,
                             input_data: torch.Tensor,
                             ideal_data: torch.Tensor,
                             loss_type: str,
                             base_path: str = "",
                             figsize: Tuple[int, int] = (15, 10),
                             calculate_psnr: bool = True,
                             calculate_ssim: bool = True,
                             calculate_peaks: bool = True,
                             peak_sigma: float = 1.0,
                             compare_architectures: bool = True) -> plt.Figure:
    """
    Plot performance metrics vs iteration for a given loss function.
    Compares Unet and no_Unet architectures.
    
    Args:
        model_configs (Dict): Dictionary containing model configurations
        input_data (torch.Tensor): Input data to run through the models
        ideal_data (torch.Tensor): Ideal/ground truth data for comparison
        loss_type (str): Loss function type ('L1', 'L2', or 'pearson')
        base_path (str): Base path where models are stored
        figsize (Tuple[int, int]): Figure size
        calculate_psnr (bool): Whether to calculate PSNR
        calculate_ssim (bool): Whether to calculate SSIM
        calculate_peaks (bool): Whether to calculate peak metrics
        peak_sigma (float): Sigma for Gaussian smoothing in peak detection
        compare_architectures (bool): If True, plot both Unet and no_Unet. If False, only plot Unet.
        
    Returns:
        plt.Figure: The created figure
    """
    # Initialize storage for metrics
    iterations = model_configs['iterations']
    metrics_data = {
        'Unet': {
            'psnr': [],
            'ssim': [],
            'avg_peak_dist': [],
            'fwhm_diff': [],
            'matched_peaks': [],
            'total_peaks': []
        },
        'no_Unet': {
            'psnr': [],
            'ssim': [],
            'avg_peak_dist': [],
            'fwhm_diff': [],
            'matched_peaks': [],
            'total_peaks': []
        }
    }
    
    # Get ideal peaks if calculating peak metrics
    # These are calculated ONCE with consistent parameters and used for all models
    ideal_peaks = None
    if calculate_peaks:
        ideal_peaks, ideal_fwhm = find_peaks_and_fwhm(ideal_data.squeeze().cpu().numpy(), sigma=peak_sigma)
    
    # Process each architecture type
    architectures = ['Unet', 'no_Unet'] if compare_architectures else ['Unet']
    
    for arch in architectures:
        model_key = f"{loss_type}_{arch}"
        
        if model_key not in model_configs['models']:
            print(f"Warning: {model_key} not found in model configs")
            continue
        
        for iteration in iterations:
            # Construct model path
            model_path = Path(base_path) / model_configs['models'][model_key].format(iteration)
            
            if not model_path.exists():
                print(f"Warning: Model not found: {model_path}")
                metrics_data[arch]['psnr'].append(np.nan)
                metrics_data[arch]['ssim'].append(np.nan)
                metrics_data[arch]['avg_peak_dist'].append(np.nan)
                metrics_data[arch]['fwhm_diff'].append(np.nan)
                metrics_data[arch]['matched_peaks'].append(np.nan)
                metrics_data[arch]['total_peaks'].append(np.nan)
                continue
            
            # Load and run model
            use_unet = (arch == 'Unet')
            model = load_model(str(model_path), use_unet)
            
            with torch.no_grad():
                output = model(input_data)
            
            # Calculate metrics
            metrics = calculate_metrics(output, ideal_data,
                                      calculate_psnr=calculate_psnr,
                                      calculate_ssim=calculate_ssim,
                                      calculate_xcorr=False,
                                      calculate_peaks=calculate_peaks,
                                      calculate_peak_sensitivity=calculate_peak_sensitivity,
                                      peak_sigma=peak_sigma)
            
            # Store metrics
            metrics_data[arch]['psnr'].append(metrics.get('psnr', np.nan))
            metrics_data[arch]['ssim'].append(metrics.get('ssim', np.nan))
            metrics_data[arch]['avg_peak_dist'].append(metrics.get('avg_peak_dist', np.nan))
            metrics_data[arch]['fwhm_diff'].append(metrics.get('fwhm_diff', np.nan))
            
            # Calculate matched peaks
            if calculate_peaks and ideal_peaks:
                # Use the optimized parameters for model output peak detection
                used_sigma = metrics.get('peak_sigma_used', peak_sigma)
                used_threshold = metrics.get('peak_threshold_used', 0.265)
                
                # Use the SAME ideal peaks for all models (calculated once above)
                output_peaks, _ = find_peaks_and_fwhm(output.squeeze().cpu().numpy(), 
                                                       sigma=used_sigma, 
                                                       threshold=used_threshold)
                matched = 0
                for ideal_peak in ideal_peaks:
                    min_dist = float('inf')
                    for output_peak in output_peaks:
                        dist = np.sqrt((ideal_peak[0]-output_peak[0])**2 + (ideal_peak[1]-output_peak[1])**2)
                        min_dist = min(min_dist, dist)
                    if min_dist < 5:
                        matched += 1
                metrics_data[arch]['matched_peaks'].append(matched)
                metrics_data[arch]['total_peaks'].append(len(ideal_peaks))
            else:
                metrics_data[arch]['matched_peaks'].append(np.nan)
                metrics_data[arch]['total_peaks'].append(np.nan)
    
    # Create subplots
    num_plots = sum([calculate_psnr, calculate_ssim])
    if calculate_peaks and ideal_peaks:
        num_plots += 3  # Add three plots: peak distance, FWHM, and matched peaks
    
    fig, axes = plt.subplots(num_plots, 1, figsize=figsize, sharex=True)
    if num_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Track if we've added the legend
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
    
    # Add x-axis label only to the last plot
    axes[-1].set_xlabel('Epochs', fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    return fig

def save_metrics_plot(fig: plt.Figure, save_path: str):
    """
    Save the metrics vs iteration plot.
    
    Args:
        fig (plt.Figure): Figure to save
        save_path (str): Path where to save the figure
    """
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_metrics_vs_iteration_averaged(model_configs: Dict,
                                       indices_list: List[Tuple[int, int, int]],
                                       loss_types: List[str],
                                       base_path: str = "",
                                       mask_path: str = '/home/beams/PTYCHOSAXS/deconvolutionNN/data/mask/mask_ZCB_9_3D.npy',
                                       data_path: str = '/net/micdata/data2/12IDC/ptychosaxs/data/diff_sim/lattice_ls400_gs1024_lsp6_r3.0_typeSC',
                                       figsize: Tuple[int, int] = (15, 30),
                                       calculate_psnr: bool = True,
                                       calculate_ssim: bool = True,
                                       calculate_peaks: bool = True,
                                       calculate_peak_sensitivity: bool = False,
                                       peak_sigma: float = 0.714,
                                       central_only: bool = True,
                                       compare_architectures: bool = False) -> plt.Figure:
    """
    Plot performance metrics vs iteration comparing different loss functions,
    averaged over multiple diffraction patterns.
    
    Args:
        model_configs (Dict): Dictionary containing model configurations
        indices_list (List[Tuple[int, int, int]]): List of (hr, kr, lr) indices
        loss_types (List[str]): List of loss function types to compare
        base_path (str): Base path where models are stored
        mask_path (str): Path to the mask file
        data_path (str): Path to the data directory
        figsize (Tuple[int, int]): Figure size
        calculate_psnr (bool): Whether to calculate PSNR
        calculate_ssim (bool): Whether to calculate SSIM
        calculate_peaks (bool): Whether to calculate peak metrics
        calculate_peak_sensitivity (bool): Whether to calculate peak sensitivity metrics for optimizing parameters
        peak_sigma (float): Sigma for Gaussian smoothing in peak detection
        central_only (bool): If True, only process central pattern (num=5)
        compare_architectures (bool): If True, plot both Unet and no_Unet
        
    Returns:
        plt.Figure: The created figure
    """
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../../NN/ptychosaxsNN/')))
    import utils.ptychosaxsNN_utils as ptNN_U
    
    # Load mask
    mask = np.load(mask_path)
    
    # Initialize storage for metrics
    iterations = model_configs['iterations']
    architectures = ['Unet', 'no_Unet'] if compare_architectures else ['Unet']
    
    # Initialize metrics storage: metrics_data[loss_type_arch][iteration_idx] = list of values
    metrics_data = {}
    for loss_type in loss_types:
        for arch in architectures:
            key = f"{loss_type}_{arch}"
            metrics_data[key] = {
                'psnr': [[] for _ in iterations],
                'ssim': [[] for _ in iterations],
                'avg_peak_dist': [[] for _ in iterations],
                'fwhm_diff': [[] for _ in iterations],
                'matched_peaks': [[] for _ in iterations],
                'total_peaks': [[] for _ in iterations]
            }
    
    # Process each pattern
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for hr, kr, lr in tqdm(indices_list, desc="Processing patterns"):
        # Define which pattern numbers to process
        pattern_nums = [5] if central_only else range(1, 10)
        
        for num in tqdm(pattern_nums):
            # Load and preprocess data
            pattern_file = f'output_hanning_conv_{hr}_{kr}_{lr}_0000{num}.npz'
            try:
                data = np.load(f'{data_path}/{pattern_file}')
            except FileNotFoundError:
                print(f"Warning: File not found: {pattern_file}")
                continue
            
            dp_pp, _, _ = ptNN_U.preprocess_ZCB_9(data['convDP'], mask)
            dp_pp_IDEAL, _, _ = ptNN_U.preprocess_ZCB_9(data['pinholeDP_extra_conv'], 
                                                         mask=np.ones(dp_pp[0][0].shape))
            
            dp_pp = dp_pp.to(device=device, dtype=torch.float)
            dp_pp_IDEAL = dp_pp_IDEAL.to(device=device, dtype=torch.float)
            
            # Get initial ideal peaks if calculating peak metrics
            # (These will be re-found with optimal parameters for each model)
            ideal_peaks = None
            if calculate_peaks:
                ideal_peaks, _ = find_peaks_and_fwhm(dp_pp_IDEAL.squeeze().cpu().numpy(), 
                                                              sigma=peak_sigma)
            
            # Process each loss type, architecture, and iteration
            for iter_idx, iteration in enumerate(iterations):
                for loss_type in loss_types:
                    for arch in architectures:
                        model_key = f"{loss_type}_{arch}"
                        data_key = f"{loss_type}_{arch}"
                        
                        if model_key not in model_configs['models']:
                            continue
                        
                        # Construct model path
                        model_path = Path(base_path) / model_configs['models'][model_key].format(iteration)
                        
                        if not model_path.exists():
                            continue
                        
                        # Load and run model
                        use_unet = (arch == 'Unet')
                        model = load_model(str(model_path), use_unet)
                        
                        with torch.no_grad():
                            output = model(dp_pp)
                        
                        # Calculate metrics
                        metrics = calculate_metrics(output, dp_pp_IDEAL,
                                                  calculate_psnr=calculate_psnr,
                                                  calculate_ssim=calculate_ssim,
                                                  calculate_xcorr=False,
                                                  calculate_peaks=calculate_peaks,
                                                  calculate_peak_sensitivity=calculate_peak_sensitivity,
                                                  peak_sigma=peak_sigma)
                        
                        # Store metrics
                        if calculate_psnr and 'psnr' in metrics:
                            metrics_data[data_key]['psnr'][iter_idx].append(metrics['psnr'])
                        if calculate_ssim and 'ssim' in metrics:
                            metrics_data[data_key]['ssim'][iter_idx].append(metrics['ssim'])
                        
                        if calculate_peaks and ideal_peaks:
                            if 'avg_peak_dist' in metrics:
                                metrics_data[data_key]['avg_peak_dist'][iter_idx].append(metrics['avg_peak_dist'])
                            if 'fwhm_diff' in metrics:
                                metrics_data[data_key]['fwhm_diff'][iter_idx].append(metrics['fwhm_diff'])
                            
                            # Use the same parameters that were used in calculate_metrics
                            used_sigma = metrics.get('peak_sigma_used', peak_sigma)
                            used_threshold = metrics.get('peak_threshold_used', 0.265)
                            
                            # Use the SAME ideal peaks for all models (calculated once above)
                            # Calculate matched peaks
                            output_peaks, _ = find_peaks_and_fwhm(output.squeeze().cpu().numpy(), 
                                                                  sigma=used_sigma, 
                                                                  threshold=used_threshold)
                            matched = 0
                            for ideal_peak in ideal_peaks:
                                min_dist = float('inf')
                                for output_peak in output_peaks:
                                    dist = np.sqrt((ideal_peak[0]-output_peak[0])**2 + 
                                                 (ideal_peak[1]-output_peak[1])**2)
                                    min_dist = min(min_dist, dist)
                                if min_dist < 5:
                                    matched += 1
                            
                            metrics_data[data_key]['matched_peaks'][iter_idx].append(matched)
                            metrics_data[data_key]['total_peaks'][iter_idx].append(len(ideal_peaks))
    
    # Average metrics across all patterns
    averaged_metrics = {}
    for key in metrics_data:
        averaged_metrics[key] = {
            'psnr': [],
            'ssim': [],
            'avg_peak_dist': [],
            'fwhm_diff': [],
            'matched_peaks': [],
            'total_peaks': []
        }
        
        for iter_idx in range(len(iterations)):
            for metric in ['psnr', 'ssim', 'avg_peak_dist', 'fwhm_diff']:
                values = metrics_data[key][metric][iter_idx]
                avg = np.mean(values) if values else np.nan
                averaged_metrics[key][metric].append(avg)
            
            # For matched peaks, calculate percentage
            matched_list = metrics_data[key]['matched_peaks'][iter_idx]
            total_list = metrics_data[key]['total_peaks'][iter_idx]
            if matched_list and total_list:
                total_matched = sum(matched_list)
                total_peaks = sum(total_list)
                averaged_metrics[key]['matched_peaks'].append(total_matched)
                averaged_metrics[key]['total_peaks'].append(total_peaks)
            else:
                averaged_metrics[key]['matched_peaks'].append(0)
                averaged_metrics[key]['total_peaks'].append(1)  # Avoid division by zero
    
    # Now create the plots using averaged data
    num_plots = sum([calculate_psnr, calculate_ssim])
    if calculate_peaks:
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
            display_name = 'NPCC' if loss_type == 'pearson' else loss_type
            data_key_unet = f"{loss_type}_Unet"
            line, = ax.plot(iterations, averaged_metrics[data_key_unet]['psnr'], 'o-', 
                          label=f'{display_name} (Unet)', linewidth=2, markersize=8)
            
            if compare_architectures:
                data_key_no_unet = f"{loss_type}_no_Unet"
                ax.plot(iterations, averaged_metrics[data_key_no_unet]['psnr'], 's--', 
                       label=f'{display_name} (no Unet)', linewidth=2, markersize=7, 
                       color=line.get_color())
        
        ax.set_ylabel('PSNR (dB)', fontsize=18, fontweight='bold')
        ax.set_title(f'PSNR vs Epochs (Averaged over {len(indices_list)} patterns)', fontsize=20, fontweight='bold')
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
            display_name = 'NPCC' if loss_type == 'pearson' else loss_type
            data_key_unet = f"{loss_type}_Unet"
            line, = ax.plot(iterations, averaged_metrics[data_key_unet]['ssim'], 'o-', 
                          label=f'{display_name} (Unet)', linewidth=2, markersize=8)
            
            if compare_architectures:
                data_key_no_unet = f"{loss_type}_no_Unet"
                ax.plot(iterations, averaged_metrics[data_key_no_unet]['ssim'], 's--', 
                       label=f'{display_name} (no Unet)', linewidth=2, markersize=7, 
                       color=line.get_color())
        
        ax.set_ylabel('SSIM', fontsize=18, fontweight='bold')
        ax.set_title(f'SSIM vs Epochs (Averaged over {len(indices_list)} patterns)', fontsize=20, fontweight='bold')
        if not legend_added:
            ax.legend(fontsize=20, loc='center right', framealpha=0.9)
            legend_added = True
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=16)
        plot_idx += 1
    
    # Plot peak metrics
    if calculate_peaks:
        # Average peak distance
        ax = axes[plot_idx]
        for loss_type in loss_types:
            display_name = 'NPCC' if loss_type == 'pearson' else loss_type
            data_key_unet = f"{loss_type}_Unet"
            line, = ax.plot(iterations, averaged_metrics[data_key_unet]['avg_peak_dist'], 'o-', 
                          label=f'{display_name} (Unet)', linewidth=2, markersize=8)
            
            if compare_architectures:
                data_key_no_unet = f"{loss_type}_no_Unet"
                ax.plot(iterations, averaged_metrics[data_key_no_unet]['avg_peak_dist'], 's--', 
                       label=f'{display_name} (no Unet)', linewidth=2, markersize=7, 
                       color=line.get_color())
        
        ax.set_ylabel('Average Peak Distance (pixels)', fontsize=18, fontweight='bold')
        ax.set_title(f'Peak Distance vs Epochs (Averaged over {len(indices_list)} patterns)', fontsize=20, fontweight='bold')
        if not legend_added:
            ax.legend(fontsize=20, loc='center right', framealpha=0.9)
            legend_added = True
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=16)
        plot_idx += 1
        
        # FWHM difference
        ax = axes[plot_idx]
        for loss_type in loss_types:
            display_name = 'NPCC' if loss_type == 'pearson' else loss_type
            data_key_unet = f"{loss_type}_Unet"
            line, = ax.plot(iterations, averaged_metrics[data_key_unet]['fwhm_diff'], 'o-', 
                          label=f'{display_name} (Unet)', linewidth=2, markersize=8)
            
            if compare_architectures:
                data_key_no_unet = f"{loss_type}_no_Unet"
                ax.plot(iterations, averaged_metrics[data_key_no_unet]['fwhm_diff'], 's--', 
                       label=f'{display_name} (no Unet)', linewidth=2, markersize=7, 
                       color=line.get_color())
        
        ax.set_ylabel('FWHM Difference (pixels)', fontsize=18, fontweight='bold')
        ax.set_title(f'FWHM Difference vs Epochs (Averaged over {len(indices_list)} patterns)', fontsize=20, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=16)
        plot_idx += 1
        
        # Matched peaks (as percentage)
        ax = axes[plot_idx]
        for loss_type in loss_types:
            display_name = 'NPCC' if loss_type == 'pearson' else loss_type
            data_key_unet = f"{loss_type}_Unet"
            matched_ratio_unet = [m/t*100 if t > 0 else np.nan 
                                 for m, t in zip(averaged_metrics[data_key_unet]['matched_peaks'], 
                                               averaged_metrics[data_key_unet]['total_peaks'])]
            line, = ax.plot(iterations, matched_ratio_unet, 'o-', 
                          label=f'{display_name} (Unet)', linewidth=2, markersize=8)
            
            if compare_architectures:
                data_key_no_unet = f"{loss_type}_no_Unet"
                matched_ratio_no_unet = [m/t*100 if t > 0 else np.nan 
                                        for m, t in zip(averaged_metrics[data_key_no_unet]['matched_peaks'], 
                                                      averaged_metrics[data_key_no_unet]['total_peaks'])]
                ax.plot(iterations, matched_ratio_no_unet, 's--', 
                       label=f'{display_name} (no Unet)', linewidth=2, markersize=7, 
                       color=line.get_color())
        
        ax.set_ylabel('Matched Peaks (%)', fontsize=18, fontweight='bold')
        ax.set_title(f'Peak Detection Rate vs Epochs (Averaged over {len(indices_list)} patterns)', fontsize=20, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=16)
        ax.set_ylim([0, 105])
        plot_idx += 1
    
    # Add x-axis label only to the last plot
    axes[-1].set_xlabel('Epochs', fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_metrics_vs_iteration_compare_loss(model_configs: Dict,
                                          input_data: torch.Tensor,
                                          ideal_data: torch.Tensor,
                                          loss_types: List[str],
                                          base_path: str = "",
                                          figsize: Tuple[int, int] = (15, 10),
                                          calculate_psnr: bool = True,
                                          calculate_ssim: bool = True,
                                          calculate_peaks: bool = True,
                                          calculate_peak_sensitivity: bool = False,
                                          peak_sigma: float = 1.0,
                                          compare_architectures: bool = False) -> plt.Figure:
    """
    Plot performance metrics vs iteration comparing different loss functions.
    
    Args:
        model_configs (Dict): Dictionary containing model configurations
        input_data (torch.Tensor): Input data to run through the models
        ideal_data (torch.Tensor): Ideal/ground truth data for comparison
        loss_types (List[str]): List of loss function types to compare (e.g., ['L1', 'L2', 'pearson'])
        base_path (str): Base path where models are stored
        figsize (Tuple[int, int]): Figure size
        calculate_psnr (bool): Whether to calculate PSNR
        calculate_ssim (bool): Whether to calculate SSIM
        calculate_peaks (bool): Whether to calculate peak metrics
        calculate_peak_sensitivity (bool): Whether to calculate peak sensitivity metrics for optimizing parameters
        peak_sigma (float): Sigma for Gaussian smoothing in peak detection
        compare_architectures (bool): If True, plot both Unet and no_Unet. If False, only plot Unet.
        
    Returns:
        plt.Figure: The created figure
    """
    # Initialize storage for metrics
    iterations = model_configs['iterations']
    metrics_data = {}
    
    # Determine which architectures to process
    architectures = ['Unet', 'no_Unet'] if compare_architectures else ['Unet']
    
    for loss_type in loss_types:
        for arch in architectures:
            key = f"{loss_type}_{arch}"
            metrics_data[key] = {
                'psnr': [],
                'ssim': [],
                'avg_peak_dist': [],
                'fwhm_diff': [],
                'matched_peaks': [],
                'total_peaks': []
            }
    
    # Get initial ideal peaks if calculating peak metrics
    # (These will be re-found with optimal parameters for each model)
    ideal_peaks = None
    if calculate_peaks:
        ideal_peaks, ideal_fwhm = find_peaks_and_fwhm(ideal_data.squeeze().cpu().numpy(), sigma=peak_sigma)
    
    # Process each loss type and architecture
    for loss_type in loss_types:
        for arch in architectures:
            model_key = f"{loss_type}_{arch}"
            data_key = f"{loss_type}_{arch}"
            
            if model_key not in model_configs['models']:
                print(f"Warning: {model_key} not found in model configs")
                continue
            
            use_unet = (arch == 'Unet')
            
            for iteration in iterations:
                # Construct model path
                model_path = Path(base_path) / model_configs['models'][model_key].format(iteration)
                
                if not model_path.exists():
                    print(f"Warning: Model not found: {model_path}")
                    metrics_data[data_key]['psnr'].append(np.nan)
                    metrics_data[data_key]['ssim'].append(np.nan)
                    metrics_data[data_key]['avg_peak_dist'].append(np.nan)
                    metrics_data[data_key]['fwhm_diff'].append(np.nan)
                    metrics_data[data_key]['matched_peaks'].append(np.nan)
                    metrics_data[data_key]['total_peaks'].append(np.nan)
                    continue
                
                # Load and run model
                model = load_model(str(model_path), use_unet)
                
                with torch.no_grad():
                    output = model(input_data)
                
                # Calculate metrics
                metrics = calculate_metrics(output, ideal_data,
                                          calculate_psnr=calculate_psnr,
                                          calculate_ssim=calculate_ssim,
                                          calculate_xcorr=False,
                                          calculate_peaks=calculate_peaks,
                                          calculate_peak_sensitivity=calculate_peak_sensitivity,
                                          peak_sigma=peak_sigma)
                
                # Store metrics
                metrics_data[data_key]['psnr'].append(metrics.get('psnr', np.nan))
                metrics_data[data_key]['ssim'].append(metrics.get('ssim', np.nan))
                metrics_data[data_key]['avg_peak_dist'].append(metrics.get('avg_peak_dist', np.nan))
                metrics_data[data_key]['fwhm_diff'].append(metrics.get('fwhm_diff', np.nan))
                
                # Calculate matched peaks
                if calculate_peaks and ideal_peaks:
                    # Use the same parameters that were used in calculate_metrics
                    used_sigma = metrics.get('peak_sigma_used', peak_sigma)
                    used_threshold = metrics.get('peak_threshold_used', 0.265)
                    
                    # Use the SAME ideal peaks for all models (calculated once above)
                    output_peaks, _ = find_peaks_and_fwhm(output.squeeze().cpu().numpy(), 
                                                           sigma=used_sigma, 
                                                           threshold=used_threshold)
                    matched = 0
                    for ideal_peak in ideal_peaks:
                        min_dist = float('inf')
                        for output_peak in output_peaks:
                            dist = np.sqrt((ideal_peak[0]-output_peak[0])**2 + (ideal_peak[1]-output_peak[1])**2)
                            min_dist = min(min_dist, dist)
                        if min_dist < 5:
                            matched += 1
                    metrics_data[data_key]['matched_peaks'].append(matched)
                    metrics_data[data_key]['total_peaks'].append(len(ideal_peaks))
                else:
                    metrics_data[data_key]['matched_peaks'].append(np.nan)
                    metrics_data[data_key]['total_peaks'].append(np.nan)
    
    # Create subplots
    num_plots = sum([calculate_psnr, calculate_ssim])
    if calculate_peaks and ideal_peaks:
        num_plots += 3  # Add three plots: peak distance, FWHM, and matched peaks
    
    fig, axes = plt.subplots(num_plots, 1, figsize=figsize, sharex=True)
    if num_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Track if we've added the legend
    legend_added = False
    
    # Plot PSNR
    if calculate_psnr:
        ax = axes[plot_idx]
        for loss_type in loss_types:
            # Plot Unet with solid line and circle markers
            data_key_unet = f"{loss_type}_Unet"
            display_name = 'NPCC' if loss_type == 'pearson' else loss_type
            line, = ax.plot(iterations, metrics_data[data_key_unet]['psnr'], 'o-', 
                          label=f'{display_name} (Unet)', linewidth=2, markersize=8)
            
            # Plot no_Unet with dashed line and square markers, same color
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
            # Plot Unet with solid line and circle markers
            data_key_unet = f"{loss_type}_Unet"
            display_name = 'NPCC' if loss_type == 'pearson' else loss_type
            line, = ax.plot(iterations, metrics_data[data_key_unet]['ssim'], 'o-', 
                          label=f'{display_name} (Unet)', linewidth=2, markersize=8)
            
            # Plot no_Unet with dashed line and square markers, same color
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
            # Plot Unet with solid line and circle markers
            data_key_unet = f"{loss_type}_Unet"
            display_name = 'NPCC' if loss_type == 'pearson' else loss_type
            line, = ax.plot(iterations, metrics_data[data_key_unet]['avg_peak_dist'], 'o-', 
                          label=f'{display_name} (Unet)', linewidth=2, markersize=8)
            
            # Plot no_Unet with dashed line and square markers, same color
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
            # Plot Unet with solid line and circle markers
            data_key_unet = f"{loss_type}_Unet"
            display_name = 'NPCC' if loss_type == 'pearson' else loss_type
            line, = ax.plot(iterations, metrics_data[data_key_unet]['fwhm_diff'], 'o-', 
                          label=f'{display_name} (Unet)', linewidth=2, markersize=8)
            
            # Plot no_Unet with dashed line and square markers, same color
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
            # Plot Unet with solid line and circle markers
            data_key_unet = f"{loss_type}_Unet"
            display_name = 'NPCC' if loss_type == 'pearson' else loss_type
            matched_ratio_unet = [m/t*100 if not np.isnan(m) and t > 0 else np.nan 
                                 for m, t in zip(metrics_data[data_key_unet]['matched_peaks'], 
                                               metrics_data[data_key_unet]['total_peaks'])]
            line, = ax.plot(iterations, matched_ratio_unet, 'o-', 
                          label=f'{display_name} (Unet)', linewidth=2, markersize=8)
            
            # Plot no_Unet with dashed line and square markers, same color
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
    
    # Add x-axis label only to the last plot
    axes[-1].set_xlabel('Epochs', fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    return fig

def calculate_cumulative_stats(model_configs: Dict,
                           indices_list: List[Tuple[int, int, int]],
                           base_path: str = "",
                           mask_path: str = '/home/beams/PTYCHOSAXS/deconvolutionNN/data/mask/mask_ZCB_9_3D.npy',
                           data_path: str = '/net/micdata/data2/12IDC/ptychosaxs/data/diff_sim/lattice_ls400_gs1024_lsp6_r3.0_typeSC',
                           calculate_psnr: bool = True,
                           calculate_ssim: bool = True,
                           calculate_xcorr: bool = False,
                           calculate_peaks: bool = True,
                           calculate_peak_sensitivity: bool = False,
                           peak_sigma: float = 1.0,
                           central_only: bool = True) -> Dict[str, Dict[str, float]]:
    """
    Calculate cumulative statistics across multiple input patterns.
    
    Args:
        model_configs (Dict): Dictionary containing model configurations
        indices_list (List[Tuple[int, int, int]]): List of (hr, kr, lr) indices
        base_path (str): Base path where models are stored
        mask_path (str): Path to the mask file
        data_path (str): Path to the data directory
        calculate_psnr (bool): Whether to calculate PSNR
        calculate_ssim (bool): Whether to calculate SSIM
        calculate_xcorr (bool): Whether to calculate cross-correlation
        calculate_peaks (bool): Whether to calculate peak metrics
        calculate_peak_sensitivity (bool): Whether to calculate peak sensitivity metrics
        peak_sigma (float): Sigma for Gaussian smoothing in peak detection
        central_only (bool): If True, only process central pattern (num=5), 
                           if False, process all patterns (num=1-9)
        
    Returns:
        Dict[str, Dict[str, float]]: Dictionary of cumulative statistics per model
    """
    # Load mask
    mask = np.load(mask_path)
    
    # Initialize statistics dictionary
    stats = {}
    
    # Get model paths
    model_info = get_model_paths_from_config(model_configs, base_path)
    if not model_info:
        raise ValueError("No valid model paths found in the configuration")
    
    # Create DataFrame for easier organization
    df = pd.DataFrame(model_info)
    df = df.sort_values(['iterations', 'loss_type'])
    
    # Initialize metrics for each model
    for _, row in df.iterrows():
        model_key = f"{row['loss_type']}_{'Unet' if row['use_unet'] else 'no_Unet'}_{row['iterations']}"
        stats[model_key] = {
            'psnr_sum': 0.0,
            'ssim_sum': 0.0,
            'xcorr_sum': 0.0,
            'peak_dist_sum': 0.0,
            'fwhm_diff_sum': 0.0,
            'total_peaks_ideal': 0,
            'total_peaks_matched': 0,
            'pattern_count': 0,
            # Initialize peak sensitivity metrics if enabled
            'optimal_sigma_sum': 0.0,
            'optimal_threshold_sum': 0.0,
            'max_f1_score_sum': 0.0,
            'peak_position_stability_sum': 0.0,
            'peak_count_stability_sum': 0.0,
            'parameter_sensitivity_sum': 0.0,
            'false_positive_rate_sum': 0.0,
            'false_negative_rate_sum': 0.0
        }
    
    # Process each pattern
    for hr, kr, lr in tqdm(indices_list):
        # Define which pattern numbers to process
        pattern_nums = [5] if central_only else range(1, 10)
        
        for num in pattern_nums:
            # Load and preprocess data
            pattern_file = f'output_hanning_conv_{hr}_{kr}_{lr}_0000{num}.npz'
            try:
                data = np.load(f'{data_path}/{pattern_file}')
            except FileNotFoundError:
                print(f"Warning: File not found: {pattern_file}")
                continue
            
            dp_pp, _, _ = ptNN_U.preprocess_ZCB_9(data['convDP'], mask)
            dp_pp_IDEAL, _, _ = ptNN_U.preprocess_ZCB_9(data['pinholeDP_extra_conv'], mask=np.ones(dp_pp[0][0].shape))
            
            # Convert to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dp_pp = dp_pp.to(device=device, dtype=torch.float)
            dp_pp_IDEAL = dp_pp_IDEAL.to(device=device, dtype=torch.float)
            
            # Process each model
            for _, row in df.iterrows():
                model_key = f"{row['loss_type']}_{'Unet' if row['use_unet'] else 'no_Unet'}_{row['iterations']}"
                
                # Load and run model
                model = load_model(row['path'], row['use_unet'])
                with torch.no_grad():
                    output = model(dp_pp)
                
                # Calculate metrics
                metrics = calculate_metrics(output, dp_pp_IDEAL,
                                         calculate_psnr=calculate_psnr,
                                         calculate_ssim=calculate_ssim,
                                         calculate_xcorr=calculate_xcorr,
                                         calculate_peaks=calculate_peaks,
                                         calculate_peak_sensitivity=calculate_peak_sensitivity,
                                         peak_sigma=peak_sigma)
                
                # Update statistics
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
                    if 'num_peaks2' in metrics:  # ideal peaks
                        stats[model_key]['total_peaks_ideal'] += metrics['num_peaks2']
                        # Count matched peaks (those within distance threshold)
                        # Use the same parameters that were used in calculate_metrics
                        used_sigma = metrics.get('peak_sigma_used', peak_sigma)
                        used_threshold = metrics.get('peak_threshold_used', 0.265)
                        
                        ideal_peaks, _ = find_peaks_and_fwhm(dp_pp_IDEAL.squeeze().cpu().numpy(), 
                                                              sigma=used_sigma, 
                                                              threshold=used_threshold)
                        output_peaks, _ = find_peaks_and_fwhm(output.squeeze().cpu().numpy(), 
                                                               sigma=used_sigma, 
                                                               threshold=used_threshold)
                        matched = 0
                        for ideal_peak in ideal_peaks:
                            min_dist = float('inf')
                            for output_peak in output_peaks:
                                dist = np.sqrt((ideal_peak[0]-output_peak[0])**2 + (ideal_peak[1]-output_peak[1])**2)
                                min_dist = min(min_dist, dist)
                            if min_dist < 5:  # Same threshold as in visualization
                                matched += 1
                        stats[model_key]['total_peaks_matched'] += matched
                
                if calculate_peak_sensitivity:
                    # Add peak sensitivity metrics
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
                'patterns_processed': count
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

def group_stats_by_model_type(stats: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Group statistics by model type (L1/L2/pearson, Unet/no_Unet), combining iterations.
    
    Args:
        stats (Dict[str, Dict[str, float]]): Dictionary of statistics per model
        
    Returns:
        Dict[str, Dict[str, float]]: Dictionary of grouped statistics per model type
    """
    grouped_stats = {}
    
    for model_key, metrics in stats.items():
        # Extract model type (e.g., 'L1_Unet', 'L2_no_Unet', etc.)
        model_type = '_'.join(model_key.split('_')[:-1])  # Remove iteration number
        
        if model_type not in grouped_stats:
            grouped_stats[model_type] = {
                'avg_psnr': [],
                'avg_ssim': [],
                'avg_xcorr': [],
                'avg_peak_dist': [],
                'avg_fwhm_diff': [],
                'peak_detection_rate': [],
                'total_peaks_matched': 0,
                'total_peaks_ideal': 0,
                'total_patterns': 0,
                # Sensitivity metrics
                'avg_optimal_sigma': [],
                'avg_optimal_threshold': [],
                'avg_max_f1_score': [],
                'avg_peak_position_stability': [],
                'avg_peak_count_stability': [],
                'avg_parameter_sensitivity': [],
                'avg_false_positive_rate': [],
                'avg_false_negative_rate': []
            }
        
        # Append individual metrics to lists for later averaging
        for metric in ['avg_psnr', 'avg_ssim', 'avg_xcorr', 'avg_peak_dist', 'avg_fwhm_diff', 'peak_detection_rate',
                      'avg_optimal_sigma', 'avg_optimal_threshold', 'avg_max_f1_score',
                      'avg_peak_position_stability', 'avg_peak_count_stability', 'avg_parameter_sensitivity',
                      'avg_false_positive_rate', 'avg_false_negative_rate']:
            if metric in metrics:
                grouped_stats[model_type][metric].append(metrics[metric])
        
        # Sum up total peaks and patterns
        grouped_stats[model_type]['total_peaks_matched'] += metrics['total_peaks_matched']
        grouped_stats[model_type]['total_peaks_ideal'] += metrics['total_peaks_ideal']
        grouped_stats[model_type]['total_patterns'] += metrics['patterns_processed']
    
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
            'total_patterns': metrics['total_patterns']
        }
        
        # Add sensitivity metrics if they exist
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

def print_cumulative_stats(stats: Dict[str, Dict[str, float]], sort_by: str = 'avg_ssim', group_by_model: bool = True):
    """
    Print cumulative statistics in a formatted table, sorted by a specified metric.
    Can group statistics by model type.
    
    Args:
        stats (Dict[str, Dict[str, float]]): Dictionary of statistics per model
        sort_by (str): Metric to sort by
        group_by_model (bool): Whether to group statistics by model type
    """
    if group_by_model:
        stats = group_stats_by_model_type(stats)
    
    # Convert to DataFrame for easier formatting
    rows = []
    for model, metrics in stats.items():
        row = {'Model': model}
        row.update(metrics)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values(by=sort_by, ascending=False)
    
    # Format the metrics for better readability
    formatted_df = df.copy()
    for col in df.columns:
        if col == 'Model':
            continue
        if col in ['total_peaks_matched', 'total_peaks_ideal', 'total_patterns']:
            formatted_df[col] = df[col].map(lambda x: f"{int(x):,}")
        else:
            formatted_df[col] = df[col].map(lambda x: f"{x:.4f}")
    
    # Print formatted table with a title indicating grouping
    print("\nCumulative Statistics {} (sorted by {}):\n".format(
        "Grouped by Model Type" if group_by_model else "Per Model and Iteration",
        sort_by
    ))
    print(formatted_df.to_string())
    
    # Print summary statistics
    if group_by_model:
        print("\nSummary:")
        print(f"Total number of model types: {len(df)}")
        print(f"Total patterns processed: {df['total_patterns'].astype(int).sum():,}")
        print(f"Total peaks detected: {df['total_peaks_matched'].astype(int).sum():,} / {df['total_peaks_ideal'].astype(int).sum():,}")
        print(f"Overall peak detection rate: {df['total_peaks_matched'].astype(int).sum() / df['total_peaks_ideal'].astype(int).sum():.4f}")

def create_stats_table_figure(stats: Dict[str, Dict[str, float]], 
                           sort_by: str = 'avg_ssim',
                           group_by_model: bool = True,
                           figsize: Tuple[float, float] = (15, 5)) -> plt.Figure:
    """
    Create a formatted table figure from the statistics.
    
    Args:
        stats (Dict[str, Dict[str, float]]): Dictionary of statistics per model
        sort_by (str): Metric to sort by
        group_by_model (bool): Whether to group statistics by model type
        figsize (Tuple[float, float]): Figure size in inches
        
    Returns:
        plt.Figure: The created figure
    """
    if group_by_model:
        stats = group_stats_by_model_type(stats)
    
    # Convert to DataFrame and sort
    rows = []
    for model, metrics in stats.items():
        row = {'Model': model}
        row.update(metrics)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values(by=sort_by, ascending=False)
    
    # Select and rename columns for display
    columns_to_show = [
        'Model',
        'avg_psnr',
        'avg_ssim',
        'peak_detection_rate',
        'avg_peak_dist',
        'avg_fwhm_diff',
        'total_peaks_matched',
        'total_peaks_ideal',
        'pattern_count' if 'pattern_count' in df.columns else 'total_patterns'
    ]
    
    # Add sensitivity metrics if they exist
    if 'avg_optimal_sigma' in df.columns:
        columns_to_show.extend([
            'avg_optimal_sigma',
            'avg_optimal_threshold',
            'avg_max_f1_score',
            'avg_false_positive_rate',
            'avg_false_negative_rate'
        ])
    
    column_labels = {
        'Model': 'Model Type',
        'avg_psnr': 'PSNR (dB)',
        'avg_ssim': 'SSIM',
        'peak_detection_rate': 'Peak Detection Rate',
        'avg_peak_dist': 'Avg Peak Distance',
        'avg_fwhm_diff': 'Avg FWHM Diff',
        'total_peaks_matched': 'Peaks Matched',
        'total_peaks_ideal': 'Total Peaks',
        'pattern_count': 'Patterns',
        'total_patterns': 'Patterns',
        'avg_optimal_sigma': 'Optimal Sigma',
        'avg_optimal_threshold': 'Optimal Threshold',
        'avg_max_f1_score': 'Peak F1 Score',
        'avg_false_positive_rate': 'False Pos Rate',
        'avg_false_negative_rate': 'False Neg Rate'
    }
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')
    
    # Format the data
    display_df = df[columns_to_show].copy()
    
    # Format numeric columns
    for col in display_df.columns:
        if col == 'Model':
            continue
        if col in ['total_peaks_matched', 'total_peaks_ideal', 'pattern_count', 'total_patterns']:
            display_df[col] = display_df[col].map(lambda x: f"{int(x):,}")
        elif col in ['avg_psnr', 'avg_peak_dist', 'avg_fwhm_diff']:
            display_df[col] = display_df[col].map(lambda x: f"{x:.2f}")
        else:
            display_df[col] = display_df[col].map(lambda x: f"{x:.4f}")
    
    # Create table
    table = ax.table(
        cellText=display_df.values,
        colLabels=[column_labels[col] for col in columns_to_show],
        cellLoc='center',
        loc='center',
        colColours=['#E6E6E6'] * len(columns_to_show)
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    
    # Adjust cell widths based on content
    for (row, col), cell in table.get_celld().items():
        if col == 0:  # Model column
            cell.set_width(0.2)
        else:
            cell.set_width(0.1)
    
    # Add title
    title = f"Model Performance Statistics (Sorted by {column_labels[sort_by]})"
    if group_by_model:
        title += " - Grouped by Model Type"
    #plt.title(title, pad=20, size=12, weight='bold')
    
    # Add summary statistics as text below the table
    if group_by_model:
        patterns_col = 'pattern_count' if 'pattern_count' in df.columns else 'total_patterns'
        summary_text = [
            f"Total number of model types: {len(df)}",
            f"Total patterns processed: {df[patterns_col].astype(int).sum():,}",
            f"Total peaks detected: {df['total_peaks_matched'].astype(int).sum():,} / {df['total_peaks_ideal'].astype(int).sum():,}",
            f"Overall peak detection rate: {df['total_peaks_matched'].astype(int).sum() / df['total_peaks_ideal'].astype(int).sum():.4f}"
        ]
        #plt.figtext(0.1, 0.02, '\n'.join(summary_text), fontsize=9, va='bottom')
    
    plt.tight_layout()
    return fig

def save_stats_table(stats: Dict[str, Dict[str, float]], 
                    save_path: str,
                    sort_by: str = 'avg_ssim',
                    group_by_model: bool = True,
                    figsize: Tuple[float, float] = (15, 5)):
    """
    Create and save a formatted table of statistics as a figure.
    
    Args:
        stats (Dict[str, Dict[str, float]]): Dictionary of statistics per model
        save_path (str): Path where to save the figure
        sort_by (str): Metric to sort by
        group_by_model (bool): Whether to group statistics by model type
        figsize (Tuple[float, float]): Figure size in inches
    """
    fig = create_stats_table_figure(stats, sort_by, group_by_model, figsize)
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

def cross_validate_models(
    model_base_path: str,
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
    cmap: str = 'jet'
) -> plt.Figure:
    """
    Cross-validate models by showing outputs from all trained model conditions
    applied to all test data conditions in a confusion matrix style grid.
    
    Creates an n×n grid where:
    - Columns represent different trained models (training conditions)
    - Rows represent different test data (test conditions)
    - Diagonal shows models tested on their own training data type
    
    Args:
        model_base_path (str): Base path for model files
        data_base_path (str): Base path for data H5 files (e.g., '/net/micdata/data2/12IDC/ptychosaxs/batch_mode/hkl/')
        mask_path (str): Path to the mask file
        hkl_index (Tuple[int, int, int]): The (h, k, l) indices for the test pattern
        lattice_types (List[str]): List of lattice types (e.g., ['SC', 'ClathII'])
        noise_statuses (List[str]): List of noise statuses (e.g., ['Noise', 'noNoise'])
        probe_sizes (List[int]): List of probe sizes (e.g., [128, 256])
        loss_function (str): Loss function used for training
        unet_status (str): 'Unet' or 'no_Unet'
        epoch (int): Epoch number to load
        pattern_num (int): Pattern number to use (1-9, typically 5 for central)
        figsize (Tuple[int, int]): Figure size
        cmap (str): Colormap for images
        
    Returns:
        plt.Figure: Figure with cross-validation grid
    """
    import sys
    import os
    import h5py
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../../NN/ptychosaxsNN/')))
    import utils.ptychosaxsNN_utils as ptNN_U
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask = np.load(mask_path)
    hr, kr, lr = hkl_index
    
    # Generate all condition combinations
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
    
    # Create figure with grid (add 1 for headers, add 2 for ground truth columns)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(n_conditions + 1, n_conditions + 3,  # +3 for label column + 2 ground truth columns
                         hspace=0.4, wspace=0.4,
                         left=0.08, right=0.98, top=0.95, bottom=0.05)
    
    # Add title
    loss_display = 'NPCC' if 'pearson' in loss_function else loss_function.replace('_loss', '')
    fig.suptitle(f'Network ({epoch} epochs, {loss_display})',
                fontsize=24, fontweight='bold')
    
    # Top-left corner (labels)
    ax_corner = fig.add_subplot(gs[0, 0])
    ax_corner.axis('off')
    ax_corner.text(0.5, 0.5, 'Lattice\n(hkl)', 
                  ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Column headers (training conditions)
    for col_idx, cond in enumerate(conditions):
        ax = fig.add_subplot(gs[0, col_idx + 1])
        ax.axis('off')
        noise_label = 'Noise' if cond['noise'] == 'Noise' else 'NoNoise'
        label_text = f"{noise_label}\n{cond['lattice']}"
        ax.text(0.5, 0.8, label_text, 
               ha='center', va='top', fontsize=12, fontweight='bold')
        # Add probe size at bottom
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
                model = load_model(model_path, use_unet)
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
        # Add probe size
        ax.text(0.8, 0.2, f"{test_cond['probe']}x{test_cond['probe']}", 
               ha='right', va='bottom', fontsize=10)
        
        # Load test data for this row
        # Path: Lattice{lattice_type}_Probe{probe_size}x{probe_size}_ZCB_9_3D__{noise_status}_hkl{hr}{kr}{lr}/sim_ZCB_9_3D_S5065_N1_steps4_dp256.h5
        data_dir = f'Lattice{test_cond["lattice"]}_Probe{test_cond["probe"]}x{test_cond["probe"]}_ZCB_9_3D__{test_cond["noise"]}_hkl{hr}{kr}{lr}'
        data_file = f'{data_base_path}{data_dir}/sim_ZCB_9_3D_S5065_N1_steps4_dp256.h5'
        
        if os.path.exists(data_file):
            try:
                with h5py.File(data_file, 'r') as h5f:
                    # Load the pattern_num-th pattern (patterns are typically 1-indexed in the file)
                    # Check what keys are available
                    if 'convDP' in h5f.keys():
                        conv_dp_data = h5f['convDP'][pattern_num - 1]  # Assuming 0-indexed array
                        # Also load ideal ground truth
                        if 'pinholeDP_raw_FFT' in h5f.keys():
                            ideal_dp_data = h5f['pinholeDP_raw_FFT'][pattern_num - 1]
                        elif 'pinholeDP_extra_conv' in h5f.keys():
                            ideal_dp_data = h5f['pinholeDP_extra_conv'][pattern_num - 1]
                        elif 'pinholeDP' in h5f.keys():
                            ideal_dp_data = h5f['pinholeDP'][pattern_num - 1]
                        else:
                            ideal_dp_data = None
                    else:
                        # Try alternative key naming
                        conv_dp_data = h5f[f'convDP_{pattern_num}'][:]
                        ideal_dp_data = h5f[f'pinholeDP_{pattern_num}'][:] if f'pinholeDP_{pattern_num}' in h5f.keys() else None
                    
                dp_pp, _, _ = ptNN_U.preprocess_ZCB_9(conv_dp_data, mask)
                dp_pp = dp_pp.to(device=device, dtype=torch.float)
                
                # Process ideal ground truth
                if ideal_dp_data is not None:
                    dp_pp_IDEAL, _, _ = ptNN_U.preprocess_ZCB_9(ideal_dp_data, 
                                                                 mask=np.ones(dp_pp[0][0].shape))
                    dp_pp_IDEAL = dp_pp_IDEAL.to(device=device, dtype=torch.float)
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
                            
                            # Add small colorbar
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
                    
                    # Highlight diagonal (same training and test condition)
                    if row_idx == col_idx:
                        for spine in ax.spines.values():
                            spine.set_edgecolor('red')
                            spine.set_linewidth(3)
                
                # Add ground truth images (convolved input and ideal output)
                # Convolved input
                ax_conv = fig.add_subplot(gs[row_idx + 1, n_conditions + 1])
                conv_input_np = dp_pp.squeeze().cpu().numpy()
                im_conv = ax_conv.imshow(conv_input_np, cmap=cmap)
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider = make_axes_locatable(ax_conv)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im_conv, cax=cax, format='%.1e')
                ax_conv.set_xticks([])
                ax_conv.set_yticks([])
                
                # Ideal ground truth
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
                # Fill row with error messages
                for col_idx in range(n_conditions + 2):  # Include ground truth columns
                    ax = fig.add_subplot(gs[row_idx + 1, col_idx + 1])
                    ax.text(0.5, 0.5, 'Test Data\nNot Found', 
                           ha='center', va='center', fontsize=8)
                    ax.set_xticks([])
                    ax.set_yticks([])
        else:
            print(f"Warning: Test data not found: {data_file}")
            # Fill row with error messages
            for col_idx in range(n_conditions + 2):  # Include ground truth columns
                ax = fig.add_subplot(gs[row_idx + 1, col_idx + 1])
                ax.text(0.5, 0.5, 'Test Data\nNot Found', 
                       ha='center', va='center', fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
    
    return fig

# Example usage:
# save_stats_table(stats, 'model_stats.pdf', sort_by='peak_detection_rate')
# # Or to try different sortings:
# for metric in ['avg_ssim', 'avg_psnr', 'peak_detection_rate']:
#     save_stats_table(stats, f'model_stats_{metric}.pdf', sort_by=metric)



#%%
# Example usage:
model_configs = {
    'iterations': [2, 10, 25, 50, 100, 500],
    'models': {
        'L1_no_Unet': 'best_model_ZCB_9_no_Unet_epoch_{}.pth',
        'L1_Unet': 'best_model_ZCB_9_Unet_epoch_{}.pth',
        'L2_no_Unet': 'best_model_ZCB_9_32_no_Unet_epoch_{}_L2.pth',
        'L2_Unet': 'best_model_ZCB_9_32_Unet_epoch_{}_L2.pth',
        'pearson_no_Unet': 'best_model_ZCB_9_no_Unet_epoch_{}_pearson_loss.pth',
        'pearson_Unet': 'best_model_ZCB_9_Unet_epoch_{}_pearson_loss.pth',
        #'pearson_no_Unet': 'best_model_ZCB_9_31_no_Unet_epoch_{}_pearson_loss.pth',
        #'pearson_Unet': 'best_model_ZCB_9_31_Unet_epoch_{}_pearson_loss.pth'
    }
}


    
# Load the input data
mask = np.load('/home/beams/PTYCHOSAXS/deconvolutionNN/data/mask/mask_ZCB_9_3D.npy')

ind=random.randint(0,10800)
ind=4111#8370#2362#8370#4111#9375#338#5840
print(f'Using index {ind}')
# preprocess diffraction pattern
#dp_pp,_,_ = ptNN_U.preprocess_ZCB_9(np.load(f'/net/micdata/data2/12IDC/ptychosaxs/data/diff_sim/32/output_hanning_conv_{ind:05d}.npz')['convDP'],mask)
#dp_pp_IDEAL,_,_ = ptNN_U.preprocess_ZCB_9(np.load(f'/net/micdata/data2/12IDC/ptychosaxs/data/diff_sim/32/output_hanning_conv_{ind:05d}.npz')['pinholeDP_extra_conv'],mask=np.ones(dp_pp[0][0].shape))
hr,kr,lr=3,1,0
dp_pp,_,_ = ptNN_U.preprocess_ZCB_9(np.load(f'/net/micdata/data2/12IDC/ptychosaxs/data/diff_sim/lattice_ls400_gs1024_lsp6_r3.0_typeSC/output_hanning_conv_{hr}_{kr}_{lr}_00006.npz')['convDP'],mask)
dp_pp_IDEAL,_,_ = ptNN_U.preprocess_ZCB_9(np.load(f'/net/micdata/data2/12IDC/ptychosaxs/data/diff_sim/lattice_ls400_gs1024_lsp6_r3.0_typeSC/output_hanning_conv_{hr}_{kr}_{lr}_00006.npz')['pinholeDP_extra_conv'],mask=np.ones(dp_pp[0][0].shape))



fig,ax = plt.subplots(1,2)
im1=ax[0].imshow(dp_pp.squeeze().cpu().numpy())
im2=ax[1].imshow(dp_pp_IDEAL.squeeze().cpu().numpy())
plt.colorbar(im1,ax=ax[0])
plt.colorbar(im2,ax=ax[1])
ax[0].set_title('Convolution')
ax[1].set_title('Ideal')
plt.show()

#%%


# Create the comparison grid
fig = create_comparison_grid_from_config(
    model_configs=model_configs,
    input_data=dp_pp.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype=torch.float),
    ideal_data=dp_pp_IDEAL.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype=torch.float),
    base_path="/net/micdata/data2/12IDC/ptychosaxs/models/ZCB_9_3D/",
    figsize=(15, 15),
    calculate_psnr=True,
    calculate_ssim=True,
    calculate_xcorr=False,
    calculate_peak_sensitivity=True,
    calculate_peaks=True
)
#save_comparison_grid(fig, 'comparison_grid.pdf')


#%%
# Plot metrics vs iteration comparing all loss functions on the same plot
fig_compare = plot_metrics_vs_iteration_compare_loss(
    model_configs=model_configs,
    input_data=dp_pp.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype=torch.float),
    ideal_data=dp_pp_IDEAL.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype=torch.float),
    loss_types=['L1', 'L2', 'pearson'],  # Compare all loss functions
    base_path="/net/micdata/data2/12IDC/ptychosaxs/models/ZCB_9_3D/",
    figsize=(15, 30),
    calculate_psnr=True,
    calculate_ssim=True,
    calculate_peaks=True,
    calculate_peak_sensitivity=True,  # Set to True to optimize peak finding parameters
    compare_architectures=True  # Set to True to also plot no_Unet (dashed lines with square markers)
)

#%%
# Plot metrics vs iteration averaged over multiple patterns (like cumulative_stats)
indices_list = [
    (1,0,0),
    (1,1,1),
    (2,1,1),
    (3,1,0),
    (3,2,1),
    (2,0,0),
    (2,2,0)
]

fig_averaged = plot_metrics_vs_iteration_averaged(
    model_configs=model_configs,
    indices_list=indices_list,
    loss_types=['L1', 'L2', 'pearson'],
    base_path="/net/micdata/data2/12IDC/ptychosaxs/models/ZCB_9_3D/",
    mask_path='/home/beams/PTYCHOSAXS/deconvolutionNN/data/mask/mask_ZCB_9_3D.npy',
    data_path='/net/micdata/data2/12IDC/ptychosaxs/data/diff_sim/lattice_ls400_gs1024_lsp6_r3.0_typeSC',
    figsize=(15, 30),
    calculate_psnr=True,
    calculate_ssim=True,
    calculate_peaks=True,
    calculate_peak_sensitivity=True,  # Set to True to optimize peak finding parameters
    peak_sigma=0.714,
    central_only=True,  # Only use central pattern (num=5) from each index
    compare_architectures=True  # Compare both Unet and no_Unet
)

# # If you want to compare architectures for a single loss function, use the original function:
# fig_L1 = plot_metrics_vs_iteration(
#     model_configs=model_configs,
#     input_data=dp_pp.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype=torch.float),
#     ideal_data=dp_pp_IDEAL.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype=torch.float),
#     loss_type='L1',
#     base_path="/net/micdata/data2/12IDC/ptychosaxs/models/ZCB_9_3D/",
#     figsize=(15, 12),
#     calculate_psnr=True,
#     calculate_ssim=True,
#     calculate_peaks=True,
#     compare_architectures=True  # Compares Unet vs no_Unet for L1 loss
# )

# # Save plots if needed
# # save_metrics_plot(fig_compare, 'metrics_vs_iteration_all_losses.png')

#%%













#%%
probe_sizes=[256]#,256]#,128]
lattice_types=['ClathII']#,'SC']#,'ClathII']
unet_statuses=['Unet']#,'no_Unet']#,'no_Unet']#,'Unet']#,'no_Unet']
loss_functions=['pearson_loss','L1','L2']#,'L1','L2']
noise_statuses=['noNoise']#,'noNoise']#,'Noise']
files=[25]#2,10,25,50,100,150,200,250,300,400,500]
base_path="/net/micdata/data2/12IDC/ptychosaxs/"
model_list=[base_path + f'batch_mode/trained_model/best_model_Lattice{lattice_type}_Probe{probe_size}x{probe_size}_ZCB_9_3D__{noise_status}_sim_ZCB_9_3D_S5065_N600_steps4_dp256_{unet_status}_epoch_{f}_{loss_function}_symmetry_0.0.pth' for f in files for loss_function in loss_functions for unet_status in unet_statuses for lattice_type in lattice_types for probe_size in probe_sizes for noise_status in noise_statuses]
model_list_info =[f'Lattice{lattice_type}_Probe{probe_size}x{probe_size}_{noise_status}_{unet_status}_{loss_function}' for loss_function in loss_functions for unet_status in unet_statuses for lattice_type in lattice_types for probe_size in probe_sizes for noise_status in noise_statuses]
# for i in range(len(model_list)):
#     print(model_list_info[i])
# Example usage:
model_configs = {
    'iterations': files,
    'models': {
        'pearson_Unet': model_list[0],
        #'pearson_no_Unet': model_list[1],
        'L1_Unet': model_list[1],
        #'L1_no_Unet': model_list[3],
        'L2_Unet': model_list[2],
        #'L2_no_Unet': model_list[5],
    }
}
# Load the input data
mask = np.load('/home/beams/PTYCHOSAXS/deconvolutionNN/data/mask/mask_ZCB_9_3D.npy')
import h5py
hr,kr,lr=1,1,1
h5file_data=f'/net/micdata/data2/12IDC/ptychosaxs/batch_mode/hkl/LatticeSC_Probe256x256_ZCB_9_3D__Noise_hkl{hr}{kr}{lr}/sim_ZCB_9_3D_S5065_N1_steps4_dp256.h5'
# Load data directly from the HDF5 file
print(f"Loading data from: {h5file_data}")

with h5py.File(h5file_data, "r") as h5f:
    # Load convDP and pinholeDP_raw_FFT as conv and ideal diffraction patterns
    conv_DPs = h5f['convDP'][:]  # Shape: (16, 256, 256)
    ideal_DPs = h5f['pinholeDP_raw_FFT'][:]  # Shape: (16, 256, 256)
    
    print(h5f.keys())
    num_patterns = len(conv_DPs)
    print(f"Loaded {num_patterns} diffraction patterns")
    print(f"Pattern shapes - conv_DPs: {conv_DPs.shape}, ideal_DPs: {ideal_DPs.shape}")

# Create dummy probe array (as before)
probe_DPs = np.ones(conv_DPs.shape)  # dummy array for testing network with a probe

# Display a random pattern to verify the data
ri = np.random.randint(0, len(conv_DPs))
ri=10
print(f"Using index {ri}")

dp_pp,_,_ = ptNN_U.preprocess_ZCB_9(conv_DPs[ri],mask)
dp_pp_IDEAL,_,_ = ptNN_U.preprocess_ZCB_9(ideal_DPs[ri],mask=np.ones(dp_pp[0][0].shape))

fig,ax = plt.subplots(1,2,figsize=(15,5))
im1=ax[0].imshow(dp_pp.squeeze().cpu().numpy(),cmap='jet')
im2=ax[1].imshow(dp_pp_IDEAL.squeeze().cpu().numpy(),cmap='jet')
plt.colorbar(im1,ax=ax[0])
plt.colorbar(im2,ax=ax[1])
ax[0].set_title('Convolution')
ax[1].set_title('Ideal')
plt.show()


# Create the comparison grid
fig = create_comparison_grid_from_config(
    model_configs=model_configs,
    input_data=dp_pp.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype=torch.float),
    ideal_data=dp_pp_IDEAL.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype=torch.float),
    base_path="/net/micdata/data2/12IDC/ptychosaxs/models/ZCB_9_3D/",
    figsize=(15, 15),
    calculate_psnr=True,
    calculate_ssim=True,
    calculate_xcorr=False,
    calculate_peaks=True
)

#%%


# Cross-validate models: test all trained models on all test data types
fig_cross_val = cross_validate_models(
    model_base_path="/net/micdata/data2/12IDC/ptychosaxs/batch_mode/trained_model/",
    data_base_path="/net/micdata/data2/12IDC/ptychosaxs/batch_mode/hkl/",
    mask_path="/home/beams/PTYCHOSAXS/deconvolutionNN/data/mask/mask_ZCB_9_3D.npy",
    hkl_index=(1, 1, 1),  # Test on (3,1,1) reflection
    lattice_types=['SC', 'ClathII'],
    noise_statuses=['Noise', 'noNoise'],
    probe_sizes=[128, 256],
    loss_function='pearson_loss',
    unet_status='Unet',
    epoch=25,
    pattern_num=10,  # Use central pattern
    figsize=(24, 20),  # Wider to accommodate ground truth columns
    cmap='jet'
)







#%%

# ##############################################################
# ##############################################################
# ##############################################################
# ##############################################################
# ##############################################################
# ##############################################################
# ##############################################################
# ##############################################################
# #/scratch/trained_model/best_model_LatticeSC_Probe256x256_ZCB_9_3D__Noise_sim_ZCB_9_3D_S5065_N600_steps4_dp256_Unet_epoch_500_L2_symmetry_0.0.pth

# probe_sizes=[256]
# lattice_types=['SC']
# unet_statuses=['Unet']
# loss_functions=['pearson_loss','L1','L2']
# noise_statuses=['Noise']
# files=[25]#,10,25,50,100,500]
# base_path="/scratch/trained_model/"
# model_list=[base_path + f'best_model_Lattice{lattice_type}_Probe{probe_size}x{probe_size}_ZCB_9_3D__{noise_status}_sim_ZCB_9_3D_S5065_N600_steps4_dp256_{unet_status}_epoch_{f}_{loss_function}_symmetry_0.0.pth' for f in files for loss_function in loss_functions for unet_status in unet_statuses for lattice_type in lattice_types for probe_size in probe_sizes for noise_status in noise_statuses]
# model_list_info =[f'Lattice{lattice_type}_Probe{probe_size}x{probe_size}_{noise_status}_{unet_status}_{loss_function}' for loss_function in loss_functions for unet_status in unet_statuses for lattice_type in lattice_types for probe_size in probe_sizes for noise_status in noise_statuses]
# # Example usage:
# model_configs = {
#     'iterations': files,
#     'models': {
#         'pearson_Unet': model_list[0],
#         'L1_Unet': model_list[1],
#         'L2_Unet': model_list[2],
#     }
# }
# # Load the input data
# mask = np.load('/home/beams/PTYCHOSAXS/deconvolutionNN/data/mask/mask_ZCB_9_3D.npy')
# import h5py
# hr,kr,lr=1,1,0
# h5file_data=f'/net/micdata/data2/12IDC/ptychosaxs/batch_mode/hkl/LatticeSC_Probe256x256_ZCB_9_3D__Noise_hkl{hr}{kr}{lr}/sim_ZCB_9_3D_S5065_N1_steps4_dp256.h5'
# # Load data directly from the HDF5 file
# print(f"Loading data from: {h5file_data}")

# with h5py.File(h5file_data, "r") as h5f:
#     # Load convDP and pinholeDP_raw_FFT as conv and ideal diffraction patterns
#     conv_DPs = h5f['convDP'][:]  # Shape: (16, 256, 256)
#     ideal_DPs = h5f['pinholeDP_raw_FFT'][:]  # Shape: (16, 256, 256)
    
#     print(h5f.keys())
#     num_patterns = len(conv_DPs)
#     print(f"Loaded {num_patterns} diffraction patterns")
#     print(f"Pattern shapes - conv_DPs: {conv_DPs.shape}, ideal_DPs: {ideal_DPs.shape}")

# # Create dummy probe array (as before)
# probe_DPs = np.ones(conv_DPs.shape)  # dummy array for testing network with a probe

# # Display a random pattern to verify the data
# ri = np.random.randint(0, len(conv_DPs))
# ri=0
# print(f"Using index {ri}")

# dp_pp,_,_ = ptNN_U.preprocess_ZCB_9(conv_DPs[ri],mask)
# dp_pp_IDEAL,_,_ = ptNN_U.preprocess_ZCB_9(ideal_DPs[ri],mask=np.ones(dp_pp[0][0].shape))

# fig,ax = plt.subplots(1,2,figsize=(15,5))
# im1=ax[0].imshow(dp_pp.squeeze().cpu().numpy(),cmap='jet')
# im2=ax[1].imshow(dp_pp_IDEAL.squeeze().cpu().numpy(),cmap='jet')
# plt.colorbar(im1,ax=ax[0])
# plt.colorbar(im2,ax=ax[1])
# ax[0].set_title('Convolution')
# ax[1].set_title('Ideal')
# plt.show()

# ##############################################################
# ##############################################################
# ##############################################################
# ##############################################################
# ##############################################################
# ##############################################################
# ##############################################################
# ##############################################################






# %%
# Define your list of indices
indices_list = [
    (1,0,0),
    # (1,1,1),
    # (2,1,1),
    # (3,1,0),
    # (3,2,1),
    # (2,0,0),
    # (2,2,0)
    # ... add more combinations as needed
]

# Calculate cumulative stats
stats = calculate_cumulative_stats(
    model_configs=model_configs,
    indices_list=indices_list,
    base_path="/net/micdata/data2/12IDC/ptychosaxs/models/ZCB_9_3D/",
    calculate_psnr=True,
    calculate_ssim=True,
    calculate_xcorr=False,
    calculate_peaks=True,
    calculate_peak_sensitivity=True,
    peak_sigma=0.714,
    central_only=True
)

# Print stats sorted by different metrics
print_cumulative_stats(stats, sort_by='avg_ssim')  # Sort by SSIM
print_cumulative_stats(stats, sort_by='peak_detection_rate')  # Sort by peak detection rate
print_cumulative_stats(stats, sort_by='avg_psnr')  # Sort by PSNR




# %%
create_stats_table_figure(stats, sort_by='peak_detection_rate', group_by_model=True)
plt.show()
# %%
save_stats_table(stats, 'model_stats.pdf', sort_by='peak_detection_rate', group_by_model=True)