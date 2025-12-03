import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from skimage.transform import resize
import torch
from tqdm import tqdm
import os
import h5py
#import hdf5plugin #THIS IMPORT CAUSES A HANGUP
from scipy.ndimage import maximum_filter, label, find_objects
#import scipy.fft as spf
import scipy.io as sio
from typing import List, Tuple
#import pyFAI
from pathlib import Path
import re
from sklearn.cluster import DBSCAN
import plotly.graph_objects as go
#import webbrowser
import plotly.graph_objects as go
from tqdm import tqdm
from skimage.transform import downscale_local_mean
from plotly.subplots import make_subplots
#import time
#import pdb
from ipywidgets import interact, FloatSlider, Button, VBox, Output
import ipywidgets as widgets
import pandas as pd
from scipy.ndimage import gaussian_filter
import glob
import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import rotate as rotate_gpu

def log10_custom(arr):
    # Create a mask for positive values
    positive_mask = arr > 0
    
    # Initialize result array
    result = np.zeros_like(arr, dtype=float)
    
    # Calculate log10 only on positive values
    log10_positive = np.log10(arr[positive_mask])
    
    # Find the minimum log10 value from the positive entries
    min_log10_value = log10_positive.min() if log10_positive.size > 0 else 0
    
    # Set positive entries to their log10 values
    result[positive_mask] = log10_positive
    
    # Set non-positive entries to the minimum log10 value
    result[~positive_mask] = min_log10_value#hostname
    
    
    return result

def set_path(path):
    return Path(path)
   
def create_circular_mask(image, center_x=0,center_y=0,radius=48):
    """
    Creates a circular mask at the center of the image.
    """
    # Get the dimensions of the image
    h, w = image.shape[:2]
    
    # Calculate the center of the image
    if center_x == 0 and center_y==0:
        center_x, center_y = w // 2, h // 2
    
    # Create a grid of x and y coordinates
    y, x = np.ogrid[:h, :w]
    
    # Calculate the distance from each pixel to the center
    distance_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    
    # Create the circular mask
    mask = distance_from_center >= radius
    
    return mask.astype(np.uint8)  # Return as uint8 (0s and 1s)

def replace_2d_array_values_by_row_indices(array, start, end):
    """
    Replaces values in a 2D numpy array with 0 if their row indices fall within the specified range.
    Used for masking diffraction patterns at edges where padding is used in ptycho recon
    
    Returns:
    np.ndarray: 2D numpy array with specified rows replaced with 0.

    """
    if not isinstance(array, np.ndarray) or array.ndim != 2:
        raise ValueError("The input must be a 2D numpy array")
    
    if not (0 <= start <= end < array.shape[0]):
        raise ValueError("Invalid range specified")

    array[start:end+1, :] = np.min(array)
    return array

def replace_2d_array_values_by_column_indices(array, start, end):
    """
    Replaces values in a 2D numpy array with 0 if their row indices fall within the specified range.
    Used for masking diffraction patterns at edges where padding is used in ptycho recon

    Returns:
    np.ndarray: 2D numpy array with specified rows replaced with 0.

    """
    if not isinstance(array, np.ndarray) or array.ndim != 2:
        raise ValueError("The input must be a 2D numpy array")
    
    if not (0 <= start <= end < array.shape[0]):
        raise ValueError("Invalid range specified")

    array[:,start:end+1] = np.min(array)
    return array
    
def preprocess_cindy(dp):#,probe):
    size=256
    dp_pp=dp
    #probe_sub=abs(spf.fftshift(spf.fft2(probe)))**2
    #dp_pp=dp-probe_sub
    dp_pp=np.asarray(replace_2d_array_values_by_column_indices(replace_2d_array_values_by_column_indices(replace_2d_array_values_by_row_indices(replace_2d_array_values_by_row_indices(dp_pp,0,16),495,511),0,16),495,511))
    dp_pp=log10_custom(dp_pp)
    #dp_pp[np.isnan(dp_pp)] = 0
    #dp_pp[dp_pp <= 0] = np.min(dp_pp[dp_pp > 0])# small positive value
    dp_pp=np.asarray(resize(dp_pp[:,:],(size,size),preserve_range=True,anti_aliasing=True))
    #dp_pp=np.log10(dp_pp)

    sf=np.max(dp_pp)-np.min(dp_pp)
    bkg=np.min(dp_pp)
    dp_pp=np.asarray((dp_pp-bkg)/(sf))
    dp_pp=torch.tensor(dp_pp.reshape(1,1,size,size))
    return dp_pp,sf,bkg

def preprocess_zhihua2(dp,mask,waxs_mask):#,probe):
    size=512
    dp_pp=dp
    dp_pp=np.asarray(dp_pp*mask)
    dp_pp=np.asarray(resize(dp_pp,(size,size),preserve_range=True,anti_aliasing=True))
    dp_pp=np.asarray(dp_pp*waxs_mask)
    dp_pp=log10_custom(dp_pp)
    
    #dp_pp=np.log10(dp_pp)

    sf=np.max(dp_pp)-np.min(dp_pp)
    bkg=np.min(dp_pp)
    dp_pp=np.asarray((dp_pp-bkg)/(sf))
    dp_pp=torch.tensor(dp_pp.reshape(1,1,size,size))
    #print(dp_pp.shape)
    #dp_pp=vignette_transform(dp_pp, center_decay=center_decay)
    #print(dp_pp.shape)
    return dp_pp,sf,bkg


def preprocess_zhihua(dp,mask,center_decay=2):#,probe):
    size=512
    dp_pp=dp
    dp_pp=np.asarray(dp_pp*mask)

    dp_pp=np.asarray(resize(dp_pp,(size,size),preserve_range=True,anti_aliasing=True))
    dp_pp=log10_custom(dp_pp)

    sf=np.max(dp_pp)-np.min(dp_pp)
    bkg=np.min(dp_pp)
    dp_pp=np.asarray((dp_pp-bkg)/(sf))
    dp_pp=torch.tensor(dp_pp.reshape(1,1,size,size))

    return dp_pp,sf,bkg

def preprocess_ZCB_9(dp,mask):
    size=256
    dp_pp=dp
    dp_pp=np.asarray(dp_pp*mask)
    dp_pp=np.asarray(resize(dp_pp,(size,size),preserve_range=True,anti_aliasing=True))
    dp_pp=log10_custom(dp_pp)

    sf=np.max(dp_pp)-np.min(dp_pp)
    bkg=np.min(dp_pp)
    dp_pp=np.asarray((dp_pp-bkg)/(sf))
    dp_pp=torch.tensor(dp_pp.reshape(1,1,size,size))
    
    return dp_pp,sf,bkg

def create_circular_waxs_mask_min_value(image, radius):
    """
    Create a modified version of the image where the central region is set to the minimum value.
    
    Args:
        image (ndarray): Input image to be masked
        radius (int): Radius of the central circular region to modify
        
    Returns:
        ndarray: Image with central circle set to minimum value
    """
    # Create a copy of the image
    masked_image = image.copy()
    
    # Find minimum value in the image (excluding zeros or NaNs if needed)
    valid_pixels = image[image > 0]  # Adjust if needed
    if len(valid_pixels) > 0:
        min_val = np.min(valid_pixels)
    else:
        min_val = 0
    
    # Create the circular mask
    h, w = image.shape
    y, x = np.indices((h, w))
    center_y, center_x = h // 2, w // 2
    distance_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    
    # Set central region to minimum value
    central_region = (distance_from_center <= radius)
    masked_image[central_region] = min_val
    
    return masked_image,central_region


def bin_ndarray(ndarray, new_shape, operation='mean'):
    """
    Bins an ndarray to a new shape by averaging or summing.
    """
    shape = ndarray.shape
    assert len(shape) == len(new_shape)
    compression_pairs = [(d, c//d) for d, c in zip(new_shape, shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray

def create_circular_mask_zero_outside(image, radius=128):
    """
    Creates a circular mask at the center of the image where values outside the circle are set to zero.
    
    Args:
        image (ndarray): Input image to be masked
        radius (int): Radius of the circle to keep (values outside will be set to zero)
        
    Returns:
        ndarray: Image with values outside the circle set to zero
    """
    # Get the dimensions of the image
    h, w = image.shape[:2]

    # Center
    center_x, center_y = w // 2, h // 2
    
    # Create a grid of x and y coordinates
    y, x = np.ogrid[:h, :w]
    
    # Calculate the distance from each pixel to the center
    distance_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    
    # Create the circular mask (True inside circle, False outside)
    mask = distance_from_center <= radius
    
    # Create a copy of the image and set values outside circle to zero
    masked_image = image.copy()
    masked_image[~mask] = 0
    
    return masked_image

def preprocess_CMT(dp,mask,radius=128):
    bin_size=256
    size=256
    dp_pp=dp
    dp_pp=np.asarray(dp_pp)
    dp_pp=bin_ndarray(dp_pp, (bin_size, bin_size), operation='mean')
    dp_pp=np.asarray(dp_pp*mask)
    #dp_pp=np.asarray(resize(dp_pp,(size,size),preserve_range=True,anti_aliasing=True))
    dp_pp=log10_custom(dp_pp)
    sf=np.max(dp_pp)-np.min(dp_pp)
    bkg=np.min(dp_pp)
    dp_pp=np.asarray((dp_pp-bkg)/(sf))*mask
    dp_pp=create_circular_mask_zero_outside(dp_pp,radius)
    dp_pp=torch.tensor(dp_pp.reshape(1,1,size,size))
    
    return dp_pp,sf,bkg

def generate_weight_mask(shape, center_decay):
    h, w = shape
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_distance = np.sqrt(center_x**2 + center_y**2)
    weight_mask = (distance / max_distance) ** center_decay
    return weight_mask

def vignette_transform(image, center_decay=2):
    h, w = image.shape[-2:]
    weight_mask = generate_weight_mask((h, w), center_decay)
    weight_mask = torch.tensor(weight_mask, dtype=image.dtype, device=image.device).unsqueeze(0)
    return image * weight_mask

def preprocess_chansong(dp,probe):
    lbound,ubound=(23,38),(235,250)
    size=256
    dp_pp=dp
    #dp_pp=np.asarray(replace_2d_array_values_by_row_indices(replace_2d_array_values_by_row_indices(dp_pp,ubound[0],ubound[1]),lbound[0],lbound[1]))
    dp_pp=log10_custom(dp_pp)
    dp_pp=np.asarray(resize(dp_pp[:,:],(size,size),preserve_range=True,anti_aliasing=True))
    sf=np.max(dp_pp)-np.min(dp_pp)
    bkg=np.min(dp_pp)
    dp_pp=np.asarray((dp_pp-bkg)/(sf))
    dp_pp=torch.tensor(dp_pp.reshape(1,1,size,size))
    probe=resize_dp(probe)
    dp_probe=torch.tensor(probe.reshape(1,1,size,size))
    dp_pp=torch.cat([dp_pp, dp_probe], dim=1)
    return dp_pp,sf,bkg

def invert_preprocess_cindy(dp,sf,bkg):
    dp_rec=dp*sf + bkg
    dp_rec=10**(dp_rec)
    return dp_rec
        
def plot_and_save_scan(dps,ptycho_scan,scanx=20,scany=15):
    
    fig, axs = plt.subplots(scany,scanx, sharex=True,sharey=True,figsize=(scanx,scany))

    # Remove vertical space between Axes
    fig.subplots_adjust(hspace=0,wspace=0)
    count=0
    inputs=[]
    outputs=[]
    sfs=[]
    bkgs=[]
    for i in tqdm(range(0,scany)):
        for j in range(0,scanx):
            dp_count=np.asarray(dps[count][1:513,259:771])
          
            dp_count_copy=dp_count.copy()
            result,sf,bkg=preprocess_cindy(dp_count_copy)
            resulta=result.to(device=ptycho_scan.device, dtype=torch.float)

            result=ptycho_scan.model(resulta).detach().to("cpu").numpy()[0][0]
            im=axs[i][j].imshow(result)
            axs[i][j].imshow(result)
            axs[i][j].axis("off")

            outputs.append(result)
            sfs.append(sf)
            bkgs.append(bkg)
            inputs.append(resulta.detach().to("cpu").numpy()[0][0])
         
            count+=1
    cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()
    return inputs,outputs,sfs,bkgs


def check_scan_completeness(base_path, scan_name, scan_id, start_i=1, end_i=29, start_j=1, end_j=36):
    """
    Check if all files in a scan sequence exist and identify unexpected files.
    """
    expected_files = []
    missing_files = []
    expected_filenames = set()
    
    # Generate all expected filenames
    for i in range(start_i, end_i + 1):
        for j in range(start_j, end_j + 1):
            filename = f"{scan_name}_{scan_id}_{i:05d}_{j:05d}.h5"
            full_path = os.path.join(base_path, filename)
            expected_files.append(full_path)
            expected_filenames.add(filename)
    
    # Check which files exist
    for file_path in expected_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    total_expected = len(expected_files)
    found = total_expected - len(missing_files)
    
    # Get a list of all actual files for comparison
    actual_files = glob.glob(os.path.join(base_path, f"{scan_name}_{scan_id}_*.h5"))
    
    # Find unexpected files (files that exist but don't match the expected pattern)
    unexpected_files = []
    for file_path in actual_files:
        filename = os.path.basename(file_path)
        if filename not in expected_filenames:
            unexpected_files.append(file_path)
    
    return {
        'complete': len(missing_files) == 0,
        'total_expected': total_expected,
        'found': found,
        'missing': missing_files,
        'completion_percentage': (found / total_expected) * 100 if total_expected > 0 else 0,
        'actual_file_count': len(actual_files),
        'unexpected_files': unexpected_files
    }

def plot_full_scan(dps, preprocess_func,mask, model, scanx=36, scany=29, dpsize=256, center=(517,575)):
    """
    Plot diffraction patterns from a 2D scan more efficiently.
    """
    # Create figure and axes once
    fig, axs = plt.subplots(scany, scanx, figsize=(scanx, scany))
    fig.subplots_adjust(hspace=0, wspace=0)
    
    # Handle different dimensions of axs
    if scany == 1 and scanx == 1:
        axs = np.array([[axs]])
    elif scany == 1:
        axs = np.array([axs])
    elif scanx == 1:
        axs = np.array([[ax] for ax in axs])
    
    # Pre-calculate indices for cropping
    y_start = center[0] - dpsize//2
    y_end = center[0] + dpsize//2
    x_start = center[1] - dpsize//2
    x_end = center[1] + dpsize//2
    
    count = 0
    inputs = []
    outputs = []
    sfs = []
    bkgs = []
    
    # Turn off all axes at once
    for ax_row in axs:
        for ax in ax_row:
            ax.axis('off')
    
    # Process in batches for better performance
    try:
        # Use tqdm for progress tracking
        pbar = tqdm(total=scanx*scany)
        
        for i in range(scany):
            for j in range(scanx):
                if count >= len(dps):
                    break
                    
                # Crop the diffraction pattern
                dp_count = dps[count][y_start:y_end, x_start:x_end]
                
                
                resultT, sfT, bkgT = preprocess_func(dp_count,mask)
                
                
                # Plot the result
                im = axs[i][j].imshow(dp_count, cmap='jet',norm=colors.LogNorm())
                #im = axs[i][j].imshow(model_new(resultT.to(device=device, dtype=torch.float)).detach().to("cpu").numpy()[0][0], cmap='jet')#,norm=colors.LogNorm())
                
                count += 1
                pbar.update(1)
                
        pbar.close()
                
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user (Ctrl+C)")
        # Continue with plotting what we have so far
    
    # Only add colorbar if we processed at least one image
    if count > 0:
        cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
    
    plt.tight_layout()
    plt.show()
    
    return 0

def plot_full_fly_scan(dps, preprocess_func, mask, model, scanx=36, scany=29, dpsize=256, center=(517,575)):
    """
    Plot diffraction patterns from a 2D scan more efficiently.
    Handles serpentine/fly scan pattern where alternate rows go in opposite directions.
    """
    # Create figure and axes once
    fig, axs = plt.subplots(scany, scanx, figsize=(scanx, scany))
    fig.subplots_adjust(hspace=0, wspace=0)
    
    # Handle different dimensions of axs
    if scany == 1 and scanx == 1:
        axs = np.array([[axs]])
    elif scany == 1:
        axs = np.array([axs])
    elif scanx == 1:
        axs = np.array([[ax] for ax in axs])
    
    # Pre-calculate indices for cropping
    y_start = center[0] - dpsize//2
    y_end = center[0] + dpsize//2
    x_start = center[1] - dpsize//2
    x_end = center[1] + dpsize//2
    
    count = 0
    inputs = []
    outputs = []
    sfs = []
    bkgs = []
    
    # Turn off all axes at once
    for ax_row in axs:
        for ax in ax_row:
            ax.axis('off')
    
    # Process in batches for better performance
    try:
        # Use tqdm for progress tracking
        pbar = tqdm(total=min(scanx*scany, len(dps)))
        
        for i in range(scany):
            # For even rows (0, 2, 4...), go left to right
            # For odd rows (1, 3, 5...), go right to left
            if i % 2 == 0:
                j_range = range(scanx)  # Left to right
            else:
                j_range = range(scanx-1, -1, -1)  # Right to left
                
            for j in j_range:
                if count < len(dps):  # Check if we still have data to plot
                    # Crop the diffraction pattern
                    dp_count = dps[count][y_start:y_end, x_start:x_end]
                    
                    resultT, sfT, bkgT = preprocess_func(dp_count, mask)
                    
                    # Plot the result
                    im = axs[i][j].imshow(dp_count, cmap='jet', norm=colors.LogNorm())
                    #im = axs[i][j].imshow(model_new(resultT.to(device=device, dtype=torch.float)).detach().to("cpu").numpy()[0][0], cmap='jet')#,norm=colors.LogNorm())
                    
                    count += 1
                    pbar.update(1)
                else:
                    # No more data to plot
                    axs[i][j].text(0.5, 0.5, 'No data', ha='center', va='center')
                
        pbar.close()
                
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user (Ctrl+C)")
        # Continue with plotting what we have so far
    
    # Only add colorbar if we processed at least one image
    if count > 0:
        cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
    
    plt.tight_layout()
    plt.show()
    
    return 0

def plot_absorption_fly_scan(dps, preprocess_func, mask, model, scanx=36, scany=29, dpsize=256, center=(517,575)):
    """
    Plot absorption values from a 2D scan in a grid layout.
    Handles serpentine/fly scan pattern where alternate rows go in opposite directions.
    """
    # Create figure and axes once
    fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    
    # Pre-calculate indices for cropping
    y_start = center[0] - dpsize//2
    y_end = center[0] + dpsize//2
    x_start = center[1] - dpsize//2
    x_end = center[1] + dpsize//2
    
    # Create a 2D array to store absorption values
    absorption_map = np.zeros((scany, scanx))
    
    count = 0
    
    # Process diffraction patterns
    try:
        # Use tqdm for progress tracking
        pbar = tqdm(total=min(scanx*scany, len(dps)))
        
        for i in range(scany):
            # For even rows (0, 2, 4...), go left to right
            # For odd rows (1, 3, 5...), go right to left
            if i % 2 == 0:
                j_range = range(scanx)  # Left to right
            else:
                j_range = range(scanx-1, -1, -1)  # Right to left
            
            for j in j_range:
                if count < len(dps):  # Check if we still have data to plot
                    # Crop the diffraction pattern
                    dp_count = dps[count][y_start:y_end, x_start:x_end]
                    
                    # Process the diffraction pattern
                    resultT, sfT, bkgT = preprocess_func(dp_count, mask)
                    
                    # Calculate absorption value (total intensity or other metric)
                    # Option 1: Use the sum of the diffraction pattern as absorption
                    absorption = np.sum(dp_count)
                    
                    # Option 2: Use the model output if it represents absorption
                    # model_output = model(resultT.to(device=device, dtype=torch.float)).detach().to("cpu").numpy()[0][0]
                    # absorption = np.mean(model_output)  # or some other metric
                    
                    # Store the absorption value
                    absorption_map[i, j] = absorption
                    
                    count += 1
                    pbar.update(1)
                
        pbar.close()
                
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user (Ctrl+C)")
    
    # Plot the absorption map
    im = axs.imshow(absorption_map, cmap='viridis')
    axs.set_title('Absorption Map')
    axs.set_xlabel('X position')
    axs.set_ylabel('Y position')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axs)
    cbar.set_label('Absorption (a.u.)')
    
    plt.tight_layout()
    plt.show()
    
    return absorption_map


def resize_dp(dp):
    return resize(dp,(256,256),preserve_range=True,anti_aliasing=True)
    #return resize(dp,(1408,1408),preserve_range=True,anti_aliasing=True)
    #return resize(dp,(256,288),preserve_range=True,anti_aliasing=True)

def norm_0to1(image):
    return np.asarray((image-np.min(image))/(np.max(image)-np.min(image)))
    
def find_peaks2d(dp, center_cut=100, n=25, threshold=0.3, plot=True):
    peaks = []
    
    # Define the shape of the image
    rows, cols = dp.shape
    
    # Calculate half of the neighborhood size
    half_n = n // 2
    
    # Iterate over each pixel, excluding border pixels based on neighborhood size
    for i in range(half_n, rows - half_n):
        for j in range(half_n, cols - half_n):
            # Extract the nxn neighborhood of the current pixel
            neighborhood = dp[i-half_n:i+half_n+1, j-half_n:j+half_n+1]
            
            # Coordinates of the center of the image
            center_x, center_y = rows // 2, cols // 2
            
            # Check if the center pixel is greater than all its neighbors, above the threshold, and unique
            if dp[i, j] > threshold and dp[i, j] == np.max(neighborhood) and np.count_nonzero(dp[i, j] == neighborhood) == 1:
                if (i-center_x)**2 + (j-center_y)**2 > center_cut**2:
                    peaks.append((i, j))
    if plot:
        fig,ax=plt.subplots()
        im=ax.imshow(dp, cmap='jet', interpolation='nearest')
        peak_y, peak_x = zip(*peaks)
        ax.scatter(peak_x, peak_y, color='red', marker='x', s=100, label='Peaks')
        plt.colorbar(im)
        plt.show()

    return peaks

def find_peaks_2d_filter(diffraction_pattern, center_cut=25,n=25, threshold=0.1,plot=True):
    """
    Find peaks in a 2D diffraction pattern.

    Parameters:
    diffraction_pattern (ndarray): The 2D diffraction pattern as a NumPy array.
    n(int): The size of the neighborhood for local maximum detection. Default is 5.
    threshold (float): Optional intensity threshold. Peaks below this value will be ignored.

    Returns:
    peaks (list of tuples): A list of (row, col) indices of the detected peaks.
    """
    # Apply a maximum filter to identify local maxima
    local_max = maximum_filter(diffraction_pattern, size=n) == diffraction_pattern
    
    # Apply an optional intensity threshold to remove low-intensity peaks
    if threshold is not None:
        local_max &= diffraction_pattern > threshold
    
    # Label the peaks
    labeled, num_objects = label(local_max)
    
    # Extract the coordinates of the peaks
    slices = find_objects(labeled)
    peaks = [(int((s[0].start + s[0].stop - 1) / 2), int((s[1].start + s[1].stop - 1) / 2)) for s in slices]
    
    # Find the center of the diffraction pattern
    center_row, center_col = np.array(diffraction_pattern.shape) // 2
    
    # Filter out peaks within the specified radius from the center
    filtered_peaks = []
    for peak in peaks:
        distance_from_center = np.sqrt((peak[0] - center_row) ** 2 + (peak[1] - center_col) ** 2)
        if distance_from_center > center_cut:
            filtered_peaks.append(peak)

    return filtered_peaks

def neighborhood_intensity(image, x, y, radius=5):
    """
    Calculate the sum of pixel intensities in a square neighborhood around (x, y).
    The neighborhood is a square of side 2*radius+1 centered on (x, y).
    
    :param image: 2D NumPy array representing the image
    :param x: X-coordinate of the peak
    :param y: Y-coordinate of the peak
    :param radius: The radius around the peak to define the neighborhood
    :return: Integrated intensity within the neighborhood
    """
    # Define the neighborhood bounds, ensuring they stay within image limits
    x_min = max(0, x - radius)
    x_max = min(image.shape[0], x + radius + 1)
    y_min = max(0, y - radius)
    y_max = min(image.shape[1], y + radius + 1)
    
    # Extract the neighborhood and calculate the sum of intensities
    neighborhood = image[x_min:x_max, y_min:y_max]
    return np.sum(neighborhood)

def circular_neighborhood_intensity(image, x, y, radius=5,plot=True):
    x_min = max(0, x - radius)
    x_max = min(image.shape[0], x + radius + 1)
    y_min = max(0, y - radius)
    y_max = min(image.shape[1], y + radius + 1)
    

    # Create a meshgrid of coordinates in the neighborhood
    X, Y = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max), indexing='ij')
    
    # Calculate the Euclidean distance from the peak (x, y)
    distances = np.sqrt((X - x)**2 + (Y - y)**2)
    
    # Mask for points within the radius
    mask = distances <= radius
    
    # Sum the pixel intensities within the mask
    neighborhood = image[x_min:x_max, y_min:y_max]
    if plot:    
        # Plot the original image with the circular neighborhood overlay
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='jet',norm=colors.LogNorm(),clim=(1,1000))
        
        # Create a circle patch to show the circular neighborhood on the image
        circle = plt.Circle((y, x), radius, color='red', fill=False, linewidth=2)
        ax.add_patch(circle)
        
        # Mark the center point
        ax.plot(y, x, 'ro')
        
        # Set labels and title
        ax.set_title(f'Circular Neighborhood (Center: [{x}, {y}], Radius: {radius})')
        plt.show()
 
    return np.sum(neighborhood[mask])
    
def read_hdf5_file(file_path):
    """
    Reads an HDF5 file and returns its contents.

    Parameters:
    file_path (str): The path to the HDF5 file.

    Returns:
    dict: A dictionary with dataset names as keys and their data as values.
    """
    data_dict = {}

    try:
        with h5py.File(file_path, 'r') as hdf_file:
            def extract_data(name, obj):
                if isinstance(obj, h5py.Dataset):
                    data_dict[name] = obj[()]

            hdf_file.visititems(extract_data)
    except Exception as e:
        print(f"An error occurred: {e}")

    return data_dict

def find_directories_with_number(base_path, number):
    """
    Finds immediate subdirectories containing a specific number in their name,
    allowing for flexible number formatting.

    Args:
    - base_path (str): The path to the directory to search.
    - number (int): The number to search for in subdirectory names.

    Returns:
    - list: A list of matching directory paths.
    """
    matching_dirs = []
    # Create a regex pattern to match the number with optional leading zeros anywhere in the name
    #number_pattern = rf"0*{number}\b"
    number_pattern = rf"(^|[^0-9])0*{number}([^0-9]|$)"

    try:
        # List only directories in the base path
        for entry in os.listdir(base_path):
            full_path = os.path.join(base_path, entry)
            # Check if the entry is a directory and matches the pattern
            if os.path.isdir(full_path) and re.search(number_pattern, entry):
                matching_dirs.append(full_path)
    except FileNotFoundError:
        print(f"The path '{base_path}' does not exist.")
    except PermissionError:
        print(f"Permission denied to access '{base_path}'.")

    return [Path(m) for m in matching_dirs]
    
def load_h5_scan_to_npy(file_path,scan,plot=True,point_data=True):
    # For loading cindy ptycho scan data
    # file_path = '/net/micdata/data2/12IDC/2021_Nov/ptycho/'
    # scan = 1125 (e.g.)
    dps=[]
    file_path_new=find_directories_with_number(file_path,scan)[0]
    for filename in tqdm(os.listdir(file_path_new)[:]):#-1]):
        filename = file_path_new / filename
        #print(filename)
        #print(read_hdf5_file(filename).keys())
        data = read_hdf5_file(filename)['entry/data/data']
        
        if point_data:
            dps.append(data)
            if plot:
                plt.figure()
                plt.imshow(data,norm=colors.LogNorm())
                plt.show()
        else:
            for j in range(0,len(data)):
                dps.append(data[j])
                if plot:
                    plt.figure()
                    plt.imshow(data[j],norm=colors.LogNorm())
                    plt.show()
    dps=np.asarray(dps)
    return dps

def load_hdf5_scan_to_npy(file_path,scan,plot=True):
    # For loading cindy ptycho scan data
    # file_path = '/net/micdata/data2/12IDC/2021_Nov/results/ML_recon/'
    # scan = 1125 (e.g.)
    dps=[]
    file_path_new=find_directories_with_number(file_path,scan)[0]
    print(file_path_new)
    for filename in os.listdir(file_path_new)[:-1]:
        filename = file_path_new / filename
        read_hdf5_file(file_path_new / filename).keys()
        if 'dp' not in read_hdf5_file(file_path_new / filename).keys(): #skip parameter file
            continue
        else:
            data = read_hdf5_file(file_path_new / filename)['dp']
            for j in range(0,len(data)):
                dps.append(data[j])
                if plot:
                    plt.figure()
                    plt.imshow(data[j],norm=colors.LogNorm())
                    plt.show()
    dps=np.asarray(dps)
    return dps
def create_azimuthal_segments(shape, center=None, num_segments=8, inner_radius=0, outer_radius=None):
    """
    Create a mask dividing a 2D array into azimuthal segments with radial constraints.
    
    Parameters:
    -----------
    shape : tuple
        Shape of the 2D array (height, width)
    center : tuple, optional
        Center coordinates (y, x). If None, uses the center of the array.
    num_segments : int
        Number of azimuthal segments to create
    inner_radius : float
        Inner radius of the annular segments (default: 0, starts from center)
    outer_radius : float
        Outer radius of the annular segments (default: None, uses 80% of the minimum dimension)
    
    Returns:
    --------
    segment_masks : list of 2D arrays
        List of boolean masks for each segment
    """
    if center is None:
        center = (shape[0] // 2, shape[1] // 2)
    
    # Default outer radius is 80% of the minimum dimension to avoid corners
    if outer_radius is None:
        outer_radius = min(center[0], center[1], 
                          shape[0] - center[0], 
                          shape[1] - center[1]) * 0.8
    
    y, x = np.ogrid[:shape[0], :shape[1]]
    y = y - center[0]
    x = x - center[1]
    
    # Calculate radius for each pixel
    radius = np.sqrt(y**2 + x**2)
    
    # Calculate angles in radians (0 to 2π)
    angles = np.arctan2(y, x) % (2 * np.pi)
    
    # Create segment masks
    segment_masks = []
    segment_size = 2 * np.pi / num_segments
    
    for i in range(num_segments):
        start_angle = i * segment_size
        end_angle = (i + 1) * segment_size
        
        # Create mask for this segment with radial constraints
        mask = (angles >= start_angle) & (angles < end_angle) & (radius >= inner_radius) & (radius <= outer_radius)
        segment_masks.append(mask)
    
    return segment_masks

def create_annular_mask(shape, peak_x,peak_y, r_outer):
    """Create an annular mask between r_inner and r_outer centered at 'center'."""
    y, x = np.ogrid[:shape[0], :shape[1]]
    dist_from_center = np.sqrt((x - peak_x)**2 + (y - peak_y)**2)
    mask = (dist_from_center <= r_outer)
    return mask
    
def ensure_inverse_peaks(peaks: List[Tuple[int, int]], tolerance: int = 4) -> List[Tuple[int, int]]:
    """
    Ensures that each peak in the list has an inverse peak, within a given tolerance.
    
    Parameters:
    - peaks: List of tuples, each representing (p1, p2) coordinates of peaks.
    - tolerance: Tolerance in pixels to check if an inverse peak is present.
    
    Returns:
    - List of peaks with ensured inverse peaks.
    """
        
    ## Example usage:
    #peaks = [(10, 20), (-8, -18), (15, 25)]
    #updated_peaks = ensure_inverse_peaks(peaks)
    #print(updated_peaks)
    #ensure_inverse_peaks(peaks_shifted)
    
    # Function to check if a point is within tolerance of another point
    def within_tolerance(point1, point2, tol):
        return abs(point1[0] - point2[0]) <= tol and abs(point1[1] - point2[1]) <= tol

    # Result list starting with original peaks
    result_peaks = peaks.copy()
    
    # Iterate over each peak and ensure its inverse is present
    for p1, p2 in peaks:
        # Calculate the inverse coordinates
        inv_p1, inv_p2 = -p1, -p2
        
        # Check if the inverse peak is within tolerance in the current list
        if not any(within_tolerance((inv_p1, inv_p2), (px, py), tolerance) for px, py in result_peaks):
            # If no peak within tolerance, add the inverse peak
            result_peaks.append((inv_p1, inv_p2))
    
    return result_peaks

def plot_scan_positions(param_file, plot=True):
    """
    Load and plot scan positions from parameter file with connecting line
    
    Args:
        param_file (str): Full path to parameter file
        plot (bool): Whether to show the plot (default: True)
        
    Example:
        x_pos, y_pos = plot_scan_positions('/net/micdata/data2/12IDC/2021_Nov/results/ML_recon/tomo_scan3/scan1053/data_roi0_Ndp512_para.hdf5', plot=True)
    
    Returns:
        tuple: Arrays of X and Y positions
    """
    # Load data
    data = read_hdf5_file(param_file)
    pos_x = data['ppX']
    pos_y = data['ppY']
    
    if plot:
        # Create figure
        plt.figure(figsize=(10, 10))
        
        # Find unique y positions to identify rows
        unique_y = np.unique(pos_y)
        
        # Store end points of each row
        row_ends = []
        next_row_starts = []
        
        # Plot horizontal lines connecting points in each row
        for i, y in enumerate(unique_y):
            # Get all points in this row
            mask = pos_y == y
            x_row = pos_x[mask]
            y_row = pos_y[mask]
            
            # Sort points by x position
            sort_idx = np.argsort(x_row)
            x_row = x_row[sort_idx]
            y_row = y_row[sort_idx]
            
            # Plot the connecting line
            plt.plot(x_row, y_row, '-', color='red', alpha=0.25, linewidth=3)
            
            # Store end points for connecting between rows
            if i < len(unique_y) - 1:
                row_ends.append((x_row[-1], y_row[-1]))
                # Get start of next row
                next_y = unique_y[i + 1]
                next_mask = pos_y == next_y
                next_x = pos_x[next_mask]
                next_y = pos_y[next_mask]
                sort_idx = np.argsort(next_x)
                next_row_starts.append((next_x[sort_idx][0], next_y[sort_idx][0]))
        
        # Draw connecting lines between rows
        for i in range(len(row_ends)):
            plt.plot([row_ends[i][0], next_row_starts[i][0]], 
                    [row_ends[i][1], next_row_starts[i][1]], 
                    '--', color='red', alpha=0.25, linewidth=1.5)
        
        # Plot all points
        plt.scatter(pos_x, pos_y, color='red', s=20)
        
        plt.title('Scan Positions')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.axis('equal')  # Make aspect ratio 1:1
        plt.grid(True)
        plt.gca().invert_yaxis()  # Invert y-axis
        plt.show()
    
    return pos_x, pos_y


def get_angle_for_scan(df, scan_number):
    '''
    Get angle for a specific scan number
    
    Example File:
    df = pd.read_csv('/home/beams/PTYCHOSAXS/NN/ptychosaxsNN/data/Sample6_tomo6_projs_1537_1789_shifts.txt', 
                 comment='#',  # Skip lines starting with #
                 names=['Angle', 'y_shift', 'x_shift', 'scanNo'])  # Specify column names
    '''
    return df.loc[df['scanNo'] == scan_number, 'Angle'].values[0]

def convert_2D_to_3D_peaks(x, y, phi_angle):
    """
    Convert 2D detector peak positions to 3D coordinates
    
    Args:
        x (float): X position on detector
        y (float): Y position on detector
        phi_angle (float): Rotation angle in degrees around y-axis
            
    Returns:
        tuple: (x3d, y3d, z3d) coordinates
    """
    # Convert angle to radians
    phi = np.radians(phi_angle)
    
    # Apply rotation around y-axis
    # x' = x*cos(phi) + z*sin(phi)
    # y' = y
    # z' = -x*sin(phi) + z*cos(phi)
    x3d = x * np.cos(phi)  # z=0 at detector, so z*sin(phi) term is 0
    y3d = y  # y-axis is rotation axis, no change
    z3d = -x * np.sin(phi)  # z=0 at detector, so z*cos(phi) term is 0
    
    return x3d, y3d, z3d

def visualize_3D_peaks(peaks_list, phi_angles, intensities_list=None, save_path=None):
    """
    Visualize peaks from multiple projections in 3D space
    
    Args:
        peaks_list (list): List of peak positions [(x1,y1), (x2,y2), ...] for each angle
        phi_angles (list): List of rotation angles in degrees
        intensities_list (list, optional): List of intensities for each peak
        save_path (str, optional): Path to save the HTML file
    """
    fig = go.Figure()
    
    # Convert all peaks to 3D coordinates
    for i, (peaks, phi) in enumerate(zip(peaks_list, phi_angles)):
        x3d_list = []
        y3d_list = []
        z3d_list = []
        hover_texts = []
        
        for j, (x, y) in enumerate(peaks):
            x3d, y3d, z3d = convert_2D_to_3D_peaks(x, y, phi)
            x3d_list.append(x3d)
            y3d_list.append(y3d)
            z3d_list.append(z3d)
            
            # Create hover text
            intensity_text = f"Intensity: {intensities_list[i][j]:.1f}<br>" if intensities_list else ""
            hover_texts.append(
                f"Detector X: {x:.1f}<br>"
                f"Detector Y: {y:.1f}<br>"
                f"Phi: {phi:.1f}°<br>"
                f"{intensity_text}"
                f"3D X: {x3d:.3f}<br>"
                f"3D Y: {y3d:.3f}<br>"
                f"3D Z: {z3d:.3f}"
            )
        
        # Define marker properties
        marker_props = {
            'size': 8,
            'opacity': 0.7,
            'line': dict(width=2, color='white')
        }
        
        if intensities_list:
            marker_props.update({
                'color': intensities_list[i],
                'colorscale': 'Viridis',
                'colorbar': dict(
                    title="Intensity",
                    len=0.8,
                    y=0.8
                )
            })
        else:
            marker_props['color'] = phi  # Color by angle if no intensities
        
        # Add scatter plot for this angle
        fig.add_trace(go.Scatter3d(
            x=x3d_list,
            y=y3d_list,
            z=z3d_list,
            mode='markers',
            name=f'Phi={phi:.1f}°',
            marker=marker_props,
            hovertemplate="%{text}<extra></extra>",
            text=hover_texts,
            showlegend=True
        ))
    
    # Update layout
    fig.update_layout(
        title="3D Diffraction Peak Positions",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode='cube',  # Equal scaling
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            bgcolor='white'
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            itemsizing='constant'
        ),
        paper_bgcolor='white'
    )
    
    # Save HTML file if path provided
    if save_path:
        fig.write_html(save_path)
        print(f"Saved interactive visualization to {save_path}")
    
    # Show figure
    fig.show()
    
    return fig

def group_3D_peaks(peaks_list, phi_angles, tolerance=0.1):
    """
    Group peaks that likely belong to the same 3D feature
    based on their positions and projection angles
    
    Args:
        peaks_list (list): List of peak positions [(x1,y1), (x2,y2), ...] for each angle
        phi_angles (list): List of rotation angles in degrees
        tolerance (float): Distance tolerance for grouping peaks in 3D space
        
    Returns:
        dict: Groups of peaks that belong to the same 3D feature
    """
    # Convert all 2D peaks to 3D coordinates
    all_points_3d = []
    peak_info = []  # Store original peak info
    
    for peaks, phi in zip(peaks_list, phi_angles):
        for x, y in peaks:
            x3d, y3d, z3d = convert_2D_to_3D_peaks(x, y, phi)
            all_points_3d.append([x3d, y3d, z3d])
            peak_info.append({
                'original_x': x,
                'original_y': y,
                'phi': phi,
                'x3d': x3d,
                'y3d': y3d,
                'z3d': z3d
            })
    
    # Convert to numpy array for clustering
    points_array = np.array(all_points_3d)
    
    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=tolerance, min_samples=2).fit(points_array)
    labels = clustering.labels_
    
    # Group peaks by cluster
    groups = {}
    for idx, label in enumerate(labels):
        if label >= 0:  # Ignore noise points (label = -1)
            if label not in groups:
                groups[label] = []
            groups[label].append(peak_info[idx])
    
    return groups

def visualize_grouped_peaks(peaks_list, phi_angles, tolerance=0.1):
    """
    Visualize peaks grouped by their 3D proximity, showing only group centers
    
    Args:
        peaks_list (list): List of peak positions [(x1,y1), (x2,y2), ...] for each angle
        phi_angles (list): List of rotation angles in degrees
        tolerance (float): Distance tolerance for grouping peaks in 3D space
    """
    
    # First, group the peaks using DBSCAN
    groups = group_3D_peaks(peaks_list, phi_angles, tolerance)
    
    # Create visualization
    fig = go.Figure()
    
    # Use different colors for different groups
    colors = plt.cm.rainbow(np.linspace(0, 1, len(groups)))
    
    # Store all centers for range calculation
    all_centers = []
    
    # Plot each group center
    for group_id, peaks in groups.items():
        positions = np.array([[p['x3d'], p['y3d'], p['z3d']] for p in peaks])
        center = np.mean(positions, axis=0)
        all_centers.append(center)
        
        # Format phi angles string
        phi_angles_str = ', '.join(f"{p['phi']:.1f}°" for p in peaks)
        
        # Add group center marker
        fig.add_trace(go.Scatter3d(
            x=[center[0]],
            y=[center[1]],
            z=[center[2]],
            mode='markers',
            name=f'Group {group_id}',
            marker=dict(
                size=15,
                symbol='diamond',
                color=f'rgb({int(colors[group_id][0]*255)},{int(colors[group_id][1]*255)},{int(colors[group_id][2]*255)})',
                opacity=1,
                line=dict(width=2, color='white')
            ),
            hovertemplate=(
                f"Group {group_id}<br>" +
                "X: %{x:.3f}<br>" +
                "Y: %{y:.3f}<br>" +
                "Z: %{z:.3f}<br>" +
                f"Points in group: {len(peaks)}<br>" +
                f"Phi angles: {phi_angles_str}<br>" +
                "<extra></extra>"
            )
        ))
    
    # Calculate ranges for all centers
    all_centers = np.array(all_centers)
    x_range = [all_centers[:, 0].min(), all_centers[:, 0].max()]
    y_range = [all_centers[:, 1].min(), all_centers[:, 1].max()]
    z_range = [all_centers[:, 2].min(), all_centers[:, 2].max()]
    
    # Add padding to ranges
    padding = 0.2  # 20% padding
    x_pad = (x_range[1] - x_range[0]) * padding
    y_pad = (y_range[1] - y_range[0]) * padding
    z_pad = (z_range[1] - z_range[0]) * padding
    
    # Update layout
    fig.update_layout(
        title="Grouped 3D Diffraction Peaks (Centers)",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode='cube',
            camera=dict(
                eye=dict(x=2, y=2, z=2)
            ),
            bgcolor='white',
            xaxis=dict(range=[x_range[0]-x_pad, x_range[1]+x_pad]),
            yaxis=dict(range=[y_range[0]-y_pad, y_range[1]+y_pad]),
            zaxis=dict(range=[z_range[0]-z_pad, z_range[1]+z_pad])
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.0,
            itemsizing='constant',
            bgcolor="rgba(255, 255, 255, 0.8)"
        ),
        paper_bgcolor='white',
        width=1000,
        height=800,
        margin=dict(r=150)
    )
    
    return fig

def convert_pixels_to_q(x, y, wavelength=1.24, detector_distance=5570, pixel_size=0.075):
    """
    Convert detector pixel coordinates to q-vectors
    
    Args:
        x (array): X coordinates in pixels (centered at 0)
        y (array): Y coordinates in pixels (centered at 0)
        wavelength (float): X-ray wavelength in Angstroms
        detector_distance (float): Sample to detector distance in mm
        pixel_size (float): Detector pixel size in mm
        
    Returns:
        tuple: (qx, qy, qz) coordinates in inverse Angstroms
    """
    # Convert pixel positions to real space coordinates (mm)
    x_mm = x * pixel_size
    y_mm = y * pixel_size
    
    # Calculate scattering angles
    r = np.sqrt(x_mm**2 + y_mm**2)
    theta = 0.5 * np.arctan2(r, detector_distance)
    
    # Calculate q magnitude
    q_mag = (4 * np.pi * np.sin(theta)) / wavelength
    
    # Calculate direction cosines
    cos_phi = x_mm / r
    cos_phi[r == 0] = 0  # Handle center point
    sin_phi = y_mm / r
    sin_phi[r == 0] = 0  # Handle center point
    
    # Calculate q components
    qx = q_mag * cos_phi
    qy = q_mag * sin_phi
    qz = q_mag * np.sin(theta)
    
    return qx, qy, qz

def convert_dp_to_3D(dp, phi_angle, intensity_threshold=0, center_cutoff=0, 
                    convert_to_q=False, wavelength=1.24, detector_distance=5570, pixel_size=0.075):
    """
    Convert a 2D diffraction pattern to 3D coordinates
    
    Args:
        dp (np.ndarray): 2D diffraction pattern of any size
        phi_angle (float): Rotation angle in degrees
        intensity_threshold (float): Minimum intensity to consider
        center_cutoff (float): Radius of central region to exclude (in pixels)
        convert_to_q (bool): If True, convert to q-space coordinates
        wavelength (float): X-ray wavelength in Angstroms
        detector_distance (float): Sample to detector distance in mm
        pixel_size (float): Detector pixel size in mm
    """
    # Get coordinates of points above threshold
    y_coords, x_coords = np.where(dp > intensity_threshold)
    intensities = dp[y_coords, x_coords]
    
    # Center coordinates around (0,0)
    x_coords = x_coords - dp.shape[1]//2
    y_coords = y_coords - dp.shape[0]//2
    
    # Apply center cutoff
    if center_cutoff > 0:
        radii = np.sqrt(x_coords**2 + y_coords**2)
        mask = radii >= center_cutoff
        x_coords = x_coords[mask]
        y_coords = y_coords[mask]
        intensities = intensities[mask]
        
        if len(x_coords) == 0:  # Return early if no points remain
            return np.array([]), np.array([])
    
    # Convert each point to 3D
    positions = []
    for x, y in zip(x_coords, y_coords):
        if convert_to_q:
            # Convert to q-space first
            qx, qy, qz = convert_pixels_to_q(np.array([x]), np.array([y]), 
                           wavelength, detector_distance, pixel_size)
            # Apply rotation around qy-axis
            phi_rad = np.radians(phi_angle)
            qx_rot = qx[0] * np.cos(phi_rad) - qz[0] * np.sin(phi_rad)
            qy_rot = qy[0]
            qz_rot = qx[0] * np.sin(phi_rad) + qz[0] * np.cos(phi_rad)
            positions.append([qx_rot, qy_rot, qz_rot])
        else:
            # Original pixel-space conversion
            phi_rad = np.radians(phi_angle)
            x3d = x * np.cos(phi_rad)
            y3d = y
            z3d = -x * np.sin(phi_rad)
            positions.append([x3d, y3d, z3d])
    
    return np.array(positions), intensities

def visualize_3D_diffraction_patterns(dps, phi_angles, intensity_threshold=0.1, downsample_factor=4, 
                                    center_cutoff=0, convert_to_q=False, wavelength=1.24, 
                                    detector_distance=5570, pixel_size=0.075):
    """
    Convert and visualize diffraction patterns in 3D space
    
    Args:
        dps (list): List of 2D diffraction patterns or deconvolved patterns
        phi_angles (list): List of rotation angles in degrees
        intensity_threshold (float): Minimum intensity to consider
        downsample_factor (int): Factor by which to downsample the patterns
        center_cutoff (float): Radius of central region to exclude (in pixels)
        convert_to_q (bool): If True, plot in q-space (Å⁻¹) instead of pixel space
        wavelength (float): X-ray wavelength in Angstroms
        detector_distance (float): Sample to detector distance in mm
        pixel_size (float): Detector pixel size in mm
    """
    
    # Check input patterns
    if not isinstance(dps, list) or len(dps) == 0:
        raise ValueError("Input dps must be a non-empty list")
    
    # Get shape of first pattern
    pattern_shape = dps[0].shape
    print(f"Input pattern shape: {pattern_shape}")
    
    print(f"Converting diffraction patterns to 3D points (center cutoff: {center_cutoff} pixels)")
    if convert_to_q:
        print(f"Converting to q-space with:")
        print(f"  Wavelength: {wavelength} Å")
        print(f"  Detector distance: {detector_distance} mm")
        print(f"  Pixel size: {pixel_size} mm")
    
    # Create visualization
    fig = go.Figure()
    
    # Use different colors for different angles
    colors = plt.cm.rainbow(np.linspace(0, 1, len(phi_angles)))
    
    # Convert and plot each diffraction pattern
    for i, (dp, phi) in enumerate(tqdm(zip(dps, phi_angles), total=len(dps))):
        positions, intensities = convert_dp_to_3D(
            dp, phi, 
            intensity_threshold=intensity_threshold,
            center_cutoff=center_cutoff,
            convert_to_q=convert_to_q,
            wavelength=wavelength,
            detector_distance=detector_distance,
            pixel_size=pixel_size
        )
        
        if len(positions) > 0:  # Only plot if points were found
            fig.add_trace(go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode='markers',
                name=f'Phi={phi:.1f}°',
                marker=dict(
                    size=3,
                    color=intensities,
                    colorscale='Viridis',
                    opacity=0.6
                ),
                hovertemplate=(
                    f"{('q' if convert_to_q else 'X')}: %{{x:.3f}}{('Å⁻¹' if convert_to_q else 'px')}<br>" +
                    f"{('q' if convert_to_q else 'Y')}: %{{y:.3f}}{('Å⁻¹' if convert_to_q else 'px')}<br>" +
                    f"{('q' if convert_to_q else 'Z')}: %{{z:.3f}}{('Å⁻¹' if convert_to_q else 'px')}<br>" +
                    "Intensity: %{marker.color:.1f}<br>" +
                    f"Phi: {phi:.1f}°<br>" +
                    "<extra></extra>"
                )
            ))
    
    # Calculate overall ranges for all points
    all_x = []
    all_y = []
    all_z = []
    for trace in fig.data:
        all_x.extend(trace.x)
        all_y.extend(trace.y)
        all_z.extend(trace.z)
    
    if all_x:  # Only update ranges if we have points
        x_range = [min(all_x), max(all_x)]
        y_range = [min(all_y), max(all_y)]
        z_range = [min(all_z), max(all_z)]
        
        padding = 0.2
        x_pad = (x_range[1] - x_range[0]) * padding
        y_pad = (y_range[1] - y_range[0]) * padding
        z_pad = (z_range[1] - z_range[0]) * padding
        
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[x_range[0]-x_pad, x_range[1]+x_pad]),
                yaxis=dict(range=[y_range[0]-y_pad, y_range[1]+y_pad]),
                zaxis=dict(range=[z_range[0]-z_pad, z_range[1]+z_pad])
            )
        )
    
    # Update axis labels based on space
    axis_labels = {
        'x': 'qx (Å⁻¹)' if convert_to_q else 'X (pixels)',
        'y': 'qy (Å⁻¹)' if convert_to_q else 'Y (pixels)',
        'z': 'qz (Å⁻¹)' if convert_to_q else 'Z (pixels)'
    }
    
    fig.update_layout(
        title="3D Diffraction Patterns" + (" (Q-space)" if convert_to_q else ""),
        scene=dict(
            xaxis_title=axis_labels['x'],
            yaxis_title=axis_labels['y'],
            zaxis_title=axis_labels['z'],
            aspectmode='cube',
            camera=dict(
                eye=dict(x=2, y=2, z=2)
            ),
            bgcolor='white'
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.0,
            itemsizing='constant'
        )
    )
    
    return fig



def visualize_3D_diffraction_patterns2(dps, phi_angles, intensity_threshold=0.1, downsample_factor=4, 
                                    center_cutoff=0, convert_to_q=False, wavelength=1.24, 
                                    detector_distance=5570, pixel_size=0.075, return_data=False,
                                    grid_size=100):
    """
    Convert and visualize diffraction patterns in 3D space and optionally return gridded data
    
    Args:
        dps (list): List of 2D diffraction patterns or deconvolved patterns
        phi_angles (list): List of rotation angles in degrees
        intensity_threshold (float): Minimum intensity to consider
        downsample_factor (int): Factor by which to downsample the patterns
        center_cutoff (float): Radius of central region to exclude (in pixels)
        convert_to_q (bool): If True, plot in q-space (Å⁻¹) instead of pixel space
        wavelength (float): X-ray wavelength in Angstroms
        detector_distance (float): Sample to detector distance in mm
        pixel_size (float): Detector pixel size in mm
        return_data (bool): If True, return the gridded data
        grid_size (int): Number of points along each dimension for gridded data
    """
    # Check input patterns
    if not isinstance(dps, list) or len(dps) == 0:
        raise ValueError("Input dps must be a non-empty list")
    
    # Create visualization
    fig = go.Figure()
    
    # Use different colors for different angles
    colors = plt.cm.rainbow(np.linspace(0, 1, len(phi_angles)))
    
    # Convert and plot each diffraction pattern
    for i, (dp, phi) in enumerate(tqdm(zip(dps, phi_angles), total=len(dps))):
        positions, intensities = convert_dp_to_3D(
            dp, phi, 
            intensity_threshold=intensity_threshold,
            center_cutoff=center_cutoff,
            convert_to_q=convert_to_q,
            wavelength=wavelength,
            detector_distance=detector_distance,
            pixel_size=pixel_size
        )
        
        if len(positions) > 0:  # Only plot if points were found
            fig.add_trace(go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode='markers',
                name=f'Phi={phi:.1f}°',
                marker=dict(
                    size=3,
                    color=intensities,
                    colorscale='Viridis',
                    opacity=0.6
                ),
                hovertemplate=(
                    f"{('q' if convert_to_q else 'X')}: %{{x:.3f}}{('Å⁻¹' if convert_to_q else 'px')}<br>" +
                    f"{('q' if convert_to_q else 'Y')}: %{{y:.3f}}{('Å⁻¹' if convert_to_q else 'px')}<br>" +
                    f"{('q' if convert_to_q else 'Z')}: %{{z:.3f}}{('Å⁻¹' if convert_to_q else 'px')}<br>" +
                    "Intensity: %{marker.color:.1f}<br>" +
                    f"Phi: {phi:.1f}°<br>" +
                    "<extra></extra>"
                )
            ))

    # Calculate overall ranges for all points
    all_x = []
    all_y = []
    all_z = []
    for trace in fig.data:
        all_x.extend(trace.x)
        all_y.extend(trace.y)
        all_z.extend(trace.z)
    
    if all_x:  # Only update ranges if we have points
        x_range = [min(all_x), max(all_x)]
        y_range = [min(all_y), max(all_y)]
        z_range = [min(all_z), max(all_z)]
        
        padding = 0.2
        x_pad = (x_range[1] - x_range[0]) * padding
        y_pad = (y_range[1] - y_range[0]) * padding
        z_pad = (z_range[1] - z_range[0]) * padding
        
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[x_range[0]-x_pad, x_range[1]+x_pad]),
                yaxis=dict(range=[y_range[0]-y_pad, y_range[1]+y_pad]),
                zaxis=dict(range=[z_range[0]-z_pad, z_range[1]+z_pad])
            )
        )

    # Update axis labels based on space
    axis_labels = {
        'x': 'qx (Å⁻¹)' if convert_to_q else 'X (pixels)',
        'y': 'qy (Å⁻¹)' if convert_to_q else 'Y (pixels)',
        'z': 'qz (Å⁻¹)' if convert_to_q else 'Z (pixels)'
    }
    
    fig.update_layout(
        title="3D Diffraction Patterns" + (" (Q-space)" if convert_to_q else ""),
        scene=dict(
            xaxis_title=axis_labels['x'],
            yaxis_title=axis_labels['y'],
            zaxis_title=axis_labels['z'],
            aspectmode='cube',
            camera=dict(
                eye=dict(x=2, y=2, z=2)
            ),
            bgcolor='white'
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.0,
            itemsizing='constant'
        )
    )

    if return_data:
        # Collect all positions and intensities
        all_positions = []
        all_intensities = []
        for dp, phi in zip(dps, phi_angles):
            positions, intensities = convert_dp_to_3D(
                dp, phi, 
                intensity_threshold=intensity_threshold,
                center_cutoff=center_cutoff,
                convert_to_q=convert_to_q,
                wavelength=wavelength,
                detector_distance=detector_distance,
                pixel_size=pixel_size
            )
            if len(positions) > 0:
                all_positions.append(positions)
                all_intensities.append(intensities)
        
        positions = np.concatenate(all_positions)
        intensities = np.concatenate(all_intensities)
        
        # Create regular grid
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        z_min, z_max = positions[:, 2].min(), positions[:, 2].max()
        
        # Add small padding
        padding = 0.05
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        x_min -= x_range * padding
        x_max += x_range * padding
        y_min -= y_range * padding
        y_max += y_range * padding
        z_min -= z_range * padding
        z_max += z_range * padding
        
        # Create regular grid
        x = np.linspace(x_min, x_max, grid_size)
        y = np.linspace(y_min, y_max, grid_size)
        z = np.linspace(z_min, z_max, grid_size)
        
        # Initialize volume
        volume = np.zeros((grid_size, grid_size, grid_size))
        
        # Convert positions to grid indices
        x_idx = np.clip(((positions[:, 0] - x_min) / (x_max - x_min) * (grid_size - 1)).astype(int), 0, grid_size-1)
        y_idx = np.clip(((positions[:, 1] - y_min) / (y_max - y_min) * (grid_size - 1)).astype(int), 0, grid_size-1)
        z_idx = np.clip(((positions[:, 2] - z_min) / (z_max - z_min) * (grid_size - 1)).astype(int), 0, grid_size-1)
        
        # Accumulate intensities in volume
        np.add.at(volume, (x_idx, y_idx, z_idx), intensities)
        
        # Optional: Apply Gaussian smoothing
        volume = gaussian_filter(volume, sigma=1.0)
        
        tomo_data = {
            'volume': volume,
            'coords': {
                'x': x,
                'y': y,
                'z': z
            },
            'extent': {
                'x': [x_min, x_max],
                'y': [y_min, y_max],
                'z': [z_min, z_max]
            },
            'units': 'Å⁻¹' if convert_to_q else 'pixels'
        }
        return fig, tomo_data
    
    return fig




def create_grid_image(dps, grid_size_row, grid_size_col):
    """
    Create a grid image from diffraction patterns
    
    Args:
        dps (list): List of diffraction patterns
        grid_size_row (int): Number of rows in grid
        grid_size_col (int): Number of columns in grid
        
    Returns:
        np.ndarray: Grid image of diffraction patterns
    """
    image_size = dps[0].shape
    grid_image = np.zeros((grid_size_row * image_size[0], 
                          grid_size_col * image_size[1]))
    
    # Create grid image
    for j in range(grid_size_row):
        for i in range(grid_size_col):
            image_idx = j * grid_size_col + i
            if image_idx < len(dps):
                grid_image[
                    j * image_size[0]:(j + 1) * image_size[0],
                    i * image_size[1]:(i + 1) * image_size[1]
                ] = dps[image_idx]
    
    return grid_image

def apply_shift_to_grid(grid_image, y_shift, x_shift, grid_size_row, grid_size_col, dp_size=256):
    """
    Apply shifts to each diffraction pattern in the grid
    
    Args:
        grid_image (np.ndarray): Original grid image
        y_shift (float): Vertical shift in pixels
        x_shift (float): Horizontal shift in pixels
        grid_size_row (int): Number of rows in grid
        grid_size_col (int): Number of columns in grid
        dp_size (int): Size of each diffraction pattern
        
    Returns:
        np.ndarray: Shifted grid image
    """
    # Calculate new size needed for shifted image
    max_y_shift = abs(y_shift) * grid_size_row
    max_x_shift = abs(x_shift) * grid_size_col
    new_height = grid_image.shape[0] + int(2 * max_y_shift)
    new_width = grid_image.shape[1] + int(2 * max_x_shift)
    
    # Create new image with padding
    shifted_grid = np.zeros((new_height, new_width))
    
    # Calculate total number of patterns from grid image dimensions
    total_patterns = (grid_image.shape[0] // dp_size) * (grid_image.shape[1] // dp_size)
    
    # Apply shifts to each diffraction pattern
    for j in range(grid_size_row):
        for i in range(grid_size_col):
            pattern_idx = j * grid_size_col + i
            if pattern_idx >= total_patterns:
                continue
                
            # Calculate cumulative shift
            cum_y_shift = int(y_shift * j)
            cum_x_shift = int(x_shift * i)
            
            # Calculate source and target positions
            src_y = j * dp_size
            src_x = i * dp_size
            tgt_y = src_y + cum_y_shift + int(max_y_shift)
            tgt_x = src_x + cum_x_shift + int(max_x_shift)
            
            # Copy diffraction pattern to new position
            shifted_grid[
                tgt_y:tgt_y + dp_size,
                tgt_x:tgt_x + dp_size
            ] = grid_image[
                src_y:src_y + dp_size,
                src_x:src_x + dp_size
            ]
    
    return shifted_grid


def visualize_shifted_grid(analyzer, y_shift, x_shift, grid_size_row, grid_size_col, log_scale=True):
    """
    Create and display shifted grid images with weighted intensity peaks
    """
    # Calculate maximum shifts
    max_y_shift = abs(y_shift * (grid_size_row - 1))
    max_x_shift = abs(x_shift * (grid_size_col - 1))
    
    print(f"Shift per position: ({y_shift:.1f}, {x_shift:.1f}) pixels")
    print(f"Maximum cumulative shifts: ({max_y_shift:.1f}, {max_x_shift:.1f}) pixels")
    
    # Create original grid
    image_size = analyzer.dps[0].shape
    original_grid = np.zeros((grid_size_row * image_size[0], 
                            grid_size_col * image_size[1]))
    
    # Create shifted grid with expanded dimensions
    shifted_grid = np.zeros((
        grid_size_row * image_size[0] + int(max_y_shift),
        grid_size_col * image_size[1] + int(max_x_shift)
    ))
    
    # Calculate intensities for each frame
    ss = []
    for image in tqdm(analyzer.dps, desc="Calculating intensities"):
        frame_intensities, _ = analyzer.calculate_frame_intensities_and_orientations(image, analyzer.dps[0])
        s = np.array([frame_intensities[i]/analyzer.intensities_sum[i]/np.max(analyzer.intensities_sum) 
                     for i in range(len(frame_intensities))])
        ss.append(s)
    
    test_ss = np.array([(s-np.min(ss))/(np.max(ss)-np.min(ss)) for s in ss])
    
    # Get peak coordinates
    peak_y, peak_x = zip(*analyzer.peaks)
    
    # Create figure
    fig = plt.figure(figsize=(20, 10))
    gs = plt.GridSpec(1, 2, width_ratios=[original_grid.shape[1], shifted_grid.shape[1]])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    
    # First create the original grid
    for j in range(grid_size_row):
        for i in range(grid_size_col):
            image_idx = j * grid_size_col + i
            if image_idx < len(analyzer.dps):
                # Fill original grid
                y_start = j * image_size[0]
                x_start = i * image_size[1]
                original_grid[
                    y_start:y_start + image_size[0],
                    x_start:x_start + image_size[1]
                ] = analyzer.dps[image_idx]
                
                # Add peaks to original grid
                ax1.scatter(
                    np.array(peak_x)*2 + x_start,
                    np.array(peak_y)*2 + y_start,
                    color='red', s=50,
                    alpha=[max(0, alpha) for alpha in test_ss[image_idx]]
                )
                
                for px, py, alpha in zip(peak_x, peak_y, test_ss[image_idx]):
                    circle = plt.Circle(
                        (px*2 + x_start, py*2 + y_start),
                        analyzer.radius,
                        color='red',
                        fill=False,
                        alpha=max(0, alpha),
                        linewidth=0.5
                    )
                    ax1.add_patch(circle)
    
    # Now shift the entire grid at once
    y_shift_total = int(max_y_shift)
    x_shift_total = int(max_x_shift)
    
    # Copy the entire original grid to the shifted position
    shifted_grid[
        y_shift_total:y_shift_total + original_grid.shape[0],
        x_shift_total:x_shift_total + original_grid.shape[1]
    ] = original_grid
    
    # Add peaks to shifted grid (shifted as a whole)
    for j in range(grid_size_row):
        for i in range(grid_size_col):
            image_idx = j * grid_size_col + i
            if image_idx < len(analyzer.dps):
                x_start = i * image_size[1] + x_shift_total
                y_start = j * image_size[0] + y_shift_total
                
                ax2.scatter(
                    np.array(peak_x)*2 + x_start,
                    np.array(peak_y)*2 + y_start,
                    color='red', s=50,
                    alpha=[max(0, alpha) for alpha in test_ss[image_idx]]
                )
                
                for px, py, alpha in zip(peak_x, peak_y, test_ss[image_idx]):
                    circle = plt.Circle(
                        (px*2 + x_start, py*2 + y_start),
                        analyzer.radius,
                        color='red',
                        fill=False,
                        alpha=max(0, alpha),
                        linewidth=0.5
                    )
                    ax2.add_patch(circle)
    
    # Plot grids with log scale
    if log_scale:
        im1 = ax1.imshow(original_grid, norm=colors.LogNorm(), cmap=analyzer.cmap)
        im2 = ax2.imshow(shifted_grid, norm=colors.LogNorm(), cmap=analyzer.cmap)
    else:
        im1 = ax1.imshow(original_grid, cmap=analyzer.cmap)
        im2 = ax2.imshow(shifted_grid, cmap=analyzer.cmap)
    
    # Add colorbars
    plt.colorbar(im1, ax=ax1)
    plt.colorbar(im2, ax=ax2)
    
    # Set titles
    ax1.set_title('Original Grid')
    ax2.set_title('Shifted Grid')
    
    plt.tight_layout()
    return fig

def visualize_tomo(tomo_data, intensity_threshold=0.1):
    """
    Visualize 3D tomographic data using interactive scatter plot
    
    Args:
        tomo_data (dict): Dictionary containing volume data and coordinates
        intensity_threshold (float): Minimum intensity to show (0-1 relative to max intensity)
    
    Returns:
        go.Figure: Interactive 3D plotly figure
    """
    # Debug: Print input data structure
    print("Tomo_data keys:", tomo_data.keys())
    
    volume = tomo_data['volume']
    x = tomo_data['coords']['x']
    y = tomo_data['coords']['y']
    z = tomo_data['coords']['z']
    units = tomo_data['units']
    
    # Debug: Print shapes and ranges
    print("Volume shape:", volume.shape)
    print("Coordinate ranges:")
    print(f"x: {x.min():.3f} to {x.max():.3f}")
    print(f"y: {y.min():.3f} to {y.max():.3f}")
    print(f"z: {z.min():.3f} to {z.max():.3f}")
    print(f"Volume range: {volume.min():.3f} to {volume.max():.3f}")
    
    # Create meshgrid of coordinates
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Debug: Print meshgrid shapes
    print("Meshgrid shapes:")
    print(f"X: {X.shape}, Y: {Y.shape}, Z: {Z.shape}")
    
    # Apply threshold and get coordinates
    max_intensity = volume.max()
    threshold = max_intensity * intensity_threshold
    mask = volume > threshold
    
    # Debug: Print threshold info
    print(f"Max intensity: {max_intensity:.3f}")
    print(f"Threshold: {threshold:.3f}")
    print(f"Points above threshold: {np.sum(mask)}")
    
    # Convert masked arrays to 1D arrays for plotting
    x_plot = X[mask]
    y_plot = Y[mask]
    z_plot = Z[mask]
    intensities_plot = volume[mask]
    
    # Debug: Print final plotting arrays
    print("Plotting arrays:")
    print(f"Number of points to plot: {len(x_plot)}")
    if len(x_plot) > 0:
        print(f"x_plot range: {x_plot.min():.3f} to {x_plot.max():.3f}")
        print(f"y_plot range: {y_plot.min():.3f} to {y_plot.max():.3f}")
        print(f"z_plot range: {z_plot.min():.3f} to {z_plot.max():.3f}")
        print(f"intensities range: {intensities_plot.min():.3f} to {intensities_plot.max():.3f}")
    
    # Create figure
    fig = go.Figure(data=[go.Scatter3d(
        x=x_plot,
        y=y_plot,
        z=z_plot,
        mode='markers',
        marker=dict(
            size=3,
            color=intensities_plot,
            colorscale='Viridis',
            opacity=0.6,
            colorbar=dict(title='Intensity')
        ),
        hovertemplate=(
            f"x: %{{x:.3f}} {units}<br>" +
            f"y: %{{y:.3f}} {units}<br>" +
            f"z: %{{z:.3f}} {units}<br>" +
            "Intensity: %{marker.color:.1f}<br>" +
            "<extra></extra>"
        )
    )])
    
    # Update layout
    fig.update_layout(
        title="3D Tomographic Reconstruction",
        scene=dict(
            xaxis_title=f"x ({units})",
            yaxis_title=f"y ({units})",
            zaxis_title=f"z ({units})",
            aspectmode='cube',
            camera=dict(
                eye=dict(x=2, y=2, z=2)
            ),
            bgcolor='white'
        )
    )
    
    return fig



def get_roi_from_frame(frame_number, grid_size_col=11):
    """
    Convert frame number to grid position
    
    Args:
        frame_number (int): Frame number in the scan
        grid_size_col (int): Number of columns in the grid
    
    Returns:
        tuple: (x, y) coordinates in the grid
    """
    # Convert frame number to grid position
    row = frame_number // grid_size_col
    col = frame_number % grid_size_col
    return (col, row)

def get_frames_from_roi(analyzers, df, roi_pos, roi_radius=512, grid_size_col=11):
    """
    Get frame indices for diffraction patterns closest to an ROI across all projections
    
    Args:
        analyzers (list): List of DiffractionAnalyzer objects
        df (pd.DataFrame): DataFrame containing shift information
        roi_pos (tuple): (x, y) position in pixels
        roi_radius (float): Radius around ROI center to consider (in pixels)
        grid_size_col (int): Number of columns in the grid
    
    Returns:
        dict: Dictionary mapping scan numbers to lists of frame indices
    """
    roi_x, roi_y = roi_pos
    print(f"Reference ROI position: ({roi_x:.1f}, {roi_y:.1f})")
    
    roi_frames = {}
    
    for analyzer in analyzers:
        scan = analyzer.scan_number
        
        # Get shifts for this scan
        y_shift = df.loc[df['scanNo'] == scan, 'y_shift'].iloc[0]
        x_shift = df.loc[df['scanNo'] == scan, 'x_shift'].iloc[0]
        
        # Calculate grid positions for all frames in this scan
        n_frames = len(analyzer.dps)
        
        frame_positions = []
        for frame in range(n_frames):
            # Convert frame number to pixel coordinates
            row = (frame // grid_size_col) * 512
            col = (frame % grid_size_col) * 512
            
            # Apply shifts
            shifted_x = col + x_shift
            shifted_y = row + y_shift
            
            frame_positions.append((shifted_x, shifted_y))
        
        # Find frames within ROI
        distances = [np.sqrt((pos[0] - roi_x)**2 + (pos[1] - roi_y)**2) 
                    for pos in frame_positions]
        
        # Get frames within radius
        close_frames = [i for i, d in enumerate(distances) if d <= roi_radius]
        
        if close_frames:
            roi_frames[scan] = close_frames
            print(f"Scan {scan}: Found frames {close_frames} within radius {roi_radius} pixels")
    
    return roi_frames



def select_roi_from_shifted_grid(analyzer, df, grid_size_row=12, grid_size_col=11):
    """
    Display shifted grid and let user select ROI using sliders
    
    Args:
        analyzer: DiffractionAnalyzer object for reference scan
        df: DataFrame with shift information
        grid_size_row, grid_size_col: Grid dimensions
    
    Returns:
        widgets.Output: Output widget that will contain the selected position
    """
    # Get shifts for this scan
    scan = analyzer.scan_number
    y_shift = df.loc[df['scanNo'] == scan, 'y_shift'].iloc[0]
    x_shift = df.loc[df['scanNo'] == scan, 'x_shift'].iloc[0]
    
    # Calculate all frame positions and intensities
    positions = []
    intensities = []
    frame_numbers = []
    
    for frame in range(len(analyzer.dps)):

        row = int((frame // grid_size_col)*512)
        col = int((frame % grid_size_col)*512)
        
        # Apply shifts
        shifted_x = col + x_shift
        shifted_y = row + y_shift
        
        positions.append((shifted_x, shifted_y))
        intensities.append(np.mean(analyzer.dps[frame]))
        frame_numbers.append(frame)
    
    positions = np.array(positions)
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(positions[:, 0], positions[:, 1], 
                        c=intensities, cmap='viridis',
                        s=100, alpha=0.6)
    
    # Add frame number annotations
    for i, (x, y) in enumerate(positions):
        ax.annotate(str(frame_numbers[i]), (x, y), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8)
    
    plt.colorbar(scatter, label='Mean Intensity')
    ax.set_title(f'Shifted Grid for Scan {scan}')
    ax.set_xlabel('X position (shifted)')
    ax.set_ylabel('Y position (shifted)')
    plt.show()
    
    # Create output widget to store the result
    output = widgets.Output()
    
    # Create sliders
    x_slider = FloatSlider(
        value=(x_max + x_min)/2,
        min=x_min,
        max=x_max,
        step=0.1,
        description='X position:',
        continuous_update=False
    )
    
    y_slider = FloatSlider(
        value=(y_max + y_min)/2,
        min=y_min,
        max=y_max,
        step=0.1,
        description='Y position:',
        continuous_update=False
    )
    
    button = Button(description="Confirm Selection")
    
    def on_button_clicked(b):
        with output:
            print(f"Selected position: ({x_slider.value:.2f}, {y_slider.value:.2f})")
            output.roi_pos = (x_slider.value, y_slider.value)
    
    button.on_click(on_button_clicked)
    
    # Display widgets
    display(VBox([x_slider, y_slider, button]))
    display(output)
    
    return output


def plot_roi_frames_interactive(analyzers, roi_frames, df):
    """
    Create interactive plot to view ROI frames across different scans
    
    Args:
        analyzers (list): List of DiffractionAnalyzer objects
        roi_frames (dict): Dictionary mapping scan numbers to frame lists
        df (DataFrame): DataFrame containing scan information
    """
    # Create mapping of analyzers by scan number
    analyzer_dict = {a.scan_number: a for a in analyzers}
    
    # Get list of scans and corresponding angles
    scans = list(roi_frames.keys())
    angles = [df.loc[df['scanNo'] == scan, 'Angle'].values[0] for scan in scans]
    
    # Create widgets
    scan_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(scans)-1,
        step=1,
        description='Scan Index:',
        continuous_update=False
    )
    
    scan_info = widgets.HTML(
        value="Scan info will appear here"
    )
    
    plot_output = widgets.Output()
    
    def update_plot(change):
        with plot_output:
            plot_output.clear_output(wait=True)
            
            scan_idx = scan_slider.value
            scan = scans[scan_idx]
            angle = angles[scan_idx]
            frame = roi_frames[scan][0]
            analyzer = analyzer_dict[scan]
            
            # Update info
            scan_info.value = f"""
            <b>Scan</b>: {scan}<br>
            <b>Angle</b>: {angle:.2f}°<br>
            <b>Frame</b>: {frame}
            """
            
            # Create new figure for each update
            plt.figure(figsize=(8, 6))
            plt.imshow(analyzer.dps[frame], norm=colors.LogNorm())
            plt.colorbar(label='Intensity')
            plt.title(f'Scan {scan}, Frame {frame}, Angle {angle:.1f}°')
            plt.tight_layout()
            plt.show()
    
    # Connect the callback to the slider
    scan_slider.observe(update_plot, names='value')
    
    # Create layout and display
    display(widgets.VBox([scan_slider, scan_info]))
    display(plot_output)
    
    # Initial plot
    update_plot(None)


def plot_diffraction_patterns_interactive(dps):
    """
    Create interactive plot to view all diffraction patterns in an array
    
    Args:
        dps (np.ndarray): Array of diffraction patterns with shape (n_frames, height, width)
    """
    # Get number of frames
    n_frames = len(dps)
    
    # Create widgets
    frame_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=n_frames-1,
        step=1,
        description='Frame:',
        continuous_update=False
    )
    
    # Add log scale toggle
    log_scale = widgets.Checkbox(
        value=True,
        description='Log Scale',
        disabled=False
    )
    
    frame_info = widgets.HTML(
        value="Frame info will appear here"
    )
    
    plot_output = widgets.Output()
    
    def update_plot(change):
        with plot_output:
            plot_output.clear_output(wait=True)
            
            frame_idx = frame_slider.value
            
            # Update info
            frame_info.value = f"""
            <b>Frame</b>: {frame_idx}<br>
            <b>Max Intensity</b>: {dps[frame_idx].max():.2e}<br>
            <b>Min Intensity</b>: {dps[frame_idx].min():.2e}
            """
            
            # Create new figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot with or without log scale
            if log_scale.value:
                im = ax.imshow(dps[frame_idx], 
                             norm=colors.LogNorm())
            else:
                im = ax.imshow(dps[frame_idx])
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='Intensity')
            
            # Add title and labels
            ax.set_title(f'Frame {frame_idx}')
            ax.set_xlabel('Pixel')
            ax.set_ylabel('Pixel')
            
            plt.tight_layout()
            plt.show()
    
    # Connect callbacks
    frame_slider.observe(update_plot, names='value')
    log_scale.observe(update_plot, names='value')
    
    # Create layout with controls side by side
    controls = widgets.HBox([frame_slider, log_scale])
    
    # Display widgets and plot
    display(widgets.VBox([controls, frame_info]))
    display(plot_output)
    
    # Initial plot
    update_plot(None)


def normalize_roi_peak_intensities(analyzers, roi_frames, tolerance=8):
    """
    Calculate normalized peak intensities for ROI frames
    
    Args:
        analyzers (list): List of DiffractionAnalyzer objects
        roi_frames (dict): Dictionary mapping scan numbers to frame lists
        tolerance (int): Pixel distance within which peaks are considered the same
    
    Returns:
        dict: Contains summed and normalized intensities for each peak group
    """
    # Collect all peak data
    all_peaks = []
    peak_intensities = []
    scan_frame_info = []
    
    for analyzer in analyzers:
        scan = analyzer.scan_number
        if scan in roi_frames:
            frame = roi_frames[scan][0]  # Take first frame
            intensities, _ = analyzer.calculate_frame_intensities_and_orientations(
                analyzer.dps[frame], 
                analyzer.dps[0],
                plot=False
            )
            
            # Store peak positions and intensities
            for peak_idx, peak in enumerate(analyzer.peaks):
                all_peaks.append(np.array(peak) * 2)  # Convert to 512x512 coordinates
                peak_intensities.append(intensities[peak_idx])
                scan_frame_info.append((scan, frame, peak_idx))
    
    # Group peaks that are within tolerance distance
    peak_groups = []
    peak_indices = list(range(len(all_peaks)))
    
    while peak_indices:
        current_idx = peak_indices.pop(0)
        current_peak = all_peaks[current_idx]
        current_group = [current_idx]
        
        # Find all peaks within tolerance
        i = 0
        while i < len(peak_indices):
            idx = peak_indices[i]
            peak = all_peaks[idx]
            distance = np.sqrt(np.sum((current_peak - peak)**2))
            
            if distance <= tolerance:
                current_group.append(idx)
                peak_indices.pop(i)
            else:
                i += 1
        
        peak_groups.append(current_group)
    
    # Add visualization of peak grouping
    plot_peak_grouping_diagnostic(all_peaks, peak_groups, peak_intensities,
                                "Original Method: Peak Grouping") 
    
    # Calculate sums and normalize intensities
    results = {
        'peak_groups': [],  # List of peak positions in each group
        'summed_intensities': [],  # Sum of intensities for each group
        'normalized_intensities': {},  # Dict of {scan: {peak_group_idx: norm_intensity}}
        'peak_group_centers': []  # Average position of peaks in each group
    }
    
    for group_idx, group in enumerate(peak_groups):
        # Calculate group properties
        group_peaks = [all_peaks[i] for i in group]
        group_intensities = [peak_intensities[i] for i in group]
        group_sum = sum(group_intensities)
        group_center = np.mean(group_peaks, axis=0)
        
        results['peak_groups'].append(group_peaks)
        results['summed_intensities'].append(group_sum)
        results['peak_group_centers'].append(group_center)
        
        # Calculate normalized intensities for each scan
        for i in group:
            scan, frame, peak_idx = scan_frame_info[i]
            if scan not in results['normalized_intensities']:
                results['normalized_intensities'][scan] = {}
            
            norm_intensity = peak_intensities[i] / group_sum if group_sum > 0 else 0
            results['normalized_intensities'][scan][group_idx] = norm_intensity
    
    # Print summary
    print(f"Found {len(peak_groups)} unique peak groups")
    for i, center in enumerate(results['peak_group_centers']):
        print(f"Peak group {i}: center at ({center[0]:.1f}, {center[1]:.1f}), "
              f"total intensity: {results['summed_intensities'][i]:.2f}")
        
    plot_normalization_comparison(results['peak_groups'], 
                            results['normalized_intensities'],
                            results['peak_group_centers'])
    return results


def normalize_roi_peak_intensities2(analyzers, roi_frames, tolerance=8):
    """
    Calculate normalized peak intensities for ROI frames and summed pattern
    
    Args:
        analyzers (list): List of DiffractionAnalyzer objects
        roi_frames (dict): Dictionary mapping scan numbers to frame lists
        tolerance (int): Pixel distance within which peaks are considered the same
    
    Returns:
        dict: Contains summed and normalized intensities for each peak group
    """
    # Create summed pattern first
    summed_pattern = None
    reference_analyzer = analyzers[0]  # Use first analyzer as reference
    
    for analyzer in analyzers:
        scan = analyzer.scan_number
        if scan in roi_frames:
            frame = roi_frames[scan][0]
            if summed_pattern is None:
                summed_pattern = analyzer.dps[frame].copy()
            else:
                summed_pattern += analyzer.dps[frame]
    
    # Collect all peak data from all analyzers
    all_peaks = []
    peak_intensities = []
    scan_frame_info = []
    
    for analyzer in analyzers:
        scan = analyzer.scan_number
        if scan in roi_frames:
            frame = roi_frames[scan][0]
            intensities, _ = analyzer.calculate_frame_intensities_and_orientations(
                analyzer.dps[frame], 
                analyzer.dps[0],
                plot=False
            )
            
            # Store peak positions and intensities
            for peak_idx, peak in enumerate(analyzer.peaks):
                all_peaks.append(np.array(peak) * 2)  # Convert to 512x512 coordinates
                peak_intensities.append(intensities[peak_idx])
                scan_frame_info.append((scan, frame, peak_idx))
    
    # Sort all peaks by intensity (highest to lowest)
    sorted_indices = np.argsort(peak_intensities)[::-1]
    peak_indices = sorted_indices.tolist()
    
    # Group peaks that are within tolerance distance, prioritizing higher intensities
    peak_groups = []
    
    while peak_indices:
        current_idx = peak_indices.pop(0)  # Take highest intensity peak
        current_peak = all_peaks[current_idx]
        current_intensity = peak_intensities[current_idx]
        current_group = [current_idx]
        
        i = 0
        while i < len(peak_indices):
            idx = peak_indices[i]
            peak = all_peaks[idx]
            intensity = peak_intensities[idx]
            distance = np.sqrt(np.sum((current_peak - peak)**2))
            
            if distance <= tolerance:
                # Always add to group but keep the highest intensity peak as center
                current_group.append(idx)
                peak_indices.pop(i)
            else:
                i += 1
        
        peak_groups.append(current_group)
        
    # Add visualization of peak grouping
    plot_peak_grouping_diagnostic(all_peaks, peak_groups, peak_intensities,
                                "Version 2: Intensity-Prioritized Peak Grouping")
    
    # Add summed pattern visualization
    plt.figure(figsize=(8, 8))
    plt.imshow(summed_pattern, norm=colors.LogNorm())
    plt.colorbar(label='Intensity')
    plt.title('Summed Pattern')
    plt.show()
    # Calculate summed pattern intensities for each unique peak position
    results = {
        'peak_groups': [],
        'summed_intensities': [],
        'normalized_intensities': {},
        'peak_group_centers': [],
        'summed_pattern': summed_pattern,
        'summed_pattern_intensities': []
    }
    
    for group_idx, group in enumerate(peak_groups):
        # Calculate group properties
        group_peaks = [all_peaks[i] for i in group]
        group_intensities = [peak_intensities[i] for i in group]
        
        # Use the position of the highest intensity peak as the center
        max_intensity_idx = np.argmax(group_intensities)
        group_center = np.round(group_peaks[max_intensity_idx]).astype(int)
        group_sum = sum(group_intensities)
        
        # Calculate intensity for this peak in summed pattern
        summed_intensity = reference_analyzer.calculate_frame_intensities_and_orientations(
            summed_pattern,
            reference_analyzer.dps[0],
            plot=False
        )[0][0]
        
        results['peak_groups'].append(group_peaks)
        results['summed_intensities'].append(group_sum)
        results['peak_group_centers'].append(group_center)
        results['summed_pattern_intensities'].append(summed_intensity)
        
        # Calculate normalized intensities for each scan
        for i in group:
            scan, frame, peak_idx = scan_frame_info[i]
            if scan not in results['normalized_intensities']:
                results['normalized_intensities'][scan] = {}
            
            # Normalize by summed pattern intensity
            norm_intensity = peak_intensities[i] / summed_intensity if summed_intensity > 0 else 0
            results['normalized_intensities'][scan][group_idx] = norm_intensity
    
    # Print summary
    print(f"Found {len(peak_groups)} unique peak groups")
    for i, center in enumerate(results['peak_group_centers']):
        print(f"Peak group {i}: center at ({center[0]}, {center[1]}), "
              f"summed intensity: {results['summed_pattern_intensities'][i]:.2f}")
    
    # Add visualization of normalization results
    plot_normalization_comparison(results['peak_groups'], 
                                results['normalized_intensities'],
                                results['peak_group_centers'])
    return results



def normalize_roi_peak_intensities3(analyzers, roi_frames, tolerance=8, intensity_threshold=0.01):
    """
    Calculate normalized peak intensities for ROI frames and summed pattern
    
    Args:
        analyzers (list): List of DiffractionAnalyzer objects
        roi_frames (dict): Dictionary mapping scan numbers to frame lists
        tolerance (int): Pixel distance within which peaks are considered the same
        intensity_threshold (float): Minimum normalized intensity for a peak to be included
    
    Returns:
        dict: Contains summed and normalized intensities for each peak group
    """
      # Create summed pattern first
    summed_pattern = None
    reference_analyzer = analyzers[0]  # Use first analyzer as reference
    
    for analyzer in analyzers:
        scan = analyzer.scan_number
        if scan in roi_frames:
            frame = roi_frames[scan][0]
            if summed_pattern is None:
                summed_pattern = analyzer.dps[frame].copy()
            else:
                summed_pattern += analyzer.dps[frame]
    
    # Collect all peak data from all analyzers
    all_peaks = []
    peak_intensities = []
    scan_frame_info = []
    
    for analyzer in analyzers:
        scan = analyzer.scan_number
        if scan in roi_frames:
            frame = roi_frames[scan][0]
            intensities, _ = analyzer.calculate_frame_intensities_and_orientations(
                analyzer.dps[frame], 
                analyzer.dps[0],
                plot=False
            )
            
            # Store peak positions and intensities
            for peak_idx, peak in enumerate(analyzer.peaks):
                all_peaks.append(np.array(peak) * 2)  # Convert to 512x512 coordinates
                peak_intensities.append(intensities[peak_idx])
                scan_frame_info.append((scan, frame, peak_idx))
    
    # Sort all peaks by intensity (highest to lowest)
    sorted_indices = np.argsort(peak_intensities)[::-1]
    peak_indices = sorted_indices.tolist()
    
    # Group peaks that are within tolerance distance, prioritizing higher intensities
    peak_groups = []
    
    while peak_indices:
        current_idx = peak_indices.pop(0)  # Take highest intensity peak
        current_peak = all_peaks[current_idx]
        current_intensity = peak_intensities[current_idx]
        current_group = [current_idx]
        
        i = 0
        while i < len(peak_indices):
            idx = peak_indices[i]
            peak = all_peaks[idx]
            intensity = peak_intensities[idx]
            distance = np.sqrt(np.sum((current_peak - peak)**2))
            
            if distance <= tolerance:
                # Always add to group but keep the highest intensity peak as center
                current_group.append(idx)
                peak_indices.pop(i)
            else:
                i += 1
        
        peak_groups.append(current_group)
    # Calculate summed pattern intensities for each unique peak position
    results = {
        'peak_groups': [],
        'summed_intensities': [],
        'normalized_intensities': {},
        'peak_group_centers': [],
        'summed_pattern': summed_pattern,
        'summed_pattern_intensities': []
    }
    
    # First pass: calculate all normalized intensities
    temp_groups = []
    for group_idx, group in enumerate(peak_groups[0:2]):
        group_peaks = [all_peaks[i] for i in group]
        group_intensities = [peak_intensities[i] for i in group]
        max_intensity_idx = np.argmax(group_intensities)
        group_center = np.round(group_peaks[max_intensity_idx]).astype(int)
        group_sum = sum(group_intensities)
        
        original_coords_peaks = np.array([[peak[0]//2, peak[1]//2] for peak in group_peaks])
        # Calculate intensity for this peak in summed pattern
        summed_intensity = reference_analyzer.calculate_frame_intensities_and_orientations_peaks(
            summed_pattern,
            reference_analyzer.dps[0]*len(reference_analyzer.dps[0]),
            peaks=original_coords_peaks,
            plot=False
        )[0]#[0]
        

        # Store temporary results including normalized intensities
        temp_group_info = {
            'peaks': group_peaks,
            'center': group_center,
            'summed_intensity': summed_intensity,
            'group_sum': group_sum,
            'normalized_intensities': {},
            'max_norm_intensity': 0.0  # Track maximum normalized intensity for this group
        }
        
        # Calculate normalized intensities for each scan
        for idx,i in enumerate(group):
            scan, frame, peak_idx = scan_frame_info[i]
            if scan in roi_frames:  # Only consider ROI frames
                norm_intensity = peak_intensities[i] / summed_intensity[idx] if summed_intensity[idx] > 0 else 0
                temp_group_info['normalized_intensities'][scan] = norm_intensity
                temp_group_info['max_norm_intensity'] = max(
                    temp_group_info['max_norm_intensity'], 
                    norm_intensity
                )
        
        temp_groups.append(temp_group_info)
    
    # Second pass: only keep groups that have significant intensity in ROI frames
    valid_group_idx = 0
    for group_info in temp_groups:
        if group_info['max_norm_intensity'] >= intensity_threshold:
            # Add to final results
            results['peak_groups'].append(group_info['peaks'])
            results['summed_intensities'].append(group_info['group_sum'])
            results['peak_group_centers'].append(group_info['center'])
            results['summed_pattern_intensities'].append(group_info['summed_intensity'])
            
            # Add normalized intensities for each scan
            for scan, norm_intensity in group_info['normalized_intensities'].items():
                if scan not in results['normalized_intensities']:
                    results['normalized_intensities'][scan] = {}
                results['normalized_intensities'][scan][valid_group_idx] = norm_intensity
            
            valid_group_idx += 1
    
    # Print summary
    print(f"Found {len(results['peak_groups'])} valid peak groups")
    for i, center in enumerate(results['peak_group_centers']):
        print(f"Peak group {i}: center at ({center[0]}, {center[1]})")
        print(f"Summed intensities for peaks in group:")
        for j, intensity in enumerate(results['summed_pattern_intensities'][i]):
            print(f"  Peak {j}: {intensity:.2f}")
            
    return results



def normalize_roi_peak_intensities4(analyzers, roi_frames, tolerance=8, intensity_threshold=0.01):
    # Create summed pattern first (keep this part)
    summed_pattern = None
    reference_analyzer = analyzers[0]
    
    for analyzer in analyzers:
        scan = analyzer.scan_number
        if scan in roi_frames:
            frame = roi_frames[scan][0]
            if summed_pattern is None:
                summed_pattern = analyzer.dps[frame].copy()
            else:
                summed_pattern += analyzer.dps[frame]
    # Collect all peak data from all analyzers
    all_peaks = []
    peak_intensities = []
    scan_frame_info = []
    
    for analyzer in analyzers:
        scan = analyzer.scan_number
        if scan in roi_frames:
            frame = roi_frames[scan][0]
            intensities, _ = analyzer.calculate_frame_intensities_and_orientations(
                analyzer.dps[frame], 
                analyzer.dps[0],
                plot=False
            )
            
            # Store peak positions and intensities
            for peak_idx, peak in enumerate(analyzer.peaks):
                all_peaks.append(np.array(peak) * 2)  # Convert to 512x512 coordinates
                peak_intensities.append(intensities[peak_idx])
                scan_frame_info.append((scan, frame, peak_idx))
    
    # Sort all peaks by intensity (highest to lowest)
    sorted_indices = np.argsort(peak_intensities)[::-1]
    peak_indices = sorted_indices.tolist()
    
    # Group peaks that are within tolerance distance, prioritizing higher intensities
    peak_groups = []
    
    while peak_indices:
        current_idx = peak_indices.pop(0)  # Take highest intensity peak
        current_peak = all_peaks[current_idx]
        current_intensity = peak_intensities[current_idx]
        current_group = [current_idx]
        
        i = 0
        while i < len(peak_indices):
            idx = peak_indices[i]
            peak = all_peaks[idx]
            intensity = peak_intensities[idx]
            distance = np.sqrt(np.sum((current_peak - peak)**2))
            
            if distance <= tolerance:
                # Always add to group but keep the highest intensity peak as center
                current_group.append(idx)
                peak_indices.pop(i)
            else:
                i += 1
        
        peak_groups.append(current_group)
    
    results = {
        'peak_groups': [],
        'peak_group_centers': [],
        'summed_pattern': summed_pattern,
        'summed_intensities': {},      # Change to store by peak position
        'frame_intensities': {},       # Add to store individual frame intensities
        'normalized_intensities': {}
    }
    
    # For each unique peak position
    for group_idx, group in enumerate(peak_groups):
        group_peaks = [all_peaks[i] for i in group]
        # Use the highest intensity peak position as the representative position
        max_intensity_idx = np.argmax([peak_intensities[i] for i in group])
        peak_center = np.round(group_peaks[max_intensity_idx]).astype(int)
        
        # Calculate summed pattern intensity at this position
        summed_intensity = reference_analyzer.calculate_frame_intensities_and_orientations_peaks(
            summed_pattern,
            reference_analyzer.dps[0]*len(reference_analyzer.dps[0]),
            peaks=np.array([[peak_center[0]//2, peak_center[1]//2]]),
            plot=False
        )[0][0]
        
        results['peak_groups'].append(group_peaks)  # Add this line
        results['peak_group_centers'].append(peak_center)
        results['summed_intensities'][group_idx] = summed_intensity
        results['frame_intensities'][group_idx] = {}
        results['normalized_intensities'][group_idx] = {}
        
        # Calculate intensity at this position for each frame
        for analyzer in analyzers:
            scan = analyzer.scan_number
            if scan in roi_frames:
                frame = roi_frames[scan][0]
                frame_intensity = reference_analyzer.calculate_frame_intensities_and_orientations_peaks(
                    analyzer.dps[frame],
                    analyzer.dps[0],
                    peaks=np.array([[peak_center[0]//2, peak_center[1]//2]]),
                    plot=False
                )[0][0]
                
                results['frame_intensities'][group_idx][scan] = frame_intensity
                results['normalized_intensities'][group_idx][scan] = (
                    frame_intensity / summed_intensity if summed_intensity > 0 else 0
                )
    
    return results



def plot_roi_frames_with_peaks(analyzers, roi_frames, normalized_results, df):
    """
    Interactive plot of ROI frames with peak groups overlaid
    
    Args:
        analyzers (list): List of DiffractionAnalyzer objects
        roi_frames (dict): Dictionary mapping scan numbers to frame lists
        normalized_results (dict): Results from normalize_roi_peak_intensities
        df (DataFrame): DataFrame containing scan information
    """
    # Create mapping of analyzers by scan number
    analyzer_dict = {a.scan_number: a for a in analyzers}
    
    # Get list of scans and corresponding angles
    scans = list(roi_frames.keys())
    angles = [df.loc[df['scanNo'] == scan, 'Angle'].values[0] for scan in scans]
    
    # Convert peak centers to integers
    peak_centers = [np.round(center).astype(int) for center in normalized_results['peak_group_centers']]
    
    # Calculate global min and max intensities for alpha normalization
    all_intensities = []
    for scan in normalized_results['normalized_intensities']:
        all_intensities.extend(normalized_results['normalized_intensities'][scan].values())
    global_min = min(all_intensities)
    global_max = max(all_intensities)
    intensity_range = global_max - global_min if global_max > global_min else 1
    
    # Create widgets
    scan_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(scans)-1,
        step=1,
        description='Scan Index:',
        continuous_update=False
    )
    
    show_peaks = widgets.Checkbox(
        value=True,
        description='Show peaks',
        disabled=False
    )
    
    scan_info = widgets.HTML(
        value="Scan info will appear here"
    )
    
    plot_output = widgets.Output()
    
    def update_plot(change):
        with plot_output:
            plot_output.clear_output(wait=True)
            
            scan_idx = scan_slider.value
            scan = scans[scan_idx]
            angle = angles[scan_idx]
            frame = roi_frames[scan][0]
            analyzer = analyzer_dict[scan]
            
            # Update info
            peak_info = ""
            if scan in normalized_results['normalized_intensities']:
                for peak_idx, norm_intensity in normalized_results['normalized_intensities'][scan].items():
                    center = peak_centers[peak_idx]
                    # Calculate globally normalized alpha
                    alpha = (norm_intensity - global_min) / intensity_range
                    alpha = max(0.01, min(1, alpha))  # Ensure minimum visibility
                    peak_info += f"Peak {peak_idx} at ({center[0]}, {center[1]}): {norm_intensity:.8f} (alpha: {alpha:.3f})<br>"
            
            scan_info.value = f"""
            <b>Scan</b>: {scan}<br>
            <b>Angle</b>: {angle:.2f}°<br>
            <b>Frame</b>: {frame}<br>
            <b>Peak Intensities</b>:<br>{peak_info}
            """
            
            # Create plot
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(analyzer.dps[frame], norm=colors.LogNorm())
            plt.colorbar(im, ax=ax, label='Intensity')
            
            # Overlay peaks if checkbox is checked
            if show_peaks.value and scan in normalized_results['normalized_intensities']:
                for peak_idx, norm_intensity in normalized_results['normalized_intensities'][scan].items():
                    center = peak_centers[peak_idx]
                    # Calculate globally normalized alpha
                    alpha = (norm_intensity - global_min) / intensity_range
                    alpha = max(0.01, min(1, alpha))  # Ensure minimum visibility
                    
                    # Plot peak position
                    ax.scatter(center[1], center[0], color='red', s=100, alpha=alpha,
                             label=f'Peak {peak_idx}')
                    # Add circle around peak
                    circle = plt.Circle((center[1], center[0]), 8, color='red', 
                                     fill=False, alpha=alpha)
                    ax.add_patch(circle)
                
                ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
            
            ax.set_title(f'Scan {scan}, Frame {frame}, Angle {angle:.1f}°')
            plt.tight_layout()
            plt.show()
    
    # Connect callbacks
    scan_slider.observe(update_plot, names='value')
    show_peaks.observe(update_plot, names='value')
    
    # Create layout and display
    controls = widgets.VBox([scan_slider, show_peaks, scan_info])
    display(controls)
    display(plot_output)
    
    # Initial plot
    update_plot(None)

def plot_summed_pattern_with_peaks(normalized_results):
    """
    Plot summed pattern with all peak groups overlaid
    
    Args:
        normalized_results (dict): Results from normalize_roi_peak_intensities2
    """
    # Get summed pattern and peak centers
    summed_pattern = normalized_results['summed_pattern']
    peak_centers = [np.round(center).astype(int) for center in normalized_results['peak_group_centers']]
    summed_intensities = normalized_results['summed_pattern_intensities'][0]
    
    # Calculate alpha values for visualization
    nonzero_intensities = [i for i in summed_intensities if i > 0]
    if nonzero_intensities:
        global_min = min(nonzero_intensities)
        global_max = max(nonzero_intensities)
    else:
        global_min = 0
        global_max = 1
    
    intensity_range = global_max - global_min if global_max > global_min else 1
    
    def calculate_alpha(intensity):
        """Helper function to calculate alpha value"""
        if intensity <= 0:
            return 0.0
        else:
            alpha = np.log1p(intensity/global_min) / np.log1p(global_max/global_min)
            return max(0, min(1, alpha))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(summed_pattern, norm=colors.LogNorm())
    plt.colorbar(im, ax=ax, label='Intensity')
    
    # Add peaks
    peak_info = []
    for peak_idx, (center, intensity) in enumerate(zip(peak_centers, summed_intensities)):
        alpha = calculate_alpha(intensity)
        if alpha > 0:
            # Plot peak position
            ax.scatter(center[1], center[0], color='red', s=100, alpha=alpha,
                      label=f'Peak {peak_idx}')
            # Add circle around peak
            circle = plt.Circle((center[1], center[0]), 8, color='red', 
                              fill=False, alpha=alpha)
            ax.add_patch(circle)
            
            # Store peak info for printing
            peak_info.append(f"Peak {peak_idx}: ({center[0]}, {center[1]}), "
                           f"intensity: {intensity:.2f}, alpha: {alpha:.3f}")
    
    ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
    ax.set_title('Summed Pattern with Peak Groups')
    plt.tight_layout()
    
    # Print peak information
    print("Peak Information:")
    for info in peak_info:
        print(info)
    
    plt.show()
    
def plot_summed_pattern_with_peaks2(normalized_results, analyzers):
    """
    Plot summed pattern with all peak groups overlaid
    
    Args:
        normalized_results (dict): Results from normalize_roi_peak_intensities2
    """
    # Get summed pattern and peak centers
    summed_pattern = normalized_results['summed_pattern']
    peak_centers = [np.round(center).astype(int) for center in normalized_results['peak_group_centers']]
    
    # Get the actual summed pattern intensities for each peak
    reference_analyzer = analyzers[0]  # You'll need to pass analyzers to this function
    peak_centers_original = [[center[0]//2, center[1]//2] for center in peak_centers]
    summed_intensities = reference_analyzer.calculate_frame_intensities_and_orientations_peaks(
        summed_pattern,
        reference_analyzer.dps[0]*len(reference_analyzer.dps[0]),
        peaks=np.array(peak_centers_original),
        plot=False
    )[0]
    
    # Calculate alpha values for visualization
    nonzero_intensities = [i for i in summed_intensities if i > 0]
    if nonzero_intensities:
        global_min = min(nonzero_intensities)
        global_max = max(nonzero_intensities)
    else:
        global_min = 0
        global_max = 1
    
    intensity_range = global_max - global_min if global_max > global_min else 1
    
    def calculate_alpha(intensity):
        """Helper function to calculate alpha value"""
        if intensity <= 0:
            return 0.0
        else:
            # Use log scale for better visualization
            alpha = np.log1p(intensity/global_min) / np.log1p(global_max/global_min)
            return max(0.1, min(1, alpha))  # Minimum alpha of 0.1 for visibility
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(summed_pattern, norm=colors.LogNorm())
    plt.colorbar(im, ax=ax, label='Intensity')
    
    # Add peaks
    peak_info = []
    for peak_idx, (center, intensity) in enumerate(zip(peak_centers, summed_intensities)):
        alpha = calculate_alpha(intensity)
        # Plot peak position
        ax.scatter(center[1], center[0], color='red', s=100, alpha=alpha,
                  label=f'Peak {peak_idx}')
        # Add circle around peak
        circle = plt.Circle((center[1], center[0]), 8, color='red', 
                          fill=False, alpha=alpha)
        ax.add_patch(circle)
        
        # Store peak info for printing
        peak_info.append(f"Peak {peak_idx}: ({center[0]}, {center[1]}), "
                       f"intensity: {intensity:.2f}, alpha: {alpha:.3f}")
    
    ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
    ax.set_title('Summed Pattern with Peak Groups')
    plt.tight_layout()
    
    # Print peak information
    print("Peak Information:")
    for info in peak_info:
        print(info)
    
    plt.show()
    
def save_normalized_results(normalized_results, analyzers, roi_frames, df, save_path):
    """
    Save normalized results to CSV and NPY files
    
    Args:
        normalized_results (dict): Results from normalize_roi_peak_intensities2
        analyzers (list): List of DiffractionAnalyzer objects
        roi_frames (dict): Dictionary mapping scan numbers to frame lists
        df (DataFrame): DataFrame containing scan information
        save_path (str or Path): Directory to save the analysis files
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for CSV
    csv_data = []
    
    # Get scan numbers and corresponding angles (only for scans in normalized_results)
    valid_scans = list(set(roi_frames.keys()) & set(normalized_results['normalized_intensities'].keys()))
    angles = {scan: df.loc[df['scanNo'] == scan, 'Angle'].values[0] for scan in valid_scans}
    
    # Sort scans by phi angle
    sorted_scans = sorted(valid_scans, key=lambda x: angles[x])
    
    # Create frame index mapping
    frame_index = {scan: idx for idx, scan in enumerate(sorted_scans)}
    
    # For each scan and peak group
    for scan in sorted_scans:
        phi = angles[scan]
        scan_frame = roi_frames[scan][0]
        frame = frame_index[scan]  # Add frame index
        
        for peak_idx in normalized_results['normalized_intensities'][scan]:
            peak_center = normalized_results['peak_group_centers'][peak_idx]
            norm_intensity = normalized_results['normalized_intensities'][scan][peak_idx]
            summed_intensity = normalized_results['summed_pattern_intensities'][peak_idx]
            
            csv_data.append({
                'scan': scan,
                'phi': phi,
                'frame': frame,
                'scan_frame': scan_frame,
                'peak_idx': peak_idx,
                'x': peak_center[1],  # Swap x,y for consistency
                'y': peak_center[0],
                'intensity': norm_intensity,
                'summed_intensity': summed_intensity
            })
    
    # Save as CSV
    df_results = pd.DataFrame(csv_data)
    csv_path = save_path / 'normalized_peak_analysis.csv'
    df_results.to_csv(csv_path, index=False)
    
    # Prepare data for NPY
    # Reshape data into more convenient arrays
    n_scans = len(sorted_scans)
    n_peaks = len(normalized_results['peak_group_centers'])
    
    peak_positions = np.array([(center[1], center[0]) for center in normalized_results['peak_group_centers']])  # Swap x,y
    norm_intensities_array = np.zeros((n_scans, n_peaks))
    phi_angles = np.array([angles[scan] for scan in sorted_scans])  # Use sorted scans
    
    # Fill intensity array using sorted scan order
    for scan_idx, scan in enumerate(sorted_scans):
        if scan in normalized_results['normalized_intensities']:
            for peak_idx in normalized_results['normalized_intensities'][scan]:
                norm_intensities_array[scan_idx, peak_idx] = normalized_results['normalized_intensities'][scan][peak_idx]
    
    npy_data = {
        'scans': np.array(sorted_scans),  # Use sorted scans
        'phi_angles': phi_angles,
        'peak_positions': peak_positions,
        'normalized_intensities': norm_intensities_array,
        'summed_pattern': normalized_results['summed_pattern'],
        'summed_intensities': np.array(normalized_results['summed_pattern_intensities'])
    }
    
    npy_path = save_path / 'normalized_peak_analysis.npy'
    np.save(npy_path, npy_data)
    
    print(f"Analysis saved to:")
    print(f"CSV: {csv_path}")
    print(f"NPY: {npy_path}")
    
    return df_results, npy_data



def plot_peak_grouping_diagnostic(all_peaks, peak_groups, peak_intensities, title="Peak Grouping Visualization"):
    """Helper function to visualize peak grouping"""
    plt.figure(figsize=(12, 8))
    
    # Plot all peaks as small dots
    all_peaks = np.array(all_peaks)
    plt.scatter(all_peaks[:, 1], all_peaks[:, 0], c='gray', alpha=0.3, s=20, label='All Peaks')
    
    # Plot each group with a different color
    colors = plt.cm.rainbow(np.linspace(0, 1, len(peak_groups)))
    for group_idx, (group, color) in enumerate(zip(peak_groups, colors)):
        group_peaks = np.array([all_peaks[i] for i in group])
        group_intensities = np.array([peak_intensities[i] for i in group])
        
        # Size points by intensity
        sizes = 50 + 200 * (group_intensities / max(peak_intensities))
        
        plt.scatter(group_peaks[:, 1], group_peaks[:, 0], 
                   c=[color], s=sizes, alpha=0.6,
                   label=f'Group {group_idx}')
        
        # Draw lines connecting peaks in the same group
        if len(group_peaks) > 1:
            center = group_peaks.mean(axis=0)
            for peak in group_peaks:
                plt.plot([center[1], peak[1]], [center[0], peak[0]], 
                        c=color, alpha=0.2)
    
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_normalization_comparison(peak_groups, normalized_intensities, peak_group_centers):
    """Helper function to visualize normalization results"""
    # Get unique scans
    scans = sorted(normalized_intensities.keys())
    n_scans = len(scans)
    n_groups = len(peak_groups)
    
    # Create intensity matrix
    intensity_matrix = np.zeros((n_scans, n_groups))
    for i, scan in enumerate(scans):
        for group_idx in normalized_intensities[scan]:
            intensity_matrix[i, group_idx] = normalized_intensities[scan][group_idx]
    
    # Plot heatmap
    plt.figure(figsize=(15, 8))
    plt.subplot(121)
    im = plt.imshow(intensity_matrix, aspect='auto', cmap='viridis')
    plt.colorbar(im, label='Normalized Intensity')
    plt.xlabel('Peak Group')
    plt.ylabel('Scan Number')
    plt.title('Normalized Intensities Heatmap')
    
    # Plot peak positions with intensity
    plt.subplot(122)
    peak_centers = np.array(peak_group_centers)
    max_intensities = np.max(intensity_matrix, axis=0)
    
    plt.scatter(peak_centers[:, 1], peak_centers[:, 0], 
               c=max_intensities, cmap='viridis',
               s=100, alpha=0.6)
    
    plt.colorbar(label='Max Normalized Intensity')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Peak Positions colored by Max Intensity')
    
    plt.tight_layout()
    plt.show()
















def plot_summed_pattern_with_peaks4(normalized_results):
    """
    Plot summed pattern with all peak groups overlaid
    
    Args:
        normalized_results (dict): Results from normalize_roi_peak_intensities4
    """
    # Get summed pattern and peak centers
    summed_pattern = normalized_results['summed_pattern']
    peak_centers = normalized_results['peak_group_centers']
    
    # Calculate alpha values for visualization
    summed_intensities = [normalized_results['summed_intensities'][i] for i in range(len(peak_centers))]
    nonzero_intensities = [i for i in summed_intensities if i > 0]
    
    if nonzero_intensities:
        global_min = min(nonzero_intensities)
        global_max = max(nonzero_intensities)
    else:
        global_min = 0
        global_max = 1
    
    def calculate_alpha(intensity):
        """Helper function to calculate alpha value"""
        if intensity <= 0:
            return 0.0
        else:
            # Use log scale for better visualization
            alpha = np.log1p(intensity/global_min) / np.log1p(global_max/global_min)
            return max(0.1, min(1, alpha))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(summed_pattern, norm=colors.LogNorm())
    plt.colorbar(im, ax=ax, label='Intensity')
    
    # Add peaks
    peak_info = []
    for peak_idx, (center, intensity) in enumerate(zip(peak_centers, summed_intensities)):
        alpha = calculate_alpha(intensity)
        # Plot peak position
        ax.scatter(center[1], center[0], color='red', s=100, alpha=alpha,
                  label=f'Peak {peak_idx}')
        # Add circle around peak
        circle = plt.Circle((center[1], center[0]), 8, color='red', 
                          fill=False, alpha=alpha)
        ax.add_patch(circle)
        
        # Store peak info for printing
        peak_info.append(f"Peak {peak_idx}: ({center[0]}, {center[1]}), "
                       f"intensity: {intensity:.2f}, alpha: {alpha:.3f}")
    
    ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
    ax.set_title('Summed Pattern with Peak Groups')
    plt.tight_layout()
    
    # Print peak information
    print("Peak Information:")
    for info in peak_info:
        print(info)
    
    plt.show()

def plot_roi_frames_with_peaks4(analyzers, roi_frames, normalized_results, df):
    """
    Interactive plot of ROI frames with peak groups overlaid
    
    Args:
        analyzers (list): List of DiffractionAnalyzer objects
        roi_frames (dict): Dictionary mapping scan numbers to frame lists
        normalized_results (dict): Results from normalize_roi_peak_intensities4
        df (DataFrame): DataFrame containing scan information
    """
    # Create mapping of analyzers by scan number
    analyzer_dict = {a.scan_number: a for a in analyzers}
    
    # Get list of scans and corresponding angles
    scans = list(roi_frames.keys())
    angles = [df.loc[df['scanNo'] == scan, 'Angle'].values[0] for scan in scans]
    
    # Get peak centers
    peak_centers = normalized_results['peak_group_centers']
    
    # Calculate global min and max intensities for alpha normalization
    all_intensities = []
    for scan in scans:
        for peak_idx in range(len(peak_centers)):
            if peak_idx in normalized_results['normalized_intensities'] and scan in normalized_results['normalized_intensities'][peak_idx]:
                all_intensities.append(normalized_results['normalized_intensities'][peak_idx][scan])
    
    global_min = min(all_intensities) if all_intensities else 0
    global_max = max(all_intensities) if all_intensities else 1
    intensity_range = global_max - global_min if global_max > global_min else 1
    
    # Create widgets
    scan_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(scans)-1,
        step=1,
        description='Scan Index:',
        continuous_update=False
    )
    
    show_peaks = widgets.Checkbox(
        value=True,
        description='Show peaks',
        disabled=False
    )
    
    scan_info = widgets.HTML(
        value="Scan info will appear here"
    )
    
    plot_output = widgets.Output()
    
    def update_plot(change):
        with plot_output:
            plot_output.clear_output(wait=True)
            
            scan_idx = scan_slider.value
            scan = scans[scan_idx]
            angle = angles[scan_idx]
            frame = roi_frames[scan][0]
            analyzer = analyzer_dict[scan]
            
            # Update info
            peak_info = ""
            for peak_idx in range(len(peak_centers)):
                if peak_idx in normalized_results['normalized_intensities'] and scan in normalized_results['normalized_intensities'][peak_idx]:
                    norm_intensity = normalized_results['normalized_intensities'][peak_idx][scan]
                    center = peak_centers[peak_idx]
                    alpha = (norm_intensity - global_min) / intensity_range
                    alpha = max(0.01, min(1, alpha))
                    peak_info += f"Peak {peak_idx} at ({center[0]}, {center[1]}): {norm_intensity:.8f} (alpha: {alpha:.3f})<br>"
            
            scan_info.value = f"""
            <b>Scan</b>: {scan}<br>
            <b>Angle</b>: {angle:.2f}°<br>
            <b>Frame</b>: {frame}<br>
            <b>Peak Intensities</b>:<br>{peak_info}
            """
            
            # Create plot
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(analyzer.dps[frame], norm=colors.LogNorm())
            plt.colorbar(im, ax=ax, label='Intensity')
            
            # Overlay peaks if checkbox is checked
            if show_peaks.value:
                for peak_idx in range(len(peak_centers)):
                    if peak_idx in normalized_results['normalized_intensities'] and scan in normalized_results['normalized_intensities'][peak_idx]:
                        norm_intensity = normalized_results['normalized_intensities'][peak_idx][scan]
                        center = peak_centers[peak_idx]
                        alpha = (norm_intensity - global_min) / intensity_range
                        alpha = max(0.01, min(1, alpha))
                        
                        ax.scatter(center[1], center[0], color='red', s=100, alpha=alpha,
                                 label=f'Peak {peak_idx}')
                        circle = plt.Circle((center[1], center[0]), 8, color='red', 
                                         fill=False, alpha=alpha)
                        ax.add_patch(circle)
                
                ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
            
            ax.set_title(f'Scan {scan}, Frame {frame}, Angle {angle:.1f}°')
            plt.tight_layout()
            plt.show()
    
    # Connect callbacks
    scan_slider.observe(update_plot, names='value')
    show_peaks.observe(update_plot, names='value')
    
    # Create layout and display
    controls = widgets.VBox([scan_slider, show_peaks, scan_info])
    display(controls)
    display(plot_output)
    
    # Initial plot
    update_plot(None)

    
def calculate_rotation_angles(h, k, l):
    """
    Calculate rotation angles to align a specific (hkl) plane with the beam direction [0,0,1]
    for a cubic lattice.
    
    Parameters:
    -----------
    h, k, l : int
        Miller indices of the desired reflection
    
    Returns:
    --------
    tuple
        Rotation angles (alpha, beta, gamma) in degrees
    """
    # Normalize the vector
    norm = np.sqrt(h**2 + k**2 + l**2)
    h, k, l = h/norm, k/norm, l/norm
    
    # Calculate rotation angles
    # First rotation: around z-axis to align projection with x-z plane
    alpha = np.arctan2(k, h) * 180/np.pi
    
    # Second rotation: around y-axis to align with z-axis
    beta = np.arccos(l) * 180/np.pi
    
    # Third rotation: around z-axis to set final orientation
    gamma = 0  # Can be adjusted if specific in-plane orientation is needed
    
    return (alpha, beta, gamma)

def add_poisson_noise_to_dp(psi, Nc_avg=np.inf, N=None, use_gpu=True):
    """
    Add Poisson noise to diffraction patterns following the MATLAB logic.
    
    Parameters:
    -----------
    psi : array-like
        Complex exit wave (probe * object)
    Nc_avg : float, optional
        Average number of counts per pixel. If inf, no noise is added.
        Default is np.inf (no noise)
    N : int, optional
        Grid size for normalization. If None, uses the size of psi.
        Default is None
    use_gpu : bool, optional
        Whether to use GPU (CuPy) or CPU (NumPy). Default is True
        
    Returns:
    --------
    dp_true : array
        True diffraction pattern (no noise)
    dp : array
        Diffraction pattern with Poisson noise added
    snr : float
        Signal-to-noise ratio
    """
    # Choose array library based on use_gpu flag
    xp = cp if use_gpu else np
    
    # Calculate true diffraction pattern
    dp_true = xp.abs(psi)**2#xp.abs(fft2(ifftshift(psi)))**2
    dp = dp_true.copy()
    
    # Initialize SNR
    snr = np.inf
    
    # Add Poisson noise if Nc_avg is finite
    if Nc_avg < np.inf:
        dp_true_temp = dp_true.copy()
        
        # Normalize to get total counts
        if N is None:
            N = int(xp.sqrt(psi.size))
        
        # Scale to desired average counts per pixel
        dp_temp = dp_true_temp / xp.sum(dp_true_temp) * (N**2 * Nc_avg)
        
        # Add Poisson noise
        if use_gpu:
            # CuPy doesn't have poisson, so we use numpy for this step
            dp_temp_np = cp.asnumpy(dp_temp)
            dp_temp_np = np.random.poisson(dp_temp_np)
            dp_temp = cp.array(dp_temp_np)
        else:
            dp_temp = np.random.poisson(dp_temp)
        
        # Scale back to original intensity
        dp_temp = dp_temp * xp.sum(dp_true_temp) / (N**2 * Nc_avg)
        
        # Calculate SNR
        if use_gpu:
            dp_true_np = cp.asnumpy(dp_true_temp)
            dp_temp_np = cp.asnumpy(dp_temp)
            snr = np.mean(dp_true_np) / np.std(dp_true_np - dp_temp_np)
        else:
            snr = np.mean(dp_true_temp) / np.std(dp_true_temp - dp_temp)
        
        dp = dp_temp
    
    return dp_true, dp, snr


def hanning_gpu(image):
    # GPU version of hanning window
    xs = cp.hanning(image.shape[0])
    ys = cp.hanning(image.shape[1])
    temp = cp.outer(xs, ys)
    return temp

def vignette_gpu(image):
    # GPU version of vignette
    rows, cols = image.shape
    X, Y = cp.meshgrid(cp.linspace(-1, 1, cols), cp.linspace(-1, 1, rows))
    distance = cp.maximum(cp.abs(X), cp.abs(Y))
    vignette_mask = cp.clip(1 - distance, 0, 1)
    vignette_image = image * vignette_mask
    return vignette_image

def load_and_prepare_data(base_directory,sample_dir,recon_path,scan_number,Niter):
    # sample_dir = 'RC02_R3D_'
    # #base_directory = '/scratch/2025_Feb/'
    # base_directory = '/net/micdata/data2/12IDC/2024_Dec/results/'
    # recon_path = 'MLc_L1_p10_g50_Ndp1280_mom0.5_pc0_noModelCon_bg0.1_vi_mm/MLc_L1_p10_g50_Ndp1280_mom0.5_bg0.1_vp4_vi_mm/'
    # scan_number = 888

    # # Load data and move to GPU
    # with h5py.File(f"{base_directory}S{scan_number:04d}/{recon_path}/recon_Niter1000.h5", 'r') as f:
    #     ob = f['object'][()]
    #     pb = f['probe'][()]
    ob = sio.loadmat(f"{base_directory}/{sample_dir}/fly{scan_number:03d}/{recon_path}/Niter{Niter}.mat")
    ob_w = cp.array(ob['object'])
    pb = cp.array(ob['probe'])
    
    ob_w = ob
    if pb.ndim == 4:
        pb1 = pb[:,:,0,0]
    else:
        pb1 = pb[:,:,0]
    
    return ob_w, pb1

def rotate_lattice_gpu(amplitude_3d, angles):
    """Rotate 3D lattice on GPU using CuPy's rotation function"""
    # Use angles directly in degrees like the CPU version
    rotated = rotate_gpu(amplitude_3d, angle=angles[0], axes=(1, 2), reshape=False)
    rotated = rotate_gpu(rotated, angle=angles[1], axes=(0, 2), reshape=False)
    rotated = rotate_gpu(rotated, angle=angles[2], axes=(0, 1), reshape=False)
    return rotated

def bin_ndarray(ndarray, new_shape, operation='mean'):
    """
    Bins an ndarray to a new shape by averaging or summing.
    """
    shape = ndarray.shape
    assert len(shape) == len(new_shape)
    compression_pairs = [(d, c//d) for d, c in zip(new_shape, shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray

def get_sample_scans(csv_path, sample_name):
    """
    Get scan information for a specific sample from the CSV file.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing scan information
    sample_name : str
        Name of the sample to search for
        
    Returns:
    --------
    df_sample : pandas.DataFrame
        DataFrame containing scan numbers and phi angles for the specified sample
    """
    # Read the CSV file without headers and specify column names
    df = pd.read_csv(csv_path, header=None, 
                     names=['scan_number', 'phi_angle', 'sample_name'])
    
    # Filter rows where sample_name matches
    df_sample = df[df['sample_name'] == sample_name]
    
    # Sort by scan number
    df_sample = df_sample.sort_values('scan_number')
    
    return df_sample