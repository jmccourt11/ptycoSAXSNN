import numpy as np
import pdb
import tifffile
import matplotlib.pyplot as plt
from matplotlib import colors
from itertools import product

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.subplots import make_subplots

from IPython.display import display
from ipywidgets import interact, FloatSlider, IntSlider
from ipywidgets import Button, VBox, HBox, IntText, Output

from tqdm import tqdm

import colorsys

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import scipy.ndimage as ndi
from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label
from scipy.ndimage import rotate
from scipy.ndimage import maximum_filter
from scipy.ndimage import shift
from scipy.optimize import minimize
import scipy.io as sio
from scipy.io import loadmat
from scipy.spatial.transform import Rotation


from skimage.measure import profile_line
from skimage.feature import match_template, peak_local_max
from skimage.registration import phase_cross_correlation
from sklearn.cluster import DBSCAN
from skimage.draw import polygon
from skimage.util import montage
from skimage.metrics import normalized_root_mse

from pathlib import Path
from datetime import datetime
import matplotlib.patches as patches
import h5py

def load_cellinfo_data(file_path):
    """
    Load and extract arrays from the 'cellinfo' structure in the given .mat file.
    
    Args:
        file_path (str): Path to the .mat file.
        
    Returns:
        dict: A dictionary where keys are field names and values are the corresponding arrays.
    """
    
    # Load the .mat file
    mat_data = loadmat(file_path)
    
    # Extract the 'cellinfo' data
    cellinfo_data = mat_data.get('cellinfo')
    
    if cellinfo_data is None:
        raise ValueError("'cellinfo' key not found in the .mat file.")
    
    # Initialize a dictionary to store the extracted data
    data_dict = {}
    
    # Iterate through each field and extract its content
    for field_name in cellinfo_data.dtype.names:
        data_dict[field_name] = cellinfo_data[field_name][0, 0]
    
    return data_dict



def generate_miller_indices(max_order=3):
    """
    Generate Miller indices (h,k,l) up to specified order.
    Excludes (0,0,0) and includes all combinations where |h|,|k|,|l| ≤ max_order.
    """
    indices = []
    for h in range(-max_order, max_order + 1):
        for k in range(-max_order, max_order + 1):
            for l in range(-max_order, max_order + 1):
                # Skip the origin
                if h == 0 and k == 0 and l == 0:
                    continue
                indices.append([h, k, l])
    return np.array(indices)




def plot_multi_reciprocal_space(
    rs_datasets,  # List of dicts: [{'magnitude': 3D array, 'Q': 4D array (shape: (nx,ny,nz,3)), 'label': str}, ...]
    cellinfo_data,
    hs, ks, ls,
    thresholds,  # List of thresholds for each dataset
    q_cutoffs,
    peak_distance_threshold=0.01,
    colormaps=['Viridis', 'Jet', 'Plasma'],
    alphas=[0.4, 0.4, 0.4],
    q_axes=[1, 2, 0],
    q_signs=[1, 1, 1],
    flatten_order='C',
    dbscan_eps=0.08, dbscan_min_samples=10
):
    """
    Plot multiple 3D reciprocal space datasets and unit cell peaks, with axis/sign/flattening troubleshooting.
    Args:
        rs_datasets: List of dicts, each with keys:
            - 'magnitude': 3D numpy array
            - 'Q': 4D numpy array (shape: (nx,ny,nz,3)), or tuple of (Qx, Qy, Qz) 3D arrays
            - 'label': str, label for the dataset
        cellinfo_data: Unit cell information
        hs, ks, ls: Miller indices for unit cell peaks
        thresholds: List of magnitude thresholds for each dataset (relative, 0-1)
        q_cutoffs: List of minimum |q| to include (float) for each dataset
        peak_distance_threshold: Max distance to consider a unit cell peak as close to a region center
        q_axes: List of indices for Qv columns to use as x, y, z
        q_signs: List of sign flips for Qv columns
        flatten_order: 'C' or 'F' for flattening order
    Returns:
        tuple: (fig, close_peaks, close_peaks_hkl, filtered_datasets)
            - fig: plotly figure object
            - close_peaks: array of peak positions
            - close_peaks_hkl: list of Miller indices
            - filtered_datasets: list of dicts containing filtered data and Q coordinates
    """
    import numpy as np
    import plotly.graph_objects as go
    from sklearn.cluster import DBSCAN

    fig = go.Figure()
    all_region_centers = []
    all_labels = []
    filtered_datasets = []

    # Plot each reciprocal space dataset
    for idx, dataset in enumerate(rs_datasets):
        magnitude = dataset['DATA']
        Q = dataset['Qv']
        label = dataset.get('label', f'Dataset {idx+1}')
        threshold = thresholds[idx]

        # --- Apply axis/sign/flattening troubleshooting ---
        if Q.ndim == 2 and Q.shape[1] == 3:
            npts = np.prod(magnitude.shape)
            if Q.shape[0] == npts:
                Qx = Q[:, q_axes[0]].reshape(magnitude.shape, order=flatten_order) * q_signs[0]
                Qy = Q[:, q_axes[1]].reshape(magnitude.shape, order=flatten_order) * q_signs[1]
                Qz = Q[:, q_axes[2]].reshape(magnitude.shape, order=flatten_order) * q_signs[2]
            else:
                Qx = Q[:, q_axes[0]] * q_signs[0]
                Qy = Q[:, q_axes[1]] * q_signs[1]
                Qz = Q[:, q_axes[2]] * q_signs[2]
        else:
            Qx = Q[..., q_axes[0]] * q_signs[0]
            Qy = Q[..., q_axes[1]] * q_signs[1]
            Qz = Q[..., q_axes[2]] * q_signs[2]

        # Create filtered version of the data
        q_mag = np.sqrt(Qx**2 + Qy**2 + Qz**2)
        mask = (q_mag > q_cutoffs[idx]) & (magnitude > threshold * np.max(magnitude))
        filtered_magnitude = magnitude.copy()
        filtered_magnitude[~mask] = 0

        filtered_datasets.append({
            'DATA': filtered_magnitude,
            'Qx': Qx,
            'Qy': Qy,
            'Qz': Qz,
            'label': label
        })

        # Flatten for plotting
        kx_flat = Qx.flatten(order=flatten_order)
        ky_flat = Qy.flatten(order=flatten_order)
        kz_flat = Qz.flatten(order=flatten_order)
        mag_flat = magnitude.flatten(order=flatten_order)

        q_mag = np.sqrt(kx_flat**2 + ky_flat**2 + kz_flat**2)
        mask = (q_mag > q_cutoffs[idx]) & (mag_flat > threshold * np.max(mag_flat))
        kx_f = kx_flat[mask]
        ky_f = ky_flat[mask]
        kz_f = kz_flat[mask]
        mag_f = mag_flat[mask]

        fig.add_trace(go.Scatter3d(
            x=kx_f, y=ky_f, z=kz_f,
            mode='markers',
            marker=dict(
                size=5,
                #color=mag_f,
                color=kx_f,
                colorscale=colormaps[idx],
                opacity=alphas[idx],
                colorbar=dict(title='X Position') if idx == 0 else None
                #colorbar=dict(title=f'{label} Magnitude') if idx == 0 else None
            ),
            name=label
        ))
        
        # factor = 4  # Try 2, 3, or higher for more aggressive downsampling
        # Qx_ds = Qx[::factor, ::factor, ::factor]
        # Qy_ds = Qy[::factor, ::factor, ::factor]
        # Qz_ds = Qz[::factor, ::factor, ::factor]
        # filtered_magnitude_ds = filtered_magnitude[::factor, ::factor, ::factor]
        # fig.add_trace(go.Isosurface(
        #     x=Qx_ds.flatten(order=flatten_order),
        #     y=Qy_ds.flatten(order=flatten_order),
        #     z=Qz_ds.flatten(order=flatten_order),
        #     value=filtered_magnitude_ds.flatten(order=flatten_order),
        #     isomin=0.1 * np.max(filtered_magnitude_ds),
        #     isomax=np.max(filtered_magnitude_ds),
        #     opacity=alphas[idx],
        #     surface_count=3,
        #     colorscale=colormaps[idx],
        #     showscale=(idx == 0),
        #     name=label
        # ))

        # Cluster and find region centers
        if len(kx_f) > 0:
            coords = np.column_stack((kx_f, ky_f, kz_f))
            coords_norm = coords / np.max(np.abs(coords))
            clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(coords_norm)
            labels_ = clustering.labels_
            region_centers = []
            for clabel in set(labels_):
                if clabel == -1:
                    continue
                mask_c = labels_ == clabel
                cluster_points = coords[mask_c]
                cluster_mags = mag_f[mask_c]
                weights = cluster_mags / np.sum(cluster_mags)
                center = np.sum(cluster_points * weights[:, np.newaxis], axis=0)
                region_centers.append(center)
            region_centers = np.array(region_centers)
            if len(region_centers) > 0:
                all_region_centers.extend(region_centers)
                all_labels.extend([label]*len(region_centers))

    # Compute unit cell peaks
    vs = []
    hkl_list = []
    for i, h in enumerate(hs):
        v = hs[i]*cellinfo_data['recilatticevectors'][0] + \
            ks[i]*cellinfo_data['recilatticevectors'][1] + \
            ls[i]*cellinfo_data['recilatticevectors'][2]
        vs.append(v)
        hkl_list.append(f"({h},{ks[i]},{ls[i]})")
    vs = np.array(vs)

    # Find unit cell peaks close to any region center
    close_peaks = []
    close_peaks_hkl = []
    used_regions = set()
    all_region_centers = np.array(all_region_centers)
    for i, peak in enumerate(vs):
        if len(all_region_centers) == 0:
            break
        distances = np.sqrt(np.sum((all_region_centers - peak)**2, axis=1))
        min_dist_idx = np.argmin(distances)
        min_dist = distances[min_dist_idx]
        if min_dist < peak_distance_threshold and (min_dist_idx, all_labels[min_dist_idx]) not in used_regions:
            close_peaks.append(peak)
            close_peaks_hkl.append(hkl_list[i])
            used_regions.add((min_dist_idx, all_labels[min_dist_idx]))
    close_peaks = np.array(close_peaks)

    if len(close_peaks) > 0:
        fig.add_trace(go.Scatter3d(
            x=close_peaks[:,0], y=close_peaks[:,1], z=close_peaks[:,2],
            #mode='markers+text',
            mode='markers',
            marker=dict(size=5, color='red', opacity=0.3, symbol='diamond'),
            #text=close_peaks_hkl,
            #textfont=dict(size=6),
            #textposition="top center",
            name='Unit Cell Peaks'
        ))
        
    # Example: list of hkl labels to highlight
    #highlight_hkls = ['(2,0,0)', '(6,0,0)', '(-6,0,0)', '(-2,0,0)']
    #highlight_hkls = ['(0,2,0)', '(0,6,0)', '(0,-2,0)', '(0,-6,0)']
    highlight_hkls = ['(0,0,2)', '(0,0,6)', '(0,0,-2)', '(0,0,-6)'] 
    # highlight_hkls = [
    #     # 6,6,6 permutations and +/-1
    #     '(0,6,6)', '(0,-6,-6)', '(0,6,-6)', '(0,-6,6)',
    #     '(0,7,6)', '(0,-7,-6)', '(0,7,-6)', '(0,-7,6)',
    #     '(0,5,6)', '(0,-5,-6)', '(0,5,-6)', '(0,-5,6)',
    #     '(0,6,7)', '(0,-6,-7)', '(0,6,-7)', '(0,-6,7)',
    #     '(0,6,5)', '(0,-6,-5)', '(0,6,-5)', '(0,-6,5)',
        
    #     '(6,0,6)', '(-6,0,-6)', '(6,0,-6)', '(-6,0,6)',
    #     '(7,0,6)', '(-7,0,-6)', '(7,0,-6)', '(-7,0,6)',
    #     '(5,0,6)', '(-5,0,-6)', '(5,0,-6)', '(-5,0,6)',
    #     '(6,0,7)', '(-6,0,-7)', '(6,0,-7)', '(-6,0,7)',
    #     '(6,0,5)', '(-6,0,-5)', '(6,0,-5)', '(-6,0,5)',
        
    #     '(6,6,0)', '(-6,-6,0)', '(6,-6,0)', '(-6,6,0)',
    #     '(7,6,0)', '(-7,-6,0)', '(7,-6,0)', '(-7,6,0)', 
    #     '(5,6,0)', '(-5,-6,0)', '(5,-6,0)', '(-5,6,0)',
    #     '(6,7,0)', '(-6,-7,0)', '(6,-7,0)', '(-6,7,0)',
    #     '(6,5,0)', '(-6,-5,0)', '(6,-5,0)', '(-6,5,0)',
        
    #     # 2,8,2 permutations and +/-1
    #     '(2,8,2)', '(-2,8,2)', '(2,-8,2)', '(-2,-8,2)',
    #     '(3,8,2)', '(-3,8,2)', '(3,-8,2)', '(-3,-8,2)',
    #     '(1,8,2)', '(-1,8,2)', '(1,-8,2)', '(-1,-8,2)',
    #     '(2,9,2)', '(-2,9,2)', '(2,-9,2)', '(-2,-9,2)',
    #     '(2,7,2)', '(-2,7,2)', '(2,-7,2)', '(-2,-7,2)',
    #     '(2,8,3)', '(-2,8,3)', '(2,-8,3)', '(-2,-8,3)',
    #     '(2,8,1)', '(-2,8,1)', '(2,-8,1)', '(-2,-8,1)',
        
    #     '(2,8,-2)', '(-2,8,-2)', '(2,-8,-2)', '(-2,-8,-2)',
    #     '(3,8,-2)', '(-3,8,-2)', '(3,-8,-2)', '(-3,-8,-2)',
    #     '(1,8,-2)', '(-1,8,-2)', '(1,-8,-2)', '(-1,-8,-2)',
    #     '(2,9,-2)', '(-2,9,-2)', '(2,-9,-2)', '(-2,-9,-2)',
    #     '(2,7,-2)', '(-2,7,-2)', '(2,-7,-2)', '(-2,-7,-2)',
    #     '(2,8,-3)', '(-2,8,-3)', '(2,-8,-3)', '(-2,-8,-3)',
    #     '(2,8,-1)', '(-2,8,-1)', '(2,-8,-1)', '(-2,-8,-1)',
        
    #     '(8,2,2)', '(8,-2,2)', '(-8,2,2)', '(-8,-2,2)',
    #     '(9,2,2)', '(9,-2,2)', '(-9,2,2)', '(-9,-2,2)',
    #     '(7,2,2)', '(7,-2,2)', '(-7,2,2)', '(-7,-2,2)',
    #     '(8,3,2)', '(8,-3,2)', '(-8,3,2)', '(-8,-3,2)',
    #     '(8,1,2)', '(8,-1,2)', '(-8,1,2)', '(-8,-1,2)',
    #     '(8,2,3)', '(8,-2,3)', '(-8,2,3)', '(-8,-2,3)',
    #     '(8,2,1)', '(8,-2,1)', '(-8,2,1)', '(-8,-2,1)',
        
    #     '(8,2,-2)', '(8,-2,-2)', '(-8,2,-2)', '(-8,-2,-2)',
    #     '(9,2,-2)', '(9,-2,-2)', '(-9,2,-2)', '(-9,-2,-2)',
    #     '(7,2,-2)', '(7,-2,-2)', '(-7,2,-2)', '(-7,-2,-2)',
    #     '(8,3,-2)', '(8,-3,-2)', '(-8,3,-2)', '(-8,-3,-2)',
    #     '(8,1,-2)', '(8,-1,-2)', '(-8,1,-2)', '(-8,-1,-2)',
    #     '(8,2,-3)', '(8,-2,-3)', '(-8,2,-3)', '(-8,-2,-3)',
    #     '(8,2,-1)', '(8,-2,-1)', '(-8,2,-1)', '(-8,-2,-1)',
        
    #     '(2,2,8)', '(-2,2,8)', '(2,-2,8)', '(-2,-2,8)',
    #     '(3,2,8)', '(-3,2,8)', '(3,-2,8)', '(-3,-2,8)',
    #     '(1,2,8)', '(-1,2,8)', '(1,-2,8)', '(-1,-2,8)',
    #     '(2,3,8)', '(-2,3,8)', '(2,-3,8)', '(-2,-3,8)',
    #     '(2,1,8)', '(-2,1,8)', '(2,-1,8)', '(-2,-1,8)',
    #     '(2,2,9)', '(-2,2,9)', '(2,-2,9)', '(-2,-2,9)',
    #     '(2,2,7)', '(-2,2,7)', '(2,-2,7)', '(-2,-2,7)',
        
    #     '(2,2,-8)', '(-2,2,-8)', '(2,-2,-8)', '(-2,-2,-8)',
    #     '(3,2,-8)', '(-3,2,-8)', '(3,-2,-8)', '(-3,-2,-8)',
    #     '(1,2,-8)', '(-1,2,-8)', '(1,-2,-8)', '(-1,-2,-8)',
    #     '(2,3,-8)', '(-2,3,-8)', '(2,-3,-8)', '(-2,-3,-8)',
    #     '(2,1,-8)', '(-2,1,-8)', '(2,-1,-8)', '(-2,-1,-8)',
    #     '(2,2,-9)', '(-2,2,-9)', '(2,-2,-9)', '(-2,-2,-9)',
    #     '(2,2,-7)', '(-2,2,-7)', '(2,-2,-7)', '(-2,-2,-7)',
        
    #     # 2,2,2 permutations and +/-1
    #     '(0,2,2)', '(0,-2,-2)', '(0,2,-2)', '(0,-2,2)',
    #     '(0,3,2)', '(0,-3,-2)', '(0,3,-2)', '(0,-3,2)',
    #     '(0,1,2)', '(0,-1,-2)', '(0,1,-2)', '(0,-1,2)',
    #     '(0,2,3)', '(0,-2,-3)', '(0,2,-3)', '(0,-2,3)',
    #     '(0,2,1)', '(0,-2,-1)', '(0,2,-1)', '(0,-2,1)',
        
    #     '(2,0,2)', '(-2,0,-2)', '(2,0,-2)', '(-2,0,2)',
    #     '(3,0,2)', '(-3,0,-2)', '(3,0,-2)', '(-3,0,2)',
    #     '(1,0,2)', '(-1,0,-2)', '(1,0,-2)', '(-1,0,2)',
    #     '(2,0,3)', '(-2,0,-3)', '(2,0,-3)', '(-2,0,3)',
    #     '(2,0,1)', '(-2,0,-1)', '(2,0,-1)', '(-2,0,1)',
        
    #     '(2,2,0)', '(-2,-2,0)', '(2,-2,0)', '(-2,2,0)',
    #     '(3,2,0)', '(-3,-2,0)', '(3,-2,0)', '(-3,2,0)',
    #     '(1,2,0)', '(-1,-2,0)', '(1,-2,0)', '(-1,2,0)',
    #     '(2,3,0)', '(-2,-3,0)', '(2,-3,0)', '(-2,3,0)',
    #     '(2,1,0)', '(-2,-1,0)', '(2,-1,0)', '(-2,1,0)'
    # ]

    # Separate peaks to highlight
    highlight_mask = [hkl in highlight_hkls for hkl in close_peaks_hkl]
    normal_mask = [not h for h in highlight_mask]

    # Normal peaks
    if any(normal_mask):
        fig.add_trace(go.Scatter3d(
            x=close_peaks[normal_mask, 0],
            y=close_peaks[normal_mask, 1],
            z=close_peaks[normal_mask, 2],
            mode='text',
            text=np.array(close_peaks_hkl)[normal_mask],
            textposition='top center',
            textfont=dict(color='black', size=10),
            name='hkl Peaks'
        ))


    print(f"Number of unit cell peaks: {len(close_peaks)}")
    print(f'close_peaks: {close_peaks_hkl}')
    
    fig.update_layout(
        title="Multi Reciprocal Space Visualization (TROUBLESHOOT MODE)",
        scene=dict(
            xaxis_title="Qx (Å⁻¹)",
            yaxis_title="Qy (Å⁻¹)",
            zaxis_title="Qz (Å⁻¹)",
            aspectmode='cube'
        ),
        width=1000,
        height=1000,
        showlegend=True
    )
        # Highlighted peaks
    if any(highlight_mask):
        fig.add_trace(go.Scatter3d(
            x=close_peaks[highlight_mask, 0],
            y=close_peaks[highlight_mask, 1],
            z=close_peaks[highlight_mask, 2],
            mode='text',
            text=np.array(close_peaks_hkl)[highlight_mask],
            textposition='top center',
            textfont=dict(color='red', size=20),  # Larger, red text
            name='Highlighted Peaks'
        ))

    return fig, close_peaks, close_peaks_hkl, filtered_datasets


def count_overlapping_peaks(
    dataset1, dataset2,
    threshold1, threshold2,
    q_cutoff1, q_cutoff2,
    overlap_distance=0.01,
    dbscan_eps=0.08, dbscan_min_samples=10,
    q_axes=[1,2,0], q_signs=[1,1,1], flatten_order='C'
):
    """
    Count the number of overlapping peaks between two 3D reciprocal space datasets.
    Returns: (n_peaks1, n_peaks2, n_overlapping)
    """
    import numpy as np
    from sklearn.cluster import DBSCAN

    def find_peaks(dataset, threshold, q_cutoff):
        DATA = dataset['DATA']
        Q = dataset['Qv']
        # --- Apply axis/sign/flattening troubleshooting ---
        if Q.ndim == 2 and Q.shape[1] == 3:
            npts = np.prod(DATA.shape)
            if Q.shape[0] == npts:
                Qx = Q[:, q_axes[0]].reshape(DATA.shape, order=flatten_order) * q_signs[0]
                Qy = Q[:, q_axes[1]].reshape(DATA.shape, order=flatten_order) * q_signs[1]
                Qz = Q[:, q_axes[2]].reshape(DATA.shape, order=flatten_order) * q_signs[2]
            else:
                Qx = Q[:, q_axes[0]] * q_signs[0]
                Qy = Q[:, q_axes[1]] * q_signs[1]
                Qz = Q[:, q_axes[2]] * q_signs[2]
        else:
            Qx = Q[..., q_axes[0]] * q_signs[0]
            Qy = Q[..., q_axes[1]] * q_signs[1]
            Qz = Q[..., q_axes[2]] * q_signs[2]
        kx_flat = Qx.flatten(order=flatten_order)
        ky_flat = Qy.flatten(order=flatten_order)
        kz_flat = Qz.flatten(order=flatten_order)
        mag_flat = DATA.flatten(order=flatten_order)
        q_mag = np.sqrt(kx_flat**2 + ky_flat**2 + kz_flat**2)
        mask = (q_mag > q_cutoff) & (mag_flat > threshold * np.max(mag_flat))
        coords = np.column_stack((kx_flat[mask], ky_flat[mask], kz_flat[mask]))
        mag_f = mag_flat[mask]
        if len(coords) == 0:
            return np.zeros((0,3))
        coords_norm = coords / np.max(np.abs(coords))
        clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(coords_norm)
        labels_ = clustering.labels_
        region_centers = []
        for clabel in set(labels_):
            if clabel == -1:
                continue
            mask_c = labels_ == clabel
            cluster_points = coords[mask_c]
            cluster_mags = mag_f[mask_c]
            weights = cluster_mags / np.sum(cluster_mags)
            center = np.sum(cluster_points * weights[:, np.newaxis], axis=0)
            region_centers.append(center)
        return np.array(region_centers)

    peaks1 = find_peaks(dataset1, threshold1, q_cutoff1)
    peaks2 = find_peaks(dataset2, threshold2, q_cutoff2)
    n_peaks1 = len(peaks1)
    n_peaks2 = len(peaks2)

    # Count overlaps
    n_overlapping = 0
    used2 = set()
    for i, p1 in enumerate(peaks1):
        dists = np.sqrt(np.sum((peaks2 - p1)**2, axis=1))
        min_idx = np.argmin(dists)
        if dists[min_idx] < overlap_distance and min_idx not in used2:
            n_overlapping += 1
            used2.add(min_idx)

    return n_peaks1, n_peaks2, n_overlapping


def peak_confusion_matrix(
    true_dataset, pred_dataset,
    true_threshold, pred_threshold,
    true_q_cutoff, pred_q_cutoff,
    overlap_distance=0.01,
    dbscan_eps=0.08, dbscan_min_samples=10,
    q_axes=[1,2,0], q_signs=[1,1,1], flatten_order='C',
    plot=False,
    overlay=False,
    cellinfo_data=None,
    miller_indices=None
):
    """
    Compute confusion matrix for peak detection in reciprocal space.
    If plot=True, show a 3D plot of true, predicted, and matched peaks.
    If overlay=True, show overlays of found peaks on thresholded reciprocal space points for both datasets.
    Returns: dict with keys 'TP', 'FP', 'FN', 'n_true', 'n_pred', and optionally 'fig', 'overlay_true_fig', 'overlay_pred_fig'
    """
    import numpy as np
    from sklearn.cluster import DBSCAN
    import plotly.graph_objects as go

    def find_peaks(dataset, threshold, q_cutoff):
        DATA = dataset['DATA']
        Q = dataset['Qv']
        if Q.ndim == 2 and Q.shape[1] == 3:
            npts = np.prod(DATA.shape)
            if Q.shape[0] == npts:
                Qx = Q[:, q_axes[0]].reshape(DATA.shape, order=flatten_order) * q_signs[0]
                Qy = Q[:, q_axes[1]].reshape(DATA.shape, order=flatten_order) * q_signs[1]
                Qz = Q[:, q_axes[2]].reshape(DATA.shape, order=flatten_order) * q_signs[2]
            else:
                Qx = Q[:, q_axes[0]] * q_signs[0]
                Qy = Q[:, q_axes[1]] * q_signs[1]
                Qz = Q[:, q_axes[2]] * q_signs[2]
        else:
            Qx = Q[..., q_axes[0]] * q_signs[0]
            Qy = Q[..., q_axes[1]] * q_signs[1]
            Qz = Q[..., q_axes[2]] * q_signs[2]
        kx_flat = Qx.flatten(order=flatten_order)
        ky_flat = Qy.flatten(order=flatten_order)
        kz_flat = Qz.flatten(order=flatten_order)
        mag_flat = DATA.flatten(order=flatten_order)
        q_mag = np.sqrt(kx_flat**2 + ky_flat**2 + kz_flat**2)
        mask = (q_mag > q_cutoff) & (mag_flat > threshold * np.max(mag_flat))
        coords = np.column_stack((kx_flat[mask], ky_flat[mask], kz_flat[mask]))
        mag_f = mag_flat[mask]
        if len(coords) == 0:
            return np.zeros((0,3))
        coords_norm = coords / np.max(np.abs(coords))
        clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(coords_norm)
        labels_ = clustering.labels_
        region_centers = []
        for clabel in set(labels_):
            if clabel == -1:
                continue
            mask_c = labels_ == clabel
            cluster_points = coords[mask_c]
            cluster_mags = mag_f[mask_c]
            weights = cluster_mags / np.sum(cluster_mags)
            center = np.sum(cluster_points * weights[:, np.newaxis], axis=0)
            region_centers.append(center)
        return np.array(region_centers)

    def get_thresholded_points(dataset, threshold, q_cutoff):
        DATA = dataset['DATA']
        Q = dataset['Qv']
        if Q.ndim == 2 and Q.shape[1] == 3:
            npts = np.prod(DATA.shape)
            if Q.shape[0] == npts:
                Qx = Q[:, q_axes[0]].reshape(DATA.shape, order=flatten_order) * q_signs[0]
                Qy = Q[:, q_axes[1]].reshape(DATA.shape, order=flatten_order) * q_signs[1]
                Qz = Q[:, q_axes[2]].reshape(DATA.shape, order=flatten_order) * q_signs[2]
            else:
                Qx = Q[:, q_axes[0]] * q_signs[0]
                Qy = Q[:, q_axes[1]] * q_signs[1]
                Qz = Q[:, q_axes[2]] * q_signs[2]
        else:
            Qx = Q[..., q_axes[0]] * q_signs[0]
            Qy = Q[..., q_axes[1]] * q_signs[1]
            Qz = Q[..., q_axes[2]] * q_signs[2]
        kx_flat = Qx.flatten(order=flatten_order)
        ky_flat = Qy.flatten(order=flatten_order)
        kz_flat = Qz.flatten(order=flatten_order)
        mag_flat = DATA.flatten(order=flatten_order)
        q_mag = np.sqrt(kx_flat**2 + ky_flat**2 + kz_flat**2)
        mask = (q_mag > q_cutoff) & (mag_flat > threshold * np.max(mag_flat))
        return kx_flat[mask], ky_flat[mask], kz_flat[mask], mag_flat[mask]

    true_peaks = find_peaks(true_dataset, true_threshold, true_q_cutoff)
    pred_peaks = find_peaks(pred_dataset, pred_threshold, pred_q_cutoff)
    n_true = len(true_peaks)
    n_pred = len(pred_peaks)

    matched_true = set()
    matched_pred = set()
    matches = []  # Store (i, j) pairs

    if n_true == 0 or n_pred == 0:
        TP = 0
        FN = n_true
        FP = n_pred
    else:
        dists = np.linalg.norm(true_peaks[:, None, :] - pred_peaks[None, :, :], axis=2)
        for i in range(n_true):
            min_j = np.argmin(dists[i])
            if dists[i, min_j] < overlap_distance and min_j not in matched_pred:
                matched_true.add(i)
                matched_pred.add(min_j)
                matches.append((i, min_j))
        TP = len(matched_true)
        FN = n_true - TP
        FP = n_pred - TP

    # Match peaks to Miller indices if cellinfo_data and miller_indices are provided
    matched_hkl_info = {}
    if cellinfo_data is not None and miller_indices is not None:
        # Calculate theoretical peak positions
        vs = []
        hkl_list = []
        for i, h in enumerate(miller_indices[:, 0]):
            v = miller_indices[i, 0]*cellinfo_data['recilatticevectors'][0] + \
                miller_indices[i, 1]*cellinfo_data['recilatticevectors'][1] + \
                miller_indices[i, 2]*cellinfo_data['recilatticevectors'][2]
            vs.append(v)
            hkl_list.append(f"({miller_indices[i, 0]},{miller_indices[i, 1]},{miller_indices[i, 2]})")
        vs = np.array(vs)
        
        # Match true peaks to Miller indices
        true_peak_hkl = []
        for i, peak in enumerate(true_peaks):
            distances = np.sqrt(np.sum((vs - peak)**2, axis=1))
            min_dist_idx = np.argmin(distances)
            min_dist = distances[min_dist_idx]
            if min_dist < 0.02:  # Threshold for matching to Miller indices
                true_peak_hkl.append(hkl_list[min_dist_idx])
            else:
                true_peak_hkl.append("Unknown")
        
        # Match predicted peaks to Miller indices
        pred_peak_hkl = []
        for i, peak in enumerate(pred_peaks):
            distances = np.sqrt(np.sum((vs - peak)**2, axis=1))
            min_dist_idx = np.argmin(distances)
            min_dist = distances[min_dist_idx]
            if min_dist < 0.02:  # Threshold for matching to Miller indices
                pred_peak_hkl.append(hkl_list[min_dist_idx])
            else:
                pred_peak_hkl.append("Unknown")
        
        # Print matched peaks information
        print("\n=== PEAK MATCHING RESULTS ===")
        print(f"True peaks matched to Miller indices:")
        for i, (peak, hkl) in enumerate(zip(true_peaks, true_peak_hkl)):
            print(f"  Peak {i+1}: {peak} -> {hkl}")
        
        print(f"\nPredicted peaks matched to Miller indices:")
        for i, (peak, hkl) in enumerate(zip(pred_peaks, pred_peak_hkl)):
            print(f"  Peak {i+1}: {peak} -> {hkl}")
        
        # Print matched pairs with their hkl
        print(f"\nMatched peak pairs (True -> Predicted):")
        for (i, j) in matches:
            print(f"  {true_peak_hkl[i]} -> {pred_peak_hkl[j]}")
        
        matched_hkl_info = {
            'true_peak_hkl': true_peak_hkl,
            'pred_peak_hkl': pred_peak_hkl,
            'matched_pairs_hkl': [(true_peak_hkl[i], pred_peak_hkl[j]) for (i, j) in matches]
        }

    results = {
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'n_true': n_true,
        'n_pred': n_pred,
        'matched_hkl_info': matched_hkl_info
    }

    if plot:
        fig = go.Figure(layout=dict(width=1000, height=800))
        # Plot all true peaks
        if n_true > 0:
            fig.add_trace(go.Scatter3d(
                x=true_peaks[:,0], y=true_peaks[:,1], z=true_peaks[:,2],
                mode='markers',
                marker=dict(size=19, color='blue', opacity=0.5, symbol='circle'),
                name='True Peaks'
            ))
        # Plot all predicted peaks
        if n_pred > 0:
            fig.add_trace(go.Scatter3d(
                x=pred_peaks[:,0], y=pred_peaks[:,1], z=pred_peaks[:,2],
                mode='markers',
                marker=dict(size=10, color='orange', opacity=0.5, symbol='diamond'),
                name='Predicted Peaks'
            ))
        # Plot matched pairs with lines
        for (i, j) in matches:
            fig.add_trace(go.Scatter3d(
                x=[true_peaks[i,0], pred_peaks[j,0]],
                y=[true_peaks[i,1], pred_peaks[j,1]],
                z=[true_peaks[i,2], pred_peaks[j,2]],
                mode='lines',
                line=dict(color='green', width=8),
                name='Matched Pair',
                showlegend=False
            ))
        fig.update_layout(
            title='3D Peaks: True (blue), Predicted (orange), Matches (green lines)',
            scene=dict(
                xaxis_title='Qx',
                yaxis_title='Qy',
                zaxis_title='Qz',
                aspectmode='cube'
            ),
            legend=dict(itemsizing='constant')
        )
        fig.show()
        results['fig'] = fig

    if overlay:
        # Overlay for true dataset
        kx_t, ky_t, kz_t, mag_t = get_thresholded_points(true_dataset, true_threshold, true_q_cutoff)
        fig_true = go.Figure()
        fig_true.add_trace(go.Scatter3d(
            x=kx_t, y=ky_t, z=kz_t,
            mode='markers',
            marker=dict(size=3, color=mag_t, colorscale='Viridis', opacity=0.2),
            name='Thresholded Points'
        ))
        if n_true > 0:
            fig_true.add_trace(go.Scatter3d(
                x=true_peaks[:,0], y=true_peaks[:,1], z=true_peaks[:,2],
                mode='markers',
                marker=dict(size=10, color='blue', opacity=0.4, symbol='circle'),
                name='True Peaks'
            ))
        fig_true.update_layout(
            title='True Peaks Overlayed on Reciprocal Space',
            scene=dict(xaxis_title='Qx', yaxis_title='Qy', zaxis_title='Qz', aspectmode='cube')
        )

        # Overlay for predicted dataset
        kx_p, ky_p, kz_p, mag_p = get_thresholded_points(pred_dataset, pred_threshold, pred_q_cutoff)
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter3d(
            x=kx_p, y=ky_p, z=kz_p,
            mode='markers',
            marker=dict(size=3, color=mag_p, colorscale='Plasma', opacity=0.2),
            name='Thresholded Points'
        ))
        if n_pred > 0:
            fig_pred.add_trace(go.Scatter3d(
                x=pred_peaks[:,0], y=pred_peaks[:,1], z=pred_peaks[:,2],
                mode='markers',
                marker=dict(size=10, color='orange', opacity=0.4, symbol='diamond'),
                name='Predicted Peaks'
            ))
        fig_pred.update_layout(
            title='Predicted Peaks Overlayed on Reciprocal Space',
            scene=dict(xaxis_title='Qx', yaxis_title='Qy', zaxis_title='Qz', aspectmode='cube')
        )

        fig_true.show()
        fig_pred.show()
        results['overlay_true_fig'] = fig_true
        results['overlay_pred_fig'] = fig_pred

    return results

def create_hsv_colorscale(n_colors=100):
    colors = []
    for i in range(n_colors + 1):  # +1 to include both endpoints
        # Convert to HSV color (hue cycles from 0 to 1)
        hue = i / n_colors
        # Full saturation and value for vibrant colors
        hsv = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        # Convert to RGB string format
        rgb = f'rgb({int(hsv[0]*255)},{int(hsv[1]*255)},{int(hsv[2]*255)})'
        colors.append([i/n_colors, rgb])
    # Add the first color again at the end to make it cyclic
    colors.append([1.0, colors[0][1]])
    return colors


def plot_arbitrary_line_profile(projection, src, dst, linewidth=1, title='Arbitrary Line Profile', show_plot=True):
    """
    Plot a line profile between two points (src, dst) in the projection.
    src, dst: (row, col) coordinates
    """
    prof = profile_line(projection, src, dst, linewidth=linewidth, mode='reflect')
    if show_plot:
        plt.figure(figsize=(8,4))
        plt.plot(prof)
        plt.xlabel('Distance along line')
        plt.ylabel('Intensity')
        plt.title(title + f' ({src} to {dst})')
        plt.show()
    
        plt.figure(figsize=(12,12))
        plt.imshow(projection, cmap='gray')
        plt.plot([src[1], dst[1]], [src[0], dst[0]], 'r-', linewidth=linewidth)
        plt.plot(src[1], src[0], 'go', label='Start')
        plt.plot(dst[1], dst[0], 'ro', label='End') 
        plt.colorbar(label='Intensity')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(title + f' ({src} to {dst})')
        plt.legend()
        plt.show()
    return prof


def plot_1d_fft(profile, pixel_size=None, title="Fourier Spectrum", show_phase=False, show_plot=True):
    """
    Compute and plot the amplitude (and optionally phase) spectrum of a 1D profile.
    pixel_size: If provided, will convert x-axis to spatial frequency (1/pixel_size units)
    """
    N = len(profile)
    fft_vals = np.fft.fft(profile)
    fft_freqs = np.fft.fftfreq(N, d=pixel_size if pixel_size else 1)
    amplitude = np.abs(fft_vals)
    phase = np.angle(fft_vals)
    
    # Only plot the positive frequencies
    if show_plot:
        pos_mask = fft_freqs >= 0
        plt.figure(figsize=(8,4))
        plt.plot(fft_freqs[pos_mask], amplitude[pos_mask], label='Amplitude')
        plt.xlabel('Frequency (1/pixel)' if pixel_size is None else f'Frequency (1/{pixel_size} units)')
        plt.ylabel('Amplitude')
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.show()
        
    if show_phase:
        plt.figure(figsize=(8,4))
        plt.plot(fft_freqs[pos_mask], phase[pos_mask], label='Phase')
        plt.xlabel('Frequency (1/pixel)' if pixel_size is None else f'Frequency (1/{pixel_size} units)')
        plt.ylabel('Phase (radians)')
        plt.title(title + " (Phase)")
        plt.grid(True)
        plt.legend()
        plt.show()
    
    return fft_freqs, amplitude, phase


def get_orientation_matrix(peak_positions, peak_values):
    """
    Convert peak positions to orientation matrix using weighted peaks
    """
    # Center the peaks around origin
    center = np.mean(peak_positions, axis=0)
    centered_peaks = peak_positions - center
    
    # Calculate covariance matrix with peak value weights
    weights = peak_values / np.sum(peak_values)
    cov_matrix = np.zeros((3, 3))
    for peak, weight in zip(centered_peaks, weights):
        cov_matrix += weight * np.outer(peak, peak)
    
    return cov_matrix

def get_axis_angle(voxel_peaks, voxel_values, ref_peaks, ref_values):
    """
    Calculate axis-angle representation between voxel orientation and reference orientation
    with improved eigenvector handling
    """
    # Get orientation matrices
    voxel_orient = get_orientation_matrix(voxel_peaks, voxel_values)
    ref_orient = get_orientation_matrix(ref_peaks, ref_values)
    
    # Get principal directions (eigenvectors) and sort by eigenvalues
    voxel_eigvals, voxel_eigvecs = np.linalg.eigh(voxel_orient)
    ref_eigvals, ref_eigvecs = np.linalg.eigh(ref_orient)
    
    # Sort eigenvectors by eigenvalue magnitude
    voxel_order = np.argsort(-np.abs(voxel_eigvals))
    ref_order = np.argsort(-np.abs(ref_eigvals))
    
    voxel_eigvecs = voxel_eigvecs[:, voxel_order]
    ref_eigvecs = ref_eigvecs[:, ref_order]
    
    # Try both orientations of each eigenvector to find best alignment
    best_rmsd = float('inf')
    best_R = None
    
    for flip_x in [-1, 1]:
        for flip_y in [-1, 1]:
            for flip_z in [-1, 1]:
                test_voxel_eigvecs = voxel_eigvecs.copy()
                test_voxel_eigvecs[:, 0] *= flip_x
                test_voxel_eigvecs[:, 1] *= flip_y
                test_voxel_eigvecs[:, 2] *= flip_z
                
                R = Rotation.align_vectors(ref_eigvecs.T, test_voxel_eigvecs.T)[0]
                rotated = R.apply(voxel_peaks)
                rmsd = np.sqrt(np.mean(np.sum((rotated - ref_peaks)**2, axis=1)))
                
                if rmsd < best_rmsd:
                    best_rmsd = rmsd
                    best_R = R
    
    # Convert best rotation to axis-angle representation
    axis, angle = best_R.as_rotvec(degrees=True), np.linalg.norm(best_R.as_rotvec(degrees=True))
    
    return axis, angle, best_rmsd


def get_axis_angle_simple(voxel_peaks, voxel_values, ref_peaks, ref_values):
    """
    Calculate rotation (axis and angle) needed to align voxel peaks with reference peaks
    
    Args:
        voxel_peaks: peak positions for voxel FFT
        voxel_values: peak intensities for voxel FFT
        ref_peaks: reference peak positions
        ref_values: reference peak intensities
    
    Returns:
        axis: unit vector representing rotation axis
        angle: rotation angle in degrees (0-180)
        rmsd: root mean square deviation after alignment
    """
    # Sort peaks by intensity and use top N peaks
    N = min(6, len(voxel_peaks), len(ref_peaks))
    voxel_order = np.argsort(-voxel_values)[:N]
    ref_order = np.argsort(-ref_values)[:N]
    
    voxel_peaks = voxel_peaks[voxel_order]
    ref_peaks = ref_peaks[ref_order]
    voxel_values = voxel_values[voxel_order]
    ref_values = ref_values[ref_order]
    
    # #only take the top 2 peaks
    # voxel_peaks=voxel_peaks[:2]
    # ref_peaks=ref_peaks[:2]
    # voxel_values=voxel_values[:2]
    # ref_values=ref_values[:2]
    
    # Get principal directions for both sets of peaks
    voxel_eigvals, voxel_eigvecs = np.linalg.eigh(get_orientation_matrix(voxel_peaks, voxel_values))
    ref_eigvals, ref_eigvecs = np.linalg.eigh(get_orientation_matrix(ref_peaks, ref_values))
    
    # Sort eigenvectors by eigenvalue magnitude
    voxel_order = np.argsort(-np.abs(voxel_eigvals))
    ref_order = np.argsort(-np.abs(ref_eigvals))
    
    voxel_eigvecs = voxel_eigvecs[:, voxel_order]
    ref_eigvecs = ref_eigvecs[:, ref_order]
    
    # Try both possible alignments
    R1 = Rotation.align_vectors(ref_eigvecs.T, voxel_eigvecs.T)[0]
    R2 = Rotation.align_vectors(-ref_eigvecs.T, voxel_eigvecs.T)[0]
    
    rotated1 = R1.apply(voxel_peaks)
    rotated2 = R2.apply(voxel_peaks)
    
    rmsd1 = np.sqrt(np.mean(np.sum((rotated1 - ref_peaks)**2, axis=1)))
    rmsd2 = np.sqrt(np.mean(np.sum((rotated2 - ref_peaks)**2, axis=1)))
    
    # Choose the better alignment
    if rmsd1 < rmsd2:
        R = R1
        rmsd = rmsd1
    else:
        R = R2
        rmsd = rmsd2
    
    # Convert to axis-angle representation
    rotvec = R.as_rotvec(degrees=True)
    angle = np.linalg.norm(rotvec)
    
    # Normalize angle to 0-180 range and adjust axis accordingly
    if angle > 180:
        angle = 360 - angle
        axis = -rotvec / np.linalg.norm(rotvec)
    else:
        axis = rotvec / np.linalg.norm(rotvec)
    
    # Force consistent axis direction (z component should be positive for z-axis rotation)
    if axis[2] < 0:
        axis = -axis
        
    return axis, angle, rmsd


# Add VTK saving functionality
def save_peaks_to_vtk(fig_RS, filename="fft_peaks.vtp"):
    """
    Save peaks from a Plotly figure to VTK format.
    
    Args:
        fig_RS: Plotly figure containing 3D scatter traces
        filename: output filename (should end in .vtp)
    """
    import vtk
    from vtk.util import numpy_support
    
    # Extract all points and their values from the Plotly figure
    all_points = []
    all_values = []
    all_voxel_coords = []
    
    for trace in fig_RS.data:
        # Extract coordinates and values
        x = np.array(trace.x)
        y = np.array(trace.y)
        z = np.array(trace.z)
        values = np.array(trace.marker.color)
        
        # Extract voxel coordinates from the trace name
        name_parts = trace.name.split(',')
        voxel_x = int(name_parts[0].split('=')[1])
        voxel_y = int(name_parts[1].split('=')[1])
        voxel_z = int(name_parts[2].split('=')[1])
        
        # Store points and values
        points = np.column_stack((x, y, z))
        all_points.append(points)
        all_values.append(values)
        all_voxel_coords.append(np.array([[voxel_x, voxel_y, voxel_z]] * len(x)))
    
    # Combine all data
    all_points = np.vstack(all_points)
    all_values = np.concatenate(all_values)
    all_voxel_coords = np.vstack(all_voxel_coords)
    
    # Create vtkPoints object
    vtk_points = vtk.vtkPoints()
    for point in all_points:
        vtk_points.InsertNextPoint(point)
    
    # Create vtkPolyData object
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    
    # Create vertex cells
    vertices = vtk.vtkCellArray()
    for i in range(len(all_points)):
        vertex = vtk.vtkVertex()
        vertex.GetPointIds().SetId(0, i)
        vertices.InsertNextCell(vertex)
    polydata.SetVerts(vertices)
    
    # Add peak values as point data
    vtk_values = numpy_support.numpy_to_vtk(all_values)
    vtk_values.SetName("peak_intensity")
    polydata.GetPointData().AddArray(vtk_values)
    
    # Add normalized values
    normalized_values = all_values / np.max(all_values)
    vtk_normalized = numpy_support.numpy_to_vtk(normalized_values)
    vtk_normalized.SetName("normalized_intensity")
    polydata.GetPointData().AddArray(vtk_normalized)
    
    # Add voxel coordinates as point data
    for i, name in enumerate(['voxel_x', 'voxel_y', 'voxel_z']):
        vtk_coord = numpy_support.numpy_to_vtk(all_voxel_coords[:, i])
        vtk_coord.SetName(name)
        polydata.GetPointData().AddArray(vtk_coord)
    
    # Write to file
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()
    
    print(f"Saved {len(all_points)} peaks to {filename}")
    print("Data fields saved:")
    print("- peak_intensity: raw peak values")
    print("- normalized_intensity: values normalized to [0,1]")
    print("- voxel_x, voxel_y, voxel_z: voxel coordinates")


def find_peaks_3d(magnitude, threshold=0.1, sigma=1):
    # Apply Gaussian filter to smooth the data
    smoothed = gaussian_filter(magnitude, sigma=sigma)
    
    # Apply a threshold
    max_intensity = smoothed.max()
    threshold_value = max_intensity * threshold
    mask = smoothed > threshold_value
    
    # Use maximum filter to find local maxima
    local_max = maximum_filter(smoothed, size=3) == smoothed
    
    # Combine mask and local maxima
    peaks = mask & local_max
    
    # Label the peaks
    labeled, num_features = label(peaks)
    
    # Extract peak positions
    peak_positions = np.argwhere(peaks)
    
    return peak_positions, smoothed[peaks]

def find_peaks_3d_cutoff(magnitude, threshold=0.1, sigma=1, center_cutoff_radius=5):
    """
    Find peaks in 3D data with central region exclusion
    
    Args:
        magnitude: 3D numpy array
        threshold: relative threshold value (0-1)
        sigma: smoothing parameter for Gaussian filter
        center_cutoff_radius: radius (in pixels) around center to exclude
    
    Returns:
        peak_positions: array of peak coordinates
        peak_values: array of peak intensities
    """
    # Apply Gaussian filter to smooth the data
    smoothed = gaussian_filter(magnitude, sigma=sigma)
    
    # Create central cutoff mask
    center_z, center_y, center_x = np.array(magnitude.shape) // 2
    z, y, x = np.ogrid[:magnitude.shape[0], :magnitude.shape[1], :magnitude.shape[2]]
    central_mask = (x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2 > center_cutoff_radius**2
    
    # Apply threshold
    max_intensity = smoothed.max()
    threshold_value = max_intensity * threshold
    threshold_mask = smoothed > threshold_value
    
    # Use maximum filter to find local maxima
    local_max = maximum_filter(smoothed, size=3) == smoothed
    
    # Combine masks: threshold, local maxima, and central cutoff
    peaks = threshold_mask & local_max & central_mask
    
    # Label the peaks
    labeled, num_features = label(peaks)
    
    # Extract peak positions
    peak_positions = np.argwhere(peaks)
    
    return peak_positions, smoothed[peaks]

def calculate_orientation(projection, kx, ky):
    """Calculate primary orientation in a 2D projection using center of mass"""
    # Find center indices
    center_x = len(kx) // 2
    center_y = len(ky) // 2
    
    # Create coordinate grids relative to center
    y_coords, x_coords = np.indices(projection.shape)
    x_coords = x_coords - center_x
    y_coords = y_coords - center_y
    
    # Find points above threshold
    max_val = np.max(projection)
    threshold_mask = projection > (max_val * 0.1)  # Example threshold
    
    # Create circular mask to exclude the central region
    radius = 5  # Example radius
    r = np.sqrt(x_coords**2 + y_coords**2)
    central_mask = r > radius
    mask = threshold_mask & central_mask
    
    if np.sum(mask) < 2:  # Need at least 2 points
        return None, None, None
    
    # Calculate center of mass
    x_com, y_com = calculate_center_of_mass(projection * mask)
    
    # Offset the COM by the center of the image
    x_com -= center_x
    y_com -= center_y
    
    # Calculate angle from center of mass
    angle = np.arctan2(y_com, x_com)
    
    # Calculate magnitude (distance from center)
    magnitude = np.sqrt(x_com**2 + y_com**2)
    
    # Convert to real frequencies
    x_freq = x_com * (kx[1] - kx[0])
    y_freq = y_com * (ky[1] - ky[0])

    return angle, magnitude, (x_freq, y_freq)

def calculate_center_of_mass(image):
    """
    Calculate the center of mass of a 2D image.

    Args:
        image (np.ndarray): 2D array representing the image.

    Returns:
        tuple: (x_com, y_com) coordinates of the center of mass.
    """
    # Create coordinate grids
    y_indices, x_indices = np.indices(image.shape)
    
    # Calculate total mass
    image = np.nan_to_num(image, nan=0.0)
    total_mass = np.sum(image)
    
    if total_mass == 0:
        raise ValueError("The total mass of the image is zero, cannot compute center of mass.")
    
    # Calculate center of mass
    x_com = np.sum(x_indices * image) / total_mass
    y_com = np.sum(y_indices * image) / total_mass
    
    return x_com, y_com


def calculate_peaks_com(peaks_x, peaks_y, peaks_z, peak_values):
    """Calculate center of mass of peaks weighted by their values"""
    if len(peaks_x) == 0:
        return None, None, None
    
    total_weight = np.sum(peak_values)
    if total_weight == 0:
        return None, None, None
    
    com_x = np.sum(peaks_x * peak_values) / total_weight
    com_y = np.sum(peaks_y * peak_values) / total_weight
    com_z = np.sum(peaks_z * peak_values) / total_weight
    
    return com_x, com_y, com_z


def extract_voxel_region(tomo_data, voxel_results, z_idx, y_idx, x_idx):
    vz, vy, vx = voxel_results['voxel_size']
    nz, ny, nx = tomo_data.shape
    
    # Calculate the start and end indices for each dimension
    z_start = z_idx * vz
    z_end = min((z_idx + 1) * vz, nz)
    y_start = y_idx * vy
    y_end = min((y_idx + 1) * vy, ny)
    x_start = x_idx * vx
    x_end = min((x_idx + 1) * vx, nx)
    
    # Extract the region
    region = tomo_data[z_start:z_end, y_start:y_end, x_start:x_end]
    
    # Pad the region if it's smaller than the voxel size
    pad_width = ((0, vz - region.shape[0]), 
                 (0, vy - region.shape[1]), 
                 (0, vx - region.shape[2]))
    region_padded = np.pad(region, pad_width, mode='constant', constant_values=0)
    
    return region_padded

def compute_fft(region, use_vignette=False):
    if use_vignette:
        vignette = create_3d_vignette(region.shape)
        region_to_fft = region * vignette
    else:
        region_to_fft = region
    
    fft_3d = np.fft.fftn(region_to_fft)
    fft_3d_shifted = np.fft.fftshift(fft_3d)
    magnitude = np.abs(fft_3d_shifted)
    phase = np.angle(fft_3d_shifted)
    
    
    kz = np.fft.fftshift(np.fft.fftfreq(region.shape[0]))
    ky = np.fft.fftshift(np.fft.fftfreq(region.shape[1]))
    kx = np.fft.fftshift(np.fft.fftfreq(region.shape[2]))
    
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing='ij')
    
    return magnitude, KX, KY, KZ
    #return phase, KX, KY, KZ


def compute_fft_q(region, use_vignette=False, pixel_size=1.0,scale=1):
    """
    Compute 3D FFT and return magnitude and q-vectors in reciprocal space
    
    Args:
        region: 3D numpy array of real space data
        use_vignette: Boolean to apply vignette filter
        pixel_size: Real space pixel size in nanometers
    
    Returns:
        magnitude: FFT magnitude
        QX, QY, QZ: Q-vectors in reciprocal space (Å⁻¹)
    """
    if use_vignette:
        vignette = create_3d_vignette(region.shape,scale)
        region_to_fft = region * vignette
    else:
        region_to_fft = region
    
    fft_3d = np.fft.fftn(region_to_fft)
    fft_3d_shifted = np.fft.fftshift(fft_3d)
    magnitude = np.abs(fft_3d_shifted)
    
    # Calculate reciprocal space frequencies
    kz = np.fft.fftshift(np.fft.fftfreq(region.shape[0]))
    ky = np.fft.fftshift(np.fft.fftfreq(region.shape[1]))
    kx = np.fft.fftshift(np.fft.fftfreq(region.shape[2]))
    
    # Convert to q-space (Å⁻¹)
    # q = 4π*sin(θ)/λ = 2π/d, where d is real space distance
    # For small angles: q ≈ 2π*θ/λ
    pixel_size_A = pixel_size  # Convert nm to Å
    qz = 2 * np.pi * kz / (pixel_size_A)
    qy = 2 * np.pi * ky / (pixel_size_A)
    qx = 2 * np.pi * kx / (pixel_size_A)
    #print('Q pixel size of FFT:', qx[1]-qx[0], qy[1]-qy[0], qz[1]-qz[0])
    
    QZ, QY, QX = np.meshgrid(qz, qy, qx, indexing='ij')
    
    return magnitude, QX, QY, QZ


def create_3d_fft_plot(magnitude, KX, KY, KZ, fft_threshold):
    max_magnitude = np.max(magnitude)
    threshold = max_magnitude * fft_threshold
    mask = magnitude > threshold
    
    fig_3d = go.Figure(data=[go.Scatter3d(
        x=KX[mask],
        y=KY[mask],
        z=KZ[mask],
        mode='markers',
        marker=dict(
            size=5,
            color=np.log10(magnitude[mask] + 1),
            colorscale='Viridis',
            opacity=0.6,
            colorbar=dict(title='Log Magnitude')
        )
    )])
    
    fig_3d.update_layout(
        title="3D FFT of Voxel",
        scene=dict(
            xaxis_title="kx",
            yaxis_title="ky",
            zaxis_title="kz",
            aspectmode='cube',
            camera=dict(eye=dict(x=2, y=2, z=2))
        ),
        width=800, height=800
    )
    
    return fig_3d

def create_2d_projections(magnitude):
    proj_xy = np.max(magnitude, axis=0)
    proj_xz = np.max(magnitude, axis=1)
    proj_yz = np.max(magnitude, axis=2)
    return proj_xy, proj_xz, proj_yz

def calculate_and_plot_orientations(proj_xy, proj_xz, proj_yz, kx, ky, kz):
    angle_xy, mag_xy, freq_xy = calculate_orientation(proj_xy, kx, ky)
    angle_xz, mag_xz, freq_xz = calculate_orientation(proj_xz, kx, kz)
    angle_yz, mag_yz, freq_yz = calculate_orientation(proj_yz, ky, kz)
    
    fig_xy = plot_projection(proj_xy, kx, ky, angle_xy, mag_xy, freq_xy, "XY Projection")
    fig_xz = plot_projection(proj_xz, kx, kz, angle_xz, mag_xz, freq_xz, "XZ Projection")
    fig_yz = plot_projection(proj_yz, ky, kz, angle_yz, mag_yz, freq_yz, "YZ Projection")
    
    return fig_xy, fig_xz, fig_yz

def analyze_voxel_fourier(tomo_data, voxel_results, z_idx, y_idx, x_idx, fft_threshold=1e-3, use_vignette=False, overlay_octants=False, plot_projections=False):
    region = extract_voxel_region(tomo_data, voxel_results, z_idx, y_idx, x_idx)
    magnitude, KX, KY, KZ = compute_fft(region, use_vignette)
    
    fig_3d = create_3d_fft_plot(magnitude, KX, KY, KZ, fft_threshold)
    
    if overlay_octants:
        octant_intensities = calculate_octant_intensities(magnitude, KX, KY, KZ)
        kx_vector, ky_vector, kz_vector = calculate_orientation_vector_from_octants(octant_intensities, KX, KY, KZ)
        plot_orientation_vector_on_fft(fig_3d, kx_vector, ky_vector, kz_vector)
    
    proj_xy, proj_xz, proj_yz = create_2d_projections(magnitude)
    fig_xy, fig_xz, fig_yz = calculate_and_plot_orientations(proj_xy, proj_xz, proj_yz, KX[0,0,:], KY[0,:,0], KZ[:,0,0])

    return fig_3d, fig_xy, fig_yz, fig_xz

def plot_3D_tomogram(tomo_data, intensity_threshold=0.1):
    """
    Create efficient 3D visualization of tomogram
    
    Args:
        tomo_data (np.ndarray): 3D tomographic data
        intensity_threshold (float): Threshold relative to max intensity (0-1)
    """
    # Ensure the shape is interpreted correctly
    nz, ny, nx = tomo_data.shape  # Assuming (z, y, x) order
    z = np.arange(nz)
    y = np.arange(ny)
    x = np.arange(nx)
    
    # Create meshgrid
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')  # Ensure correct order
    
    # Apply threshold
    max_intensity = tomo_data.max()
    threshold = max_intensity * intensity_threshold
    mask = tomo_data > threshold
    
    # Get masked coordinates and intensities
    x_plot = X[mask]
    y_plot = Y[mask]
    z_plot = Z[mask]
    intensities_plot = tomo_data[mask]
    
    # Create figure
    fig = go.Figure(data=[go.Scatter3d(
        x=x_plot,
        y=y_plot,
        z=z_plot,
        mode='markers',
        marker=dict(
            size=3,
            color=intensities_plot,
            colorscale='Greys',  # 'Viridis',
            opacity=0.8,
            colorbar=dict(title='Intensity')
        ),
        hovertemplate=(
            "x: %{x}<br>" +
            "y: %{y}<br>" +
            "z: %{z}<br>" +
            "Intensity: %{marker.color:.1f}<br>" +
            "<extra></extra>"
        )
    )])
    
    # Update layout
    fig.update_layout(
        title="3D Tomogram Visualization",
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode='cube',
            camera=dict(
                eye=dict(x=2, y=2, z=2)
            ),
            bgcolor='white'
        ),
        width=800,
        height=800
    )
    
    return fig

def analyze_tomogram_voxels(tomo_data, voxel_size=(10, 10, 10)):
    """
    Break tomogram into voxels and analyze, including partial voxels at edges
    
    Args:
        tomo_data (np.ndarray): 3D tomographic data
        voxel_size (tuple): Size of voxels in (z, y, x)
    
    Returns:
        dict: Voxel analysis results
    """
    nz, ny, nx = tomo_data.shape
    vz, vy, vx = voxel_size
    
    # Calculate number of voxels needed to cover entire volume
    n_voxels_z = int(np.ceil(nz / vz))
    n_voxels_y = int(np.ceil(ny / vy))
    n_voxels_x = int(np.ceil(nx / vx))
    
    # Initialize arrays for voxel statistics
    voxel_means = np.zeros((n_voxels_z, n_voxels_y, n_voxels_x))
    voxel_maxes = np.zeros_like(voxel_means)
    voxel_stds = np.zeros_like(voxel_means)
    
    # Calculate statistics for each voxel
    for iz in range(n_voxels_z):
        z_start = iz * vz
        z_end = min((iz + 1) * vz, nz)
        
        for iy in range(n_voxels_y):
            y_start = iy * vy
            y_end = min((iy + 1) * vy, ny)
            
            for ix in range(n_voxels_x):
                x_start = ix * vx
                x_end = min((ix + 1) * vx, nx)
                
                voxel = tomo_data[z_start:z_end, 
                                y_start:y_end, 
                                x_start:x_end]
                
                voxel_means[iz, iy, ix] = np.mean(voxel)
                voxel_maxes[iz, iy, ix] = np.max(voxel)
                voxel_stds[iz, iy, ix] = np.std(voxel)
    
    return {
        'means': voxel_means,
        'maxes': voxel_maxes,
        'stds': voxel_stds,
        'voxel_size': voxel_size,
        'n_voxels': (n_voxels_z, n_voxels_y, n_voxels_x)
    }

def plot_voxel_region(tomo_data, voxel_results, z_idx, y_idx, x_idx, intensity_threshold=0.1):
    """
    Plot a specific voxel region and its location in the full tomogram
    """
    vz, vy, vx = voxel_results['voxel_size']
    
    # Extract voxel region
    region = tomo_data[z_idx*vz:(z_idx+1)*vz, 
                      y_idx*vy:(y_idx+1)*vy, 
                      x_idx*vx:(x_idx+1)*vx]
    
    # Create coordinate arrays for this region
    z, y, x = np.meshgrid(np.arange(vz), np.arange(vy), np.arange(vx), indexing='ij')
    
    # Apply threshold to region
    max_intensity = region.max()
    threshold = max_intensity * intensity_threshold
    mask = region > threshold
    
    # Get masked coordinates and intensities for region
    x_plot = x[mask] + x_idx*vx
    y_plot = y[mask] + y_idx*vy
    z_plot = z[mask] + z_idx*vz
    intensities_plot = region[mask]
    
    # Create figure with two subplots
    fig = go.Figure()
    
    # Plot 1: Voxel region
    fig.add_trace(go.Scatter3d(
        x=x_plot, y=y_plot, z=z_plot,
        mode='markers',
        marker=dict(
            size=2,
            color=intensities_plot,
            colorscale='Viridis',
            opacity=0.6#,
            #colorbar=dict(title='Intensity', x=0.45)
        ),
        name='Voxel Region',
        showlegend=False
    ))
    
    # Plot 2: Full tomogram with highlighted region
    # Apply threshold to full tomogram
    full_max = tomo_data.max()
    full_threshold = full_max * intensity_threshold
    full_mask = tomo_data > full_threshold
    
    # Create meshgrid for full tomogram - corrected indexing
    Z, Y, X = np.meshgrid(np.arange(tomo_data.shape[0]), 
                         np.arange(tomo_data.shape[1]), 
                         np.arange(tomo_data.shape[2]), 
                         indexing='ij')
    
    # Add full tomogram points with low opacity
    fig.add_trace(go.Scatter3d(
        x=X[full_mask],
        y=Y[full_mask],
        z=Z[full_mask],
        mode='markers',
        marker=dict(
            size=2,
            color='gray',
            opacity=0.1
        ),
        name='Full Tomogram',
        showlegend=False
    ))
    
    # Add box to highlight voxel region
    box_x = [x_idx*vx, (x_idx+1)*vx, (x_idx+1)*vx, x_idx*vx, x_idx*vx, 
            x_idx*vx, (x_idx+1)*vx, (x_idx+1)*vx, x_idx*vx, x_idx*vx]
    box_y = [y_idx*vy, y_idx*vy, (y_idx+1)*vy, (y_idx+1)*vy, y_idx*vy,
            y_idx*vy, y_idx*vy, (y_idx+1)*vy, (y_idx+1)*vy, y_idx*vy]
    box_z = [z_idx*vz, z_idx*vz, z_idx*vz, z_idx*vz, z_idx*vz,
            (z_idx+1)*vz, (z_idx+1)*vz, (z_idx+1)*vz, (z_idx+1)*vz, (z_idx+1)*vz]
    
    fig.add_trace(go.Scatter3d(
        x=box_x, y=box_y, z=box_z,
        mode='lines',
        line=dict(color='red', width=4),
        name='Region Box',
        showlegend=False
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Voxel Region (z={z_idx}, y={y_idx}, x={x_idx})",
        scene=dict(
            aspectmode='cube',
            camera=dict(eye=dict(x=2, y=2, z=2))
        ),
        width=1000, height=800,
        showlegend=False
    )
    
    # Create local view figure
    fig_local = go.Figure(data=[go.Scatter3d(
        x=x[mask], y=y[mask], z=z[mask],
        mode='markers',
        marker=dict(
            size=10,
            color=region[mask],
            colorscale='Viridis',
            opacity=0.6#,
            #colorbar=dict(title='Intensity')
        )
    )])
    
    fig_local.update_layout(
        title=f"Local Voxel View (z={z_idx}, y={y_idx}, x={x_idx})",
        scene=dict(
            aspectmode='cube',
            camera=dict(eye=dict(x=2, y=2, z=2))
        ),
        width=800, height=800
    )
    
    return fig, fig_local

def create_3d_vignette(shape,scale=1):
    """
    Create a 3D cosine window vignette
    
    Args:
        shape (tuple): Shape of the 3D volume (z, y, x)
    
    Returns:
        np.ndarray: 3D vignette array
    """
    z, y, x = np.ogrid[0:shape[0], 0:shape[1], 0:shape[2]]
    
    # Create normalized coordinates (-1 to 1)
    z_norm = 2 * z / (shape[0] - 1) - 1
    y_norm = 2 * y / (shape[1] - 1) - 1
    x_norm = 2 * x / (shape[2] - 1) - 1
    
    # Calculate radial distance from center (squared)
    r_squared = x_norm**2 + y_norm**2 + z_norm**2
    
    # Create cosine window
    vignette = np.cos(np.pi/2 * np.sqrt(r_squared)*scale)
    vignette = np.clip(vignette, 0, 1)
    
    return vignette



def calculate_octant_intensities(magnitude, KX, KY, KZ):
    """
    Calculate the total intensity for each of the 8 octants in the 3D FFT magnitude.
    
    Args:
        magnitude (np.ndarray): 3D FFT magnitude.
        KX, KY, KZ (np.ndarray): Frequency coordinates.
    
    Returns:
        dict: Total intensity for each octant.
    """
    # Initialize dictionary to store total intensity for each octant
    octant_intensities = {
        '+++' : 0,
        '++-' : 0,
        '+-+' : 0,
        '+--' : 0,
        '-++' : 0,
        '-+-' : 0,
        '--+' : 0,
        '---' : 0
    }
    
    # Determine the octant for each point and sum the magnitudes
    for i in range(magnitude.shape[0]):
        for j in range(magnitude.shape[1]):
            for k in range(magnitude.shape[2]):
                key = ('+' if KX[i, j, k] >= 0 else '-') + \
                      ('+' if KY[i, j, k] >= 0 else '-') + \
                      ('+' if KZ[i, j, k] >= 0 else '-')
                octant_intensities[key] += magnitude[i, j, k]
    
    return octant_intensities

def calculate_orientation_vector_from_octants(octant_intensities, KX, KY, KZ):
    """
    Calculate the orientation vector pointing to the octant with the highest intensity.
    
    Args:
        octant_intensities (dict): Total intensity for each octant.
        KX, KY, KZ (np.ndarray): Frequency coordinates.
    
    Returns:
        tuple: (kx_vector, ky_vector, kz_vector) coordinates of the orientation vector.
    """
    # Find the octant with the highest intensity
    max_octant = max(octant_intensities, key=octant_intensities.get)
    
    # Define the center of each octant in frequency space
    kx_center, ky_center, kz_center = 0, 0, 0
    kx_max, ky_max, kz_max = KX.max(), KY.max(), KZ.max()
    kx_min, ky_min, kz_min = KX.min(), KY.min(), KZ.min()

    octant_centers = {
        '+++': ((kx_max + kx_center) / 2, (ky_max + ky_center) / 2, (kz_max + kz_center) / 2),
        '++-': ((kx_max + kx_center) / 2, (ky_max + ky_center) / 2, (kz_min + kz_center) / 2),
        '+-+': ((kx_max + kx_center) / 2, (ky_min + ky_center) / 2, (kz_max + kz_center) / 2),
        '+--': ((kx_max + kx_center) / 2, (ky_min + ky_center) / 2, (kz_min + kz_center) / 2),
        '-++': ((kx_min + kx_center) / 2, (ky_max + ky_center) / 2, (kz_max + kz_center) / 2),
        '-+-': ((kx_min + kx_center) / 2, (ky_max + ky_center) / 2, (kz_min + kz_center) / 2),
        '--+': ((kx_min + kx_center) / 2, (ky_min + ky_center) / 2, (kz_max + kz_center) / 2),
        '---': ((kx_min + kx_center) / 2, (ky_min + ky_center) / 2, (kz_min + kz_center) / 2)
    }
    
    # Get the center of the octant with the highest intensity
    kx_vector, ky_vector, kz_vector = octant_centers[max_octant]
    
    return kx_vector, ky_vector, kz_vector

def plot_orientation_vector_on_fft(fig, kx_vector, ky_vector, kz_vector):
    """
    Add an orientation vector to the 3D FFT plot.
    
    Args:
        fig (go.Figure): Plotly figure.
        kx_vector, ky_vector, kz_vector (float): Coordinates of the orientation vector.
    """
    fig.add_trace(go.Cone(
        x=[0], y=[0], z=[0],
        u=[kx_vector], v=[ky_vector], w=[kz_vector],
        sizemode="absolute",
        sizeref=0.1,
        anchor="tail",
        colorscale='Reds',
        showscale=False
    ))

def plot_projection(projection, kx, ky, angle, magnitude, freq_coords, title):
    """Plot a 2D projection with an orientation vector."""
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(z=projection,
                   x=kx, y=ky,
                   colorscale='Viridis',
                   showscale=False)
    )
    
    if angle is not None:
        x_com, y_com = freq_coords
        arrow_length = 0.25 * magnitude
        
        x_end = x_com + arrow_length * np.cos(angle)
        y_end = y_com + arrow_length * np.sin(angle)
        
        fig.add_trace(
            go.Scatter(
                x=[x_com, x_end],
                y=[y_com, y_end],
                mode='lines',
                line=dict(color='red', width=4),
                showlegend=False
            )
        )
    
    fig.update_layout(
        title=title,
        xaxis_title="kx",
        yaxis_title="ky",
        width=600, height=600
    )
    
    return fig

def calculate_wedge_intensities_with_radius(array, num_wedges=32, min_radius=2.5, max_radius=7.5):
    """
    Calculate the intensity of discrete wedges on a 2D array within a radius range.
    
    Args:
        array (np.ndarray): 2D array representing the image.
        num_wedges (int): Number of wedges to divide the array into.
        min_radius (float): Minimum radius for the wedge calculation.
        max_radius (float): Maximum radius for the wedge calculation.
    
    Returns:
        list: Intensities of each wedge.
    """
    # Get the center of the array
    center_y, center_x = np.array(array.shape) // 2
    
    # Create coordinate grids
    y_indices, x_indices = np.indices(array.shape)
    x_indices = x_indices - center_x
    y_indices = y_indices - center_y
    
    # Calculate distances and angles for each point
    distances = np.sqrt(x_indices**2 + y_indices**2)
    angles = np.arctan2(y_indices, x_indices)
    
    # Normalize angles to [0, 2*pi]
    angles = (angles + 2 * np.pi) % (2 * np.pi)
    
    # Calculate wedge boundaries
    wedge_boundaries = np.linspace(0, 2 * np.pi, num_wedges + 1)
    
    # Calculate intensities for each wedge within the radius range
    wedge_intensities = []
    for i in range(num_wedges):
        # Create a mask for the current wedge and radius range
        mask = ((angles >= wedge_boundaries[i]) & (angles < wedge_boundaries[i + 1]) &
                (distances >= min_radius) & (distances <= max_radius))
        
        # Sum the intensities within the wedge
        wedge_intensity = np.sum(array[mask])
        wedge_intensities.append(wedge_intensity)
    
    return wedge_intensities

def visualize_wedges(array, num_wedges=32, min_radius=2.5, max_radius=7.5, show_first_wedge=False):
    """
    Visualize the wedges within a specified radius range on a 2D array.
    
    Args:
        array (np.ndarray): 2D array representing the image.
        num_wedges (int): Number of wedges to divide the array into.
        min_radius (float): Minimum radius for the wedge calculation.
        max_radius (float): Maximum radius for the wedge calculation.
        show_first_wedge (bool): Whether to overlay only the first wedge.
    """
    # Get the center of the array
    center_y, center_x = np.array(array.shape) // 2
    
    # Create coordinate grids
    y_indices, x_indices = np.indices(array.shape)
    x_indices = x_indices - center_x
    y_indices = y_indices - center_y
    
    # Calculate distances and angles for each point
    distances = np.sqrt(x_indices**2 + y_indices**2)
    angles = np.arctan2(y_indices, x_indices)
    
    # Normalize angles to [0, 2*pi]
    angles = (angles + 2 * np.pi) % (2 * np.pi)
    
    # Calculate wedge boundaries
    wedge_boundaries = np.linspace(0, 2 * np.pi, num_wedges + 1)
    
    # Create a mask for the wedges within the radius range
    mask = np.zeros_like(array, dtype=bool)
    for i in range(num_wedges):
        wedge_mask = ((angles >= wedge_boundaries[i]) & (angles < wedge_boundaries[i + 1]) &
                      (distances >= min_radius) & (distances <= max_radius))
        mask |= wedge_mask
    
    # Create a mask for the first wedge
    first_wedge_mask = ((angles >= wedge_boundaries[0]) & (angles < wedge_boundaries[1]) &
                        (distances >= min_radius) & (distances <= max_radius))
    
    # Plot the original array
    plt.imshow(array, cmap='gray', origin='lower')
    plt.colorbar(label='Intensity')
    
    # Overlay the wedge mask
    if show_first_wedge:
        plt.imshow(first_wedge_mask, cmap='cool', alpha=0.8, origin='lower')
    else:
        plt.imshow(mask, cmap='cool', alpha=0.5, origin='lower')
    
    # Plot the center and radius boundaries
    circle1 = plt.Circle((center_x, center_y), min_radius, color='red', fill=False, linestyle='--')
    circle2 = plt.Circle((center_x, center_y), max_radius, color='red', fill=False, linestyle='--')
    plt.gca().add_patch(circle1)
    plt.gca().add_patch(circle2)
    
    plt.title('Wedge Visualization')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()



def calculate_3d_wedge_intensities(array, num_azimuthal_wedges, num_polar_wedges, min_radius, max_radius):
    """
    Calculate the intensity of discrete wedges on a 3D array within a radius range.
    
    Args:
        array (np.ndarray): 3D array representing the data.
        num_azimuthal_wedges (int): Number of azimuthal wedges (phi).
        num_polar_wedges (int): Number of polar wedges (theta).
        min_radius (float): Minimum radius for the wedge calculation.
        max_radius (float): Maximum radius for the wedge calculation.
    
    Returns:
        np.ndarray: Intensities of each wedge.
    """
    # Get the center of the array
    center_z, center_y, center_x = np.array(array.shape) // 2
    
    # Create coordinate grids
    z_indices, y_indices, x_indices = np.indices(array.shape)
    x_indices = x_indices - center_x
    y_indices = y_indices - center_y
    z_indices = z_indices - center_z
    
    # Calculate spherical coordinates
    distances = np.sqrt(x_indices**2 + y_indices**2 + z_indices**2)
    azimuthal_angles = np.arctan2(y_indices, x_indices)  # phi
    polar_angles = np.arccos(z_indices / (distances + 1e-10))  # theta
    
    # Normalize angles
    azimuthal_angles = (azimuthal_angles + 2 * np.pi) % (2 * np.pi)
    
    # Calculate wedge boundaries
    azimuthal_boundaries = np.linspace(0, 2 * np.pi, num_azimuthal_wedges + 1)
    polar_boundaries = np.linspace(0, np.pi, num_polar_wedges + 1)
    
    # Calculate intensities for each wedge
    wedge_intensities = np.zeros((num_azimuthal_wedges, num_polar_wedges))
    for i in range(num_azimuthal_wedges):
        for j in range(num_polar_wedges):
            # Create a mask for the current wedge
            mask = ((azimuthal_angles >= azimuthal_boundaries[i]) & (azimuthal_angles < azimuthal_boundaries[i + 1]) &
                    (polar_angles >= polar_boundaries[j]) & (polar_angles < polar_boundaries[j + 1]) &
                    (distances >= min_radius) & (distances <= max_radius))
            
            # Sum the intensities within the wedge
            wedge_intensity = np.sum(array[mask])
            wedge_intensities[i, j] = wedge_intensity
    
    return wedge_intensities


def extract_3d_data_from_figure(fig):
    """
    Extract 3D data from a Plotly figure.
    
    Args:
        fig (go.Figure): Plotly 3D figure.
    
    Returns:
        np.ndarray: 3D array of intensities.
    """
    # Assuming the data is in the first trace
    x = fig.data[0]['x']
    y = fig.data[0]['y']
    z = fig.data[0]['z']
    intensity = fig.data[0]['marker']['color']
    
    # Create a 3D grid based on the unique x, y, z values
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    z_unique = np.unique(z)
    
    # Initialize a 3D array
    array_3d = np.zeros((len(z_unique), len(y_unique), len(x_unique)))
    
    # Fill the 3D array with intensity values
    for xi, yi, zi, inten in zip(x, y, z, intensity):
        x_idx = np.where(x_unique == xi)[0][0]
        y_idx = np.where(y_unique == yi)[0][0]
        z_idx = np.where(z_unique == zi)[0][0]
        array_3d[z_idx, y_idx, x_idx] = inten
    
    return array_3d


def compute_orientation_tensor(magnitude, KX, KY, KZ):
    """
    Compute the orientation tensor for a voxel based on its 3D FFT magnitude.

    Args:
        magnitude (np.ndarray): 3D FFT magnitude.
        KX, KY, KZ (np.ndarray): Frequency coordinates.

    Returns:
        np.ndarray: 3x3 orientation tensor.
    """
    magnitude_normalized = magnitude / np.max(magnitude)
    
    # Calculate the center of mass in Fourier space
    total_intensity = np.sum(magnitude_normalized)
    if total_intensity == 0:
        return np.zeros((3, 3))


    
    kx_com = np.sum(KX * magnitude_normalized) / total_intensity
    ky_com = np.sum(KY * magnitude_normalized) / total_intensity
    kz_com = np.sum(KZ * magnitude_normalized) / total_intensity

    # Construct the orientation tensor
    orientation_tensor = np.array([
        [kx_com**2, kx_com*ky_com, kx_com*kz_com],
        [ky_com*kx_com, ky_com**2, ky_com*kz_com],
        [kz_com*kx_com, kz_com*ky_com, kz_com**2]
    ])

    return orientation_tensor

def generate_voxel_indices(x_range, y_range, z_range):
    """
    Generate voxel indices for a cubic range.

    Parameters:
    - x_range: tuple of (start, end) for the x dimension
    - y_range: tuple of (start, end) for the y dimension
    - z_range: tuple of (start, end) for the z dimension

    Returns:
    - List of tuples representing voxel indices within the specified range.
    """
    voxel_indices = [
        (z, y, x)
        for z in range(x_range[0], x_range[1])
        for y in range(y_range[0], y_range[1])
        for x in range(z_range[0], z_range[1])
    ]
    return voxel_indices



def initialize_combined_figure(nrows):
    """
    Initialize a combined figure with multiple rows of 3D subplots.
    
    Args:
        nrows (int): Number of rows in the combined figure
        
    Returns:
        plotly.graph_objects.Figure: Initialized figure with proper layout
    """
    # Create subplot titles for each row
    subplot_titles = []
    for i in range(nrows):
        subplot_titles.extend([f"Voxel Region {i+1}", f"FFT Peaks {i+1}"])
    
    # Create specs for 3D scenes
    specs = [[{'type': 'scene'}, {'type': 'scene'}] for _ in range(nrows)]
    
    # Initialize figure
    fig_combined = make_subplots(
        rows=nrows, 
        cols=2,
        subplot_titles=subplot_titles,
        specs=specs,
        vertical_spacing=0.02,
        horizontal_spacing=0.05
    )
    
    # Update overall layout
    fig_combined.update_layout(
        title="Voxel Analysis Along Z-Axis",
        width=1200,
        height=400*nrows,
        showlegend=False
    )
    
    # Update each subplot's layout
    for i in range(nrows):
        # Calculate vertical position for this row
        y_max = 1.0 - (i/nrows)
        y_min = 1.0 - ((i+1)/nrows)
        
        # Update scene for left subplot (Voxel Region)
        scene_name = f'scene{i*2 + 1}' if i > 0 else 'scene'
        fig_combined.update_layout(**{
            scene_name: dict(
                aspectmode='cube',
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                domain=dict(x=[0, 0.45], y=[y_min, y_max]),
                camera=dict(
                    eye=dict(x=0, y=2.5, z=0),  # front view
                    up=dict(x=0, y=0, z=1)
                )
            )
        })
        
        # Update scene for right subplot (FFT Peaks)
        scene_name = f'scene{i*2 + 2}'
        fig_combined.update_layout(**{
            scene_name: dict(
                aspectmode='cube',
                xaxis_title="KX",
                yaxis_title="KY",
                zaxis_title="KZ",
                domain=dict(x=[0.55, 1.0], y=[y_min, y_max]),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),  # isometric view
                    up=dict(x=0, y=0, z=1)
                )
            )
        })
    
    return fig_combined

def visualize_single_voxel_orientation(tomo_data, voxel_results, crystal_peaks, h, k, l, 
                                     z_idx, y_idx, x_idx,threshold=0.1, sigma=0.5, cutoff=3, pixel_size=18):
    """
    Analyze orientation for a single voxel and overlay on tomogram
    Parameters:
        z_idx, y_idx, x_idx: indices of the voxel to analyze
    """
    # Analyze single voxel
    voxel_data = extract_voxel_region(tomo_data, voxel_results, z_idx, y_idx, x_idx)
    _, angle, rmsd = test_orientation_analysis(voxel_data, crystal_peaks, 
                                             z_idx, y_idx, x_idx, 
                                             h, k, l,threshold=threshold, sigma=sigma, cutoff=cutoff, pixel_size=pixel_size, 
                                             visualize=False)
    
    # Create coordinate arrays for full tomogram
    Z, Y, X = np.meshgrid(np.arange(tomo_data.shape[0]), 
                         np.arange(tomo_data.shape[1]), 
                         np.arange(tomo_data.shape[2]), 
                         indexing='ij')
    
    # Apply threshold to full tomogram
    full_max = tomo_data.max()
    full_threshold = full_max * 0.8
    full_mask = tomo_data > full_threshold
    # Create figure
    fig = go.Figure()
    
    # Add full tomogram points with low opacity
    fig.add_trace(go.Scatter3d(
        x=X[full_mask],
        y=Y[full_mask],
        z=Z[full_mask],
        mode='markers',
        marker=dict(
            size=2,
            color='gray',
            opacity=0.1
        ),
        name='Background',
        showlegend=False
    ))
    
    # Add analyzed voxel
    vz, vy, vx = voxel_results['voxel_size']
    z_start, z_end = z_idx*vz, (z_idx+1)*vz
    y_start, y_end = y_idx*vy, (y_idx+1)*vy
    x_start, x_end = x_idx*vx, (x_idx+1)*vx
    
    # Extract voxel region directly using array indexing
    voxel_region = tomo_data[z_start:z_end, y_start:y_end, x_start:x_end]
    voxel_mask = voxel_region > full_threshold
    
    # Create coordinate arrays for the voxel
    z_coords, y_coords, x_coords = np.where(voxel_mask)
    
    # Adjust coordinates to global position
    z_coords += z_start
    y_coords += y_start
    x_coords += x_start
    
    if len(z_coords) > 0:
        # Create cyclic colorscale over 120 degrees
        cyclic_angle = angle % 120  # Make angle cyclic over 120 degrees
        normalized_angle = cyclic_angle / 120  # Normalize to [0,1]
        
        # Custom colorscale that's continuous from start to end
        colors = [
            [0, 'rgb(68,1,84)'],       # Viridis start
            [0.25, 'rgb(59,82,139)'],
            [0.5, 'rgb(33,144,141)'],
            [0.75, 'rgb(94,201,98)'],
            [1, 'rgb(68,1,84)']        # Back to start for continuity
        ]
        
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=3,
                color=[normalized_angle] * len(z_coords),  # Same angle for all points
                colorscale=colors,
                opacity=0.3,
                colorbar=dict(
                    title='Orientation Angle (°)',
                    tickmode='array',
                    ticktext=['0°', '30°', '60°', '90°', '120°'],
                    tickvals=[0, 0.25, 0.5, 0.75, 1.0]
                )
            ),
            name=f'Voxel ({z_idx},{y_idx},{x_idx})',
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Orientation = {angle:.1f}°, RMSD = {rmsd:.2e}",
            y=0.95
        ),
        scene=dict(
            aspectmode='data',
            camera=dict(eye=dict(x=2, y=2, z=2)),
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        width=1000, height=800,
        showlegend=True
    )
    
    return fig


def visualize_line_orientation(tomo_data, voxel_results, crystal_peaks, h, k, l, 
                             z_idx, y_range, x_idx,threshold=0.1, sigma=0.5, cutoff=3, pixel_size=18):
    """
    Analyze orientation for a line of voxels along y-axis
    Parameters:
        z_idx, x_idx: fixed indices for z and x
        y_range: range of y indices to analyze
    """
    # Create coordinate arrays for full tomogram
    Z, Y, X = np.meshgrid(np.arange(tomo_data.shape[0]), 
                         np.arange(tomo_data.shape[1]), 
                         np.arange(tomo_data.shape[2]), 
                         indexing='ij')
    
    # Apply threshold to full tomogram
    full_max = tomo_data.max()
    full_threshold = full_max * 0.7
    full_mask = tomo_data > full_threshold
    
    # Create figure
    fig = go.Figure()
    
    # Add full tomogram points with low opacity
    fig.add_trace(go.Scatter3d(
        x=X[full_mask],
        y=Y[full_mask],
        z=Z[full_mask],
        mode='markers',
        marker=dict(
            size=2,
            color='gray',
            opacity=0.1
        ),
        name='Background',
        showlegend=False
    ))
    
    # Process each voxel in the y-range
    vz, vy, vx = voxel_results['voxel_size']
    all_angles = []
    all_coords = []
    
    for y_idx in y_range:
        # Extract and analyze voxel
        voxel_data = extract_voxel_region(tomo_data, voxel_results, z_idx, y_idx, x_idx)
        try:
            _, angle, rmsd = test_orientation_analysis(voxel_data, crystal_peaks, 
                                                     z_idx, y_idx, x_idx, 
                                                     h, k, l,threshold=threshold, sigma=sigma, cutoff=cutoff, pixel_size=pixel_size, 
                                                     visualize=False)
            
            # Get voxel coordinates
            z_start, z_end = z_idx*vz, (z_idx+1)*vz
            y_start, y_end = y_idx*vy, (y_idx+1)*vy
            x_start, x_end = x_idx*vx, (x_idx+1)*vx
            
            voxel_region = tomo_data[z_start:z_end, y_start:y_end, x_start:x_end]
            voxel_mask = voxel_region > full_threshold
            
            z_coords, y_coords, x_coords = np.where(voxel_mask)
            
            # Adjust coordinates to global position
            z_coords += z_start
            y_coords += y_start
            x_coords += x_start
            
            # Store results
            cyclic_angle = angle % 120
            normalized_angle = cyclic_angle / 120
            
            all_angles.extend([normalized_angle] * len(z_coords))
            all_coords.extend(zip(x_coords, y_coords, z_coords))
            
        except Exception as e:
            print(f"Error processing voxel (z={z_idx},y={y_idx},x={x_idx}): {e}")
            continue
    
    if all_coords:
        # Convert coordinates to arrays
        x_coords, y_coords, z_coords = zip(*all_coords)
        
        # Custom colorscale
        colors = [
            [0, 'rgb(68,1,84)'],
            [0.25, 'rgb(59,82,139)'],
            [0.5, 'rgb(33,144,141)'],
            [0.75, 'rgb(94,201,98)'],
            [1, 'rgb(68,1,84)']
        ]
        
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=3,
                color=all_angles,
                colorscale=colors,
                opacity=0.3,
                colorbar=dict(
                    title='Orientation Angle (°)',
                    tickmode='array',
                    ticktext=['0°', '30°', '60°', '90°', '120°'],
                    tickvals=[0, 0.25, 0.5, 0.75, 1.0]
                )
            ),
            name='Analyzed Voxels',
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Orientation Analysis Along Y-axis (z={z_idx}, y={y_range[0]}-{y_range[-1]}, x={x_idx})",
            y=0.95
        ),
        scene=dict(
            aspectmode='data',
            camera=dict(eye=dict(x=2, y=2, z=2)),
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        width=1000, height=800,
        showlegend=True
    )
    
    return fig


def visualize_section_orientation(tomo_data, voxel_results, crystal_peaks, h, k, l, 
                                z_range, y_range, x_range, threshold=0.1, sigma=0.5, cutoff=3, pixel_size=18,cyclic_period=None):
    """
    Analyze orientation for a 3D section of voxels
    Parameters:
        z_range, y_range, x_range: ranges of indices to analyze
        cyclic_period: if set, angles will be made cyclic with this period (in degrees)
                      if None, raw angles will be used
    """
    # Create coordinate arrays for full tomogram
    Z, Y, X = np.meshgrid(np.arange(tomo_data.shape[0]), 
                         np.arange(tomo_data.shape[1]), 
                         np.arange(tomo_data.shape[2]), 
                         indexing='ij')
    
    # Apply threshold to full tomogram
    full_max = tomo_data.max()
    full_threshold = full_max * 0.75
    full_mask = tomo_data > full_threshold
    
    # Create figure
    fig = go.Figure()
    
    # Add full tomogram points with low opacity
    fig.add_trace(go.Scatter3d(
        x=X[full_mask],
        y=Y[full_mask],
        z=Z[full_mask],
        mode='markers',
        marker=dict(
            size=2,
            color='gray',
            opacity=0.05,
        ),
        name='Background',
        showlegend=False
    ))
    
    # Process each voxel in the section
    vz, vy, vx = voxel_results['voxel_size']
    all_angles = []
    all_coords = []
    all_rmsds = []
    
    # Progress tracking
    total_voxels = len(z_range) * len(y_range) * len(x_range)
    processed = 0
    
    # Track angle range for normalization
    min_angle = float('inf')
    max_angle = float('-inf')
    
    for z_idx in z_range:
        for y_idx in y_range:
            for x_idx in x_range:
                processed += 1
                if processed % 10 == 0:
                    print(f"Processing voxel {processed}/{total_voxels}")
                
                voxel_data = extract_voxel_region(tomo_data, voxel_results, z_idx, y_idx, x_idx)
                try:
                    _, angle, rmsd = test_orientation_analysis(voxel_data, crystal_peaks, 
                                                             z_idx, y_idx, x_idx, 
                                                             h, k, l,threshold=threshold, sigma=sigma, cutoff=cutoff, pixel_size=pixel_size, 
                                                             visualize=False)
                    
                    # Process angle based on cyclic_period
                    if cyclic_period is not None:
                        angle = angle % cyclic_period
                    
                    # Update angle range
                    min_angle = min(min_angle, angle)
                    max_angle = max(max_angle, angle)
                    
                    # Get voxel coordinates
                    z_start, z_end = z_idx*vz, (z_idx+1)*vz
                    y_start, y_end = y_idx*vy, (y_idx+1)*vy
                    x_start, x_end = x_idx*vx, (x_idx+1)*vx
                    
                    voxel_region = tomo_data[z_start:z_end, y_start:y_end, x_start:x_end]
                    voxel_mask = voxel_region > full_threshold
                    
                    z_coords, y_coords, x_coords = np.where(voxel_mask)
                    
                    # Adjust coordinates to global position
                    z_coords += z_start
                    y_coords += y_start
                    x_coords += x_start
                    
                    all_angles.extend([angle] * len(z_coords))
                    all_coords.extend(zip(x_coords, y_coords, z_coords))
                    all_rmsds.extend([rmsd] * len(z_coords))
                    
                except Exception as e:
                    print(f"Error processing voxel (z={z_idx},y={y_idx},x={x_idx}): {e}")
                    continue
    
    if all_coords:
        # Convert coordinates to arrays
        x_coords, y_coords, z_coords = zip(*all_coords)
        
        # Normalize angles to [0,1] for colorscale
        angle_range = max_angle - min_angle
        if angle_range > 0:
            normalized_angles = [(a - min_angle) / angle_range for a in all_angles]
        else:
            normalized_angles = [0.5] * len(all_angles)
        
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=3,
                color=normalized_angles,
                colorscale=create_hsv_colorscale(5),  # Creates 5 color stops plus the cyclic endpoint,
                opacity=0.05,
                colorbar=dict(
                    title='Orientation Angle (°)',
                    tickmode='array',
                    ticktext=[f'{min_angle:.0f}°', 
                             f'{(min_angle + angle_range/4):.0f}°',
                             f'{(min_angle + angle_range/2):.0f}°',
                             f'{(min_angle + 3*angle_range/4):.0f}°',
                             f'{max_angle:.0f}°'],
                    tickvals=[0, 0.25, 0.5, 0.75, 1.0]
                )
            ),
            name='Analyzed Voxels',
            showlegend=False
        ))
    
    # Update layout
    period_text = f" (Cyclic {cyclic_period}°)" if cyclic_period is not None else ""
    fig.update_layout(
        title=dict(
            text=f"Orientation Analysis{period_text} for Section:<br>z={z_range[0]}-{z_range[-1]}, y={y_range[0]}-{y_range[-1]}, x={x_range[0]}-{x_range[-1]}",
            y=0.95
        ),
        scene=dict(
            aspectmode='data',
            camera=dict(eye=dict(x=2, y=2, z=2)),
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        width=1000, height=800,
        showlegend=True
    )
    
    return fig

# Test case with visualization
def test_orientation_analysis(voxel_data, crystal_peaks, z_idx, y_idx, x_idx, h, k, l, threshold=0.1, sigma=0.5, cutoff=3, pixel_size=18, visualize=True):
    """
    Test and visualize all three steps:
    1. Original peaks
    2. After (110) alignment
    3. After rotation around (110)
    """
    # Get voxel FFT peaks and setup (same as before)
    magnitude, KX, KY, KZ = compute_fft_q(voxel_data, use_vignette=True, pixel_size=pixel_size)
    voxel_peaks, voxel_values = find_peaks_3d_cutoff(magnitude, 
                                                    threshold=threshold,
                                                    sigma=sigma, 
                                                    center_cutoff_radius=cutoff)

    voxel_peaks_q = np.array([
        KX[voxel_peaks[:, 0], voxel_peaks[:, 1], voxel_peaks[:, 2]],
        KY[voxel_peaks[:, 0], voxel_peaks[:, 1], voxel_peaks[:, 2]],
        KZ[voxel_peaks[:, 0], voxel_peaks[:, 1], voxel_peaks[:, 2]]
    ]).T

    
    # Get (110) direction and first rotation (same as before)
    crystal_110_idx = np.where((h == 1) & (k == 1) & (l == 0))[0][0]
    v2 = crystal_peaks[crystal_110_idx]
    v2_norm = v2 / np.linalg.norm(v2)
    
    strongest_idx = np.argmax(voxel_values)
    v1 = voxel_peaks_q[strongest_idx]
    v1_norm = v1 / np.linalg.norm(v1)
    
    # First rotation
    rot_axis1 = np.cross(v1_norm, v2_norm)
    if np.all(rot_axis1 == 0):
        R1 = Rotation.from_rotvec([0, 0, 0])
    else:
        rot_axis1 = rot_axis1 / np.linalg.norm(rot_axis1)
        cos_angle = np.dot(v1_norm, v2_norm)
        initial_angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        R1 = Rotation.from_rotvec(rot_axis1 * initial_angle)
    
    # Apply first rotation
    aligned_peaks = R1.apply(voxel_peaks_q)
    
    # Second rotation around (110)
    best_angle = 0
    best_rmsd = float('inf')
    final_rotated = None
    
    for angle in np.arange(0, 360, 1):
        R2 = Rotation.from_rotvec(v2_norm * np.deg2rad(angle))
        rotated = R2.apply(aligned_peaks)
        
        total_rmsd = 0
        for rot_peak in rotated:
            dists = np.linalg.norm(crystal_peaks - rot_peak, axis=1)
            total_rmsd += np.min(dists)**2
        rmsd = np.sqrt(total_rmsd / len(rotated))
        
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_angle = angle
            final_rotated = rotated
    
    if visualize:
        # Create three subplots
        fig = make_subplots(rows=1, cols=3, 
                           subplot_titles=('Original Peaks', 
                                        'After (110) Alignment',
                                        f'After {best_angle:.1f}° Rotation'),
                           specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]])
        
        # Plot settings
        scale = 0.15
        sizes = 5 + (voxel_values - voxel_values.min()) / (voxel_values.max() - voxel_values.min()) * 15
        
        # Plot 1: Original peaks (same as before)
        fig.add_trace(
            go.Scatter3d(x=crystal_peaks[:,0], y=crystal_peaks[:,1], z=crystal_peaks[:,2],
                         mode='markers', marker=dict(size=8, color='blue', opacity=0.1),
                         name='Crystal Structure'), row=1, col=1)
        
        fig.add_trace(
            go.Scatter3d(x=voxel_peaks_q[:,0], y=voxel_peaks_q[:,1], z=voxel_peaks_q[:,2],
                         mode='markers', marker=dict(size=sizes, color='red', opacity=0.6),
                         name='Original Voxel Peaks'), row=1, col=1)
        
        # Plot 2: After (110) alignment (same as before)
        fig.add_trace(
            go.Scatter3d(x=crystal_peaks[:,0], y=crystal_peaks[:,1], z=crystal_peaks[:,2],
                         mode='markers', marker=dict(size=8, color='blue', opacity=0.1),
                         showlegend=False), row=1, col=2)
        
        fig.add_trace(
            go.Scatter3d(x=aligned_peaks[:,0], y=aligned_peaks[:,1], z=aligned_peaks[:,2],
                         mode='markers', marker=dict(size=sizes, color='orange', opacity=0.6),
                         name='(110) Aligned Peaks'), row=1, col=2)
        
        # Plot 3: After rotation around (110)
        fig.add_trace(
            go.Scatter3d(x=crystal_peaks[:,0], y=crystal_peaks[:,1], z=crystal_peaks[:,2],
                         mode='markers', marker=dict(size=8, color='blue', opacity=0.1),
                         showlegend=False), row=1, col=3)
        
        fig.add_trace(
            go.Scatter3d(x=final_rotated[:,0], y=final_rotated[:,1], z=final_rotated[:,2],
                         mode='markers', marker=dict(size=sizes, color='green', opacity=0.6),
                         name='Final Aligned Peaks'), row=1, col=3)
        
        # Add (110) axis to all plots
        for col in [1, 2, 3]:
            fig.add_trace(
                go.Scatter3d(x=[0, v2_norm[0] * scale], y=[0, v2_norm[1] * scale], z=[0, v2_norm[2] * scale],
                            mode='lines', line=dict(color='yellow', width=5),
                            name='(110) Axis' if col==1 else None,
                            showlegend=col==1), row=1, col=col)
        
        # Highlight strongest peak in all plots
        fig.add_trace(
            go.Scatter3d(x=[voxel_peaks_q[strongest_idx,0]], y=[voxel_peaks_q[strongest_idx,1]], 
                         z=[voxel_peaks_q[strongest_idx,2]], mode='markers',
                         marker=dict(size=15, color='purple', symbol='diamond'),
                         name='Strongest Peak'), row=1, col=1)
        
        fig.add_trace(
            go.Scatter3d(x=[aligned_peaks[strongest_idx,0]], y=[aligned_peaks[strongest_idx,1]], 
                         z=[aligned_peaks[strongest_idx,2]], mode='markers',
                         marker=dict(size=15, color='purple', symbol='diamond'),
                         showlegend=False), row=1, col=2)
        
        fig.add_trace(
            go.Scatter3d(x=[final_rotated[strongest_idx,0]], y=[final_rotated[strongest_idx,1]], 
                         z=[final_rotated[strongest_idx,2]], mode='markers',
                         marker=dict(size=15, color='purple', symbol='diamond'),
                         showlegend=False), row=1, col=3)
        
        fig.update_layout(
            title=f"Peak Alignment Steps (Voxel {z_idx},{y_idx},{x_idx})<br>RMSD: {best_rmsd:.3e} nm⁻¹",
            scene1=dict(aspectmode='cube'),
            scene2=dict(aspectmode='cube'),
            scene3=dict(aspectmode='cube'),
            width=1800
        )
        
          
        return fig, best_angle, best_rmsd
    else:
        return None, best_angle, best_rmsd

def _hkil_to_hkl3(h, k, i, l):
    # Miller–Bravais to 3-index reciprocal basis coefficients.
    # In a hexagonal lattice, a* and b* span 120°, with i = -(h+k).
    # One simple mapping is just to drop i because it's redundant:
    # (h,k,i,l) with i=-(h+k) corresponds to (h, k, l) in 3-index form.
    # Keep this helper to accept 4-index input ergonomically.
    assert np.isclose(i, -(h+k)), "For hex, i must be -(h+k)"
    return h, k, l

def _axis_from_indices(cellinfo_data, mode, indices, lattice_type="3index"):
    """
    mode: 'plane' for (hkl)/(hkil) normal, 'direction' for [uvw]
    indices: tuple of length 3 (h,k,l or u,v,w) or 4 (h,k,i,l) for hex planes
    lattice_type: '3index' or 'hkil' (only used when mode='plane')
    """
    if mode == 'plane':
        # Always use reciprocal lattice vectors for a plane normal
        Bstar = np.asarray(cellinfo_data['recilatticevectors'])  # shape (3,3), Cartesian
        if len(indices) == 4:  # (h,k,i,l) Miller–Bravais
            h, k, i, l = indices
            h3, k3, l3 = _hkil_to_hkl3(h, k, i, l)
            hkl = np.array([h3, k3, l3], dtype=float)
        else:
            hkl = np.array(indices, dtype=float)
        v = hkl[0]*Bstar[0] + hkl[1]*Bstar[1] + hkl[2]*Bstar[2]
    elif mode == 'direction':
        # Use direct lattice vectors for a direction [uvw]
        A = np.asarray(cellinfo_data['latticevectors'])         # shape (3,3), Cartesian
        uvw = np.array(indices, dtype=float)
        v = uvw[0]*A[0] + uvw[1]*A[1] + uvw[2]*A[2]
    else:
        raise ValueError("mode must be 'plane' or 'direction'")
    return v

def project_and_plot_along_hkl_2(
    data,
    cellinfo_data,
    *indices,                        # (h,k,l) or (h,k,i,l) for hex planes; or (u,v,w) for directions
    title_prefix="",
    mode="plane",                    # 'plane' → (hkl)/(hkil) normal; 'direction' → [uvw]
    q_vectors=None,                  # for reciprocal-space labeling
    voxel_size=None,                 # (dx, dy, dz) in nm for real-space labeling; if None → pixels
    show_plot=True,
    return_vector=False
):
    """
    Rotate a 3D tomogram to align with a specified plane normal (hkl/hkil)
    or direction [uvw], project it, and plot.
    """

    # 1) Build the axis vector
    v_axis = _axis_from_indices(cellinfo_data, mode, indices)

    if np.linalg.norm(v_axis) == 0:
        raise ValueError(f"indices {indices} produce a zero vector for mode='{mode}'")

    # Center of rotation
    center = (np.array(data.shape) - 1) / 2.0

    # 2) Orthonormal basis with u_proj = z'
    u_proj = v_axis / np.linalg.norm(v_axis)
    helper_vec = np.array([0, 0, 1]) if abs(np.dot(u_proj, [0,0,1])) < 0.99 else np.array([0, 1, 0])
    new_x_dir = np.cross(helper_vec, u_proj); new_x_dir /= np.linalg.norm(new_x_dir)
    new_y_dir = np.cross(u_proj, new_x_dir)

    R = np.array([new_x_dir, new_y_dir, u_proj])   # rows = new basis in old coords
    transform_matrix = R.T
    offset = center - transform_matrix @ center

    # 3) Rotate
    rotated = ndi.affine_transform(data, transform_matrix, offset=offset, order=1, output_shape=data.shape)

    # 4) Project along z' (the chosen axis)
    projection = np.sum(rotated, axis=2)

    # 5) Plotting / axes
    fig = go.Figure()

    if (mode == "plane") and (q_vectors is not None):
        # Label reciprocal axes by projecting bounds of q into (x',y')
        if isinstance(q_vectors, (tuple, list)):
            QX, QY, QZ = q_vectors
            q_vecs = np.column_stack((QX.flatten(), QY.flatten(), QZ.flatten()))
        else:
            q_vecs = np.asarray(q_vectors)

        q_min, q_max = q_vecs.min(0), q_vecs.max(0)
        corners = np.array([
            [q_min[0], q_min[1], q_min[2]], [q_max[0], q_min[1], q_min[2]],
            [q_min[0], q_max[1], q_min[2]], [q_max[0], q_max[1], q_min[2]],
            [q_min[0], q_min[1], q_max[2]], [q_max[0], q_min[1], q_max[2]],
            [q_min[0], q_max[1], q_max[2]], [q_max[0], q_max[1], q_max[2]],
        ])
        x_coords = corners @ new_x_dir
        y_coords = corners @ new_y_dir

        fig.add_trace(go.Heatmap(
            z=projection.T,
            x=np.linspace(x_coords.min(), x_coords.max(), projection.shape[0]),
            y=np.linspace(y_coords.min(), y_coords.max(), projection.shape[1]),
            colorscale='viridis'
        ))
        xlabel, ylabel = "q′₁ (Å⁻¹)", "q′₂ (Å⁻¹)"

    else:
        # Real-space (or pixel) labeling; allow anisotropic voxels
        H, W = projection.shape[0], projection.shape[1]
        if voxel_size is not None:
            # voxel_size = (dx, dy, dz) in nm in the original (x,y,z) grid
            dx, dy, dz = voxel_size
            # After rotation, x′ and y′ are linear combos of old axes; for display,
            # a reasonable, simple labeling is to assume isotropic in-plane sampling
            # by projecting the original spacing onto new_x_dir/new_y_dir magnitudes.
            # A more exact approach would build a mapping grid; this keeps it simple.
            # Use average in-plane scale:
            scale_x = np.sqrt((new_x_dir[0]*dx)**2 + (new_x_dir[1]*dy)**2 + (new_x_dir[2]*dz)**2)
            scale_y = np.sqrt((new_y_dir[0]*dx)**2 + (new_y_dir[1]*dy)**2 + (new_y_dir[2]*dz)**2)
            x = np.arange(W) * scale_x
            y = np.arange(H) * scale_y
            fig.add_trace(go.Heatmap(z=projection.T, x=x, y=y, colorscale='gray'))
            xlabel, ylabel = "x′ (nm)", "y′ (nm)"
        else:
            fig.add_trace(go.Heatmap(z=projection.T, colorscale='gray'))
            xlabel, ylabel = "x′ (pixels)", "y′ (pixels)"

    fig.update_layout(
        title=f"{title_prefix} Projection along {mode}={indices}",
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        width=800, height=700,
        xaxis=dict(constrain="domain"),
        yaxis=dict(constrain="domain"),
    )
    if show_plot:
        fig.show()

    return (projection, rotated, v_axis, center) if return_vector else (projection, rotated)


def project_and_plot_along_hkl(data, cellinfo_data, h, k, l, title_prefix="", is_reciprocal=False, q_vectors=None,
                               pixel_size=None, show_plot=True, return_vector=False):
    """
    Rotate a 3D tomogram to align with a specified hkl vector, project it, and plot.

    Args:
        data (np.ndarray): 3D tomogram data (real or reciprocal space).
        cellinfo_data (dict): Dictionary with unit cell info, must contain 'recilatticevectors' and 'latticevectors'.
        h, k, l (int): Miller indices of the projection direction.
        title_prefix (str): Prefix for the plot title.
        is_reciprocal (bool): Flag if the data is in reciprocal space.
        q_vectors (np.ndarray): Q vectors for reciprocal space data. Shape (N, 3) or tuple of QX,QY,QZ.
        pixel_size (float): Real space pixel size for labeling axes.
        return_vector (bool): If True, also return the projection vector and center used.
    """
    # 1. Calculate projection vector for the given (h,k,l)
    if is_reciprocal:
        #print("Using reciprocal space vectors")
        vectors = cellinfo_data['recilatticevectors']
    else:
        #print("Using real space vectors")
        vectors = cellinfo_data['latticevectors'] 
    v_hkl = h * vectors[0] + k * vectors[1] + l * vectors[2]
    #print(f"v_hkl: {v_hkl}")
    if np.linalg.norm(v_hkl) == 0:
        print(f"Error: (h,k,l)=({h},{k},{l}) vector is zero.")
        return None, None

    # Use geometric center as the origin
    center = (np.array(data.shape) - 1) / 2.0
    #print(f"Geometric center: {center}")

    # 2. Create rotation matrix to align an axis (e.g., z) with v_hkl
    u_proj = v_hkl / np.linalg.norm(v_hkl) # This is the new z-axis direction

    # Find two orthogonal vectors to u_proj to form a new basis
    if np.abs(np.dot(u_proj, np.array([0, 0, 1]))) < 0.99:
        helper_vec = np.array([0, 0, 1])
    else:
        helper_vec = np.array([0, 1, 0])
        
    new_x_dir = np.cross(helper_vec, u_proj)
    new_x_dir /= np.linalg.norm(new_x_dir)
    
    new_y_dir = np.cross(u_proj, new_x_dir)
    
    # Rotation matrix to transform from original to new basis
    # Rows are the new basis vectors
    R = np.array([new_x_dir, new_y_dir, u_proj])

    # The matrix for affine_transform should be the inverse, mapping output coords to input coords
    # which is the transpose of R.
    transform_matrix = R.T

    # The rotation should be about the geometric center of the volume.
    offset = center - transform_matrix @ center
    
    # 3. Rotate the data
    rotated_data = ndi.affine_transform(data, transform_matrix, offset=offset, order=1, output_shape=data.shape)
    
    # 4. Project along the new z-axis (the hkl direction)
    projection = np.sum(rotated_data, axis=2) # The last axis corresponds to u_proj direction
    
    # 5. Plot the projection
    fig = go.Figure()

    # Determine axis labels and extent
    if is_reciprocal and q_vectors is not None:
        if isinstance(q_vectors, tuple) or isinstance(q_vectors, list):
             QX, QY, QZ = q_vectors
             q_vec_array = np.column_stack((QX.flatten(), QY.flatten(), QZ.flatten()))
        else:
             q_vec_array = q_vectors

        q_min = np.min(q_vec_array, axis=0)
        q_max = np.max(q_vec_array, axis=0)
        
        corners = np.array([
            [q_min[0], q_min[1], q_min[2]], [q_max[0], q_min[1], q_min[2]],
            [q_min[0], q_max[1], q_min[2]], [q_max[0], q_max[1], q_min[2]],
            [q_min[0], q_min[1], q_max[2]], [q_max[0], q_min[1], q_max[2]],
            [q_min[0], q_max[1], q_max[2]], [q_max[0], q_max[1], q_max[2]]
        ])
        
        x_coords = corners @ new_x_dir
        y_coords = corners @ new_y_dir
        
        xlabel = "Projection Axis 1 (Å⁻¹)"
        ylabel = "Projection Axis 2 (Å⁻¹)"
        x_range = [np.min(x_coords), np.max(x_coords)]
        y_range = [np.min(y_coords), np.max(y_coords)]
        
        fig.add_trace(go.Heatmap(z=projection.T, x=np.linspace(x_range[0], x_range[1], projection.shape[0]), y=np.linspace(y_range[0], y_range[1], projection.shape[1]), colorscale='viridis'))

    elif not is_reciprocal and pixel_size is not None:
        H, W = projection.shape[0], projection.shape[1]
        x = np.arange(W) * pixel_size
        y = np.arange(H) * pixel_size
        xlabel = "Projection Axis 1 (nm)"
        ylabel = "Projection Axis 2 (nm)"
        fig.add_trace(go.Heatmap(z=projection.T, x=x, y=y, colorscale='gray'))
        
        # Add scale bar (positioned in bottom-left corner with padding)
        # Note: y-axis is reversed, so y[-1] (max value) appears at visual bottom
        scale_length_nm = 2000  # Length of scale bar in nm
        # Calculate padding to avoid overlap with axis labels
        # Use a percentage of the y-range to ensure proper spacing regardless of crop size
        y_range_nm = y[-1] - y[0]
        padding_y_nm = max(y_range_nm * 0.08, scale_length_nm * 0.5)  # 8% of y-range or 50% of scale bar, whichever is larger
        padding_x_nm = (x[-1] - x[0]) * 0.05  # 5% of x-range for horizontal padding
        scale_x_start = x[0] + padding_x_nm  # Start from left edge + padding
        # Position at visual bottom: use y[-1] (max value) minus padding
        # With reversed y-axis, smaller y values appear higher, so subtract padding from max
        scale_y_pos = y[-1] - padding_y_nm
        
        # Ensure scale bar doesn't extend beyond image bounds
        if scale_x_start + scale_length_nm > x[-1]:
            scale_length_nm = x[-1] - scale_x_start - padding_x_nm * 0.5
        
        # Add scale bar line
        fig.add_shape(
            type="line",
            x0=scale_x_start,
            x1=scale_x_start + scale_length_nm,
            y0=scale_y_pos,
            y1=scale_y_pos,
            line=dict(color="white", width=5)
        )
        
        # Add scale bar label (positioned above the scale bar)
        # Since y-axis is reversed, "above" means smaller y values (subtract offset)
        label_offset_nm = scale_length_nm * 0.2  # Fixed offset relative to scale bar
        fig.add_annotation(
            x=scale_x_start + scale_length_nm/2,
            y=scale_y_pos - label_offset_nm,  # Subtract since y-axis is reversed
            text=f"{int(scale_length_nm/1000)} μm",
            showarrow=False,
            font=dict(color="white", size=20)
        )
    else:
        xlabel = "Projection Axis 1 (pixels)"
        ylabel = "Projection Axis 2 (pixels)"
        fig.add_trace(go.Heatmap(z=projection.T, colorscale='gray'))

        # 6. Overlay unit cell axes on the projection
    # Get unit cell axes (a, b, c)
    if is_reciprocal:
        unit_cell_axes = cellinfo_data['recilatticevectors']
    else:
        unit_cell_axes = cellinfo_data['latticevectors']
    
    # Rotate the unit cell axes using the same rotation matrix
    # R transforms from original basis to new basis (new_x_dir, new_y_dir, u_proj)
    # Debug: Check original axes
    #print(f"\nOriginal unit cell axes:")
    for i, (name, axis) in enumerate(zip(['a', 'b', 'c'], unit_cell_axes)):
        axis_norm = np.linalg.norm(axis)
        #print(f"  {name} axis: {axis}, length: {axis_norm:.6f}")
    
    # Check if axes are orthogonal (for cubic system)
    if len(unit_cell_axes) == 3:
        a, b, c = unit_cell_axes
        ab_angle = np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)) * 180 / np.pi
        bc_angle = np.arccos(np.clip(np.dot(b, c) / (np.linalg.norm(b) * np.linalg.norm(c)), -1, 1)) * 180 / np.pi
        ca_angle = np.arccos(np.clip(np.dot(c, a) / (np.linalg.norm(c) * np.linalg.norm(a)), -1, 1)) * 180 / np.pi
        #print(f"  Angles between original axes: a-b={ab_angle:.2f}°, b-c={bc_angle:.2f}°, c-a={ca_angle:.2f}°")
        #print(f"  (For cubic: all should be 90° and all lengths equal)")
    
    rotated_axes = [R @ axis for axis in unit_cell_axes]
    
    # Debug: Check rotated axes
    #print(f"\nRotated unit cell axes (in new coordinate system):")
    for i, (name, axis) in enumerate(zip(['a', 'b', 'c'], rotated_axes)):
        axis_norm = np.linalg.norm(axis)
        #print(f"  {name} axis: {axis}, length: {axis_norm:.6f}")
    
    # Project axes onto the 2D plane by removing component along u_proj
    # The projection plane is spanned by new_x_dir and new_y_dir
    # In the rotated coordinate system, u_proj is the z-axis
    # R has rows [new_x_dir, new_y_dir, u_proj], so R @ axis gives:
    # [dot(axis, new_x_dir), dot(axis, new_y_dir), dot(axis, u_proj)]
    # The first two components are the projection coordinates!
    projected_axes_2d = []
    for axis in rotated_axes:
        # In the rotated coordinate system, the z-component is along u_proj
        # So we just take the x and y components (first two elements)
        x_coord = axis[0]  # component along new_x_dir
        y_coord = axis[1]  # component along new_y_dir
        projected_axes_2d.append((x_coord, y_coord))
    
    # Determine the center of the projection in 2D coordinates
    # Note: projection.T is used in heatmap, so:
    # - x-axis corresponds to columns of projection.T = rows of projection (shape[0])
    # - y-axis corresponds to rows of projection.T = columns of projection (shape[1])
    if is_reciprocal and q_vectors is not None:
        # Use center of q_range
        center_2d_x = (x_range[0] + x_range[1]) / 2
        center_2d_y = (y_range[0] + y_range[1]) / 2
    elif not is_reciprocal and pixel_size is not None:
        center_2d_x = (x[0] + x[-1]) / 2  -50
        center_2d_y = (y[0] + y[-1]) / 2  -50
    else:
        # Use center of projection array
        # For pixel coordinates: x = 0 to projection.shape[0]-1, y = 0 to projection.shape[1]-1
        center_2d_x = projection.shape[0] / 2
        center_2d_y = projection.shape[1] / 2
    
    # Scale factor for arrow length (make arrows visible but not too long)
    # Use a fraction of the projection size
    if is_reciprocal and q_vectors is not None:
        max_range = max(np.abs(x_range[1] - x_range[0]), np.abs(y_range[1] - y_range[0]))
        scale_factor = max_range * 0.15
    elif not is_reciprocal and pixel_size is not None:
        max_range = max(np.abs(x[-1] - x[0]), np.abs(y[-1] - y[0]))
        scale_factor = max_range * 0.15
    else:
        # Scale based on projection dimensions
        max_dim = max(projection.shape[0], projection.shape[1])
        scale_factor = max_dim * 0.15
    
    # Scale axes proportionally to their projected lengths (not normalized to same length)
    # This preserves the relative lengths of the axes in the projection
    axis_colors = ['red', 'green', 'blue']  # a=red, b=green, c=blue
    axis_labels = ['a', 'b', 'c']
    
    # Find the maximum projected length to scale all axes relative to it
    max_proj_length = max([np.sqrt(x**2 + y**2) for x, y in projected_axes_2d]) if projected_axes_2d else 1.0
    
    # Debug: Print projected lengths and angles for verification
    if len(projected_axes_2d) == 3:
        proj_lengths = [np.sqrt(x**2 + y**2) for x, y in projected_axes_2d]
        #print(f"\nUnit cell axes projection along ({h},{k},{l}):")
        #for i, (name, length) in enumerate(zip(['a', 'b', 'c'], proj_lengths)):
        #    print(f"  {name} axis projected length: {length:.6f}")
        # Calculate angles between projected axes
        if all(pl > 1e-10 for pl in proj_lengths):
            vecs = [np.array([x, y]) / np.sqrt(x**2 + y**2) for (x, y), pl in zip(projected_axes_2d, proj_lengths) if pl > 1e-10]
            if len(vecs) >= 2:
                angles = []
                for i in range(len(vecs)):
                    for j in range(i+1, len(vecs)):
                        angle = np.arccos(np.clip(np.dot(vecs[i], vecs[j]), -1, 1)) * 180 / np.pi
                        angles.append(angle)
                #if angles:
                    #print(f"  Angles between projected axes: {[f'{a:.2f}°' for a in angles]}")
        #print()
    
    for i, (x_coord, y_coord) in enumerate(projected_axes_2d):
        proj_length = np.sqrt(x_coord**2 + y_coord**2)
        if proj_length > 1e-10:  # Avoid division by zero for axes parallel to projection direction
            # Scale proportionally to the maximum projected length
            # This ensures all arrows are visible but maintains relative lengths
            if max_proj_length > 1e-10:
                relative_scale = proj_length / max_proj_length
            else:
                relative_scale = 1.0
            
            x_scaled = x_coord / proj_length * scale_factor * relative_scale
            y_scaled = y_coord / proj_length * scale_factor * relative_scale
            
            # Add arrow annotation
            fig.add_annotation(
                x=center_2d_x + x_scaled,
                y=center_2d_y + y_scaled,
                ax=center_2d_x,
                ay=center_2d_y,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2.5,
                arrowcolor=axis_colors[i],
                showarrow=True,
                name=f"{axis_labels[i]} axis"
            )
    
    fig.update_layout(
        title=f"{title_prefix} Projection along ({h},{k},{l})",
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        width=800, height=700,
        xaxis=dict(constrain="domain"),
        yaxis=dict(constrain="domain", autorange="reversed")  # Flip y-axis vertically
    )
    if show_plot:
        fig.show()
    
    if return_vector:
        return projection, rotated_data, v_hkl, center
    return projection, rotated_data

def plot_hkl_vector_in_tomogram(data, cellinfo_data, h, k, l, is_reciprocal=False, scale=0.5, show_plot=True, plot_tomogram=True, intensity_threshold=0.8):
    """
    Plot a given hkl vector as an arrow in the tomogram, with the origin at the tomogram's geometric center.
    Optionally overlay the tomogram as a 3D scatter of high-intensity points.
    The camera view is set to look along the hkl vector.
    Args:
        data (np.ndarray): 3D tomogram data (real or reciprocal space).
        cellinfo_data (dict): Dictionary with unit cell info.
        h, k, l (int): Miller indices.
        is_reciprocal (bool): Use reciprocal lattice vectors if True, else real space.
        scale (float): Length scaling factor for the arrow.
        show_plot (bool): Whether to show the plot.
        plot_tomogram (bool): Whether to plot the tomogram points in the same figure.
        intensity_threshold (float): Relative threshold for tomogram points (0-1).
    Returns:
        fig: Plotly figure object.
    """
    center = (np.array(data.shape) - 1) / 2.0
    if is_reciprocal:
        vectors = cellinfo_data['recilatticevectors']
    else:
        vectors = cellinfo_data['latticevectors']
    v_hkl = h * vectors[0] + k * vectors[1] + l * vectors[2]
    v_hkl_norm = v_hkl / np.linalg.norm(v_hkl)
    v_hkl_scaled = v_hkl_norm * 2.5# * np.min(data.shape)  # Camera distance factor
    # Arrow from geometric center
    start = center
    end = center + v_hkl_norm * scale * np.min(data.shape)
    # Plot
    import plotly.graph_objects as go
    fig = go.Figure()
    if plot_tomogram:
        # Plot tomogram as gray points above threshold
        max_intensity = data.max()
        threshold = max_intensity * intensity_threshold
        mask = data > threshold
        z, y, x = np.where(mask)
        intensities = data[mask]
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=2, color='gray', opacity=0.2),
            name='Tomogram'
        ))
    fig.add_trace(go.Scatter3d(
        x=[start[2], end[2]],
        y=[start[1], end[1]],
        z=[start[0], end[0]],
        mode='lines+markers',
        line=dict(color='red', width=6),
        marker=dict(size=6, color='blue'),
        name=f'hkl=({h},{k},{l})'
    ))
    fig.add_trace(go.Scatter3d(
        x=[start[2]], y=[start[1]], z=[start[0]],
        mode='markers', marker=dict(size=8, color='green'), name='Geometric Center'
    ))
    # Set camera to look along the hkl vector
    camera_eye = dict(x=float(v_hkl_scaled[0]), y=float(v_hkl_scaled[1]), z=float(v_hkl_scaled[2]))
    fig.update_layout(
        title=f'hkl=({h},{k},{l}) vector in tomogram',
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='cube',
            camera=dict(eye=camera_eye)
        ),
        width=800, height=800
    )
    if show_plot:
        fig.show()
    return fig

def plot_unit_cell_in_tomogram(data, cellinfo_data, plot_tomogram=True, intensity_threshold=0.8, show_plot=True, pixel_size=56):
    """
    Plot the unit cell box (from latticevectors) overlaid on the tomogram, using the geometric center as the origin.
    Args:
        data (np.ndarray): 3D tomogram data.
        cellinfo_data (dict): Must contain 'latticevectors' in nm units.
        plot_tomogram (bool): Whether to plot the tomogram points.
        intensity_threshold (float): Threshold for tomogram points.
        show_plot (bool): Whether to show the plot.
        pixel_size (float): Pixel size in nm.
    Returns:
        fig: Plotly figure object.
    """
    import plotly.graph_objects as go
    center = (np.array(data.shape) - 1) / 2.0
    # Convert lattice vectors from nm to pixels
    a, b, c = [np.array(v) / pixel_size for v in cellinfo_data['latticevectors']]
    # Define 8 corners of the unit cell, centered at the geometric center
    corners = [
        center + 0*a + 0*b + 0*c,
        center + 1*a + 0*b + 0*c,
        center + 1*a + 1*b + 0*c,
        center + 0*a + 1*b + 0*c,
        center + 0*a + 0*b + 1*c,
        center + 1*a + 0*b + 1*c,
        center + 1*a + 1*b + 1*c,
        center + 0*a + 1*b + 1*c,
    ]
    corners = [np.array(pt) for pt in corners]
    # Define edges as pairs of corner indices
    edges = [
        (0,1),(1,2),(2,3),(3,0), # bottom face
        (4,5),(5,6),(6,7),(7,4), # top face
        (0,4),(1,5),(2,6),(3,7)  # vertical edges
    ]
    fig = go.Figure()
    if plot_tomogram:
        max_intensity = data.max()
        threshold = max_intensity * intensity_threshold
        mask = data > threshold
        z, y, x = np.where(mask)
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=2, color='gray', opacity=0.2),
            name='Tomogram'))
    # Plot unit cell edges
    for i,j in edges:
        fig.add_trace(go.Scatter3d(
            x=[corners[i][2], corners[j][2]],
            y=[corners[i][1], corners[j][1]],
            z=[corners[i][0], corners[j][0]],
            mode='lines',
            line=dict(color='orange', width=4),
            name='Unit Cell' if i==0 and j==1 else None,
            showlegend=(i==0 and j==1)
        ))
    fig.add_trace(go.Scatter3d(
        x=[center[2]], y=[center[1]], z=[center[0]],
        mode='markers', marker=dict(size=8, color='green'), name='Geometric Center'))
    fig.update_layout(
        title='Unit Cell in Tomogram',
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='cube'
        ),
        width=800, height=800
    )
    if show_plot:
        fig.show()
    return fig

def plot_hkl_projection_grid(tomo_data, magnitude_test, cellinfo_data, pixel_size=56):
    """
    Plot a grid of real and reciprocal space projections for all hkl in [-1,0,1]^3 (excluding (0,0,0)).
    tomo_data: 3D real space data
    magnitude_test: 3D reciprocal space mask/data
    cellinfo_data: dict with 'recilatticevectors'
    pixel_size: for real space axis labeling
    """
    import matplotlib.pyplot as plt
    from itertools import product
    
    hkls = [hkl for hkl in product([-1,0,1], repeat=3) if hkl != (0,0,0)]
    n = len(hkls)
    ncols = 9
    nrows = 2 * ((n + ncols - 1) // ncols)  # 2 rows per hkl set (real/recip)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5*ncols, 2.5*nrows), squeeze=False)
    
    for idx, (h,k,l) in tqdm(enumerate(hkls),total=len(hkls),desc="Plotting hkl projections..."):
        row_real = 2 * (idx // ncols)
        col = idx % ncols
        row_recip = row_real + 1
        # Real space
        proj_real, _ = project_and_plot_along_hkl(
            tomo_data, cellinfo_data, h, k, l, title_prefix="", is_reciprocal=False, q_vectors=None, \
                pixel_size=pixel_size,show_plot=False)
        # Reciprocal space
        proj_recip, _ = project_and_plot_along_hkl(
            magnitude_test, cellinfo_data, h, k, l, title_prefix="", is_reciprocal=True, q_vectors=None, \
                pixel_size=pixel_size,show_plot=False)
        # Plot real
        ax_real = axes[row_real, col]
        im1 = ax_real.imshow(proj_real.T, cmap='gray', aspect='auto')
        ax_real.set_title(f"hkl=[{h},{k},{l}]\nReal")
        ax_real.axis('off')
        # Plot reciprocal
        ax_recip = axes[row_recip, col]
        im2 = ax_recip.imshow(proj_recip.T, cmap='jet', aspect='auto')
        ax_recip.set_title(f"hkl=[{h},{k},{l}]\nRecip")
        ax_recip.axis('off')
    # Remove unused axes
    for i in range(idx+1, (nrows//2)*ncols):
        axes[2*(i//ncols), i%ncols].axis('off')
        axes[2*(i//ncols)+1, i%ncols].axis('off')
    plt.tight_layout()
    plt.show()
    return fig

def plot_combined_reciprocal_space(tomo_data, tomo_data_RS, tomo_data_RS_qs, cellinfo_data, hs, ks, ls, threshold_D=0.5, threshold_tomo_FFT=0.002, q_cutoff=0.07, peak_distance_threshold=0.01):
    """
    Create a combined plot showing reciprocal space data, FFT, and unit cell peaks.
    
    Args:
        tomo_data: Real space tomogram data
        tomo_data_RS: Reciprocal space data
        tomo_data_RS_qs: Q-vectors for reciprocal space data
        cellinfo_data: Unit cell information
        hs, ks, ls: Miller indices for unit cell peaks
        threshold_D: Threshold for reciprocal space data magnitude filtering
        threshold_tomo_FFT: Threshold for FFT magnitude filtering
        q_cutoff: Radius of sphere to mask out (in Å⁻¹) - only applied to reciprocal space data
        peak_distance_threshold: Maximum distance to consider a unit cell peak as "close" to high-intensity points
    """
    # Create figure
    fig = go.Figure()
    
    # 1. Plot reciprocal space data
    magnitude = tomo_data_RS
    KX, KY, KZ = tomo_data_RS_qs[:,0], tomo_data_RS_qs[:,1], tomo_data_RS_qs[:,2]
    
    # Flatten and filter
    kx_flat = KX.flatten()
    ky_flat = KY.flatten()
    kz_flat = KZ.flatten()
    magnitude_flat = magnitude.flatten()
    
    # Calculate q magnitude for each point
    q_magnitude = np.sqrt(kx_flat**2 + ky_flat**2 + kz_flat**2)
    
    # Apply both magnitude threshold and q_cutoff mask only to reciprocal space data
    mask = (q_magnitude > q_cutoff)
    kx_filtered = kx_flat[mask]
    ky_filtered = ky_flat[mask]
    kz_filtered = kz_flat[mask]
    magnitude_filtered = magnitude_flat[mask]
    
    # Add reciprocal space data
    fig.add_trace(go.Scatter3d(
        x=kx_filtered,
        y=ky_filtered,
        z=kz_filtered,
        mode='markers',
        marker=dict(
            size=5,
            color=magnitude_filtered,
            colorscale='Jet',
            opacity=0.4,
            colorbar=dict(title='RS Magnitude')
        ),
        name='Reciprocal Space Data'
    ))
    
    # 2. Compute and plot FFT
    magnitude_fft, KX_fft, KY_fft, KZ_fft = compute_fft_q(tomo_data, use_vignette=True, pixel_size=56)
    
    # Flatten and filter FFT data (only magnitude threshold, no q_cutoff)
    kx_fft_flat = KX_fft.flatten()
    ky_fft_flat = KY_fft.flatten()
    kz_fft_flat = KZ_fft.flatten()
    magnitude_fft_flat = magnitude_fft.flatten()
    
    # Apply only magnitude threshold to FFT data
    mask_fft = (magnitude_fft_flat > threshold_tomo_FFT * np.max(magnitude_fft))
    kx_fft_filtered = kx_fft_flat[mask_fft]
    ky_fft_filtered = ky_fft_flat[mask_fft]
    kz_fft_filtered = kz_fft_flat[mask_fft]
    magnitude_fft_filtered = magnitude_fft_flat[mask_fft]
    
    # Add FFT data
    fig.add_trace(go.Scatter3d(
        x=kx_fft_filtered,
        y=ky_fft_filtered,
        z=kz_fft_filtered,
        mode='markers',
        marker=dict(
            size=5,
            color=magnitude_fft_filtered,
            colorscale='Plasma',
            opacity=0.4,
            colorbar=dict(title='FFT Magnitude')
        ),
        name='FFT Data'
    ))
    
    # 3. Plot unit cell peaks (only those near high-intensity points)
    vs = []
    hkl_list = []  # Store hkl indices for each peak
    for i, h in enumerate(hs):
        v = hs[i]*cellinfo_data['recilatticevectors'][0] + \
            ks[i]*cellinfo_data['recilatticevectors'][1] + \
            ls[i]*cellinfo_data['recilatticevectors'][2]
        vs.append(v)
        hkl_list.append(f"({h},{ks[i]},{ls[i]})")
    
    vs = np.array(vs)
    
    # Process reciprocal space data clusters
    rs_points = np.column_stack((kx_filtered, ky_filtered, kz_filtered))
    if len(rs_points) > 0:
        # Normalize coordinates for clustering
        rs_coords_normalized = rs_points / np.max(np.abs(rs_points))
        
        # Perform clustering for RS data
        rs_clustering = DBSCAN(eps=0.1, min_samples=2).fit(rs_coords_normalized)
        rs_labels = rs_clustering.labels_
        
        # Find centers of RS high-intensity regions
        rs_region_centers = []
        for label in set(rs_labels):
            if label == -1:  # Skip noise points
                continue
            
            # Get points in this cluster
            mask = rs_labels == label
            cluster_points = rs_points[mask]
            cluster_magnitudes = magnitude_filtered[mask]
            
            # Calculate weighted center of mass
            weights = cluster_magnitudes / np.sum(cluster_magnitudes)
            center = np.sum(cluster_points * weights[:, np.newaxis], axis=0)
            rs_region_centers.append(center)
        
        rs_region_centers = np.array(rs_region_centers)
        
        # Add RS region centers for visualization
        fig.add_trace(go.Scatter3d(
            x=rs_region_centers.T[0],
            y=rs_region_centers.T[1],
            z=rs_region_centers.T[2],
            mode='markers',
            marker=dict(
                size=8,
                color='green',
                opacity=0.6,
                symbol='circle'
            ),
            name='RS Region Centers'
        ))
    
    # Process FFT data clusters
    fft_points = np.column_stack((kx_fft_filtered, ky_fft_filtered, kz_fft_filtered))
    if len(fft_points) > 0:
        # Normalize coordinates for clustering
        fft_coords_normalized = fft_points / np.max(np.abs(fft_points))
        
        # Perform clustering for FFT data
        fft_clustering = DBSCAN(eps=0.02, min_samples=3).fit(fft_coords_normalized)
        fft_labels = fft_clustering.labels_
        
        # Find centers of FFT high-intensity regions
        fft_region_centers = []
        for label in set(fft_labels):
            if label == -1:  # Skip noise points
                continue
            
            # Get points in this cluster
            mask = fft_labels == label
            cluster_points = fft_points[mask]
            cluster_magnitudes = magnitude_fft_filtered[mask]
            
            # Calculate weighted center of mass
            weights = cluster_magnitudes / np.sum(cluster_magnitudes)
            center = np.sum(cluster_points * weights[:, np.newaxis], axis=0)
            fft_region_centers.append(center)
        
        fft_region_centers = np.array(fft_region_centers)
        
        # Add FFT region centers for visualization
        fig.add_trace(go.Scatter3d(
            x=fft_region_centers.T[0],
            y=fft_region_centers.T[1],
            z=fft_region_centers.T[2],
            mode='markers',
            marker=dict(
                size=4,
                color='blue',
                opacity=0.5,
                symbol='circle'
            ),
            name='FFT Region Centers'
        ))
    
    # Combine all region centers
    all_region_centers = []
    if len(rs_points) > 0:
        all_region_centers.extend(rs_region_centers)
    if len(fft_points) > 0:
        all_region_centers.extend(fft_region_centers)
    all_region_centers = np.array(all_region_centers)
    
    # Find unit cell peaks that are close to region centers
    close_peaks = []
    close_peaks_hkl = []
    used_regions = set()
    
    for i, peak in enumerate(vs):
        # Calculate distances to all region centers
        distances = np.sqrt(np.sum((all_region_centers - peak)**2, axis=1))
        min_dist_idx = np.argmin(distances)
        min_dist = distances[min_dist_idx]
        
        # If the peak is close enough and this region hasn't been matched yet
        if min_dist < peak_distance_threshold and min_dist_idx not in used_regions:
            close_peaks.append(peak)
            close_peaks_hkl.append(hkl_list[i])
            used_regions.add(min_dist_idx)
    
    close_peaks = np.array(close_peaks)
    
    if len(close_peaks) > 0:
        # Add filtered unit cell peaks with labels
        fig.add_trace(go.Scatter3d(
            x=close_peaks.T[0],
            y=close_peaks.T[1],
            z=close_peaks.T[2],
            mode='markers+text',
            marker=dict(
                size=5,
                color='red',
                opacity=0.3,
                symbol='diamond'
            ),
            text=close_peaks_hkl,
            textfont=dict(size=6),
            textposition="top center",
            name='Unit Cell Peaks'
        ))
    
    # Update layout
    fig.update_layout(
        title="Combined Reciprocal Space Visualization",
        scene=dict(
            xaxis_title="Qx (Å⁻¹)",
            yaxis_title="Qy (Å⁻¹)",
            zaxis_title="Qz (Å⁻¹)",
            aspectmode='cube'
        ),
        width=1000,
        height=1000,
        showlegend=True
    )
    
    return fig



def square_unitcell_hexagons_template(size=32, hex_radius=10, spacing=2, center_dip=True, angle_deg=0):
    """
    Create a template with 4 hexagons arranged in a square unit cell.
    - size: output image size (pixels)
    - hex_radius: radius of each hexagon (pixels)
    - spacing: gap between hexagons (pixels)
    - center_dip: if True, add a central dip to each hexagon
    - angle_deg: rotation of each hexagon
    """
    img = np.zeros((size, size))
    # Calculate centers for 2x2 grid
    offset = hex_radius + spacing//2
    centers = [
        (offset, offset),
        (offset, size - offset),
        (size - offset, offset),
        (size - offset, size - offset)
    ]
    for cy, cx in centers:
        hex_img = polygon_template(size=size, center_dip=center_dip, angle_deg=angle_deg, shape=6)
        # Mask to only keep the hexagon at the right position
        mask = np.zeros((size, size))
        y, x = np.ogrid[:size, :size]
        mask[((y-cy)**2 + (x-cx)**2) < hex_radius**2] = 1
        img += hex_img * mask
    img = np.clip(img, 0, 1)
    return img
def align_patches_to_reference(patches, upsample_factor=100):
    reference = np.mean(patches, axis=0)
    aligned_patches = []
    for patch in patches:
        shift_est, _, _ = phase_cross_correlation(reference, patch, upsample_factor=upsample_factor)
        patch_aligned = shift(patch, shift_est)
        aligned_patches.append(patch_aligned)
    return aligned_patches

def align_patch_by_rotation(patch, template, angles=np.arange(0, 360, 6)):
    """Find best rotation of patch to match template."""
    best_score = np.inf
    best_angle = 0
    best_rotated = patch.copy()

    for angle in angles:
        rotated = rotate(patch, angle, reshape=False, mode='reflect')
        if rotated.shape != template.shape:
            continue
        score = normalized_root_mse(template, rotated)
        if score < best_score:
            best_score = score
            best_angle = angle
            best_rotated = rotated
    return best_rotated, best_angle
def extract_and_align_rotated_patches(image, coords, patch_size, template, rotation_scan=True):
    half = patch_size // 2
    aligned_patches = []
    angles = []
    for y, x in coords:
        if y - half < 0 or y + half >= image.shape[0] or x - half < 0 or x + half >= image.shape[1]:
            continue
        if patch_size % 2 == 0:
            patch = image[y - half:y + half, x - half:x + half]
        else:
            patch = image[y - half:y + half + 1, x - half:x + half + 1]
        if patch.shape != template.shape:
            continue
        if rotation_scan:
            patch, angle = align_patch_by_rotation(patch, template)
            angles.append(angle)
        else:
            angles.append(0)
        shift_est, _, _ = phase_cross_correlation(template, patch, upsample_factor=10)
        patch_aligned = shift(patch, shift_est)
        aligned_patches.append(patch_aligned)
    return aligned_patches, angles
def polygon_template(size=32, center_dip=False, angle_deg=0, shape=6, background_value=0.1, edge_blur_sigma=1.0, add_noise=False, noise_std=0.01, random_seed=None):
    center = (size - 1) / 2
    t = np.linspace(0, 2 * np.pi, shape + 1)
    angle_rad = np.deg2rad(angle_deg)
    x = center * np.cos(t + angle_rad) + center
    y = center * np.sin(t + angle_rad) + center
    rr, cc = polygon(y, x)
    rr = np.clip(rr.astype(int), 0, size - 1)
    cc = np.clip(cc.astype(int), 0, size - 1)
    img = np.zeros((size, size))
    hex_mask = np.zeros((size, size), dtype=bool)
    hex_mask[rr, cc] = True
    img[hex_mask] = 1.0
    if center_dip:
        yy, xx = np.meshgrid(np.arange(size), np.arange(size))
        r = np.sqrt((yy - center)**2 + (xx - center)**2)
        img[hex_mask] *= 1 - np.exp(-(r[hex_mask] / (0.3*size))**2)
    # Optionally blur the edges (affects all, but we will restore background after)
    if edge_blur_sigma > 0:
        from scipy.ndimage import gaussian_filter
        img = gaussian_filter(img, sigma=edge_blur_sigma)
    # Set background value only outside the hexagon
    img[~hex_mask] = background_value
    # Optionally add noise
    if add_noise:
        if random_seed is not None:
            np.random.seed(random_seed)
        noise = np.random.randn(size, size) * noise_std
        img[~hex_mask] += noise[~hex_mask]
    return img

def extract_aligned_patches(image, coords, patch_size, template):
    half = patch_size // 2
    patches = []
    for y, x in coords:
        if y - half < 0 or y + half >= image.shape[0] or x - half < 0 or x + half >= image.shape[1]:
            continue
        if patch_size % 2 == 0:
            patch = image[y - half:y + half, x - half:x + half]
        else:
            patch = image[y - half:y + half + 1, x - half:x + half + 1]
        if patch.shape != template.shape:
            continue
        shift_est, _, _ = phase_cross_correlation(template, patch, upsample_factor=10)
        patch_aligned = shift(patch, shift_est)
        patches.append(patch_aligned)
    return patches

def single_particle_analysis_rotation_scan(
    image, patch_size=32, center_dip=True,
    angle_range=np.arange(0, 60, 6), threshold_abs=0.6,
    min_distance=None, score_thresh=0.5, show_plot=True,
    cluster_k=None, show_gallery=True, shape=6,
    cluster_by_angle_only=False,
    random_seed=None,
    reference_template=None  # <-- new argument
):
    if min_distance is None:
        min_distance = patch_size // 2

    best_coords = []
    best_scores = []
    best_angle = None
    best_template = None
    if reference_template is not None:
        # Use the provided reference template for all angles
        best_template = reference_template
        best_scores = []
        best_coords = []
        best_angle = 0
        
        # Try different rotations of the template
        for angle in angle_range:
            rotated_template = rotate_patch(reference_template, angle)
            result = match_template(image, rotated_template, pad_input=True)
            coords = peak_local_max(result, min_distance=min_distance, threshold_abs=threshold_abs)
            scores = result[coords[:, 0], coords[:, 1]]
            
            if len(scores) > len(best_scores):
                best_coords = coords
                best_scores = scores 
                best_angle = angle
                best_template = rotated_template
    else:
        for angle in angle_range:
            template = polygon_template(patch_size, center_dip=center_dip, angle_deg=angle, shape=shape,\
                add_noise=True, noise_std=0.05, random_seed=random_seed)
            result = match_template(image, template, pad_input=True)
            coords = peak_local_max(result, min_distance=min_distance, threshold_abs=threshold_abs)
            scores = result[coords[:, 0], coords[:, 1]]
            if len(scores) > len(best_scores):
                best_coords = coords
                best_scores = scores
                best_angle = angle
                best_template = template

    print(f"Best angle = {best_angle}°, {len(best_coords)} particles detected")

    # Filter by correlation score
    filtered_coords = best_coords[best_scores > score_thresh]
    filtered_scores = best_scores[best_scores > score_thresh]
    print(f"{len(filtered_coords)} particles kept after score filtering (>{score_thresh})")

    # Align patches
    patches, angles = extract_and_align_rotated_patches(image, filtered_coords, patch_size, best_template, rotation_scan=True)
    if not patches:
        print("No patches extracted.")
        return None, filtered_coords, []

    # Align all patches to the average BEFORE clustering
    aligned_patches = align_patches_to_reference(patches, upsample_factor=100)
    avg_particle = np.mean(aligned_patches, axis=0)

    if show_plot:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(best_template, cmap='gray')
        axs[0].set_title(f"Best Template @ {best_angle}°")
        axs[1].imshow(image, cmap='gray')
        axs[1].plot(filtered_coords[:, 1], filtered_coords[:, 0], 'r.', markersize=3)
        axs[1].set_title("Detected Particles")
        axs[2].imshow(avg_particle, cmap='gray')
        axs[2].set_title("Averaged Particle")
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    if show_gallery:
        plt.figure(figsize=(6, 6))
        patch_stack = np.array(aligned_patches[:36])
        gallery = montage(patch_stack, fill=0, grid_shape=(6, 6))
        plt.imshow(gallery, cmap='gray')
        plt.title("Top 36 Patches")
        plt.axis('off')
        plt.show()

    if cluster_k is not None and cluster_k >= 1:
        angles_arr = np.array(angles).reshape(-1, 1) / 360.0  # Normalize angle to [0,1]
        if cluster_by_angle_only:
            features = angles_arr
        else:
            reshaped = np.array([p.flatten() for p in aligned_patches])
            features = np.hstack([reshaped, angles_arr])  # Combine appearance and angle

        pca = PCA(n_components=min(10, features.shape[1])) if not cluster_by_angle_only else None
        reduced = pca.fit_transform(features) if pca is not None else features
        labels = KMeans(n_clusters=cluster_k, random_state=0).fit_predict(reduced)
        print(f"Patches clustered into {cluster_k} classes.")

        mean_cluster_list=[]
        if cluster_k>1:
            for k in range(cluster_k):
                cluster = [aligned_patches[i] for i in range(len(aligned_patches)) if labels[i] == k]
                if cluster:
                    mean_cluster = np.mean(cluster, axis=0)
                    plt.figure()
                    plt.imshow(mean_cluster, cmap='gray')
                    plt.title(f"Class {k}")
                    plt.axis('off')
                    plt.show()
                print("total particles in cluster: ", len(cluster))
                mean_cluster_list.append(mean_cluster)
        else:
            print("only one cluster, using average of all patches")
            mean_cluster_list.append(np.mean(aligned_patches, axis=0))

        # Overlay detected particles colored by cluster
        import matplotlib.cm as cm
        colors = cm.get_cmap('tab10', cluster_k)
        plt.figure(figsize=(6, 6))
        plt.imshow(image, cmap='gray')
        for k in range(cluster_k):
            idxs = np.where(labels == k)[0]
            if len(idxs) > 0:
                plt.scatter(filtered_coords[idxs, 1], filtered_coords[idxs, 0],
                            color=colors(k), label=f'Cluster {k}', s=20, alpha=0.8)
        plt.title('Detected Particles Colored by Cluster')
        plt.axis('off')
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6, 6))
        for k in range(cluster_k):
            idxs = np.where(labels == k)[0]
            cluster_angles = np.array(angles)[idxs]
            print(f"Cluster {k}: mean angle = {np.mean(cluster_angles):.2f}°, median = {np.median(cluster_angles):.2f}°")
            plt.hist(cluster_angles, bins=12, alpha=0.5, label=f'Cluster {k}')
        plt.xlabel('Best matching angle (deg)')
        plt.ylabel('Count')
        plt.legend()
        plt.title('Distribution of best matching angles per cluster')
        plt.show()

    return avg_particle, filtered_coords, aligned_patches, mean_cluster_list

def crop_patch(image, center, patch_size):
    """
    Crop a square patch from the image centered at 'center' (y, x) with size 'patch_size'.
    Pads with zeros if the patch is at the edge.
    """
    y, x = center
    half = patch_size // 2
    y1, y2 = max(0, y - half), min(image.shape[0], y + half + 1)
    x1, x2 = max(0, x - half), min(image.shape[1], x + half + 1)
    patch = image[y1:y2, x1:x2]
    # Pad if at the edge
    pad_y = patch_size - patch.shape[0]
    pad_x = patch_size - patch.shape[1]
    if pad_y > 0 or pad_x > 0:
        patch = np.pad(patch, ((0, pad_y), (0, pad_x)), mode='constant')
    # Plot the patch location on the original image
    plt.figure(figsize=(12,6))
    
    # Original image with patch location
    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    
    # Draw red square at patch location
    half = patch_size // 2
    rect = plt.Rectangle((x-half, y-half), patch_size, patch_size, 
                        fill=False, color='red', linewidth=2)
    plt.gca().add_patch(rect)
    plt.title('Selected Patch Location')
    plt.axis('off')
    
    # Plot extracted patch
    plt.subplot(122)
    plt.imshow(patch, cmap='gray')
    plt.title('Extracted Patch')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    return patch


def select_patch_interactive(image, patch_size=33):
    """
    Allows the user to select the center of a patch interactively from an image.
    Returns the cropped patch and the center coordinates.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,8))
    plt.imshow(image, cmap='gray')
    plt.title(f"Click to select the center of the reference patch (size={patch_size})")
    coords = plt.ginput(1)
    plt.close()
    if not coords:
        print("No point selected.")
        return None, None
    x, y = int(coords[0][0]), int(coords[0][1])
    patch = crop_patch(image, (y, x), patch_size)
    return patch, (y, x)


def rotate_patch(patch, angle):
    """
    Rotate a patch by a given angle (degrees), keeping the same shape.
    """
    from scipy.ndimage import rotate
    return rotate(patch, angle, reshape=False, mode='reflect')



def max_rotational_correlation(img1, img2, angles=np.arange(0, 360, 1)):
    best_corr = -1
    best_angle = 0
    for angle in angles:
        rotated = rotate(img2, angle, reshape=False)
        corr = np.corrcoef(img1.flatten(), rotated.flatten())[0, 1]
        if corr > best_corr:
            best_corr = corr
            best_angle = angle
    return best_corr, best_angle

def max_positional_correlation(img1, img2, max_shift=10):
    """Find best x,y shift to align img2 with img1"""
    best_corr = -1
    best_shift = (0,0)
    
    for dy in range(-max_shift, max_shift+1):
        for dx in range(-max_shift, max_shift+1):
            shifted = shift(img2, (dy,dx))
            corr = np.corrcoef(img1.flatten(), shifted.flatten())[0,1]
            if corr > best_corr:
                best_corr = corr
                best_shift = (dy,dx)
                
    return best_corr, best_shift


def pentagon_template_fit(shape, center, scale, angle_deg, amplitude=1.0, background=0.0):
    # shape: (height, width) of output image
    # center: (y, x) center of pentagon
    # scale: size of pentagon (relative to image) 
    # angle_deg: rotation angle in degrees
    # amplitude: scaling factor
    # background: offset
    size = int(scale)
    template = polygon_template(size=size, center_dip=True, angle_deg=angle_deg, shape=5)
    # Place the template in a blank image at the specified center
    img = np.ones(shape) * background
    y0, x0 = int(center[0]), int(center[1])
    half = size // 2
    y1, y2 = max(0, y0-half), min(shape[0], y0+half)
    x1, x2 = max(0, x0-half), min(shape[1], x0+half)
    t_y1, t_y2 = half-(y0-y1), half+(y2-y0)
    t_x1, t_x2 = half-(x0-x1), half+(x2-x0)
    # Add the scaled template
    img[y1:y2, x1:x2] += amplitude * template[t_y1:t_y2, t_x1:t_x2]
    return img



def hexagon_template_fit(shape, center, scale, angle_deg, amplitude=1.0, background=0.0):
    # shape: (height, width) of output image
    # center: (y, x) center of pentagon
    # scale: size of pentagon (relative to image)
    # angle_deg: rotation angle in degrees
    # amplitude: scaling factor
    # background: offset
    size = int(scale)
    template = polygon_template(size=size, center_dip=True, angle_deg=angle_deg, shape=6)
    # Place the template in a blank image at the specified center
    img = np.ones(shape) * background
    y0, x0 = int(center[0]), int(center[1])
    half = size // 2
    y1, y2 = max(0, y0-half), min(shape[0], y0+half)
    x1, x2 = max(0, x0-half), min(shape[1], x0+half)
    t_y1, t_y2 = half-(y0-y1), half+(y2-y0)
    t_x1, t_x2 = half-(x0-x1), half+(x2-x0)
    # Add the scaled template
    img[y1:y2, x1:x2] += amplitude * template[t_y1:t_y2, t_x1:t_x2]
    return img

def fit_objective(params, data):
    center_y, center_x, scale, angle, amplitude, background = params
    template_img = pentagon_template_fit(
        data.shape, (center_y, center_x), scale, angle, amplitude, background
    )
    # Use sum of squared differences
    return np.sum((data - template_img)**2)


def switch_backend(backend='widget'):
    """
    Switch matplotlib backend between 'widget' and 'inline'
    Args:
        backend (str): Either 'widget' or 'inline'
    """
    from IPython import get_ipython
    ipython = get_ipython()
    
    if backend.lower() == 'widget':
        ipython.magic('matplotlib widget')
    elif backend.lower() == 'inline':
        ipython.magic('matplotlib inline')
    else:
        raise ValueError("Backend must be either 'widget' or 'inline'")
    
# Interactive line selector (requires matplotlib widget backend)
# Set the backend to widget
# %matplotlib widget
# Function to switch matplotlib backends
class LineSelector:
    def __init__(self, image, n_lines=6):
        self.image = image
        self.n_lines = n_lines
        self.lines = []
        self.current_points = []
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.out = Output()
        self.done_button = Button(description="Done")
        self.clear_button = Button(description="Clear Last Line")
        self.n_lines_box = IntText(value=n_lines, description='Lines:')
        self.done_button.on_click(self.finish)
        self.clear_button.on_click(self.clear_last)
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.ax.imshow(self.image, cmap='gray')
        self.ax.set_title(f"Click start and end for each line ({self.n_lines} total)")
        switch_backend('widget') # Set the backend to widget for interactive plotting
        display(VBox([HBox([self.done_button, self.clear_button, self.n_lines_box]), self.out]))
        plt.show()
        self.finished = False

    def onclick(self, event):
        if self.finished:
            return
        if event.inaxes != self.ax:
            return
        self.current_points.append((int(event.ydata), int(event.xdata)))  # (row, col)
        if len(self.current_points) == 2:
            self.lines.append(tuple(self.current_points))
            y0, x0 = self.current_points[0]
            y1, x1 = self.current_points[1]
            self.ax.plot([x0, x1], [y0, y1], 'r-', linewidth=2)
            self.ax.plot([x0, x1], [y0, y1], 'go')
            self.fig.canvas.draw()
            self.current_points = []
            with self.out:
                print(f"Line {len(self.lines)}: {self.lines[-1]}")
            if len(self.lines) >= self.n_lines_box.value:
                self.finish(None)

    def finish(self, b):
        self.finished = True
        with self.out:
            print("Selection finished.")
            print("Lines:", self.lines)

    def clear_last(self, b):
        if self.lines:
            self.lines.pop()
            self.ax.clear()
            self.ax.imshow(self.image, cmap='gray')
            for (y0, x0), (y1, x1) in self.lines:
                self.ax.plot([x0, x1], [y0, y1], 'r-', linewidth=2)
                self.ax.plot([x0, x1], [y0, y1], 'go')
            self.fig.canvas.draw()
            with self.out:
                print("Last line removed.")


def load_and_analyze_tomogram(tomogram_path, cellinfo_path, pixel_size=20, intensity_threshold=0.8, 
                            rotation_axis='z', rotation_angle=0, fft_threshold=0.0005, 
                            center_cutoff=0.015, voxel_size=(25,25,25), output_dir=None, 
                            save_outputs=True, crop_tomogram=True, crop_size=None, hkl_projection=[1,0,1]):
    """
    Load and analyze tomogram data with FFT analysis and cell info visualization.
    
    Parameters:
    -----------
    tomogram_path : str
        Path to the tomogram TIFF file
    cellinfo_path : str 
        Path to the cellinfo.mat file
    pixel_size : float
        Size of pixels in nm
    intensity_threshold : float
        Threshold for 3D tomogram visualization
    rotation_axis : str
        Axis for rotation ('x', 'y', or 'z')
    rotation_angle : float
        Angle to rotate tomogram in degrees
    fft_threshold : float
        Threshold factor for FFT magnitude
    center_cutoff : float
        Radius for center cutoff in FFT
    voxel_size : tuple
        Size of voxels for analysis (z,y,x)
    output_dir : str, optional
        Base directory for saving outputs. If None, creates one based on tomogram filename
    save_outputs : bool
        Whether to save outputs to disk (default: True)
    
    Returns:
    --------
    dict containing:
        tomo_data : ndarray
            The loaded and processed tomogram data
        magnitude : ndarray
            FFT magnitude data
        voxel_results : dict
            Results of voxel analysis
        cellinfo_data : dict
            Loaded cell information
        figures : dict
            Generated matplotlib and plotly figures:
            - tomogram: 3D tomogram visualization
            - fft: 3D FFT visualization
            - hkl_vector: HKL vector plot
            - unit_cell: Unit cell visualization
            - projection_combined: Combined projection analysis
            - hkl_grid: HKL projection grid
        projections : dict
            Projection data in real and reciprocal space
        output_paths : dict (if save_outputs=True)
            Paths to all saved files including plots, HDF5 file, and summary
        output_dir : str (if save_outputs=True)
            Base directory containing all outputs
    
    Output Directory Structure:
    ---------------------------
    analysis_<tomogram_name>_<timestamp>/
    ├── plots/
    │   ├── 3d_visualizations/
    │   │   ├── tomogram_3d.html (interactive)
    │   │   ├── fft_3d.html (interactive)
    │   │   └── hkl_vector_in_tomogram.html (interactive)
    │   ├── projections/
    │   │   ├── projection_real_space.png
    │   │   ├── projection_real_space_fft.png
    │   │   ├── projection_reciprocal_space.png
    │   │   ├── projection_combined_analysis.png
    │   │   └── hkl_projection_grid.png
    │   └── fft_analysis/
    ├── data/
    │   └── analysis_complete.h5
    ├── unit_cell_info/
    │   └── unit_cell_in_tomogram.html (interactive)
    └── analysis_summary.txt
    """
    
    # Setup output directory structure
    if save_outputs:
        if output_dir is None:
            # Create output directory based on tomogram filename and timestamp
            tomo_name = Path(tomogram_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(tomogram_path).parent / f"analysis_{tomo_name}_{timestamp}"
        else:
            output_dir = Path(output_dir)
        
        # Create subdirectories
        dirs = {
            'base': output_dir,
            'plots': output_dir / 'plots',
            'plots_3d': output_dir / 'plots' / '3d_visualizations',
            'plots_projections': output_dir / 'plots' / 'projections',
            'plots_fft': output_dir / 'plots' / 'fft_analysis',
            'data': output_dir / 'data',
            'unit_cell': output_dir / 'unit_cell_info',
        }
        
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Output directory created: {output_dir}")
    else:
        dirs = None
    # Load tomogram
    tomo_data = tifffile.imread(tomogram_path)
    
    print(f"Tomogram shape: {tomo_data.shape}")
    print(f"Pixel size: {pixel_size} nm")
    
    # Crop tomogram from the center
    if crop_tomogram:
        shape = tomo_data.shape
        center = np.array(shape) // 2
        start = center - crop_size // 2
        end = start + crop_size
        
        tomo_data = tomo_data[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        print(f"Tomogram shape after crop: {tomo_data.shape}")
    else:
        print("Tomogram not cropped")
    # Rotate if needed
    if rotation_axis == 'x':
        tomo_data = rotate(tomo_data, rotation_angle, axes=(1, 2), reshape=False)
    elif rotation_axis == 'y':
        tomo_data = rotate(tomo_data, rotation_angle, axes=(0, 2), reshape=False)
    elif rotation_axis == 'z':
        tomo_data = rotate(tomo_data, rotation_angle, axes=(0, 1), reshape=False)
    
    # Create 3D tomogram plot
    fig_tomo = plot_3D_tomogram(tomo_data, intensity_threshold=intensity_threshold)
    
    # Compute FFT
    magnitude, KX, KY, KZ = compute_fft_q(tomo_data, use_vignette=True, pixel_size=pixel_size, scale=1)
    
    # Process FFT data
    threshold = fft_threshold * np.max(magnitude)
    kx_flat = KX.flatten()
    ky_flat = KY.flatten()
    kz_flat = KZ.flatten()
    magnitude_flat = magnitude.flatten()
    
    radial_distance = np.sqrt(kx_flat**2 + ky_flat**2 + kz_flat**2)
    mask = (magnitude_flat > threshold) & (radial_distance > center_cutoff)
    
    kx_filtered = kx_flat[mask]
    ky_filtered = ky_flat[mask]
    kz_filtered = kz_flat[mask]
    magnitude_filtered = magnitude_flat[mask]
    
    # Create FFT plot
    fig_fft = go.Figure(data=go.Scatter3d(
        x=kx_filtered, y=ky_filtered, z=kz_filtered,
        mode='markers',
        marker=dict(size=4, color=magnitude_filtered, colorscale='Viridis', opacity=0.8)
    ))
    
    # Add cutoff sphere visualization
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x_sphere = center_cutoff * np.outer(np.cos(u), np.sin(v))
    y_sphere = center_cutoff * np.outer(np.sin(u), np.sin(v))
    z_sphere = center_cutoff * np.outer(np.ones(np.size(u)), np.cos(v))
    
    #fig_fft.add_trace(go.Surface(x=x_sphere, y=y_sphere, z=z_sphere, opacity=0.2))
    fig_fft.update_layout(scene=dict(aspectmode='cube'), width=800, height=800)
    
    # Load cell info and analyze
    cellinfo_data = load_cellinfo_data(cellinfo_path)
    
    # Calculate unit cells info
    limiting_axes = np.min(tomo_data.shape)
    tomo_nm_size = pixel_size * limiting_axes
    n_unit_cells = tomo_nm_size // (cellinfo_data['Vol'][0][0]**(1/3))
    
    print(f'~n unit cells per tomogram: {n_unit_cells}')
    
    # Perform voxel analysis
    voxel_results = analyze_tomogram_voxels(tomo_data, voxel_size=voxel_size)
    print(f"Number of voxels (z, y, x): {voxel_results['n_voxels']}")
    print(f'~m unit cells per voxel: {n_unit_cells*voxel_size[0]/limiting_axes}')
    
    # Generate reciprocal lattice points
    hkl = np.array(list(product([-2, -1, 0, 1, 2], repeat=3)))
    hs, ks, ls = hkl[:, 0], hkl[:, 1], hkl[:, 2]
    
    print("hs:", hs)
    print("ks:", ks)
    print("ls:", ls)
    
    # Calculate reciprocal lattice vectors
    vs = []
    for i, h in enumerate(hs):
        v = (hs[i]*cellinfo_data['recilatticevectors'][0] + 
             ks[i]*cellinfo_data['recilatticevectors'][1] + 
             ls[i]*cellinfo_data['recilatticevectors'][2])
        vs.append(v)
    vs = np.array(vs)
    
    # Add reciprocal lattice points to FFT plot
    fig_fft.add_trace(go.Scatter3d(
        x=vs.T[0], y=vs.T[1], z=vs.T[2],
        mode='markers',
        marker=dict(size=5, color='red', opacity=0.1),
        name='Cell Info'
    ))

    fig_fft.update_layout(
        title="3D FFT Magnitude with Threshold",
        scene=dict(
            xaxis_title="KX",
            yaxis_title="KY", 
            zaxis_title="KZ",
            aspectmode='cube'
        ),
        width=800, height=800
    )

    # Create magnitude threshold mask
    threshold_factor = 1e-3
    magnitude_test = magnitude > threshold_factor*np.max(magnitude)
    

    center = np.array(magnitude.shape) // 2
    R = 48
    x, y, z = np.ogrid[:magnitude.shape[0], :magnitude.shape[1], :magnitude.shape[2]]
    sphere_mask = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 >= R**2
    magnitude_test = magnitude_test & sphere_mask
    
    # Set projection direction
    h, k, l = hkl_projection[0], hkl_projection[1], hkl_projection[2]
    
    # Transpose data to match MATLAB orientation
    tomogram_test = tomo_data.T
    magnitude_test = magnitude_test.T
    
    project_and_plot_along_hkl_2(
        tomogram_test-np.mean(tomogram_test),
        cellinfo_data,
        h, k, l,                        # (h,k,l) or (h,k,i,l) for hex planes; or (u,v,w) for directions
        title_prefix="Test",
        mode="plane",                    # 'plane' → (hkl)/(hkil) normal; 'direction' → [uvw]
        q_vectors=None,                  # for reciprocal-space labeling
        voxel_size=None,                 # (dx, dy, dz) in nm for real-space labeling; if None → pixels
        show_plot=True,
        return_vector=False
    )
        
    # Generate projections
    projection_test, rotated_tomo_test = project_and_plot_along_hkl(
        tomogram_test-np.mean(tomogram_test), cellinfo_data, h, k, l,
        title_prefix="Test", is_reciprocal=False, q_vectors=None, pixel_size=pixel_size
    )
    
    projection_test_reciprocal, rotated_tomo_test_reciprocal = project_and_plot_along_hkl(
        magnitude_test-np.mean(magnitude_test), cellinfo_data, h, k, l,
        title_prefix="Test", is_reciprocal=True, q_vectors=None, pixel_size=pixel_size
    )
    
    # Plot HKL vector and unit cell
    fig_hkl_vector = plot_hkl_vector_in_tomogram(tomogram_test, cellinfo_data, h, k, l,
        is_reciprocal=False, scale=0.1, plot_tomogram=True)
    
    fig_unit_cell = plot_unit_cell_in_tomogram(tomogram_test, cellinfo_data, plot_tomogram=True,
        intensity_threshold=intensity_threshold, pixel_size=pixel_size)
    
    # Plot real space projection and its FFT
    fig_proj_combined, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(projection_test.T, cmap='gray')
    ax[0].set_title('Real Space Projection')
    ax[0].axis('off')
    
    # Add scale bar (positioned in bottom-left corner with padding)
    scalebar_length_nm = 2000  # Length in nm
    scalebar_length_pixels = scalebar_length_nm / pixel_size
    scalebar_width_pixels = 10  # Fixed width in pixels for better visibility
    scalebar_color = 'white'
    
    # Position scale bar in bottom-left corner with padding
    padding = 30  # Padding from edges in pixels
    scalebar_x_start = padding
    scalebar_y_pos = projection_test.T.shape[0] - padding - scalebar_width_pixels
    
    # Ensure scale bar doesn't go outside image bounds
    if scalebar_x_start + scalebar_length_pixels > projection_test.T.shape[1]:
        scalebar_length_pixels = projection_test.T.shape[1] - scalebar_x_start - 10
        scalebar_length_nm = scalebar_length_pixels * pixel_size
    
    rect = patches.Rectangle(
        (scalebar_x_start, scalebar_y_pos),
        scalebar_length_pixels,
        scalebar_width_pixels,
        facecolor=scalebar_color,
        edgecolor='black',
        linewidth=1
    )
    ax[0].add_patch(rect)
    
    # Add scale bar label (positioned above the scale bar)
    label_y_pos = scalebar_y_pos - 15  # Position label above scale bar
    ax[0].text(
        scalebar_x_start + scalebar_length_pixels/2,
        label_y_pos,
        f'{int(scalebar_length_nm/1000)} μm',
        color=scalebar_color,
        ha='center',
        va='top',
        fontsize=20,
        weight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5, edgecolor='white', linewidth=1)
    )
    ax[1].imshow((np.abs(np.fft.fftshift(np.fft.fft2(projection_test.T)))**2),
        cmap='jet', norm=colors.LogNorm())
    ax[1].set_title('FFT of Real Space Projection')
    ax[1].axis('off')
    fig_proj_combined.suptitle(f'Projection Analysis [h={h}, k={k}, l={l}]', fontsize=14)
    plt.show()
    
    # Plot HKL projection grid
    fig_grid = plot_hkl_projection_grid(tomo_data.T, magnitude_test.T,
        cellinfo_data, pixel_size=pixel_size)
    
    # Save all outputs if requested
    output_paths = {}
    if save_outputs:
        print("\nSaving outputs...")
        
        # Save Plotly figures (HTML format)
        print("Saving 3D visualizations...")
        fig_tomo_path = dirs['plots_3d'] / 'tomogram_3d.html'
        fig_tomo.write_html(str(fig_tomo_path))
        output_paths['tomogram_3d_html'] = str(fig_tomo_path)
        
        fig_fft_path = dirs['plots_fft'] / 'fft_3d.html'
        fig_fft.write_html(str(fig_fft_path))
        output_paths['fft_3d_html'] = str(fig_fft_path)
        
        # Save projection images
        print("Saving projections...")
        proj_real_path = dirs['plots_projections'] / 'projection_real_space.png'
        plt.figure(figsize=(10, 10))
        plt.imshow(projection_test, cmap='gray')
        plt.title(f'Real Space Projection (h={h}, k={k}, l={l})')
        plt.colorbar()
        plt.savefig(proj_real_path, dpi=300, bbox_inches='tight')
        plt.close()
        output_paths['projection_real'] = str(proj_real_path)
        
        proj_real_fft_path = dirs['plots_projections'] / 'projection_real_space_fft.png'
        plt.figure(figsize=(10, 10))
        plt.imshow(np.abs(np.fft.fftshift(np.fft.fft2(projection_test)))**2,
                   cmap='jet', norm=colors.LogNorm())
        plt.title(f'FFT of Real Space Projection (h={h}, k={k}, l={l})')
        plt.colorbar()
        plt.savefig(proj_real_fft_path, dpi=300, bbox_inches='tight')
        plt.close()
        output_paths['projection_real_fft'] = str(proj_real_fft_path)
        
        proj_reciprocal_path = dirs['plots_projections'] / 'projection_reciprocal_space.png'
        plt.figure(figsize=(10, 10))
        plt.imshow(projection_test_reciprocal, cmap='gray')
        plt.title(f'Reciprocal Space Projection (h={h}, k={k}, l={l})')
        plt.colorbar()
        plt.savefig(proj_reciprocal_path, dpi=300, bbox_inches='tight')
        plt.close()
        output_paths['projection_reciprocal'] = str(proj_reciprocal_path)
        
        # Save combined projection plot
        proj_combined_path = dirs['plots_projections'] / 'projection_combined_analysis.png'
        fig_proj_combined.savefig(proj_combined_path, dpi=300, bbox_inches='tight')
        output_paths['projection_combined'] = str(proj_combined_path)
        
        # Save HKL projection grid (matplotlib figure)
        grid_path = dirs['plots_projections'] / 'hkl_projection_grid.png'
        fig_grid.savefig(grid_path, dpi=300, bbox_inches='tight')
        output_paths['hkl_projection_grid'] = str(grid_path)
        
        # Save HKL vector plot (Plotly figure - save as HTML)
        hkl_vector_path = dirs['plots_3d'] / 'hkl_vector_in_tomogram.html'
        fig_hkl_vector.write_html(str(hkl_vector_path))
        output_paths['hkl_vector_plot'] = str(hkl_vector_path)
        
        # Save unit cell plot (Plotly figure - save as HTML)
        unit_cell_path = dirs['unit_cell'] / 'unit_cell_in_tomogram.html'
        fig_unit_cell.write_html(str(unit_cell_path))
        output_paths['unit_cell_plot'] = str(unit_cell_path)
        
        print(f"Saved {len([k for k in output_paths.keys() if 'plot' in k or 'grid' in k or 'html' in k])} plot files")
        
        # Save HDF5 file with all data
        print("Saving HDF5 file with all data...")
        h5_path = dirs['data'] / 'analysis_complete.h5'
        with h5py.File(h5_path, 'w') as h5f:
            # Create main groups
            grp_tomo = h5f.create_group('tomogram')
            grp_fft = h5f.create_group('fft')
            grp_projections = h5f.create_group('projections')
            grp_voxels = h5f.create_group('voxel_analysis')
            grp_cell = h5f.create_group('cell_info')
            grp_params = h5f.create_group('parameters')
            
            # Save tomogram data
            grp_tomo.create_dataset('data', data=tomo_data, compression='gzip')
            grp_tomo.create_dataset('rotated_real_space', data=rotated_tomo_test, compression='gzip')
            grp_tomo.create_dataset('rotated_reciprocal_space', data=rotated_tomo_test_reciprocal, compression='gzip')
            grp_tomo.attrs['shape'] = tomo_data.shape
            grp_tomo.attrs['pixel_size_nm'] = pixel_size
            
            # Save FFT data
            grp_fft.create_dataset('magnitude', data=magnitude, compression='gzip')
            grp_fft.create_dataset('KX', data=KX, compression='gzip')
            grp_fft.create_dataset('KY', data=KY, compression='gzip')
            grp_fft.create_dataset('KZ', data=KZ, compression='gzip')
            grp_fft.create_dataset('magnitude_mask', data=magnitude_test, compression='gzip')
            grp_fft.attrs['threshold_factor'] = fft_threshold
            grp_fft.attrs['center_cutoff'] = center_cutoff
            
            # Save projection data
            grp_projections.create_dataset('real_space', data=projection_test, compression='gzip')
            grp_projections.create_dataset('reciprocal_space', data=projection_test_reciprocal, compression='gzip')
            grp_projections.attrs['h'] = h
            grp_projections.attrs['k'] = k
            grp_projections.attrs['l'] = l
            
            # Save voxel analysis results
            for key, value in voxel_results.items():
                if isinstance(value, np.ndarray):
                    grp_voxels.create_dataset(key, data=value, compression='gzip')
                elif isinstance(value, (tuple, list)):
                    grp_voxels.attrs[key] = value
                else:
                    grp_voxels.attrs[key] = value
            
            # Save cell info data
            for key, value in cellinfo_data.items():
                if isinstance(value, np.ndarray):
                    grp_cell.create_dataset(key, data=value, compression='gzip')
                else:
                    try:
                        grp_cell.attrs[key] = value
                    except:
                        # Some MATLAB types might not convert directly
                        grp_cell.attrs[key] = str(value)
            
            # Save parameters
            grp_params.attrs['tomogram_path'] = tomogram_path
            grp_params.attrs['cellinfo_path'] = cellinfo_path
            grp_params.attrs['pixel_size'] = pixel_size
            grp_params.attrs['intensity_threshold'] = intensity_threshold
            grp_params.attrs['rotation_axis'] = rotation_axis
            grp_params.attrs['rotation_angle'] = rotation_angle
            grp_params.attrs['fft_threshold'] = fft_threshold
            grp_params.attrs['center_cutoff'] = center_cutoff
            grp_params.attrs['voxel_size'] = voxel_size
            grp_params.attrs['analysis_timestamp'] = datetime.now().isoformat()
            
        output_paths['h5_file'] = str(h5_path)
        
        # Save analysis summary as text file
        print("Saving analysis summary...")
        summary_path = dirs['base'] / 'analysis_summary.txt'
        limiting_axes = np.min(tomo_data.shape)
        tomo_nm_size = pixel_size * limiting_axes
        n_unit_cells = tomo_nm_size // (cellinfo_data['Vol'][0][0]**(1/3))
        m_unit_cells_per_voxel = n_unit_cells * voxel_size[0] / limiting_axes
        
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TOMOGRAM ANALYSIS SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("INPUT FILES:\n")
            f.write(f"  Tomogram: {tomogram_path}\n")
            f.write(f"  Cell Info: {cellinfo_path}\n\n")
            
            f.write("TOMOGRAM PROPERTIES:\n")
            f.write(f"  Shape: {tomo_data.shape}\n")
            f.write(f"  Pixel Size: {pixel_size} nm\n")
            f.write(f"  Rotation: {rotation_angle}° around {rotation_axis}-axis\n")
            f.write(f"  Intensity Threshold: {intensity_threshold}\n\n")
            
            f.write("FFT ANALYSIS:\n")
            f.write(f"  FFT Threshold Factor: {fft_threshold}\n")
            f.write(f"  Center Cutoff Radius: {center_cutoff}\n")
            f.write(f"  Max Magnitude: {np.max(magnitude):.6e}\n\n")
            
            f.write("VOXEL ANALYSIS:\n")
            f.write(f"  Voxel Size: {voxel_size}\n")
            f.write(f"  Number of Voxels (z, y, x): {voxel_results['n_voxels']}\n\n")
            
            f.write("UNIT CELL INFORMATION:\n")
            f.write(f"  ~n unit cells per tomogram: {n_unit_cells:.2f}\n")
            f.write(f"  ~m unit cells per voxel: {m_unit_cells_per_voxel:.2f}\n\n")
            
            f.write("PROJECTION:\n")
            f.write(f"  HKL indices: ({h}, {k}, {l})\n\n")
            
            f.write("OUTPUT FILES:\n")
            for key, path in output_paths.items():
                f.write(f"  {key}: {path}\n")
            f.write("\n" + "=" * 80 + "\n")
        
        output_paths['summary'] = str(summary_path)
        
        print(f"\nAll outputs saved to: {output_dir}")
        print(f"HDF5 file: {h5_path}")
        print(f"Summary file: {summary_path}")
    
    result = {
        'tomo_data': tomo_data,
        'magnitude': magnitude,
        'voxel_results': voxel_results,
        'cellinfo_data': cellinfo_data,
        'figures': {
            'tomogram': fig_tomo,
            'fft': fig_fft,
            'hkl_vector': fig_hkl_vector,
            'unit_cell': fig_unit_cell,
            'projection_combined': fig_proj_combined,
            'hkl_grid': fig_grid
        },
        'projections': {
            'real_space': projection_test,
            'reciprocal_space': projection_test_reciprocal,
            'rotated_tomo_real': rotated_tomo_test,
            'rotated_tomo_reciprocal': rotated_tomo_test_reciprocal
        }
    }
    
    if save_outputs:
        result['output_paths'] = output_paths
        result['output_dir'] = str(output_dir)
    
    return result



