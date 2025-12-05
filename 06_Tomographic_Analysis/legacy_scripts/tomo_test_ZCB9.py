#%%
import numpy as np
import scipy.io as sio
import plotly.graph_objects as go
from itertools import product
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from matplotlib import colors

# %%

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

from scipy.ndimage import gaussian_filter


basedir='/scratch/2025_Feb/'

# tomo_data_RS_file=sio.loadmat(f'{basedir}/FFT_RS_128.mat')
# tomo_data_RS=tomo_data_RS_file['DATA_m']
# tomo_data_RS_qs=tomo_data_RS_file['Qv_m']
# tomo_data_RS[np.isnan(tomo_data_RS)]=0
# tomo_data_RS_qs[np.isnan(tomo_data_RS_qs)]=0
# tomo_data_RS = gaussian_filter(tomo_data_RS, sigma=0.)#.75) 

tomo_data_RS_file_256=sio.loadmat(f'{basedir}/FFT_RS_256_NEW.mat')
tomo_data_RS_256=tomo_data_RS_file_256['DATA256']
tomo_data_RS_qs_256=tomo_data_RS_file_256['Qv256']
tomo_data_RS_256[np.isnan(tomo_data_RS_256)]=0
tomo_data_RS_qs_256[np.isnan(tomo_data_RS_qs_256)]=1
tomo_data_RS_256 = gaussian_filter(tomo_data_RS_256, sigma=0.) 

#tomo_data_RS_file_DECONV=sio.loadmat(f'{basedir}/DECONV_RS_256_NEW.mat')


#tomo_data_RS_file_DECONV=sio.loadmat(f'{basedir}/DECONV_RS_256_PROCESSED.mat')
#tomo_data_RS_file_DECONV=sio.loadmat(f'{basedir}/DECONV_RS_256_PROCESSED_250_PEARSON_LOSS.mat')
#tomo_data_RS_file_DECONV=sio.loadmat(f'{basedir}/DECONV_RS_256_PROCESSED_50_L1.mat')
#tomo_data_RS_file_DECONV=sio.loadmat(f'{basedir}/DECONV_RS_256_PROCESSED_50_L2.mat')
#tomo_data_RS_file_DECONV=sio.loadmat(f'{basedir}/DECONV_RS_256_PROCESSED_50_PEARSON_SC.mat')

basedir_deconv='/net/micdata/data2/12IDC/ptychosaxs/batch_mode_250/RSM/'
lattice_name='ClathII'#'ClathII'#'SC'
iteration_number=250
noise_type='Noise'
loss_type='pearson_loss'#'L1'#'pearson_loss'
tomo_data_RS_file_DECONV=sio.loadmat(f'{basedir_deconv}/best_model_Lattice{lattice_name}_Probe256x256_ZCB_9_3D__{noise_type}_sim_ZCB_9_3D_S5065_N600_steps4_dp256_Unet_epoch_{iteration_number}_{loss_type}_symmetry_0.0/DECONV_RSM_PROCESSED.mat')

#tomo_data_RS_file_DECONV=sio.loadmat(f'{basedir}/DECONV_RS_256_PROCESSED_PREVIOUS_NN.mat')
#tomo_data_RS_file_DECONV=sio.loadmat(f'{basedir}/DECONV_RS_256_VARIANCE.mat')
#tomo_data_RS_file_DECONV=sio.loadmat(f'{basedir}/DECONV_RS_256_VARIANCE_THRESHOLD.mat')
tomo_data_RS_DECONV=tomo_data_RS_file_DECONV['DATA_d']
tomo_data_RS_qs_DECONV=tomo_data_RS_file_DECONV['Qv_d']
tomo_data_RS_DECONV[np.isnan(tomo_data_RS_DECONV)]=0
tomo_data_RS_qs_DECONV[np.isnan(tomo_data_RS_qs_DECONV)]=1
tomo_data_RS_DECONV = gaussian_filter(tomo_data_RS_DECONV, sigma=0.0) 


#cellinfo_data = load_cellinfo_data(f'{basedir}/cellinfo_ZCB9_3D.mat')
cellinfo_data = load_cellinfo_data(f'{basedir}/cellinfo_FCC_forFFTs.mat')
#cellinfo_data = load_cellinfo_data(f'{basedir}/cellinfo_ZCB_9_20251022.mat')
cellinfo_data = load_cellinfo_data(f'{basedir}/cellinfo_ZCB_9_FFT_20251024.mat')
#cellinfo_data = load_cellinfo_data(f'{basedir}/cellinfo_ZCB_9_DECONV_20251024.mat')
cellinfo_data = load_cellinfo_data(f'{basedir}/cellinfo_ZCB_9_FFT_250_pearson_20251027.mat')


def rotate_tomo(data, angle_deg, axis='z'):
    """
    Rotate a 3D tomogram by a given angle (in degrees) about the specified axis.
    Args:
        data: 3D numpy array
        angle_deg: rotation angle in degrees
        axis: 'x', 'y', or 'z' (axis about which to rotate)
    Returns:
        Rotated 3D numpy array (same shape as input, with reshape=False)
    """
    if axis == 'x':
        return rotate(data, angle_deg, axes=(1, 2), reshape=False)
    elif axis == 'y':
        return rotate(data, angle_deg, axes=(0, 2), reshape=False)
    elif axis == 'z':
        return rotate(data, angle_deg, axes=(0, 1), reshape=False)
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

# Example usage:
rotated_data = rotate_tomo(tomo_data_RS_DECONV, 0, axis='z')

rs_datasets=[
   #{'DATA':tomo_data_RS, 'Qv':tomo_data_RS_qs, 'label':'RS 128'},
    {'DATA':tomo_data_RS_256, 'Qv':tomo_data_RS_qs_256, 'label':'RS 256'},
    {'DATA':rotated_data, 'Qv':tomo_data_RS_qs_DECONV, 'label':'RS 256 DECONV'}
]


# rs_datasets=[
#     {'DATA':tomo_data_RS, 'Qv':tomo_data_RS_qs, 'label':'RS 128'}
# ]





# Generate indices up to 3rd order
miller_indices = generate_miller_indices(8)
hs = miller_indices[:, 0]
ks = miller_indices[:, 1]
ls = miller_indices[:, 2]
# fig, close_peaks, close_peaks_hkl, filtered_datasets = plot_multi_reciprocal_space(
#     rs_datasets,  # List of dicts: [{'magnitude': 3D array, 'Q': 4D array (shape: (nx,ny,nz,3)), 'label': str}, ...]
#     cellinfo_data,
#     hs, ks, ls,
#     #thresholds=[0.00016, 0.0002, 0.0145],  # List of thresholds for each dataset
#     thresholds=[0.00014+0.00006],#, 0.00014+0.00006, 0.0145],  # List of thresholds for each dataset
#     q_cutoffs=[0.02],#, 0.07, 0.07],
#     peak_distance_threshold=0.0085,
#     colormaps=['inferno'], #'viridis', 'jet'],
#     alphas=[0.1]#, 0.3, 1.0]
# )
# fig.show()

parula = sio.loadmat('/home/beams0/PTYCHOSAXS/NN/ptychosaxsNN/utils/parula.mat')['cmap']
parula_colors = ['rgb({:.0f},{:.0f},{:.0f})'.format(r*255, g*255, b*255) for r, g, b in parula]

parula_blue = parula_colors[:20]  # slice lower values (bluer)
parula_orange = parula_colors[-50:-30]  # slice higher values (yellows/oranges)


jet_green_colorscale = [
    (0.0, 'rgb(0, 255, 128)'),
    (0.25, 'rgb(0, 255, 96)'),
    (0.5, 'rgb(0, 255, 64)'),
    (0.75, 'rgb(0, 255, 32)'),
    (1.0, 'rgb(0, 255, 0)')
]

jet_red_colorscale = [
    (0.0, 'rgb(255, 128, 0)'),
    (0.25, 'rgb(255, 64, 0)'),
    (0.5, 'rgb(255, 32, 0)'),
    (0.75, 'rgb(255, 16, 0)'),
    (1.0, 'rgb(255, 0, 0)')
]


## BEST SETTINGS SO FAR
# thresholds=[0.0001, 0.0134]
# q_cutoffs=[0.073,0.073]
# peak_distance_threshold=0.0105
# dbscan_eps=0.08
# dbscan_min_samples=12


q_cutoffs=[0.074,0.074]
q_cutoffs=[0.065,0.065]

if lattice_name=='ClathII':
    if loss_type=='pearson_loss':
        #thresholds=[0.0001, 0.1] #pearson 25 epochs 59/67, 0 false positives
        thresholds=[0.0001, 0.103] #pearson 250 epochs 60/67, 1 false positive
    elif loss_type=='L2':
        thresholds=[0.0001, 0.056] #L2 50 epochs 58/67 peaks, 0 false positives
    elif loss_type=='L1':
        thresholds=[0.0001, 0.06] #L1 50 epochs 61/67 peaks, 0 false positives
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")
elif lattice_name=='SC':
    if loss_type=='pearson_loss':
        if noise_type=='noNoise':
            thresholds=[0.0001, 0.1] #pearson SC 50 epochs
        elif noise_type=='Noise':
            thresholds=[0.000098, 0.04] #pearson SC 50 epochs
        else:
            raise ValueError(f"Invalid noise type: {noise_type}")
    elif loss_type=='L2':
        if noise_type=='noNoise':
            thresholds=[0.0001, 0.032] #L2 SC 50 epochs
        elif noise_type=='Noise':
            thresholds=[0.0001, 0.032] #L2 SC 50 epochs
        else:
            raise ValueError(f"Invalid noise type: {noise_type}")
    elif loss_type=='L1':
        if noise_type=='noNoise':
            thresholds=[0.0001, 0.08] #L1 SC 50 epochs
        elif noise_type=='Noise':
            thresholds=[0.0001, 0.031] #L1 SC 50 epochs
        else:
            raise ValueError(f"Invalid noise type: {noise_type}")
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")
else:
    raise ValueError(f"Invalid lattice name: {lattice_name}")


peak_distance_threshold=0.05#0.0105
dbscan_eps=0.07#0.08
dbscan_min_samples=7

fig, close_peaks, close_peaks_hkl, filtered_datasets = plot_multi_reciprocal_space(
    rs_datasets,  # List of dicts: [{'magnitude': 3D array, 'Q': 4D array (shape: (nx,ny,nz,3)), 'label': str}, ...]
    cellinfo_data,
    hs, ks, ls,
    #thresholds=[0.0003, 0.00015, 0.0130],  # List of thresholds for each dataset
    #q_cutoffs=[0.02,0.07,0.07],
    thresholds=thresholds,  # List of thresholds for each dataset
    q_cutoffs=q_cutoffs,
    peak_distance_threshold=peak_distance_threshold,
    #colormaps=['inferno','viridis', 'jet'],
    colormaps=[parula_blue, parula_orange],
    alphas=[0.6,0.5],#[0.3,0.6,0.6]
    dbscan_eps=dbscan_eps, dbscan_min_samples=dbscan_min_samples
) 
fig.show()
n_peaks1, n_peaks2, n_overlapping = count_overlapping_peaks(
    rs_datasets[0], rs_datasets[1],
    threshold1=thresholds[0], threshold2=thresholds[1],
    q_cutoff1=q_cutoffs[0], q_cutoff2=q_cutoffs[1],
    overlap_distance=peak_distance_threshold,
    dbscan_eps=dbscan_eps, dbscan_min_samples=dbscan_min_samples,
    q_axes=[1,2,0], q_signs=[1,1,1], flatten_order='C'
)

print(f"Number of overlapping peaks: {n_overlapping}")
print(f"Number of peaks in dataset 1: {n_peaks1}")
print(f"Number of peaks in dataset 2: {n_peaks2}")


results = peak_confusion_matrix(
    true_dataset=rs_datasets[0], pred_dataset=rs_datasets[1],
    true_threshold=thresholds[0], pred_threshold=thresholds[1],
    true_q_cutoff=q_cutoffs[0], pred_q_cutoff=q_cutoffs[1],
    overlap_distance=peak_distance_threshold,
    dbscan_eps=dbscan_eps, dbscan_min_samples=dbscan_min_samples,
    q_axes=[1,2,0], q_signs=[1,1,1], flatten_order='C',
    plot=True,
    overlay=True,
    cellinfo_data=cellinfo_data,
    miller_indices=miller_indices
)

print("TP: ", results['TP'])
print("FP: ", results['FP'])
print("FN: ", results['FN'])
print("n_true: ", results['n_true'])
print("n_pred: ", results['n_pred'])
# %%
