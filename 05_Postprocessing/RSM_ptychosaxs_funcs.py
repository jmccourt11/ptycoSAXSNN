#%%
##################################
#IMPORT LIBRARIES
##################################
import os
import sys
import time
import random
import importlib
import numpy as np
import pandas as pd
import h5py
import torch
from pathlib import Path
from tqdm import tqdm
from itertools import product
from scipy.io import loadmat, savemat
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter, rotate
import plotly.graph_objects as go


# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'NN/ptychosaxsNN/')))
import utils.ptychosaxsNN_utils as ptNN_U
importlib.reload(ptNN_U)

# Add the models directory to the path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'deconvolutionNN/src/models/')))

#%%


##################################
#STEP 1: DECONVOLUTION
##################################
#%%


def retry_h5_operation(operation, max_retries=5, initial_delay=1):
    """Retry an H5 operation with exponential backoff"""
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return operation()
        except (OSError, IOError) as e:
            last_exception = e
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed, retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            continue
    
    print(f"Failed after {max_retries} attempts: {str(last_exception)}")
    return None

def get_processed_scans(h5_file):
    """Get list of already processed scan numbers from H5 file"""
    processed_scans = set()
    
    def read_h5():
        with h5py.File(h5_file, 'r') as f:
            if 'raw' in f:
                return set(int(k.split('_')[1]) for k in f['raw'].keys())
        return set()
    
    try:
        if os.path.exists(h5_file):
            result = retry_h5_operation(read_h5)
            if result is not None:
                processed_scans = result
            else:
                print("Warning: Could not read existing H5 file after retries. Starting from scratch.")
    except Exception as e:
        print(f"Warning: Error reading H5 file: {str(e)}. Starting from scratch.")
    
    return processed_scans

def save_to_h5(output_file, result):
    """Save a single result to H5 file with retry mechanism"""
    def write_h5():
        # First, ensure groups exist and get existing keys
        with h5py.File(output_file, 'a') as f:
            for group_name in ['raw', 'preprocessed', 'deconvolved']:
                if group_name not in f:
                    f.create_group(group_name)
        
        # Save each group separately with fresh file handles
        for group_name in ['raw', 'preprocessed', 'deconvolved']:
            with h5py.File(output_file, 'a') as f:
                group = f[group_name]
                dataset_name = f'scan_{result["scan_number"]}'
                if dataset_name in group:
                    del group[dataset_name]
                group.create_dataset(dataset_name, data=result[group_name])
                f.flush()  # Ensure data is written to disk
        return True
    
    success = retry_h5_operation(write_h5)
    if not success:
        print(f"Warning: Could not save scan {result['scan_number']} to H5 file after retries.")
        # Save to a temporary NPZ file as backup
        backup_file = f'backup_scan_{result["scan_number"]}.npz'
        np.savez(backup_file, **result)
        print(f"Saved backup to {backup_file}")

def load_indices_from_h5(indices_file, scan_number):
    """Load indices for a specific scan from H5 file"""
    try:
        with h5py.File(indices_file, 'r') as f:
            # Assuming the indices are stored in a dataset named like 'scan_5004'
            dataset_name = f'scan_{scan_number}'
            if dataset_name in f:
                return f[dataset_name][:]
            else:
                print(f"Warning: No indices found for scan {scan_number}")
                return None
    except Exception as e:
        print(f"Error loading indices for scan {scan_number}: {str(e)}")
        return None

def process_scan(scan_number, base_path, model, device, center, dpsize, mask, indices_file=None):
    """Process a single scan"""
    try:
        # Load mask
        print(f"Loading scan {scan_number}")
        dps = ptNN_U.load_h5_scan_to_npy(base_path, scan_number, plot=False, point_data=True)
        
        # Load indices if file provided
        selected_indices = None
        if indices_file:
            selected_indices = load_indices_from_h5(indices_file, scan_number)
            if selected_indices is None:
                print(f"Warning: Proceeding with all indices for scan {scan_number}")
            else:
                print(f"Using {len(selected_indices)} selected indices for scan {scan_number}")
                # Filter dps based on indices
                dps = dps[selected_indices]
        
        # Process in batches
        batch_size = 32
        n_batches = len(dps) // batch_size + (1 if len(dps) % batch_size != 0 else 0)
        
        raw_arrays = []
        input_arrays = []
        output_arrays = []
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(dps))
            batch_dps = dps[start_idx:end_idx]
            current_batch_size = len(batch_dps)
            
            # Pre-allocate arrays
            raw_list = np.zeros((current_batch_size, dpsize, dpsize))
            input_tensors = []
            sfs = []
            bkgs = []
            
            # Prepare data
            for j, dp in enumerate(batch_dps):
                cropped_dp = dp[center[0]-dpsize//2:center[0]+dpsize//2,
                              center[1]-dpsize//2:center[1]+dpsize//2]
                raw_list[j] = cropped_dp
                resultT, sfT, bkgT = ptNN_U.preprocess_ZCB_9(cropped_dp, mask)
                input_tensors.append(resultT)
                sfs.append(sfT)
                bkgs.append(bkgT)
            
            # Stack tensors for batch processing
            input_batch = torch.cat(input_tensors, dim=0)
            
            # Process batch through model
            with torch.no_grad():
                output_batch = model(input_batch.to(device=device, dtype=torch.float))
            
            # Convert outputs back to numpy
            output_list = output_batch.cpu().numpy()
            
            # Calculate log-scale data using vectorized operations
            sfs = np.array(sfs)[:, np.newaxis, np.newaxis]
            bkgs = np.array(bkgs)[:, np.newaxis, np.newaxis]
            
            output_list_log = 10**(output_list[:, 0] * sfs + bkgs)
            input_list_log = 10**(np.array([t.numpy()[0][0] for t in input_tensors]) * sfs + bkgs)
            
            raw_arrays.append(raw_list)
            input_arrays.append(input_list_log)
            output_arrays.append(output_list_log)
        
        # Combine results
        raw_list = np.concatenate(raw_arrays)
        input_list_log = np.concatenate(input_arrays)
        output_list_log = np.concatenate(output_arrays)
        
        # Sum the patterns
        summed_raw = np.sum(raw_list, axis=0)
        summed_preprocessed = np.sum(input_list_log, axis=0)
        summed_deconvolved = np.sum(output_list_log, axis=0)
        
        return {
            'scan_number': scan_number,
            'raw': summed_raw,
            'preprocessed': summed_preprocessed,
            'deconvolved': summed_deconvolved
        }
        
    except Exception as e:
        print(f"Error processing scan {scan_number}: {str(e)}")
        return None

def save_verification_plot(results, output_dir='plots'):
    """Save verification plot for a scan"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        
        # Create masked arrays to handle zeros and negative values
        deconvolved_data = np.ma.masked_less_equal(results['deconvolved'], 0)
        preprocessed_data = np.ma.masked_less_equal(results['preprocessed'], 0)
        raw_data = np.ma.masked_less_equal(results['raw'], 0)
        
        # Add epsilon to valid data
        deconvolved_data = np.where(~deconvolved_data.mask, deconvolved_data + epsilon, deconvolved_data)
        preprocessed_data = np.where(~preprocessed_data.mask, preprocessed_data + epsilon, preprocessed_data)
        raw_data = np.where(~raw_data.mask, raw_data + epsilon, raw_data)
        
        im1 = ax[0].imshow(deconvolved_data, cmap='jet', norm=colors.LogNorm())
        ax[0].set_title(f'Deconvolved - Scan {results["scan_number"]}')
        im2 = ax[1].imshow(preprocessed_data, cmap='jet', norm=colors.LogNorm())
        ax[1].set_title(f'Preprocessed - Scan {results["scan_number"]}')
        im3 = ax[2].imshow(raw_data, cmap='jet', norm=colors.LogNorm())
        ax[2].set_title(f'Raw - Scan {results["scan_number"]}')
        
        plt.colorbar(im1, ax=ax[0])
        plt.colorbar(im2, ax=ax[1])
        plt.colorbar(im3, ax=ax[2])
        
        plt.savefig(os.path.join(output_dir, f'scan_{results["scan_number"]}_summary_TEST.png'))
        plt.close()
    except Exception as e:
        print(f"Warning: Could not save verification plot for scan {results['scan_number']}: {str(e)}")

def remove_scans_from_h5(h5_file, scan_numbers_to_remove, create_backup=True):
    """Remove specific scan numbers from all groups in the H5 file"""
    # Create backup first
    if create_backup:
        backup_file = f"{h5_file}.backup_{time.strftime('%Y%m%d_%H%M%S')}"
        try:
            import shutil
            shutil.copy2(h5_file, backup_file)
            print(f"Created backup at: {backup_file}")
        except Exception as e:
            print(f"Warning: Could not create backup: {str(e)}")
            return False

    def delete_scans():
        with h5py.File(h5_file, 'a') as f:
            removed_scans = []
            for group_name in ['raw', 'preprocessed', 'deconvolved']:
                if group_name in f:
                    group = f[group_name]
                    for scan_num in scan_numbers_to_remove:
                        dataset_name = f'scan_{scan_num}'
                        if dataset_name in group:
                            print(f"Removing {dataset_name} from {group_name}")
                            del group[dataset_name]
                            if scan_num not in removed_scans:
                                removed_scans.append(scan_num)
                            f.flush()
            return removed_scans

    removed_scans = retry_h5_operation(delete_scans)
    if removed_scans:
        print(f"Successfully removed scans: {removed_scans}")
        return True
    else:
        print("Failed to remove scans. Please check the file.")
        return False

def add_metadata_to_h5(h5_file, df):
    """Add angles and scan numbers to existing H5 file"""
    try:
        # Create a dictionary mapping scan numbers to angles
        scan_to_angle = dict(zip(df['scanNo'], df['Angle']))
        
        def update_h5():
            with h5py.File(h5_file, 'a') as f:
                # Get existing scan numbers from the 'raw' group
                existing_scans = []
                if 'raw' in f:
                    existing_scans = [int(k.split('_')[1]) for k in f['raw'].keys()]
                
                # Create metadata groups if they don't exist
                if 'metadata' not in f:
                    metadata = f.create_group('metadata')
                else:
                    metadata = f['metadata']
                
                # Create or update angles and scan_numbers datasets
                angles = []
                scan_numbers = []
                
                for scan_num in existing_scans:
                    if scan_num in scan_to_angle:
                        angles.append(scan_to_angle[scan_num])
                        scan_numbers.append(scan_num)
                
                # Save or update the datasets
                if 'angles' in metadata:
                    del metadata['angles']
                if 'scan_numbers' in metadata:
                    del metadata['scan_numbers']
                    
                metadata.create_dataset('angles', data=np.array(angles))
                metadata.create_dataset('scan_numbers', data=np.array(scan_numbers))
                
                return len(angles)
        
        num_added = retry_h5_operation(update_h5)
        if num_added is not None:
            print(f"Successfully added metadata for {num_added} scans")
            return True
    except Exception as e:
        print(f"Error adding metadata: {str(e)}")
        return False


##################################
#STEP 2: CONVERT h5 TO MAT AND PREPROCESS
##################################

def find_zero_arrays(array_3d):
    """
    Find indices where 2D arrays are all zeros in a 3D array
    
    Parameters:
    -----------
    array_3d : numpy.ndarray
        3D array where the first two dimensions are the 2D arrays
        
    Returns:
    --------
    list
        List of indices where the 2D arrays are all zeros
    """
    # Sum across the first two dimensions to check if entire 2D array is zero
    sums = np.sum(np.sum(array_3d, axis=0), axis=0)
    # Find indices where sum is zero (meaning entire 2D array is zero)
    zero_indices = np.where(sums == 0)[0]
    return zero_indices.tolist()


# ##################################
# #STEP 2.5: WORK WITH MAT FILES IN LOCAL MATLAB USING BYEONGDU'S SOFTWARE
# ##################################

# MATLAB CODE
# %saxs.waveln = 0.124;
# %saxs.SDD = 3000; %5.57


# %detector="Dectris Eiger 500K";
# saxs.SDD = 10.200;%9770%10000%5570=5.57 m
# saxs.waveln=0.123984; %1.23984 \AA
# saxs.tthi = 0;
# saxs.ai = 0;
# saxs.edensity = 0;
# saxs.beta = 0;
# ROIX = [1:256];%[1:302];
# %ROIX = [2:301];
# ROIY = [1:256];%[1:359];
# %ROIY = [30:330];
# %saxs.center = [size(ROIX,2)/2,180]
# saxs.center=[127.5,129.5];
# %saxs.psize = 0.172;
# qN = 100;%size(ROIY,2);
# qmax = 0.2;%*10;%0.4;%0.2;

# %recon_pixel_size=28.72086*2; %nm
# %pixel_size=2*pi/recon_pixel_size/numel(ROIX);%512;
# %
# % saxs.psize=pixel_size; %0.075=75 um
# %saxs.pxQ = pixel_size; %q-space
# %saxs.psize=saxs.waveln*saxs.pxQ*saxs.SDD/(2*pi);%real-space
# saxs.psize=172*10^(-6);

# % img = [];
# % for i=1:191
# %     fn = sprintf('SesMEOHrot_00044_%0.5i.h5', i);
# %     d = SAXSimageviwerLoadimage(fn);
# %     img(:, :, i) = d.image;
# %     phi(i) = i;
# % end

# disp("Data loading one.")


# %%
# %inp_data.mask = mask;
# inp_data.mask = ones(size(img(:,:,1)));
# inp_data.img_mtrx = img;
# inp_data.phi = phi;
# inp_data.norm_factor = ones(size(phi));
# inp_data.isfliped = false;
# inp_data.background = false;
# inp_data.gen_back = false;
# inp_data.switch_axes = false;

# % Switch axes if requested
# if inp_data.switch_axes
#     inp_data.img_mtrx = permute(inp_data.img_mtrx, [2 1 3]);
#     inp_data.mask = permute(inp_data.mask, [2 1]);
# end

# [Qv_d, DATA_d] = construct_RecpSpace_fromImgMtrx(inp_data, saxs, ROIX, ROIY, qN, qmax);

# % Save DATA to MAT file
# %save('reciprocal_space_data.mat', 'DATA');
# %disp('Data saved to reciprocal_space_data.mat');
 
# draw_3dmap(DATA_d,[Qv_d(:,2),Qv_d(:,1),Qv_d(:,3)])
# load('C:\Users\b304014\Software\blee\data\2025_Feb\data\ZCB_9_3D\cellinfo_FCC_forFFTs.mat')
# save('C:\Users\b304014\Software\blee\data\2025_Feb\data\ZCB_9_3D\DECONV_RS_256_PROCESSED.mat','DATA_d','Qv_d')

# scp C:\Users\b304014\Software\blee\data\2025_Feb\data\ZCB_9_3D\DECONV_RS_256_PROCESSED.mat ptychosaxs@refiner.xray.aps.anl.gov:/scratch/2025_Feb/


##################################
#STEP 3: PEAK DETECTION AND CONFUSION MATRIX
##################################


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
    #highlight_hkls = ['(0,0,2)', '(0,0,6)', '(0,0,-2)', '(0,0,-6)'] 
    highlight_hkls = ['(0,6,6)', '(0,-6,-6)', '(2,2,-8)', '(-2,-2,8)', '(2,8,-2)', '(-2,-8,2)']

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