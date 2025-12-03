#%%
import os
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import colors
import sys
from tqdm import tqdm
import h5py
import time
import pandas as pd
# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../../NN/ptychosaxsNN/')))
import utils.ptychosaxsNN_utils as ptNN_U
import importlib
importlib.reload(ptNN_U)

# Add the models directory to the path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../src/models/')))

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

#%%
def main():
    # Parameters
    scan_name = 'ZCB_9_3D'
    base_path = "/net/micdata/data2/12IDC/2025_Feb/ptycho/"
    #scan_numbers = range(5004, 5370)  # From 5004 to 5369
    # Read the file, skipping the first row (which starts with #) and using the second row as headers
    df = pd.read_csv('/net/micdata/data2/12IDC/2025_Feb/misc/ZCB_9_3D_/ZCB9_3D_alignment_shifts_28nm.txt', 
                 comment='#',  # Skip lines starting with #
                 names=['Angle', 'y_shift', 'x_shift', 'scanNo'])  # Specify column names
    scan_numbers = df['scanNo'].values.tolist()

    #scan_number=[5065]
    center = (517, 575)
    dpsize = 256
    
    # Model and mask paths
    #model_path = "/net/micdata/data2/12IDC/ptychosaxs/models/ZCB_9_3D/best_model_ZCB_9_Unet_epoch_500_pearson_loss.pth"
    model_path = '/net/micdata/data2/12IDC/ptychosaxs/batch_mode/trained_model/best_model_LatticeClathII_Probe256x256_ZCB_9_3D__Noise_sim_ZCB_9_3D_S5065_N600_steps4_dp256_Unet_epoch_25_pearson_loss_symmetry_0.0.pth'
    mask_path = '/home/beams/PTYCHOSAXS/deconvolutionNN/data/mask/mask_ZCB_9_3D.npy' 
    # Load model and mask once
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from encoder1 import recon_model as recon_model2
    model = recon_model2()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # Load mask
    mask = np.load(mask_path)
    
    # Output H5 file
    output_file = f'combined_diffraction_patterns_TEST_variance_threshold_20251022.h5'
    
    # Get already processed scans
    processed_scans = get_processed_scans(output_file)
    remaining_scans = [s for s in scan_numbers if s not in processed_scans]
    print(f"Found {len(processed_scans)} already processed scans. {len(remaining_scans)} scans remaining.")
    
    # Add indices file parameter
    #indices_file = "/home/beams/PTYCHOSAXS/NN/ptychosaxsNN/utils/ZCB_9_3D_sample_indices.h5" 
    #indices_file = "/home/beams/PTYCHOSAXS/NN/ptychosaxsNN/utils/ZCB_9_3D_sample_indices_variance.h5" 
    indices_file = "/home/beams/PTYCHOSAXS/NN/ptychosaxsNN/utils/ZCB_9_3D_sample_indices_variance_threshold.h5" 
    
    # Add delay parameter
    delay_between_scans = 0.1  # seconds
    
    # Process remaining scans sequentially with delay
    results = []
    for scan_number in tqdm(remaining_scans, desc="Processing scans"):
        result = process_scan(
            scan_number=scan_number,
            base_path=base_path,
            model=model,
            device=device,
            center=center,
            dpsize=dpsize,
            mask=mask,
            indices_file=indices_file
        )
        if result is not None:
            results.append(result)
            save_verification_plot(result)
            save_to_h5(output_file, result)
        
        if scan_number != remaining_scans[-1]:  # Don't delay after the last scan
            print(f"\nWaiting {delay_between_scans} seconds before processing next scan...", flush=True)
            time.sleep(delay_between_scans)
    
    print("Processing complete. Results saved to", output_file)

#%%
if __name__ == "__main__":
    # Comment out main() and uncomment this to remove scans
    # In your interactive session
    #h5_file = 'combined_diffraction_patterns_TEST.h5'
    #scans_to_remove = [5058, 5059,5060]  # replace with your scan numbers
    #remove_scans_from_h5(h5_file, scans_to_remove)
    main()
#%%
with h5py.File('combined_diffraction_patterns_TEST_variance_threshold_20251022.h5', 'r') as f:
    print(f['metadata']['angles'][()])
#%%
# Add metadata to H5 file
h5_file = 'combined_diffraction_patterns_TEST_variance_threshold_20251022.h5'
df = pd.read_csv('/net/micdata/data2/12IDC/2025_Feb/misc/ZCB_9_3D_/ZCB9_3D_alignment_shifts_28nm.txt', 
                 comment='#',
                 names=['Angle', 'y_shift', 'x_shift', 'scanNo'])
add_metadata_to_h5(h5_file, df)
# %%
