#%%
import numpy as np
import h5py
import os
import pdb

# Define scan range and reconstruction path
recon_dir = "ZCB_2_3D_"
scan_nums = np.arange(3021, 3384)
recon_path = "Ndp128_LSQML_c1000_m0.5_p15_cp_mm_opr3_ic_pc_ul2"

# File containing reference scan_numbers and angles
#reference_file = f"/net/micdata/data2/12IDC/2025_Feb/misc/{recon_dir}/{recon_dir}_phase_projections_{recon_path}_Niter1000.h5"
reference_file=f"/net/micdata/data2/12IDC/2025_Feb/misc/{recon_dir}/{recon_dir}phase_projections_{recon_path}_Niter1000.h5"

# Output file path
save_dir = f"/net/micdata/data2/12IDC/2025_Feb/misc/{recon_dir}/"
save_path = os.path.join(save_dir, f"{recon_dir}phase_projections_{recon_path}_Niter1000_complex_JM.h5")
print(save_path)
#%%
# Load scan_numbers and angles from the reference file
with h5py.File(reference_file, 'r') as ref_h5:
    scan_numbers = ref_h5['scan_numbers'][:]
    angles = ref_h5['angles'][:]

# Initialize list for cropped complex projections
cropped_projections = []
successful_scan_nums = []  # Track successful scan numbers instead of indices

for scan_num in scan_nums:
    file_path = f"/net/micdata/data2/12IDC/2025_Feb/ptychi_recons/S{scan_num:04d}/{recon_path}/recon_Niter1000.h5"
    try:
        with h5py.File(file_path, 'r') as f:
            obj = f['object'][()][0]
            pixel_size = f['obj_pixel_size_m'][()]

            # Optional check: ensure obj is 2D and has expected size
            h, w = obj.shape[-2:]
            ch, cw = 230, 228
            sh, sw = h // 2 - ch // 2, w // 2 - cw // 2
            cropped_obj = obj[sh:sh+ch, sw:sw+cw]

            cropped_projections.append(cropped_obj.astype(np.complex64))
            successful_scan_nums.append(scan_num)  # Track successful scan numbers
    except Exception as e:
        print(f"Error reading scan {scan_num}: {e}")
        continue

# Stack into 3D array
projections_array = np.stack(cropped_projections, axis=0)

# Create boolean mask for successful scans
successful_mask = np.isin(scan_numbers, successful_scan_nums)

# Filter scan_numbers and angles using the mask
filtered_scan_numbers = scan_numbers[successful_mask]
filtered_angles = angles[successful_mask]

# Save combined data into new HDF5 file
with h5py.File(save_path, 'w') as out_h5:
    out_h5.create_dataset('scan_numbers', data=filtered_scan_numbers)
    out_h5.create_dataset('angles', data=filtered_angles)
    out_h5.create_dataset('projections', data=projections_array)

    # Save pixel size from the last successful load
    out_h5.create_dataset('pixel_size', data=pixel_size)

print(f"Saved combined HDF5 file to: {save_path}")
print(f"Number of successful scans: {len(successful_scan_nums)}")
print(f"Number of failed scans: {len(scan_nums) - len(successful_scan_nums)}")

# %%
