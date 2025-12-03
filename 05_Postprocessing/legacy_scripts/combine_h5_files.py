#%%
"""
Script to combine multiple H5 files into a single H5 file.
Loads files from /net/micdata/data2/12IDC/2025_Jul/SAXS/1120/SZC6_3DSAXS_1120_00003_{num:05d}.h5
where num ranges from 42 to 721.
"""

import h5py
import numpy as np
import os
import time
import gc
from tqdm import tqdm


#%%
import tifffile as tif
import matplotlib.pyplot as plt
from matplotlib import colors
#obj=tif.imread('/net/micdata/data2/12IDC/2025_Jul/ptychi_recons/S0323/Ndp128_LSQML_s1000_gaussian_p5_cp_mm_opr2_ic_pc1_g_ul2/object_ph/object_ph_Niter1000.tiff')
with h5py.File('/net/micdata/data2/12IDC/2025_Jul/ptychi_recons/S0324/Ndp128_LSQML_s1000_gaussian_p5_cp_mm_opr2_ic_pc1_g_ul2/recon_Niter1000.h5','r') as f:
    obj=f['object'][()][0]
fig,ax=plt.subplots(1,2,figsize=(20,10))
ax[0].imshow(np.angle(obj),cmap='gray')
ax[1].imshow(np.abs(np.fft.fftshift(np.fft.fft2(obj)))**2,norm=colors.LogNorm(),cmap='jet')
plt.show()

#%%

def combine_h5_files(base_path, start_num, end_num, output_path, data_key='entry/data/data', 
                     crop_center=None, crop_size=None):
    """
    Combine multiple H5 files into a single H5 file with optional cropping.
    
    Parameters:
        base_path (str): Base path pattern with {num:05d} placeholder
        start_num (int): Starting file number
        end_num (int): Ending file number (inclusive)
        output_path (str): Path for the output combined H5 file
        data_key (str): Key name in the H5 files containing the data
        crop_center (tuple): (row, col) center position for cropping. If None, no cropping.
        crop_size (int): Size of the square crop region. If None, no cropping.
    """
    
    # First pass: check files and get dimensions
    print("Checking files and determining dimensions...")
    valid_files = []
    data_shape = None
    data_dtype = None
    
    for num in range(start_num, end_num + 1):
        file_path = base_path.format(num=num)
        
        if os.path.exists(file_path):
            try:
                with h5py.File(file_path, 'r') as f:
                    if data_key in f:
                        data = f[data_key]
                        original_shape = data.shape
                        
                        # Apply cropping if specified
                        if crop_center is not None and crop_size is not None:
                            row_center, col_center = crop_center
                            half_size = crop_size // 2
                            
                            # Calculate crop bounds
                            row_start = max(0, row_center - half_size)
                            row_end = min(original_shape[0], row_center + half_size)
                            col_start = max(0, col_center - half_size)
                            col_end = min(original_shape[1], col_center + half_size)
                            
                            # Ensure we get exactly crop_size x crop_size
                            if row_end - row_start != crop_size or col_end - col_start != crop_size:
                                print(f"Warning: Crop region extends beyond image bounds for {file_path}")
                                # Adjust bounds to maintain crop_size
                                if row_end - row_start < crop_size:
                                    diff = crop_size - (row_end - row_start)
                                    if row_start >= diff:
                                        row_start -= diff
                                    else:
                                        row_end = row_start + crop_size
                                if col_end - col_start < crop_size:
                                    diff = crop_size - (col_end - col_start)
                                    if col_start >= diff:
                                        col_start -= diff
                                    else:
                                        col_end = col_start + crop_size
                            
                            effective_shape = (crop_size, crop_size)
                        else:
                            effective_shape = original_shape
                        
                        if data_shape is None:
                            data_shape = effective_shape
                            data_dtype = data.dtype
                            print(f"Original data shape: {original_shape}, dtype: {data_dtype}")
                            if crop_center is not None and crop_size is not None:
                                print(f"Cropped data shape: {data_shape}")
                                print(f"Crop center: {crop_center}, crop size: {crop_size}")
                        
                        # Verify all files have the same effective shape
                        if effective_shape == data_shape:
                            valid_files.append((num, file_path))
                        else:
                            print(f"Warning: File {file_path} has different effective shape {effective_shape}, skipping")
                    else:
                        print(f"Warning: File {file_path} does not contain '{data_key}' key, skipping")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")
    
    print(f"Found {len(valid_files)} valid files")
    
    if len(valid_files) == 0:
        print("No valid files found!")
        return
    
    # Second pass: load and combine data
    print("Loading and combining data...")
    
    # Create output dataset with shape (num_files, height, width)
    combined_shape = (len(valid_files),) + data_shape
    
    with h5py.File(output_path, 'w') as output_f:
        # Create the main combined dataset with optimized chunking and compression
        # Use chunking that aligns with our processing pattern
        chunk_shape = (min(50, len(valid_files)),) + data_shape  # Chunk by our processing size
        combined_dataset = output_f.create_dataset(
            'combined_data', 
            shape=combined_shape, 
            dtype=data_dtype,
            chunks=chunk_shape,
            compression='lzf',  # Faster compression than gzip
            shuffle=True,       # Improves compression for scientific data
            fletcher32=True     # Add checksum for data integrity
        )
        
        # Create metadata datasets
        file_numbers = output_f.create_dataset('file_numbers', data=[num for num, _ in valid_files])
        file_paths = output_f.create_dataset('file_paths', data=[path.encode('utf-8') for _, path in valid_files])
        
        # Pre-calculate crop bounds if cropping is enabled
        crop_bounds = None
        if crop_center is not None and crop_size is not None:
            row_center, col_center = crop_center
            half_size = crop_size // 2
            crop_bounds = {
                'row_start': row_center - half_size,
                'row_end': row_center + half_size,
                'col_start': col_center - half_size,
                'col_end': col_center + half_size
            }
        
        # Process files in chunks to reduce memory pressure and improve performance
        chunk_size = 50  # Process 50 files at a time
        start_time = time.time()
        
        for chunk_start in range(0, len(valid_files), chunk_size):
            chunk_start_time = time.time()
            chunk_end = min(chunk_start + chunk_size, len(valid_files))
            chunk_files = valid_files[chunk_start:chunk_end]
            
            chunk_num = chunk_start//chunk_size + 1
            total_chunks = (len(valid_files)-1)//chunk_size + 1
            print(f"Processing chunk {chunk_num}/{total_chunks}")
            
            # Load chunk data into memory first
            chunk_data = []
            for num, file_path in tqdm(chunk_files, desc=f"Loading chunk {chunk_num}"):
                try:
                    with h5py.File(file_path, 'r') as f:
                        data = f[data_key][:]
                        
                        # Apply cropping if specified
                        if crop_bounds is not None:
                            row_start = max(0, crop_bounds['row_start'])
                            row_end = min(data.shape[0], crop_bounds['row_end'])
                            col_start = max(0, crop_bounds['col_start'])
                            col_end = min(data.shape[1], crop_bounds['col_end'])
                            
                            # Ensure we get exactly crop_size x crop_size
                            if row_end - row_start < crop_size:
                                diff = crop_size - (row_end - row_start)
                                if row_start >= diff:
                                    row_start -= diff
                                else:
                                    row_end = row_start + crop_size
                            if col_end - col_start < crop_size:
                                diff = crop_size - (col_end - col_start)
                                if col_start >= diff:
                                    col_start -= diff
                                else:
                                    col_end = col_start + crop_size
                            
                            # Crop the data
                            data = data[row_start:row_end, col_start:col_end]
                        
                        chunk_data.append(data.copy())  # Make a copy to avoid reference issues
                        del data  # Explicitly delete to help garbage collection
                        
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    # Fill with zeros if there's an error
                    chunk_data.append(np.zeros(data_shape, dtype=data_dtype))
            
            # Write chunk to dataset in one operation (faster than individual writes)
            chunk_array = np.array(chunk_data)
            combined_dataset[chunk_start:chunk_start + len(chunk_data)] = chunk_array
            
            # Force garbage collection and clear chunk data
            del chunk_data, chunk_array
            gc.collect()
            
            # Flush the HDF5 file to disk periodically
            output_f.flush()
            
            # Performance monitoring
            chunk_time = time.time() - chunk_start_time
            total_time = time.time() - start_time
            avg_time_per_chunk = total_time / chunk_num
            estimated_remaining = avg_time_per_chunk * (total_chunks - chunk_num)
            
            print(f"Chunk {chunk_num} completed in {chunk_time:.1f}s. "
                  f"Estimated remaining time: {estimated_remaining/60:.1f} minutes")
        
        # Add attributes
        output_f.attrs['source_pattern'] = base_path
        output_f.attrs['start_num'] = start_num
        output_f.attrs['end_num'] = end_num
        output_f.attrs['num_files'] = len(valid_files)
        output_f.attrs['data_shape'] = data_shape
        output_f.attrs['original_data_key'] = data_key
        if crop_center is not None and crop_size is not None:
            output_f.attrs['crop_center'] = crop_center
            output_f.attrs['crop_size'] = crop_size
        
        print(f"Successfully created combined H5 file: {output_path}")
        print(f"Combined dataset shape: {combined_shape}")
        print(f"Datasets created:")
        print(f"  - 'combined_data': {combined_shape} {data_dtype}")
        print(f"  - 'file_numbers': ({len(valid_files)},) int")
        print(f"  - 'file_paths': ({len(valid_files)},) string")

def main():
    # Configuration
    #base_path = "/net/micdata/data2/12IDC/2025_Jul/SAXS/1120/SZC6_3DSAXS_1120_00003_{num:05d}.h5"
    #base_path = "/net/micdata/data2/12IDC/2025_Jul/SAXS/346/SZC3SAXS346_00001_{num:05d}.h5"
    base_path = "/net/micdata/data2/12IDC/2025_Jul/SAXS/566/test_00001_{num:05d}.h5"
    start_num = 1
    end_num = 721  # Change to 721 for full range
    #output_path = "/net/micdata/data2/12IDC/2025_Jul/misc/combined_SZC6_3DSAXS_1120_00003.h5"
    output_path = "/net/micdata/data2/12IDC/2025_Jul/misc/combined_SZC8SAXS566_00001.h5"
    
    # Cropping parameters
    crop_center = (773, 627)#(627, 775)  # (row, col) center position
    crop_size = 256 #128  # Size of square crop
    
    print(f"Combining H5 files from {start_num:05d} to {end_num:05d}")
    print(f"Base path pattern: {base_path}")
    print(f"Output file: {output_path}")
    print(f"Cropping: {crop_size}x{crop_size} centered at {crop_center}")
    
    combine_h5_files(base_path, start_num, end_num, output_path, 
                     crop_center=crop_center, crop_size=crop_size)

if __name__ == "__main__":
    main()

# %%
