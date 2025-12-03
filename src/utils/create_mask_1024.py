#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import sys
import os
# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../')))
import utils.ptychosaxsNN_utils as ptNN_U



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
#%%

scan_number=318
base_path='/net/micdata/data2/12IDC/2025_Jul/ptycho/'
data = ptNN_U.load_h5_scan_to_npy(base_path, scan_number, plot=False, point_data=True)
plt.imshow(data[0],norm=colors.LogNorm())
plt.colorbar()
plt.show()
dps=data.copy()
# Create a mask for the data
# Center and ncols and nrows will be different for different scans
# ZC4
#%%
ncols=71#41#36
nrows=31#31#29
center=(773, 626)#(718,742)
dps_size = dps[0].shape
center_offset_y=dps_size[0]//2-center[0]
center_offset_x=dps_size[1]//2-center[1]
dpsize = 1024

dps_cropped = dps[:, 
    dps_size[0]//2-center_offset_y - dpsize//2:dps_size[0]//2-center_offset_y + dpsize//2,
    dps_size[1]//2-center_offset_x - dpsize//2:dps_size[1]//2-center_offset_x + dpsize//2
]

# Remove hot pixels
for i, dp in enumerate(dps_cropped):
    dp[dp >= 2**16-1] = np.min(dp)
plt.imshow(dps_cropped[0],norm=colors.LogNorm())
plt.colorbar()
plt.show()


#%%
bin_factor = 1024
dps_binned = bin_ndarray(dps_cropped[0], (bin_factor, bin_factor), operation='sum')

# Create initial mask (True everywhere)
mask = np.ones_like(dps_binned, dtype=bool)

# Remove negative values
mask[dps_binned < 0] = False

# Find horizontal and vertical stripes of zeros
# For horizontal stripes
for i in range(bin_factor):
    if np.all(dps_binned[i, :] == 0):
        mask[i, :] = False

# For vertical stripes
for j in range(bin_factor):
    if np.all(dps_binned[:, j] == 0):
        mask[:, j] = False

# Visualize the original data, mask, and masked data
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Original data
im1 = ax1.imshow(dps_binned, norm=colors.LogNorm())
ax1.set_title('Original Data')
plt.colorbar(im1, ax=ax1)

# Mask
im2 = ax2.imshow(mask, cmap='gray')
ax2.set_title('Mask')
plt.colorbar(im2, ax=ax2)

# Masked data - convert to float first
masked_data = dps_binned.astype(float)
masked_data[~mask] = np.nan
im3 = ax3.imshow(masked_data, norm=colors.LogNorm())
ax3.set_title('Masked Data')
plt.colorbar(im3, ax=ax3)

plt.tight_layout()
plt.show()

save_mask=True
if save_mask:
    mask_filename = '/home/beams/PTYCHOSAXS/deconvolutionNN/data/mask/mask_2025_Jul_ZC2_1024.npy'
    np.save(mask_filename, mask)
# %%
