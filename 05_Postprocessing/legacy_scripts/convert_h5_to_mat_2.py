#%%
import h5py
import numpy as np
from scipy.io import loadmat,savemat
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
import random
#%%

# scp ptychosaxs@refiner.xray.aps.anl.gov:/home/beams/PTYCHOSAXS/deconvolutionNN/src/utils/combined_diffraction_patterns_TEMP.h5 "C:\Users\b304014\Software\blee\data\combined_test.h5"
# scp ptychosaxs@refiner.xray.aps.anl.gov:/home/beams/PTYCHOSAXS/deconvolutionNN/src/utils/combined_diffraction_patterns_TEST_variance.h5 "C:\Users\b304014\Software\blee\data\deconvolved_DPs_variance.h5" 

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

#output_file = "C:\\Users\\b304014\\Software\\blee\\data\\deconvolved_DPs_variance_20251022.h5"
output_file = "C:\\Users\\b304014\\Software\\blee\\data\\temp\\combined_diffraction_patterns_best_model_LatticeSC_Probe256x256_ZCB_9_3D__Noise_sim_ZCB_9_3D_S5065_N600_steps4_dp256_Unet_epoch_50_pearson_loss.h5"
save_string = output_file.split("\\")[-1].split(".")[0]
print(save_string)
#%%
scan_num_start=5003
size=366
skip_indices=[0]#[1,2,3]
result_mtx=np.zeros((256,256,size))
angles=np.zeros((size))
valid_idx = 0  # Counter for valid entries
for i in range(0,size):
    if i in skip_indices:
        print(f"Skipping scan {scan_num_start+i}")
        continue
    else:
        with h5py.File(output_file, 'r') as f:
            try:
                #print(f"Processing scan {scan_num_start+i}")
                data = f['deconvolved'][f'scan_{scan_num_start+i}'][()]
                result_mtx[:,:,valid_idx]=data
                
                # Get the scan number and find its index in the scan_numbers array
                current_scan = scan_num_start + i
                scan_numbers = f['metadata']['scan_numbers'][()]
                angle_idx = np.where(scan_numbers == current_scan)[0]
                
                if len(angle_idx) > 0:
                    angles[valid_idx] = f['metadata']['angles'][()][angle_idx[0]]
                else:
                    print(f"Warning: No angle found for scan {current_scan}. Using previous angle.")
                    angles[valid_idx] = angles[valid_idx-1] if valid_idx > 0 else 0
                
                valid_idx += 1
            except KeyError as e:
                print(f"KeyError processing scan {scan_num_start+i}: {str(e)}")
            except IndexError as e:
                print(f"IndexError processing scan {scan_num_start+i}: {str(e)}")
            except Exception as e:
                print(f"Error processing scan {scan_num_start+i}: {str(e)}")
                
zero_idx = find_zero_arrays(result_mtx)
print(f"Found zero arrays at indices: {zero_idx}")

# Find the index with maximum sum
sums = np.array([np.sum(result_mtx[:,:,i]) for i in range(result_mtx.shape[2])])
max_idx = np.argmax(sums)
print(f"Found maximum sum at index: {max_idx}")

# Combine zero indices with max index
indices_to_remove = list(set(zero_idx + [max_idx]))
print(f"Removing indices: {indices_to_remove}")

# Create masks for valid arrays
valid_mask = np.ones(result_mtx.shape[2], dtype=bool)
valid_mask[indices_to_remove] = False

# Remove invalid arrays and their corresponding angles
result_mtx = result_mtx[:,:,valid_mask]
angles = angles[valid_mask]

print(f"Removed {len(indices_to_remove)} arrays. New shape: {result_mtx.shape}")

# Apply Gaussian smoothing to each frame
sigma = 1  # Adjust this value to control the amount of smoothing
for i in range(result_mtx.shape[2]):
    result_mtx[:,:,i] = gaussian_filter(result_mtx[:,:,i], sigma=sigma)

final = {'img':result_mtx,'phi':angles}

#%%
plt.imshow(result_mtx[:,:,0], norm=LogNorm())
plt.colorbar()
plt.show()
#%%
result_mtx_processed = result_mtx.copy()

# Calculate q values
detector_pixel_size = 172e-6  # 75 microns in meters
sample_detector_distance = 10  # meters
wavelength = 1.24e-10  # angstroms to meters
num_pixels = result_mtx.shape[0]  # Assuming square detector

# Create pixel coordinate arrays
x = np.arange(-num_pixels//2, num_pixels//2)
X, Y = np.meshgrid(x, x)
R = np.sqrt(X**2 + Y**2)

# Calculate scattering angle and q
theta = np.arctan(detector_pixel_size * R / sample_detector_distance)
q = 4 * np.pi * np.sin(theta) / wavelength

# Number of slices you want to plot
num_slices_to_plot = 5
# Randomly select indices to plot
indices_to_plot = random.sample(range(result_mtx.shape[2]), num_slices_to_plot)

scale_exponent = 0
norm_max=True
subtract_background=False
save_mat=False
for num in range(result_mtx.shape[2]):
    # Calculate q values, assuming X, Y, R, theta, and q are already defined as needed above
    
    # Background subtraction using the median for each slice
    background = np.median(result_mtx[:, :, num])
    # Background estimation using Gaussian filtering
    #background = gaussian_filter(result_mtx[:, :, num], sigma=10)
    if subtract_background:
        result_mtx_selected = result_mtx[:, :, num] - background
    else:
        result_mtx_selected = result_mtx[:, :, num]

    # Remove NaN values
    result_mtx_selected = np.nan_to_num(result_mtx_selected)
    
    # Intensity normalization
    if norm_max:
        max_intensity = np.max(result_mtx_selected)
        if max_intensity != 0:
            result_mtx_selected /= max_intensity
    else:
        result_mtx_selected = result_mtx_selected

    # Scaling by q^4 (example)
    scaled_image = result_mtx_selected * q**scale_exponent

    # Plot only selected slices
    if num in indices_to_plot:
        plt.imshow(scaled_image, norm=LogNorm(vmin=1e-3))
        plt.colorbar(label='Normalized Intensity * q^4')
        plt.title(f'Normalized Q^4-scaled diffraction pattern at {angles[num]:.2f}°')
        plt.show()
        
    result_mtx_processed[:,:,num] = scaled_image

final = {'img':result_mtx_processed,'phi':angles}

if save_mat:
    savemat("C:\\Users\\b304014\\Software\\blee\\data\\temp\\"+save_string+"_processed.mat",final)
    print("Saved mat file")
else:
    print("Not saving mat file")
#%%

