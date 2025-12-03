# %%
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import h5py
from matplotlib.widgets import RectangleSelector, Button, Slider
import os
from scipy.io import loadmat
from tqdm import tqdm
import pandas as pd
import os
from scipy.io import savemat
from skimage.restoration import unwrap_phase
from scipy.ndimage import gaussian_filter
import matplotlib.colors as colors
import math
import random
#%matplotlib widget
def find_zero_arrays(array_3d):
    sums = np.sum(np.sum(array_3d, axis=0), axis=0)
    zero_indices = np.where(sums == 0)[0]
    return zero_indices.tolist()
def read_and_fft_tiff(
    tiff_path,
    roi=None,  # roi can be a tuple (rect) or dict (circle)
    frame=0,   # for multi-frame tiffs, select which frame to use
    vignette=True,  # apply a 2D Hann window if True
    mode='mean',
    center=None,
    h5_index_choice=None
):
    """
    Reads a TIFF file, selects an ROI (rectangular or circular), applies a vignette, and computes the FFT.

    Parameters:
        tiff_path (str): Path to the TIFF file.
        roi (tuple, dict, or None): Rectangle (start_row, end_row, start_col, end_col) or circle {'type': 'circle', 'center': (row, col), 'radius': r}. If None, use full image.
        frame (int): Frame index for multi-frame TIFFs.
        vignette (bool): Whether to apply a 2D Hann window before FFT.

    Returns:
        fft_result (np.ndarray): The complex FFT result.
        fft_magnitude (np.ndarray): The magnitude spectrum (log-scaled).
        image (np.ndarray): The image or ROI used (after vignette if applied).
    """
    # Read the TIFF file
    if tiff_path.endswith('.mat'):
        print('reading mat file')
        img = loadmat(tiff_path)['object_roi']
    elif tiff_path.endswith('.h5'):
        print('reading hdf5 file')
        with h5py.File(tiff_path, 'r') as f:
            img = f['projections'][()][h5_index_choice]
    else:
        print('reading tiff file')
        img = tifffile.imread(tiff_path)

    if img.ndim == 3:  # multi-frame
        img = img[frame]
    
    # Select ROI if specified
    if roi is not None:
        if isinstance(roi, tuple):
            # Rectangle
            start_row, end_row, start_col, end_col = roi
            img = img[start_row:end_row, start_col:end_col]
        elif isinstance(roi, dict):
            if roi.get('type') == 'circle':
                cy, cx = roi['center']
                r = roi['radius']
                y1, y2 = int(cy - r), int(cy + r)
                x1, x2 = int(cx - r), int(cx + r)
                img = img[max(0, y1):min(img.shape[0], y2), max(0, x1):min(img.shape[1], x2)]
                # Apply circular mask
                h, w = img.shape
                Y, X = np.ogrid[:h, :w]
                mask = (Y - (cy - y1))**2 + (X - (cx - x1))**2 <= r**2
                img = img * mask
            elif roi.get('type') == 'centered_rect':
                cy, cx = roi['center']
                size = roi.get('size', (256, 256))
                Ny, Nx = size
                y1 = int(round(cy - Ny // 2))
                y2 = int(round(cy + (Ny + 1) // 2))
                x1 = int(round(cx - Nx // 2))
                x2 = int(round(cx + (Nx + 1) // 2))
                # Clamp to image bounds
                y1 = max(0, y1)
                y2 = min(img.shape[0], y2)
                x1 = max(0, x1)
                x2 = min(img.shape[1], x2)
                img = img[y1:y2, x1:x2]
            else:
                raise ValueError('ROI dict must have type "circle" or "centered_rect"')
        else:
            raise ValueError('ROI must be a tuple (rectangle) or dict')
    
    # if center is not None:
    #     img = img[center[0]-256:center[0]+256, center[1]-256:center[1]+256]
        
    # Apply vignette (2D Hann window)
    if vignette:
        h, w = img.shape
        win_row = np.hanning(h)
        win_col = np.hanning(w)
        window = np.outer(win_row, win_col)
        img = img * window



    # # Step 1: Extract amplitude and phase
    A = np.abs(img)
    phi = np.angle(img)

    # # Step 2: Unwrap phase
    phi_wrapped = np.angle(img)
    phi_unwrapped = unwrap_phase(phi_wrapped)
    
    # # Step 2.5: Subtract smooth or constant background from amplitude
    # A_cleaned = A - np.min(A)  # or Gaussian blur, see below
    # A_cleaned = np.clip(A_cleaned, 0, None)  # Remove negative artifacts
    
    # background_estimate = gaussian_filter(A, sigma=20)
    # A_cleaned = A - background_estimate
    # A_cleaned = np.clip(A_cleaned, 0, None)
    
    # #A_cleaned=A
    # # Step 3: Recombine
    obj_clean = A * np.exp(1j * phi_unwrapped)

    
    # img= A * np.exp(1j * phi_unwrapped)
    img = obj_clean

    # Compute FFT
    if mode=='mean':
        fft_result = np.fft.fftshift(np.fft.fft2(img-np.mean(img)))
    elif mode=='max':
        fft_result = np.fft.fftshift(np.fft.fft2(img-np.max(img)))
    elif mode=='min':
        fft_result = np.fft.fftshift(np.fft.fft2(img-np.min(img)))
    else:
        fft_result = np.fft.fftshift(np.fft.fft2(img))
    # plt.figure(1,2,figsize=(10,5))
    # plt.subplot(1,2,1)
    # plt.imshow(, cmap='gray')
    # plt.subplot(1,2,2)
    # plt.imshow(fft_result, cmap='jet')
    # plt.show()
    fft_magnitude = np.abs(fft_result)**2
    fft_magnitude = fft_magnitude/np.max(fft_magnitude)

    return fft_result, fft_magnitude, img

def scan_fft_over_image(
    img,
    roi_height,
    roi_width,
    step_y,
    step_x,
    vignette=True,
    show_progress=True
):
    """
    Scan the image with a fixed-size ROI and compute FFTs at each position.
    Returns a list of (y, x, fft_magnitude) for each ROI.
    """
    h, w = img.shape
    results = []
    y_positions = range(0, h - roi_height + 1, step_y)
    x_positions = range(0, w - roi_width + 1, step_x)
    for i, y in enumerate(y_positions):
        # Snake pattern: reverse x direction every other row
        if i % 2 == 0:
            xs = x_positions
        else:
            xs = reversed(list(x_positions))
        for x in xs:
            roi = img[y:y+roi_height, x:x+roi_width]
            if vignette:
                win_row = np.hanning(roi_height)
                win_col = np.hanning(roi_width)
                window = np.outer(win_row, win_col)
                roi = roi * window
            fft_result = np.fft.fftshift(np.fft.fft2(roi))
            fft_mag = np.log1p(np.abs(fft_result))
            results.append({'y': y, 'x': x, 'fft_mag': fft_mag})
        if show_progress:
            print(f'Scanned row {i+1}/{len(list(y_positions))}')
    return np.array(results)

#%%
base_dir = '/net/micdata/data2/12IDC/'
exp_dir = '2025_Jul/'
recon_dir = 'results/'
sample_dir = 'ZC6_3D_/'
scan_num = 945
#recon_path = 'roi1_Ndp512/MLc_L1_p10_g1000_Ndp256_mom0.5_pc0_maxPosError500nm_bg0.1_vi_mm/'
iterations=2000
vignette=False
mode=None#'min'#'mean'#'min'

h5_path = f'{base_dir}/{exp_dir}/{recon_dir}/{sample_dir}/fly{scan_num:03d}/data_roi0_Ndp256_para.hdf5'
#h5_path = f'{base_dir}/{exp_dir}/{recon_dir}/{sample_dir}/fly{scan_num:03d}/data_roi1_Ndp512_para.hdf5'
with h5py.File(h5_path, 'r') as f:
    sdd=f['detector_distance'][()] #m
    angle=f['angle'][()] #deg
    energy=f['energy'][()] #keV
Ndp=128
#wavelength=1.239842e-10 # m
wavelength = (12.3984/energy)*10**(-10) # nm
delta_p=172e-6 # m
pixel_size=wavelength*sdd/(Ndp*delta_p)
print(f'pixel size: {pixel_size*1e9} nm')
print(f'sdd: {sdd} m')
print(f'angle: {angle} deg')
print(f'energy: {energy} keV')
print(f'wavelength: {wavelength} nm')
print(f'delta_p: {delta_p} m')
print(f'Ndp: {Ndp}')

# if Ndp==256:
#     #recon_path+='MLc_L1_p10_g100_Ndp256_mom0.5_pc800_maxPosError500nm_bg0.1_vp4_vi_mm/'
#     recon_path+='/MLc_L1_p10_g200_Ndp256_mom0.5_pc800_model_scale_asymmetry_rotation_shear_vp4_vi_mm/'
    
# #tiff_path = f'{base_dir}/{exp_dir}/{recon_dir}/{sample_dir}/fly{scan_num:04d}/{recon_path}/O_phase_roi/O_phase_roi_Niter{iterations}.tiff'
# tiff_path = f'{base_dir}/{exp_dir}/{recon_dir}/{sample_dir}/fly{scan_num:03d}/{recon_path}/O_phase_roi/O_phase_roi_Niter{iterations}.tiff'


#tiff_path = f'{base_dir}/{exp_dir}/{recon_dir}/{sample_dir}/fly{scan_num:04d}/{recon_path}/Niter{iterations}.mat'
tiff_path = 'misc/ZC6_3D_/ZC6_3D_phase_projections_Ndp128_LSQML_c1000_m0.5_gaussian_p5_cp_mm_opr2_ic_pc1_g_ul2_Niter1000.h5'
print(tiff_path)

# with h5py.File(tiff_path, 'r') as f:
#     print(f.keys())
#     plt.figure()
#     plt.imshow(np.angle(f['projections'][()][96]), cmap='gray')
#     plt.show()
#%%

# combined_h5_path='/home/beams/PTYCHOSAXS/deconvolutionNN/src/utils/combined_diffraction_patterns_TEST.h5'
# with h5py.File(combined_h5_path, 'r') as f:
#     dp = f['deconvolved'][f'scan_{scan_num}'][()]
# plt.imshow(dp, cmap='jet',norm=colors.LogNorm())
# plt.colorbar()
# plt.show()

#%%
roi = {'type': 'centered_rect', 'center': (170, 300), 'size': (300, 300)}
fft_results=[]
for i in tqdm(range(90,91)):
    # Full image FFT
    fft_result, fft_mag, img = read_and_fft_tiff(tiff_path,vignette=vignette,mode=mode,roi=roi,h5_index_choice=i)
    fft_results.append(fft_result)

fig,ax=plt.subplots(1,2,figsize=(15,5))
plt.title('FFT Magnitude')
ax[0].imshow(fft_mag, cmap='jet',norm=colors.LogNorm())
ax[1].imshow(np.abs(img), cmap='gray')
ax[1].set_title('Phase')
plt.show()

#%%





#%%
with h5py.File(tiff_path, 'r') as f:
    scan_numbers=f['scan_numbers'][()]
    angles=f['angles'][()]




# --- Now allocate result_mtx with the correct shape ---
# Only allocate space for scans we know exist in CSV
size = len(scan_numbers)
result_mtx = np.zeros(fft_results[0].shape + (size,))


for i in range(0,size):
    try:
        print(f"Processing scan {i}")
        data = fft_results[i]
        result_mtx[..., i] = data
        angles[i] = angle[scan_num]
    except Exception as e:
        print(f"Error processing scan {scan_num}: {e}")
        continue

























#%%
pixel_size=39.3 #nm
with h5py.File('/net/micdata/data2/12IDC/2025_Jul/misc/combined_SZC2SAXS344_00001_64x64.h5', 'r') as f:
    combined_data=f['combined_data'][()]
# with h5py.File('/net/micdata/data2/12IDC/2025_Jul/misc/combined_SZC6_3DSAXS_1120_00003.h5', 'r') as f:
#     combined_data=f['combined_data'][()]
    
angles=np.linspace(0,180,len(combined_data))    
#result_mtx = np.zeros(fft_results[0].shape + (len(combined_data),))
result_mtx=np.zeros(combined_data[0].shape + (len(combined_data),))

pxR=pixel_size*1e9 #nm
pxQ = (2*np.pi/pxR/combined_data[0].shape[0]) #nm^-1

# Create distance matrix from center
center_y, center_x = result_mtx.shape[0]//2, result_mtx.shape[1]//2
y, x = np.ogrid[-center_y:result_mtx.shape[0]-center_y, -center_x:result_mtx.shape[1]-center_x]
distance_matrix = np.sqrt(x*x + y*y) * pxQ

for i in range(0,len(combined_data)):
    try:
        print(f"Processing scan {i}")
        data = combined_data[i]
        # Scale the data by distance from center in q-space
        result_mtx[..., i] = data* distance_matrix**(2)
    except Exception as e:
        print(f"Error processing scan {i}: {e}")
        continue

#%%
fig,ax=plt.subplots(1,2,figsize=(15,5))
ri=random.randint(0,len(combined_data)-1)
print(ri)
im1=ax[0].imshow(result_mtx[:,:,ri]-np.mean(result_mtx[:,:,ri]), cmap='jet',norm=colors.LogNorm())
im2=ax[1].imshow(combined_data[ri], cmap='jet',norm=colors.LogNorm())
plt.colorbar(im1)
plt.colorbar(im2)
plt.show()
#%%



# --- Remove all-zero arrays and max sum array ---
zero_idx = find_zero_arrays(result_mtx)
print(f"Found zero arrays at indices: {zero_idx}")

sums = np.array([np.sum(result_mtx[:, :, i]) for i in range(result_mtx.shape[2])])
max_idx = np.argmax(sums)
print(f"Found maximum sum at index: {max_idx}")

indices_to_remove = list(set(zero_idx + [max_idx]))
print(f"Removing indices: {indices_to_remove}")

valid_mask = np.ones(result_mtx.shape[2], dtype=bool)
valid_mask[indices_to_remove] = False

result_mtx = result_mtx[:, :, valid_mask]
angles = angles[valid_mask]

#subtract mean from each pattern in result_mtx
result_mtx = result_mtx - np.mean(result_mtx, axis=(0,1))

print(f"Removed {len(indices_to_remove)} arrays. New shape: {result_mtx.shape}")
mat_file = '2025_Jul_ZC2SAXS344_3D_SAXS_TEST_64x64mean.mat'
#%%
# --- Save to .mat file ---
final = {'img': result_mtx, 'phi': angles}
savemat(mat_file, final)
print(f"Saved to {mat_file}")
# %%
from scipy.io import loadmat
load_mat_file = '2025_Jul_ZC2SAXS344_3D_SAXS_TEST_64x64mean.mat'
data = loadmat(load_mat_file)
print(data['phi'].shape)
print(data['img'].shape)




#%%
# Find center by looking at peaks in a random image
ri = 225#random.randint(0, len(data['phi'])-1)
img = data['img'][:,:,ri]

# Find peaks above threshold
threshold = np.mean(img) + 2*np.std(img)  # Adjust threshold as needed
peaks = np.where(img > threshold)

# Calculate center as average of peak positions
center_y = int(np.mean(peaks[0]))
center_x = int(np.mean(peaks[1]))

print(f"Estimated center: ({center_x}, {center_y})")

# Plot image with center marked
plt.figure(figsize=(10,8))
plt.imshow(img, cmap='jet', norm=colors.LogNorm())
plt.plot(center_x, center_y, 'r+', markersize=20, label='Estimated Center')
plt.colorbar()
plt.legend()
plt.title(f'Image {ri} with Estimated Center')
plt.show()

# Verify symmetry by plotting horizontal and vertical profiles through center
plt.figure(figsize=(12,4))
plt.subplot(121)
plt.plot(img[center_y,:], label='Horizontal Profile')
plt.axvline(center_x, color='r', linestyle='--', label='Center')
plt.legend()
plt.title('Horizontal Profile Through Center')

plt.subplot(122) 
plt.plot(img[:,center_x], label='Vertical Profile')
plt.axvline(center_y, color='r', linestyle='--', label='Center')
plt.legend()
plt.title('Vertical Profile Through Center')
plt.show()




# %%































#%%






























#%%
plt.figure(figsize=(15,15))
tiff_path='/net/micdata/data2/12IDC/2025_Jul/ptychi_recons/S0318_BACKUP/S0318/Ndp128_LSQML_s1000_gaussian_p5_cp_mm_opr2_ic_pc1_g_ul2/object_ph/object_ph_Niter1000.tiff'
TIFF_IMAGE = tifffile.imread(tiff_path)

# Plot real space projection and its FFT
projection_test = TIFF_IMAGE
fig_proj_combined, ax = plt.subplots(1, 2, figsize=(20, 10))

# Plot real space projection with inverted colors
ax[0].imshow(projection_test, cmap='gray_r')  # Using gray_r for inverted grayscale
ax[0].set_title('Real Space Projection')
ax[0].axis('off')

# Calculate scale bar dimensions
pixel_size=39.3
scalebar_length_nm = 2000  # Length in nm
scalebar_length_pixels = int(scalebar_length_nm / pixel_size)  # Convert to pixels
scalebar_width_pixels = 5  # Fixed width in pixels

# Position scale bar in bottom right corner
padding = 20
scalebar_x = projection_test.shape[1] - scalebar_length_pixels - padding
scalebar_y = projection_test.shape[0] - padding - scalebar_width_pixels

# Add scale bar rectangle
rect = patches.Rectangle(
    (scalebar_x, scalebar_y),
    scalebar_length_pixels,
    scalebar_width_pixels,
    facecolor='white',
    edgecolor='none'
)
ax[0].add_patch(rect)

# Add scale bar label
ax[0].text(
    scalebar_x + scalebar_length_pixels/2,
    scalebar_y - 2*scalebar_width_pixels,
    f'{int(scalebar_length_nm/1000)} μm',
    color='white',
    ha='center',
    va='top',
    fontsize=12
)

# Plot FFT
fft_result = np.abs(np.fft.fftshift(np.fft.fft2(projection_test)))**2
fft_result_norm = fft_result / np.max(fft_result)  # Normalize to avoid log(0)
ax[1].imshow(fft_result_norm, cmap='jet', norm=colors.LogNorm(vmin=1e-10, vmax=1))
ax[1].set_title('FFT of Real Space Projection')
ax[1].axis('off')

plt.tight_layout()
plt.show()

#%%
# ROI FFT (e.g., rows 100:200, cols 150:250)
if Ndp==256:
    #roi = (300, 400, 325, 425) #256
    roi = {'type': 'centered_rect', 'center': (340, 340), 'size': (400, 400)}
else:
    #roi = (300//2, 400//2, 250//2, 350//2) #128
    roi = {'type': 'centered_rect', 'center': (170, 170), 'size': (200, 200)}
fft_result_roi, fft_mag_roi, img_roi = read_and_fft_tiff(tiff_path, roi=roi,vignette=vignette,mode=mode)

plt.figure()
plt.subplot(1,2,1)
plt.title('Image with ROI')
if tiff_path.endswith('.mat'):
    plt.imshow(np.angle(img), cmap='gray')
else:
    plt.imshow(img, cmap='gray')

# Overlay ROI rectangle
if isinstance(roi, tuple):
    start_row, end_row, start_col, end_col = roi
    rect_xy = (start_col, start_row)
    rect_width = end_col - start_col
    rect_height = end_row - start_row
elif isinstance(roi, dict) and roi.get('type') == 'centered_rect':
    cy, cx = roi['center']
    Ny, Nx = roi.get('size', (256, 256))
    rect_x = int(round(cx - Nx // 2))
    rect_y = int(round(cy - Ny // 2))
    rect_width = Nx
    rect_height = Ny
    rect_xy = (rect_x, rect_y)
else:
    rect_xy = (0, 0)
    rect_width = 0
    rect_height = 0

roi_rect = patches.Rectangle(
    rect_xy, rect_width, rect_height,
    linewidth=2, edgecolor='r', facecolor='none'
)
plt.gca().add_patch(roi_rect)

plt.subplot(1,2,2)
plt.title('FFT Magnitude')
plt.imshow(fft_mag, cmap='jet',norm=colors.LogNorm())
plt.colorbar()
plt.show()

plt.figure()
plt.subplot(1,2,1)
plt.title('ROI')
if tiff_path.endswith('.mat'):
    plt.imshow(np.angle(img_roi), cmap='gray')
else:
    plt.imshow(img_roi, cmap='gray')
plt.subplot(1,2,2)
plt.title('FFT Magnitude')
if tiff_path.endswith('.mat'):
    plt.imshow(fft_mag_roi, cmap='jet',norm=colors.LogNorm())
else:
    plt.imshow(fft_mag_roi, cmap='jet',norm=colors.LogNorm())
plt.colorbar()
plt.show()








#%%

recon_pixel_size = pixel_size[0]
pxQ_FFT = 2*np.pi/recon_pixel_size/fft_mag_roi.shape[0]
pxQ_DP = 4*np.pi/wavelength[0]*np.sin(1/2*np.arctan(delta_p/sdd[0]))

# Create extent values based on pixel sizes
dp_height, dp_width = dp.shape
dp_extent = [-dp_width*pxQ_DP/2*1e-10, dp_width*pxQ_DP/2*1e-10, 
             -dp_height*pxQ_DP/2*1e-10, dp_height*pxQ_DP/2*1e-10]

fft_height, fft_width = fft_mag_roi.shape
fft_extent = [-fft_width*pxQ_FFT/2*1e-10, fft_width*pxQ_FFT/2*1e-10,
              -fft_height*pxQ_FFT/2*1e-10, fft_height*pxQ_FFT/2*1e-10]

plt.figure()
plt.xlabel(r'$q_{\perp,x}$ ($\AA^{-1}$)', fontsize=16, fontname='serif')
plt.ylabel(r'$q_{\perp,y}$ ($\AA^{-1}$)', fontsize=16, fontname='serif')  
plt.imshow(fft_mag_roi-np.min(fft_mag_roi), extent=fft_extent, cmap='jet', norm=colors.LogNorm(), alpha=1.0)
plt.imshow(dp, extent=dp_extent, cmap='jet', norm=colors.LogNorm(), alpha=0.3)
plt.xticks([-0.01,-0.005,0,0.005,0.01],fontsize=12)
plt.yticks([-0.01,-0.005,0,0.005,0.01],fontsize=12)
ax = plt.gca()
linewidth=2.5
ax.spines['bottom'].set_linewidth(linewidth)
ax.spines['top'].set_linewidth(linewidth)
ax.spines['left'].set_linewidth(linewidth)
ax.spines['right'].set_linewidth(linewidth)
ax.tick_params(width=linewidth, length=6)
plt.show()





#%%

# Compute sum of FFTs over all ROIs of a projection

# Example usage:
img = tifffile.imread(tiff_path)
if img.ndim == 3:
    img = img[0]
roi_height, roi_width = 200, 200
step_y, step_x = 50, 50
fft_results = scan_fft_over_image(img, roi_height, roi_width, step_y, step_x, vignette=True)
print(f"Total FFTs computed: {len(fft_results)}")
fft_total=np.zeros(fft_results[0]['fft_mag'].shape) 
for i in range(0,len(fft_results)):
    fft_total+=fft_results[i]['fft_mag']
plt.figure()
plt.imshow(np.log1p(fft_total), cmap='jet')
plt.show()













#%%
# INTERACTIVE ROI SELECTION AND FFT CALCULATION

# Interactive ROI selection and FFT calculation
class InteractiveProjectionFFT:
    def __init__(self, tiff_paths, vignette=True):
        """
        tiff_paths: list of tiff file paths (one per projection)
        """
        self.tiff_paths = tiff_paths
        self.vignette = vignette
        self.current_proj = 0
        self.roi = None
        self.roi_type = 'rectangle'
        self.circle_selector_active = False
        self.circle_artist = None
        self.circle_params = None

        # Load first image
        self.img_orig = tifffile.imread(self.tiff_paths[self.current_proj])
        if self.img_orig.ndim == 3:
            self.img_orig = self.img_orig[0]
        self.img = self.apply_vignette(self.img_orig) if vignette else self.img_orig

        # Set up figure and widgets
        self.fig, (self.ax_img, self.ax_fft) = plt.subplots(1, 2, figsize=(10, 5))
        plt.subplots_adjust(bottom=0.18)
        self.img_disp = self.ax_img.imshow(self.img, cmap='gray')
        self.ax_img.set_title('Select ROI')
        self.fft_disp = self.ax_fft.imshow(np.zeros_like(self.img), cmap='jet')
        self.ax_fft.set_title('FFT Magnitude (ROI)')

        # Rectangle selector
        self.rect_selector = RectangleSelector(
            self.ax_img, self.onselect_rectangle, useblit=True,
            button=[1], minspanx=5, minspany=5, spancoords='pixels', interactive=True
        )
        self.rect_selector.set_active(True)
        self.cid_circle = self.fig.canvas.mpl_connect('button_press_event', self.on_circle_press)
        self.cid_circle_release = self.fig.canvas.mpl_connect('button_release_event', self.on_circle_release)
        self.cid_circle_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_circle_motion)
        self.circle_press = None
        self.roi_rect_patch = None

        # Close button
        close_ax = self.fig.add_axes([0.85, 0.01, 0.1, 0.05])
        self.close_button = Button(close_ax, 'Close')
        self.close_button.on_clicked(self.close_figure)

        # Toggle ROI type button
        toggle_ax = self.fig.add_axes([0.7, 0.01, 0.13, 0.05])
        self.toggle_button = Button(toggle_ax, 'Toggle ROI (Rect/Circle)')
        self.toggle_button.on_clicked(self.toggle_roi_type)

        # Projection slider
        slider_ax = self.fig.add_axes([0.15, 0.01, 0.5, 0.03])
        self.slider = Slider(slider_ax, 'Projection', 0, len(self.tiff_paths)-1, valinit=0, valstep=1)
        self.slider.on_changed(self.on_slider_change)

        plt.tight_layout()
        plt.show()

    def apply_vignette(self, img):
        h, w = img.shape
        win_row = np.hanning(h)
        win_col = np.hanning(w)
        window = np.outer(win_row, win_col)
        return img * window

    def on_slider_change(self, val):
        idx = int(val)
        self.current_proj = idx
        try:
            self.img_orig = tifffile.imread(self.tiff_paths[self.current_proj])
        except Exception as e:
            #print(f"Failed to read image {self.tiff_paths[self.current_proj]}: {e}")
            self.img_orig = np.ones(self.img_orig.shape)
            
        if self.img_orig.ndim == 3:
            self.img_orig = self.img_orig[0]
        self.img = self.apply_vignette(self.img_orig) if self.vignette else self.img_orig
        self.img_disp.set_data(self.img)
        self.ax_img.set_title(f'Select ROI (Projection {self.current_proj})')
        # Remove ROI overlays and FFT
        if self.roi_rect_patch:
            self.roi_rect_patch.remove()
            self.roi_rect_patch = None
        if self.circle_artist:
            self.circle_artist.remove()
            self.circle_artist = None
        self.fft_disp.set_data(np.zeros_like(self.img))
        self.fig.canvas.draw_idle()

    def toggle_roi_type(self, event):
        if self.roi_type == 'rectangle':
            self.roi_type = 'circle'
            self.rect_selector.set_active(False)
            self.circle_selector_active = True
            self.ax_img.set_title('Select ROI (Circle: click center, drag to edge)')
        else:
            self.roi_type = 'rectangle'
            self.rect_selector.set_active(True)
            self.circle_selector_active = False
            self.ax_img.set_title('Select ROI (Rectangle)')
        self.fig.canvas.draw_idle()

    def close_figure(self, event):
        plt.close(self.fig)

    def onselect_rectangle(self, eclick, erelease):
        if self.roi_type != 'rectangle':
            return
        x1, y1 = int(np.floor(eclick.xdata)), int(np.floor(eclick.ydata))
        x2, y2 = int(np.floor(erelease.xdata)), int(np.floor(erelease.ydata))
        # Ensure proper order
        start_row, end_row = sorted([y1, y2])
        start_col, end_col = sorted([x1, x2])
        # Make end exclusive (add 1)
        end_row += 1
        end_col += 1
        # Clamp to image bounds
        start_row = max(0, start_row)
        end_row = min(self.img_orig.shape[0], end_row)
        start_col = max(0, start_col)
        end_col = min(self.img_orig.shape[1], end_col)
        # Extract ROI from original image, then apply vignette and FFT
        img_roi = self.img_orig[start_row:end_row, start_col:end_col]
        self.roi = img_roi
        if self.vignette:
            h, w = img_roi.shape
            win_row = np.hanning(h)
            win_col = np.hanning(w)
            window = np.outer(win_row, win_col)
            img_roi = img_roi * window
        fft_result = np.fft.fftshift(np.fft.fft2(img_roi))
        fft_mag_roi = np.log1p(np.abs(fft_result))
        # Update ROI rectangle
        if self.roi_rect_patch:
            self.roi_rect_patch.remove()
        self.roi_rect_patch = patches.Rectangle(
            (start_col, start_row), end_col - start_col, end_row - start_row,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        self.ax_img.add_patch(self.roi_rect_patch)
        # Remove circle if present
        if self.circle_artist:
            self.circle_artist.remove()
            self.circle_artist = None
        # Update FFT plot
        self.fft_disp.remove()  # Remove the old image
        self.fft_disp = self.ax_fft.imshow(
            fft_mag_roi, cmap='jet', origin='upper',
            vmin=np.min(fft_mag_roi), vmax=np.max(fft_mag_roi)
        )
        self.ax_fft.set_title('FFT Magnitude (ROI)')
        self.ax_fft.set_xlim(0, fft_mag_roi.shape[1])
        self.ax_fft.set_ylim(fft_mag_roi.shape[0], 0)
        self.fig.canvas.draw_idle()

    def on_circle_press(self, event):
        if not self.circle_selector_active or event.inaxes != self.ax_img:
            return
        self.circle_press = (event.xdata, event.ydata)
        if self.circle_artist:
            self.circle_artist.remove()
            self.circle_artist = None

    def on_circle_release(self, event):
        if not self.circle_selector_active or self.circle_press is None or event.inaxes != self.ax_img:
            return
        x0, y0 = self.circle_press
        x1, y1 = event.xdata, event.ydata
        if x0 is None or y0 is None or x1 is None or y1 is None:
            return
        # Calculate center and radius
        cx, cy = x0, y0
        radius = np.hypot(x1 - x0, y1 - y0)
        self.circle_params = (cy, cx, radius)
        # Extract circular ROI from original image
        y1b = int(max(0, cy - radius))
        y2b = int(min(self.img_orig.shape[0], cy + radius))
        x1b = int(max(0, cx - radius))
        x2b = int(min(self.img_orig.shape[1], cx + radius))
        img_roi = self.img_orig[y1b:y2b, x1b:x2b]
        h, w = img_roi.shape
        Y, X = np.ogrid[:h, :w]
        mask = (Y - (cy - y1b))**2 + (X - (cx - x1b))**2 <= radius**2
        img_roi = img_roi * mask
        self.roi = img_roi
        if self.vignette:
            win_row = np.hanning(h)
            win_col = np.hanning(w)
            window = np.outer(win_row, win_col)
            img_roi = img_roi * window
        fft_result = np.fft.fftshift(np.fft.fft2(img_roi))
        fft_mag_roi = np.log1p(np.abs(fft_result))
        # Remove rectangle if present
        if self.roi_rect_patch:
            self.roi_rect_patch.remove()
            self.roi_rect_patch = None
        # Draw circle
        if self.circle_artist:
            self.circle_artist.remove()
        self.circle_artist = patches.Circle((cx, cy), radius, linewidth=2, edgecolor='r', facecolor='none')
        self.ax_img.add_patch(self.circle_artist)
        # Update FFT plot
        self.fft_disp.remove()
        self.fft_disp = self.ax_fft.imshow(
            fft_mag_roi, cmap='jet', origin='upper',
            vmin=np.min(fft_mag_roi), vmax=np.max(fft_mag_roi)
        )
        self.ax_fft.set_title('FFT Magnitude (ROI)')
        self.ax_fft.set_xlim(0, fft_mag_roi.shape[1])
        self.ax_fft.set_ylim(fft_mag_roi.shape[0], 0)
        self.fig.canvas.draw_idle()
        self.circle_press = None

    def on_circle_motion(self, event):
        if not self.circle_selector_active or self.circle_press is None or event.inaxes != self.ax_img:
            return
        x0, y0 = self.circle_press
        x1, y1 = event.xdata, event.ydata
        if x0 is None or y0 is None or x1 is None or y1 is None:
            return
        cx, cy = x0, y0
        radius = np.hypot(x1 - x0, y1 - y0)
        # Remove previous circle
        if self.circle_artist:
            self.circle_artist.remove()
        self.circle_artist = patches.Circle((cx, cy), radius, linewidth=2, edgecolor='r', facecolor='none', alpha=0.5)
        self.ax_img.add_patch(self.circle_artist)
        self.fig.canvas.draw_idle()

# # To use the interactive plotter, uncomment the following line:
# InteractiveFFT(tiff_path, vignette=vignette)



scan_numbers = np.arange(5004,5369,1)
tiff_paths = [f'/scratch/2025_Feb/results/ZCB_9_3D_/fly{scan:04d}/roi0_Ndp256/MLc_L1_p10_gInf_Ndp128_mom0.5_pc0_maxPosError500nm_bg0.1_vi_mm/O_phase_roi/O_phase_roi_Niter1000.tiff' for scan in scan_numbers]
InteractiveProjectionFFT(tiff_paths, vignette=True)















# %%

# BATCH FFT OF ALL SCAN PROJECTIONS TO H5

def pad_to_shape(img, target_shape):
    """Pad a 2D array to the target shape, centered."""
    y, x = img.shape
    ty, tx = target_shape
    pad_y = max(0, ty - y)
    pad_x = max(0, tx - x)
    pad_width = ((pad_y//2, pad_y - pad_y//2), (pad_x//2, pad_x - pad_x//2))
    return np.pad(img, pad_width, mode='constant')

def batch_ffts_to_h5(scan_numbers, base_dir, exp_dir, recon_dir, sample_dir, recon_path, iterations, output_h5_path,roi, vignette=True, save_complex=False,mode=None):
    """
    Calculate FFTs for a list of scan numbers and save the FFT magnitudes (and optionally complex FFTs) to an HDF5 file.
    All images are padded to the largest shape found among all scans before FFT.
    """
    output_dir = os.path.dirname(output_h5_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    # # First pass: find largest shape
    # max_shape = [0, 0]
    # img_shapes = {}
    # for scan_num in scan_numbers:
    #     #tiff_path = f'{base_dir}/{exp_dir}/{recon_dir}/{sample_dir}/fly{scan_num:04d}/{recon_path}/O_phase_roi/O_phase_roi_Niter{iterations}.tiff'
    #     tiff_path = f'{base_dir}/{exp_dir}/{recon_dir}/{sample_dir}/fly{scan_num:04d}/{recon_path}/Niter{iterations}.mat'
    #     try:
    #         _, _, img = read_and_fft_tiff(tiff_path, vignette=False,center=(302,358))
    #         img_shapes[scan_num] = img.shape
    #         print(img.shape)
    #         if img.shape[0] > max_shape[0]:
    #             max_shape[0] = img.shape[0]
    #         if img.shape[1] > max_shape[1]:
    #             max_shape[1] = img.shape[1]
    #     except Exception as e:
    #         print(f"(Shape scan) Failed for scan {scan_num}: {e}")
    # max_shape = tuple(max_shape)
    # print(f"Padding all images to shape: {max_shape}")
    # Second pass: process and save
    with h5py.File(output_h5_path, 'w') as f:
        for scan_num in scan_numbers:
            #tiff_path = f'{base_dir}/{exp_dir}/{recon_dir}/{sample_dir}/fly{scan_num:04d}/{recon_path}/O_phase_roi/O_phase_roi_Niter{iterations}.tiff'
            tiff_path = f'{base_dir}/{exp_dir}/{recon_dir}/{sample_dir}/fly{scan_num:04d}/{recon_path}/Niter{iterations}.mat'
            print(tiff_path)
            try:
                fft_result,fft_mag,img=read_and_fft_tiff(tiff_path, roi=roi,vignette=vignette,mode=mode)
                grp = f.create_group(f'scan_{scan_num}')
                grp.create_dataset('fft_magnitude', data=fft_mag, compression='gzip')
                if save_complex:
                    grp.create_dataset('fft_complex', data=fft_result, compression='gzip')
                #grp.create_dataset('image', data=img, compression='gzip')
                print(f"Processed scan {scan_num}")
                # fig,ax = plt.subplots(1,2)
                # ax[0].imshow(np.angle(img), cmap='gray')
                # ax[1].imshow(fft_mag, cmap='jet',norm=colors.LogNorm())
                # plt.show()
            except Exception as e:
                print(f"Failed for scan {scan_num}: {e}")

# %%
vignette=True
mode=None#'mean'#'min'
scan_numbers = np.arange(5004,5369,1) #5004 start scan number
#roi = {'type': 'centered_rect', 'center': (340, 340), 'size': (400, 400)}
roi = {'type': 'centered_rect', 'center': (170, 170), 'size': (200, 200)}
batch_ffts_to_h5(
    scan_numbers=scan_numbers,
    base_dir='/scratch/',
    exp_dir='2025_Feb/',
    recon_dir='results/',
    sample_dir='ZCB_9_3D_/',
    recon_path='roi0_Ndp256/MLc_L1_p10_gInf_Ndp128_mom0.5_pc0_maxPosError500nm_bg0.1_vi_mm/',#MLc_L1_p10_g100_Ndp256_mom0.5_pc800_maxPosError500nm_bg0.1_vp4_vi_mm/',
    iterations=1000,
    output_h5_path='fft_results_128_complex_object_roi_TEST.h5',
    vignette=True,
    save_complex=False,
    mode=None,
    roi=roi
)













#ZCB_9_3D
# csv_path = '/net/micdata/data2/12IDC/2025_Feb/indices_phi_samplename_all.txt'
# df = pd.read_csv(csv_path, comment='#', names=['Angle', 'y_shift', 'x_shift', 'scanNo'])

# CONSTRUCT A MAT FILE OF FFTs FROM H5 FILE


# %%

import h5py
import numpy as np
import pandas as pd
from scipy.io import savemat

def find_zero_arrays(array_3d):
    sums = np.sum(np.sum(array_3d, axis=0), axis=0)
    zero_indices = np.where(sums == 0)[0]
    return zero_indices.tolist()

# --- User parameters ---
h5_file = "fft_results_128_complex_object_roi_TEST.h5"
mat_file = "fft_results_128_complex_object_roi_TEST.mat"
scan_num_start = 5004 # start scan number
scan_num_end = 5369  # exclusive
size = scan_num_end - scan_num_start
skip_indices = []  # e.g., [1,2,3]

# --- Load angles from CSV ---
csv_path = '/net/micdata/data2/12IDC/2025_Feb/misc/ZCB_9_3D_/ZCB9_3D_alignment_shifts_28nm.txt'
df = pd.read_csv(csv_path, comment='#', names=['Angle', 'y_shift', 'x_shift', 'scanNo'])
df['scanNo'] = df['scanNo'].astype(int)

# Filter df to only include scans within our range
df = df[(df['scanNo'] >= scan_num_start) & (df['scanNo'] < scan_num_end)]
valid_scan_numbers = set(df['scanNo'])
print(f"Found {len(valid_scan_numbers)} scans with angles in CSV file")

# Build a mapping from scanNo to angle
scan_to_angle = dict(zip(df['scanNo'], df['Angle']))

# --- Determine FFT shape from the first available scan ---
with h5py.File(h5_file, 'r') as f:
    # Find first scan that exists in both HDF5 and CSV
    for scan_num in valid_scan_numbers:
        try:
            data = f[f'scan_{scan_num}']['fft_magnitude'][()]
            fft_shape = data.shape
            print(f"Found first valid scan: {scan_num}")
            break
        except Exception:
            continue
    else:
        raise RuntimeError("No valid FFTs found in the HDF5 file.")

print(f"FFT shape: {fft_shape}")

# --- Now allocate result_mtx with the correct shape ---
# Only allocate space for scans we know exist in CSV
size = len(valid_scan_numbers)
result_mtx = np.zeros(fft_shape + (size,))
angles = np.zeros((size,))
valid_idx = 0

with h5py.File(h5_file, 'r') as f:
    for scan_num in sorted(valid_scan_numbers):
        try:
            print(f"Processing scan {scan_num}")
            data = f[f'scan_{scan_num}']['fft_magnitude'][()]
            result_mtx[..., valid_idx] = data
            angles[valid_idx] = scan_to_angle[scan_num]  # This will always exist now
            valid_idx += 1
        except Exception as e:
            print(f"Error processing scan {scan_num}: {e}")
            continue

# Trim arrays if we skipped any scans
if valid_idx < size:
    print(f"Only processed {valid_idx} out of {size} scans")
    result_mtx = result_mtx[..., :valid_idx]
    angles = angles[:valid_idx]

# Now result_mtx is the correct shape!

# --- Remove all-zero arrays and max sum array ---
zero_idx = find_zero_arrays(result_mtx)
print(f"Found zero arrays at indices: {zero_idx}")

sums = np.array([np.sum(result_mtx[:, :, i]) for i in range(result_mtx.shape[2])])
max_idx = np.argmax(sums)
print(f"Found maximum sum at index: {max_idx}")

indices_to_remove = list(set(zero_idx + [max_idx]))
print(f"Removing indices: {indices_to_remove}")

valid_mask = np.ones(result_mtx.shape[2], dtype=bool)
valid_mask[indices_to_remove] = False

result_mtx = result_mtx[:, :, valid_mask]
angles = angles[valid_mask]

print(f"Removed {len(indices_to_remove)} arrays. New shape: {result_mtx.shape}")

# --- Save to .mat file ---
final = {'img': result_mtx, 'phi': angles}
savemat(mat_file, final)
print(f"Saved to {mat_file}")
# %%
from scipy.io import loadmat
load_mat_file = 'fft_results_128_complex_object_roi_TEST.mat'
data =loadmat(load_mat_file)
print(data['phi'].shape)
print(data['img'].shape)
# %%
