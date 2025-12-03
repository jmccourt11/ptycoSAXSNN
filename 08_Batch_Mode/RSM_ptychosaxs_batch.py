#%%
##################################
#IMPORT LIBRARIES
##################################
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from RSM_ptychosaxs_funcs import *
from encoder1 import recon_model as recon_model2
#%%

##################################
#STEP 1: DECONVOLUTION
##################################

# Parameters
scan_name = 'ZCB_9_3D'
base_path = "/net/micdata/data2/12IDC/2025_Feb/ptycho/"
# Read the file, skipping the first row (which starts with #) and using the second row as headers
df = pd.read_csv('/net/micdata/data2/12IDC/2025_Feb/misc/ZCB_9_3D_/ZCB9_3D_alignment_shifts_28nm.txt', 
                comment='#',  # Skip lines starting with #
                names=['Angle', 'y_shift', 'x_shift', 'scanNo'])  # Specify column names
scan_numbers = df['scanNo'].values.tolist()

#scan_number=[5065]
center = (517, 575)
dpsize = 256

# Model and mask paths
batch_mode_path = '/net/micdata/data2/12IDC/ptychosaxs/batch_mode_250/'
mask_path = '/home/beams/PTYCHOSAXS/deconvolutionNN/data/mask/mask_ZCB_9_3D.npy' 
# model_paths = [batch_mode_path + 'trained_model/best_model_LatticeClathII_Probe256x256_ZCB_9_3D__Noise_sim_ZCB_9_3D_S5065_N600_steps4_dp256_Unet_epoch_250_pearson_loss_symmetry_0.0.pth',
#                batch_mode_path + 'trained_model/best_model_LatticeClathII_Probe256x256_ZCB_9_3D__Noise_sim_ZCB_9_3D_S5065_N600_steps4_dp256_Unet_epoch_50_pearson_loss_symmetry_0.0.pth',
#                batch_mode_path + 'trained_model/best_model_LatticeClathII_Probe256x256_ZCB_9_3D__Noise_sim_ZCB_9_3D_S5065_N600_steps4_dp256_Unet_epoch_50_L2_symmetry_0.0.pth',
#                batch_mode_path + 'trained_model/best_model_LatticeClathII_Probe256x256_ZCB_9_3D__Noise_sim_ZCB_9_3D_S5065_N600_steps4_dp256_Unet_epoch_50_L1_symmetry_0.0.pth']#,]
# model_paths =  [batch_mode_path + 'trained_model/best_model_LatticeSC_Probe256x256_ZCB_9_3D__Noise_sim_ZCB_9_3D_S5065_N600_steps4_dp256_Unet_epoch_250_pearson_loss_symmetry_0.0.pth',
#                batch_mode_path + 'trained_model/best_model_LatticeSC_Probe256x256_ZCB_9_3D__Noise_sim_ZCB_9_3D_S5065_N600_steps4_dp256_Unet_epoch_50_pearson_loss_symmetry_0.0.pth',
#                batch_mode_path + 'trained_model/best_model_LatticeSC_Probe256x256_ZCB_9_3D__Noise_sim_ZCB_9_3D_S5065_N600_steps4_dp256_Unet_epoch_50_L2_symmetry_0.0.pth',
#                batch_mode_path + 'trained_model/best_model_LatticeSC_Probe256x256_ZCB_9_3D__Noise_sim_ZCB_9_3D_S5065_N600_steps4_dp256_Unet_epoch_50_L1_symmetry_0.0.pth',
#                batch_mode_path + 'trained_model/best_model_LatticeSC_Probe256x256_ZCB_9_3D__noNoise_sim_ZCB_9_3D_S5065_N600_steps4_dp256_Unet_epoch_250_pearson_loss_symmetry_0.0.pth',
#                batch_mode_path + 'trained_model/best_model_LatticeSC_Probe256x256_ZCB_9_3D__noNoise_sim_ZCB_9_3D_S5065_N600_steps4_dp256_Unet_epoch_50_pearson_loss_symmetry_0.0.pth',
#                batch_mode_path + 'trained_model/best_model_LatticeSC_Probe256x256_ZCB_9_3D__noNoise_sim_ZCB_9_3D_S5065_N600_steps4_dp256_Unet_epoch_50_L2_symmetry_0.0.pth',
#                batch_mode_path + 'trained_model/best_model_LatticeSC_Probe256x256_ZCB_9_3D__noNoise_sim_ZCB_9_3D_S5065_N600_steps4_dp256_Unet_epoch_50_L1_symmetry_0.0.pth']

model_paths =  [batch_mode_path + 'trained_model/best_model_LatticeSC_Probe256x256_ZCB_9_3D__Noise_sim_ZCB_9_3D_S5065_N600_steps4_dp256_Unet_epoch_25_pearson_loss_symmetry_0.0.pth',
               batch_mode_path + 'trained_model/best_model_LatticeSC_Probe256x256_ZCB_9_3D__Noise_sim_ZCB_9_3D_S5065_N600_steps4_dp256_Unet_epoch_25_L2_symmetry_0.0.pth',
               batch_mode_path + 'trained_model/best_model_LatticeSC_Probe256x256_ZCB_9_3D__Noise_sim_ZCB_9_3D_S5065_N600_steps4_dp256_Unet_epoch_25_L1_symmetry_0.0.pth',
               batch_mode_path + 'trained_model/best_model_LatticeSC_Probe256x256_ZCB_9_3D__noNoise_sim_ZCB_9_3D_S5065_N600_steps4_dp256_Unet_epoch_25_pearson_loss_symmetry_0.0.pth',
               batch_mode_path + 'trained_model/best_model_LatticeSC_Probe256x256_ZCB_9_3D__noNoise_sim_ZCB_9_3D_S5065_N600_steps4_dp256_Unet_epoch_25_L2_symmetry_0.0.pth',
               batch_mode_path + 'trained_model/best_model_LatticeSC_Probe256x256_ZCB_9_3D__noNoise_sim_ZCB_9_3D_S5065_N600_steps4_dp256_Unet_epoch_25_L1_symmetry_0.0.pth',
               batch_mode_path + 'trained_model/best_model_LatticeSC_Probe256x256_ZCB_9_3D__Noise_sim_ZCB_9_3D_S5065_N600_steps4_dp256_Unet_epoch_250_L1_symmetry_0.0.pth',
               batch_mode_path + 'trained_model/best_model_LatticeSC_Probe256x256_ZCB_9_3D__Noise_sim_ZCB_9_3D_S5065_N600_steps4_dp256_Unet_epoch_250_L2_symmetry_0.0.pth']
# Load mask
mask = np.load(mask_path)
    
#%%
for model_path in model_paths:
    try:
        # Extract model name from path and create corresponding directory
        model_name = model_path.split('/')[-1].replace('.pth', '')
        model_dir = os.path.join(batch_mode_path, 'RSM', model_name)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(os.path.join(model_dir, 'plots'), exist_ok=True)

        # Set output paths using model directory
        output_file = os.path.join(model_dir, 'combined_diffraction_patterns.h5')
        print(f"Processing model: {model_path}")
        print(f"Saving to: {output_file}")

        # Load model and mask once
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = recon_model2()
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()

        # Get already processed scans
        processed_scans = get_processed_scans(output_file)
        remaining_scans = [s for s in scan_numbers if s not in processed_scans]
        print(f"Found {len(processed_scans)} already processed scans. {len(remaining_scans)} scans remaining.")

        # Add indices file parameter
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
                save_verification_plot(result, output_dir=os.path.join(model_dir, 'plots'))
                save_to_h5(output_file, result)
            
            if scan_number != remaining_scans[-1]:  # Don't delay after the last scan
                print(f"\nWaiting {delay_between_scans} seconds before processing next scan...", flush=True)
                time.sleep(delay_between_scans)

        print("Processing complete. Results saved to", output_file)

        # Add metadata to H5 file
        add_metadata_to_h5(output_file, df)
    except Exception as e:
        print(f"Error processing model {model_path}: {str(e)}")
        continue
        
    
# # %%

# scp ptychosaxs@refiner.xray.aps.anl.gov:/home/beams/PTYCHOSAXS/deconvolutionNN/src/utils/combined_diffraction_patterns_TEMP.h5 "C:\Users\b304014\Software\blee\data\combined_test.h5"
# scp ptychosaxs@refiner.xray.aps.anl.gov:/home/beams/PTYCHOSAXS/deconvolutionNN/src/utils/combined_diffraction_patterns_TEST_variance.h5 "C:\Users\b304014\Software\blee\data\deconvolved_DPs_variance.h5" 



# ##################################
# #STEP 2: CONVERT h5 TO MAT AND PREPROCESS
# ##################################

# #output_file = "C:\\Users\\b304014\\Software\\blee\\data\\deconvolved_DPs_variance_20251022.h5"
# output_file = "C:\\Users\\b304014\\Software\\blee\\data\\2025_Feb\\data\\deconvolved_DPs_variance.h5"
# scan_num_start=5003
# size=366
# skip_indices=[0]#[1,2,3]
# result_mtx=np.zeros((256,256,size))
# angles=np.zeros((size))
# valid_idx = 0  # Counter for valid entries
# for i in range(0,size):
#     if i in skip_indices:
#         print(f"Skipping scan {scan_num_start+i}")
#         continue
#     else:
#         with h5py.File(output_file, 'r') as f:
#             try:
#                 #print(f"Processing scan {scan_num_start+i}")
#                 data = f['deconvolved'][f'scan_{scan_num_start+i}'][()]
#                 result_mtx[:,:,valid_idx]=data
                
#                 # Get the scan number and find its index in the scan_numbers array
#                 current_scan = scan_num_start + i
#                 scan_numbers = f['metadata']['scan_numbers'][()]
#                 angle_idx = np.where(scan_numbers == current_scan)[0]
                
#                 if len(angle_idx) > 0:
#                     angles[valid_idx] = f['metadata']['angles'][()][angle_idx[0]]
#                 else:
#                     print(f"Warning: No angle found for scan {current_scan}. Using previous angle.")
#                     angles[valid_idx] = angles[valid_idx-1] if valid_idx > 0 else 0
                
#                 valid_idx += 1
#             except KeyError as e:
#                 print(f"KeyError processing scan {scan_num_start+i}: {str(e)}")
#             except IndexError as e:
#                 print(f"IndexError processing scan {scan_num_start+i}: {str(e)}")
#             except Exception as e:
#                 print(f"Error processing scan {scan_num_start+i}: {str(e)}")
                
# zero_idx = find_zero_arrays(result_mtx)
# print(f"Found zero arrays at indices: {zero_idx}")

# # Find the index with maximum sum
# sums = np.array([np.sum(result_mtx[:,:,i]) for i in range(result_mtx.shape[2])])
# max_idx = np.argmax(sums)
# print(f"Found maximum sum at index: {max_idx}")

# # Combine zero indices with max index
# indices_to_remove = list(set(zero_idx + [max_idx]))
# print(f"Removing indices: {indices_to_remove}")

# # Create masks for valid arrays
# valid_mask = np.ones(result_mtx.shape[2], dtype=bool)
# valid_mask[indices_to_remove] = False

# # Remove invalid arrays and their corresponding angles
# result_mtx = result_mtx[:,:,valid_mask]
# angles = angles[valid_mask]

# print(f"Removed {len(indices_to_remove)} arrays. New shape: {result_mtx.shape}")

# # Apply Gaussian smoothing to each frame
# sigma = 1  # Adjust this value to control the amount of smoothing
# for i in range(result_mtx.shape[2]):
#     result_mtx[:,:,i] = gaussian_filter(result_mtx[:,:,i], sigma=sigma)

# final = {'img':result_mtx,'phi':angles}
# #savemat("C:\\Users\\b304014\\Software\\blee\\data\\deconvolved_DPs_variance_20251022.mat",final)

# #%%
# result_mtx_processed = result_mtx.copy()

# # Calculate q values
# detector_pixel_size = 172e-6  # 75 microns in meters
# sample_detector_distance = 10  # meters
# wavelength = 1.24e-10  # angstroms to meters
# num_pixels = result_mtx.shape[0]  # Assuming square detector

# # Create pixel coordinate arrays
# x = np.arange(-num_pixels//2, num_pixels//2)
# X, Y = np.meshgrid(x, x)
# R = np.sqrt(X**2 + Y**2)

# # Calculate scattering angle and q
# theta = np.arctan(detector_pixel_size * R / sample_detector_distance)
# q = 4 * np.pi * np.sin(theta) / wavelength

# # Number of slices you want to plot
# num_slices_to_plot = 5
# # Randomly select indices to plot
# indices_to_plot = random.sample(range(result_mtx.shape[2]), num_slices_to_plot)


# for num in range(result_mtx.shape[2]):
#     # Calculate q values, assuming X, Y, R, theta, and q are already defined as needed above
    
#     # Background subtraction using the median for each slice
#     background = np.median(result_mtx[:, :, num])
#     # Background estimation using Gaussian filtering
#     #background = gaussian_filter(result_mtx[:, :, num], sigma=10)
#     result_mtx_selected = result_mtx[:, :, num] - background

#     # Remove NaN values
#     result_mtx_selected = np.nan_to_num(result_mtx_selected)
    
#     # Intensity normalization
#     max_intensity = np.max(result_mtx_selected)
#     if max_intensity != 0:
#         result_mtx_selected /= max_intensity

#     # Scaling by q^4 (example)
#     scaled_image = result_mtx_selected * q**4

#     # Plot only selected slices
#     if num in indices_to_plot:
#         plt.imshow(scaled_image, norm=LogNorm())
#         plt.colorbar(label='Normalized Intensity * q^4')
#         plt.title(f'Normalized Q^4-scaled diffraction pattern at {angles[num]:.2f}°')
#         plt.show()
        
#     result_mtx_processed[:,:,num] = scaled_image

# final = {'img':result_mtx_processed,'phi':angles}
# savemat("C:\\Users\\b304014\\Software\\blee\\data\\deconvolved_DPs_variance_20251022_processed_previous_NN.mat",final)
# #%%


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


# ##################################
# #STEP 3: PEAK DETECTION AND CONFUSION MATRIX
# ##################################


# def load_cellinfo_data(file_path):
#     """
#     Load and extract arrays from the 'cellinfo' structure in the given .mat file.
    
#     Args:
#         file_path (str): Path to the .mat file.
        
#     Returns:
#         dict: A dictionary where keys are field names and values are the corresponding arrays.
#     """
    
#     # Load the .mat file
#     mat_data = loadmat(file_path)
    
#     # Extract the 'cellinfo' data
#     cellinfo_data = mat_data.get('cellinfo')
    
#     if cellinfo_data is None:
#         raise ValueError("'cellinfo' key not found in the .mat file.")
    
#     # Initialize a dictionary to store the extracted data
#     data_dict = {}
    
#     # Iterate through each field and extract its content
#     for field_name in cellinfo_data.dtype.names:
#         data_dict[field_name] = cellinfo_data[field_name][0, 0]
    
#     return data_dict



# def generate_miller_indices(max_order=3):
#     """
#     Generate Miller indices (h,k,l) up to specified order.
#     Excludes (0,0,0) and includes all combinations where |h|,|k|,|l| ≤ max_order.
#     """
#     indices = []
#     for h in range(-max_order, max_order + 1):
#         for k in range(-max_order, max_order + 1):
#             for l in range(-max_order, max_order + 1):
#                 # Skip the origin
#                 if h == 0 and k == 0 and l == 0:
#                     continue
#                 indices.append([h, k, l])
#     return np.array(indices)




# def plot_multi_reciprocal_space(
#     rs_datasets,  # List of dicts: [{'magnitude': 3D array, 'Q': 4D array (shape: (nx,ny,nz,3)), 'label': str}, ...]
#     cellinfo_data,
#     hs, ks, ls,
#     thresholds,  # List of thresholds for each dataset
#     q_cutoffs,
#     peak_distance_threshold=0.01,
#     colormaps=['Viridis', 'Jet', 'Plasma'],
#     alphas=[0.4, 0.4, 0.4],
#     q_axes=[1, 2, 0],
#     q_signs=[1, 1, 1],
#     flatten_order='C',
#     dbscan_eps=0.08, dbscan_min_samples=10
# ):
#     """
#     Plot multiple 3D reciprocal space datasets and unit cell peaks, with axis/sign/flattening troubleshooting.
#     Args:
#         rs_datasets: List of dicts, each with keys:
#             - 'magnitude': 3D numpy array
#             - 'Q': 4D numpy array (shape: (nx,ny,nz,3)), or tuple of (Qx, Qy, Qz) 3D arrays
#             - 'label': str, label for the dataset
#         cellinfo_data: Unit cell information
#         hs, ks, ls: Miller indices for unit cell peaks
#         thresholds: List of magnitude thresholds for each dataset (relative, 0-1)
#         q_cutoffs: List of minimum |q| to include (float) for each dataset
#         peak_distance_threshold: Max distance to consider a unit cell peak as close to a region center
#         q_axes: List of indices for Qv columns to use as x, y, z
#         q_signs: List of sign flips for Qv columns
#         flatten_order: 'C' or 'F' for flattening order
#     Returns:
#         tuple: (fig, close_peaks, close_peaks_hkl, filtered_datasets)
#             - fig: plotly figure object
#             - close_peaks: array of peak positions
#             - close_peaks_hkl: list of Miller indices
#             - filtered_datasets: list of dicts containing filtered data and Q coordinates
#     """
#     import numpy as np
#     import plotly.graph_objects as go
#     from sklearn.cluster import DBSCAN

#     fig = go.Figure()
#     all_region_centers = []
#     all_labels = []
#     filtered_datasets = []

#     # Plot each reciprocal space dataset
#     for idx, dataset in enumerate(rs_datasets):
#         magnitude = dataset['DATA']
#         Q = dataset['Qv']
#         label = dataset.get('label', f'Dataset {idx+1}')
#         threshold = thresholds[idx]

#         # --- Apply axis/sign/flattening troubleshooting ---
#         if Q.ndim == 2 and Q.shape[1] == 3:
#             npts = np.prod(magnitude.shape)
#             if Q.shape[0] == npts:
#                 Qx = Q[:, q_axes[0]].reshape(magnitude.shape, order=flatten_order) * q_signs[0]
#                 Qy = Q[:, q_axes[1]].reshape(magnitude.shape, order=flatten_order) * q_signs[1]
#                 Qz = Q[:, q_axes[2]].reshape(magnitude.shape, order=flatten_order) * q_signs[2]
#             else:
#                 Qx = Q[:, q_axes[0]] * q_signs[0]
#                 Qy = Q[:, q_axes[1]] * q_signs[1]
#                 Qz = Q[:, q_axes[2]] * q_signs[2]
#         else:
#             Qx = Q[..., q_axes[0]] * q_signs[0]
#             Qy = Q[..., q_axes[1]] * q_signs[1]
#             Qz = Q[..., q_axes[2]] * q_signs[2]

#         # Create filtered version of the data
#         q_mag = np.sqrt(Qx**2 + Qy**2 + Qz**2)
#         mask = (q_mag > q_cutoffs[idx]) & (magnitude > threshold * np.max(magnitude))
#         filtered_magnitude = magnitude.copy()
#         filtered_magnitude[~mask] = 0

#         filtered_datasets.append({
#             'DATA': filtered_magnitude,
#             'Qx': Qx,
#             'Qy': Qy,
#             'Qz': Qz,
#             'label': label
#         })

#         # Flatten for plotting
#         kx_flat = Qx.flatten(order=flatten_order)
#         ky_flat = Qy.flatten(order=flatten_order)
#         kz_flat = Qz.flatten(order=flatten_order)
#         mag_flat = magnitude.flatten(order=flatten_order)

#         q_mag = np.sqrt(kx_flat**2 + ky_flat**2 + kz_flat**2)
#         mask = (q_mag > q_cutoffs[idx]) & (mag_flat > threshold * np.max(mag_flat))
#         kx_f = kx_flat[mask]
#         ky_f = ky_flat[mask]
#         kz_f = kz_flat[mask]
#         mag_f = mag_flat[mask]

#         fig.add_trace(go.Scatter3d(
#             x=kx_f, y=ky_f, z=kz_f,
#             mode='markers',
#             marker=dict(
#                 size=5,
#                 #color=mag_f,
#                 color=kx_f,
#                 colorscale=colormaps[idx],
#                 opacity=alphas[idx],
#                 colorbar=dict(title='X Position') if idx == 0 else None
#                 #colorbar=dict(title=f'{label} Magnitude') if idx == 0 else None
#             ),
#             name=label
#         ))
        
#         # factor = 4  # Try 2, 3, or higher for more aggressive downsampling
#         # Qx_ds = Qx[::factor, ::factor, ::factor]
#         # Qy_ds = Qy[::factor, ::factor, ::factor]
#         # Qz_ds = Qz[::factor, ::factor, ::factor]
#         # filtered_magnitude_ds = filtered_magnitude[::factor, ::factor, ::factor]
#         # fig.add_trace(go.Isosurface(
#         #     x=Qx_ds.flatten(order=flatten_order),
#         #     y=Qy_ds.flatten(order=flatten_order),
#         #     z=Qz_ds.flatten(order=flatten_order),
#         #     value=filtered_magnitude_ds.flatten(order=flatten_order),
#         #     isomin=0.1 * np.max(filtered_magnitude_ds),
#         #     isomax=np.max(filtered_magnitude_ds),
#         #     opacity=alphas[idx],
#         #     surface_count=3,
#         #     colorscale=colormaps[idx],
#         #     showscale=(idx == 0),
#         #     name=label
#         # ))

#         # Cluster and find region centers
#         if len(kx_f) > 0:
#             coords = np.column_stack((kx_f, ky_f, kz_f))
#             coords_norm = coords / np.max(np.abs(coords))
#             clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(coords_norm)
#             labels_ = clustering.labels_
#             region_centers = []
#             for clabel in set(labels_):
#                 if clabel == -1:
#                     continue
#                 mask_c = labels_ == clabel
#                 cluster_points = coords[mask_c]
#                 cluster_mags = mag_f[mask_c]
#                 weights = cluster_mags / np.sum(cluster_mags)
#                 center = np.sum(cluster_points * weights[:, np.newaxis], axis=0)
#                 region_centers.append(center)
#             region_centers = np.array(region_centers)
#             if len(region_centers) > 0:
#                 all_region_centers.extend(region_centers)
#                 all_labels.extend([label]*len(region_centers))

#     # Compute unit cell peaks
#     vs = []
#     hkl_list = []
#     for i, h in enumerate(hs):
#         v = hs[i]*cellinfo_data['recilatticevectors'][0] + \
#             ks[i]*cellinfo_data['recilatticevectors'][1] + \
#             ls[i]*cellinfo_data['recilatticevectors'][2]
#         vs.append(v)
#         hkl_list.append(f"({h},{ks[i]},{ls[i]})")
#     vs = np.array(vs)

#     # Find unit cell peaks close to any region center
#     close_peaks = []
#     close_peaks_hkl = []
#     used_regions = set()
#     all_region_centers = np.array(all_region_centers)
#     for i, peak in enumerate(vs):
#         if len(all_region_centers) == 0:
#             break
#         distances = np.sqrt(np.sum((all_region_centers - peak)**2, axis=1))
#         min_dist_idx = np.argmin(distances)
#         min_dist = distances[min_dist_idx]
#         if min_dist < peak_distance_threshold and (min_dist_idx, all_labels[min_dist_idx]) not in used_regions:
#             close_peaks.append(peak)
#             close_peaks_hkl.append(hkl_list[i])
#             used_regions.add((min_dist_idx, all_labels[min_dist_idx]))
#     close_peaks = np.array(close_peaks)

#     if len(close_peaks) > 0:
#         fig.add_trace(go.Scatter3d(
#             x=close_peaks[:,0], y=close_peaks[:,1], z=close_peaks[:,2],
#             #mode='markers+text',
#             mode='markers',
#             marker=dict(size=5, color='red', opacity=0.3, symbol='diamond'),
#             #text=close_peaks_hkl,
#             #textfont=dict(size=6),
#             #textposition="top center",
#             name='Unit Cell Peaks'
#         ))
        
#     # Example: list of hkl labels to highlight
#     #highlight_hkls = ['(2,0,0)', '(6,0,0)', '(-6,0,0)', '(-2,0,0)']
#     #highlight_hkls = ['(0,2,0)', '(0,6,0)', '(0,-2,0)', '(0,-6,0)']
#     #highlight_hkls = ['(0,0,2)', '(0,0,6)', '(0,0,-2)', '(0,0,-6)'] 
#     highlight_hkls = ['(0,6,6)', '(0,-6,-6)', '(2,2,-8)', '(-2,-2,8)', '(2,8,-2)', '(-2,-8,2)']

#     # Separate peaks to highlight
#     highlight_mask = [hkl in highlight_hkls for hkl in close_peaks_hkl]
#     normal_mask = [not h for h in highlight_mask]

#     # Normal peaks
#     if any(normal_mask):
#         fig.add_trace(go.Scatter3d(
#             x=close_peaks[normal_mask, 0],
#             y=close_peaks[normal_mask, 1],
#             z=close_peaks[normal_mask, 2],
#             mode='text',
#             text=np.array(close_peaks_hkl)[normal_mask],
#             textposition='top center',
#             textfont=dict(color='black', size=10),
#             name='hkl Peaks'
#         ))


#     print(f"Number of unit cell peaks: {len(close_peaks)}")
#     print(f'close_peaks: {close_peaks_hkl}')
    
#     fig.update_layout(
#         title="Multi Reciprocal Space Visualization (TROUBLESHOOT MODE)",
#         scene=dict(
#             xaxis_title="Qx (Å⁻¹)",
#             yaxis_title="Qy (Å⁻¹)",
#             zaxis_title="Qz (Å⁻¹)",
#             aspectmode='cube'
#         ),
#         width=1000,
#         height=1000,
#         showlegend=True
#     )
#         # Highlighted peaks
#     if any(highlight_mask):
#         fig.add_trace(go.Scatter3d(
#             x=close_peaks[highlight_mask, 0],
#             y=close_peaks[highlight_mask, 1],
#             z=close_peaks[highlight_mask, 2],
#             mode='text',
#             text=np.array(close_peaks_hkl)[highlight_mask],
#             textposition='top center',
#             textfont=dict(color='red', size=20),  # Larger, red text
#             name='Highlighted Peaks'
#         ))

#     return fig, close_peaks, close_peaks_hkl, filtered_datasets


# def count_overlapping_peaks(
#     dataset1, dataset2,
#     threshold1, threshold2,
#     q_cutoff1, q_cutoff2,
#     overlap_distance=0.01,
#     dbscan_eps=0.08, dbscan_min_samples=10,
#     q_axes=[1,2,0], q_signs=[1,1,1], flatten_order='C'
# ):
#     """
#     Count the number of overlapping peaks between two 3D reciprocal space datasets.
#     Returns: (n_peaks1, n_peaks2, n_overlapping)
#     """
#     import numpy as np
#     from sklearn.cluster import DBSCAN

#     def find_peaks(dataset, threshold, q_cutoff):
#         DATA = dataset['DATA']
#         Q = dataset['Qv']
#         # --- Apply axis/sign/flattening troubleshooting ---
#         if Q.ndim == 2 and Q.shape[1] == 3:
#             npts = np.prod(DATA.shape)
#             if Q.shape[0] == npts:
#                 Qx = Q[:, q_axes[0]].reshape(DATA.shape, order=flatten_order) * q_signs[0]
#                 Qy = Q[:, q_axes[1]].reshape(DATA.shape, order=flatten_order) * q_signs[1]
#                 Qz = Q[:, q_axes[2]].reshape(DATA.shape, order=flatten_order) * q_signs[2]
#             else:
#                 Qx = Q[:, q_axes[0]] * q_signs[0]
#                 Qy = Q[:, q_axes[1]] * q_signs[1]
#                 Qz = Q[:, q_axes[2]] * q_signs[2]
#         else:
#             Qx = Q[..., q_axes[0]] * q_signs[0]
#             Qy = Q[..., q_axes[1]] * q_signs[1]
#             Qz = Q[..., q_axes[2]] * q_signs[2]
#         kx_flat = Qx.flatten(order=flatten_order)
#         ky_flat = Qy.flatten(order=flatten_order)
#         kz_flat = Qz.flatten(order=flatten_order)
#         mag_flat = DATA.flatten(order=flatten_order)
#         q_mag = np.sqrt(kx_flat**2 + ky_flat**2 + kz_flat**2)
#         mask = (q_mag > q_cutoff) & (mag_flat > threshold * np.max(mag_flat))
#         coords = np.column_stack((kx_flat[mask], ky_flat[mask], kz_flat[mask]))
#         mag_f = mag_flat[mask]
#         if len(coords) == 0:
#             return np.zeros((0,3))
#         coords_norm = coords / np.max(np.abs(coords))
#         clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(coords_norm)
#         labels_ = clustering.labels_
#         region_centers = []
#         for clabel in set(labels_):
#             if clabel == -1:
#                 continue
#             mask_c = labels_ == clabel
#             cluster_points = coords[mask_c]
#             cluster_mags = mag_f[mask_c]
#             weights = cluster_mags / np.sum(cluster_mags)
#             center = np.sum(cluster_points * weights[:, np.newaxis], axis=0)
#             region_centers.append(center)
#         return np.array(region_centers)

#     peaks1 = find_peaks(dataset1, threshold1, q_cutoff1)
#     peaks2 = find_peaks(dataset2, threshold2, q_cutoff2)
#     n_peaks1 = len(peaks1)
#     n_peaks2 = len(peaks2)

#     # Count overlaps
#     n_overlapping = 0
#     used2 = set()
#     for i, p1 in enumerate(peaks1):
#         dists = np.sqrt(np.sum((peaks2 - p1)**2, axis=1))
#         min_idx = np.argmin(dists)
#         if dists[min_idx] < overlap_distance and min_idx not in used2:
#             n_overlapping += 1
#             used2.add(min_idx)

#     return n_peaks1, n_peaks2, n_overlapping


# def peak_confusion_matrix(
#     true_dataset, pred_dataset,
#     true_threshold, pred_threshold,
#     true_q_cutoff, pred_q_cutoff,
#     overlap_distance=0.01,
#     dbscan_eps=0.08, dbscan_min_samples=10,
#     q_axes=[1,2,0], q_signs=[1,1,1], flatten_order='C',
#     plot=False,
#     overlay=False,
#     cellinfo_data=None,
#     miller_indices=None
# ):
#     """
#     Compute confusion matrix for peak detection in reciprocal space.
#     If plot=True, show a 3D plot of true, predicted, and matched peaks.
#     If overlay=True, show overlays of found peaks on thresholded reciprocal space points for both datasets.
#     Returns: dict with keys 'TP', 'FP', 'FN', 'n_true', 'n_pred', and optionally 'fig', 'overlay_true_fig', 'overlay_pred_fig'
#     """
#     import numpy as np
#     from sklearn.cluster import DBSCAN
#     import plotly.graph_objects as go

#     def find_peaks(dataset, threshold, q_cutoff):
#         DATA = dataset['DATA']
#         Q = dataset['Qv']
#         if Q.ndim == 2 and Q.shape[1] == 3:
#             npts = np.prod(DATA.shape)
#             if Q.shape[0] == npts:
#                 Qx = Q[:, q_axes[0]].reshape(DATA.shape, order=flatten_order) * q_signs[0]
#                 Qy = Q[:, q_axes[1]].reshape(DATA.shape, order=flatten_order) * q_signs[1]
#                 Qz = Q[:, q_axes[2]].reshape(DATA.shape, order=flatten_order) * q_signs[2]
#             else:
#                 Qx = Q[:, q_axes[0]] * q_signs[0]
#                 Qy = Q[:, q_axes[1]] * q_signs[1]
#                 Qz = Q[:, q_axes[2]] * q_signs[2]
#         else:
#             Qx = Q[..., q_axes[0]] * q_signs[0]
#             Qy = Q[..., q_axes[1]] * q_signs[1]
#             Qz = Q[..., q_axes[2]] * q_signs[2]
#         kx_flat = Qx.flatten(order=flatten_order)
#         ky_flat = Qy.flatten(order=flatten_order)
#         kz_flat = Qz.flatten(order=flatten_order)
#         mag_flat = DATA.flatten(order=flatten_order)
#         q_mag = np.sqrt(kx_flat**2 + ky_flat**2 + kz_flat**2)
#         mask = (q_mag > q_cutoff) & (mag_flat > threshold * np.max(mag_flat))
#         coords = np.column_stack((kx_flat[mask], ky_flat[mask], kz_flat[mask]))
#         mag_f = mag_flat[mask]
#         if len(coords) == 0:
#             return np.zeros((0,3))
#         coords_norm = coords / np.max(np.abs(coords))
#         clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(coords_norm)
#         labels_ = clustering.labels_
#         region_centers = []
#         for clabel in set(labels_):
#             if clabel == -1:
#                 continue
#             mask_c = labels_ == clabel
#             cluster_points = coords[mask_c]
#             cluster_mags = mag_f[mask_c]
#             weights = cluster_mags / np.sum(cluster_mags)
#             center = np.sum(cluster_points * weights[:, np.newaxis], axis=0)
#             region_centers.append(center)
#         return np.array(region_centers)

#     def get_thresholded_points(dataset, threshold, q_cutoff):
#         DATA = dataset['DATA']
#         Q = dataset['Qv']
#         if Q.ndim == 2 and Q.shape[1] == 3:
#             npts = np.prod(DATA.shape)
#             if Q.shape[0] == npts:
#                 Qx = Q[:, q_axes[0]].reshape(DATA.shape, order=flatten_order) * q_signs[0]
#                 Qy = Q[:, q_axes[1]].reshape(DATA.shape, order=flatten_order) * q_signs[1]
#                 Qz = Q[:, q_axes[2]].reshape(DATA.shape, order=flatten_order) * q_signs[2]
#             else:
#                 Qx = Q[:, q_axes[0]] * q_signs[0]
#                 Qy = Q[:, q_axes[1]] * q_signs[1]
#                 Qz = Q[:, q_axes[2]] * q_signs[2]
#         else:
#             Qx = Q[..., q_axes[0]] * q_signs[0]
#             Qy = Q[..., q_axes[1]] * q_signs[1]
#             Qz = Q[..., q_axes[2]] * q_signs[2]
#         kx_flat = Qx.flatten(order=flatten_order)
#         ky_flat = Qy.flatten(order=flatten_order)
#         kz_flat = Qz.flatten(order=flatten_order)
#         mag_flat = DATA.flatten(order=flatten_order)
#         q_mag = np.sqrt(kx_flat**2 + ky_flat**2 + kz_flat**2)
#         mask = (q_mag > q_cutoff) & (mag_flat > threshold * np.max(mag_flat))
#         return kx_flat[mask], ky_flat[mask], kz_flat[mask], mag_flat[mask]

#     true_peaks = find_peaks(true_dataset, true_threshold, true_q_cutoff)
#     pred_peaks = find_peaks(pred_dataset, pred_threshold, pred_q_cutoff)
#     n_true = len(true_peaks)
#     n_pred = len(pred_peaks)

#     matched_true = set()
#     matched_pred = set()
#     matches = []  # Store (i, j) pairs

#     if n_true == 0 or n_pred == 0:
#         TP = 0
#         FN = n_true
#         FP = n_pred
#     else:
#         dists = np.linalg.norm(true_peaks[:, None, :] - pred_peaks[None, :, :], axis=2)
#         for i in range(n_true):
#             min_j = np.argmin(dists[i])
#             if dists[i, min_j] < overlap_distance and min_j not in matched_pred:
#                 matched_true.add(i)
#                 matched_pred.add(min_j)
#                 matches.append((i, min_j))
#         TP = len(matched_true)
#         FN = n_true - TP
#         FP = n_pred - TP

#     # Match peaks to Miller indices if cellinfo_data and miller_indices are provided
#     matched_hkl_info = {}
#     if cellinfo_data is not None and miller_indices is not None:
#         # Calculate theoretical peak positions
#         vs = []
#         hkl_list = []
#         for i, h in enumerate(miller_indices[:, 0]):
#             v = miller_indices[i, 0]*cellinfo_data['recilatticevectors'][0] + \
#                 miller_indices[i, 1]*cellinfo_data['recilatticevectors'][1] + \
#                 miller_indices[i, 2]*cellinfo_data['recilatticevectors'][2]
#             vs.append(v)
#             hkl_list.append(f"({miller_indices[i, 0]},{miller_indices[i, 1]},{miller_indices[i, 2]})")
#         vs = np.array(vs)
        
#         # Match true peaks to Miller indices
#         true_peak_hkl = []
#         for i, peak in enumerate(true_peaks):
#             distances = np.sqrt(np.sum((vs - peak)**2, axis=1))
#             min_dist_idx = np.argmin(distances)
#             min_dist = distances[min_dist_idx]
#             if min_dist < 0.02:  # Threshold for matching to Miller indices
#                 true_peak_hkl.append(hkl_list[min_dist_idx])
#             else:
#                 true_peak_hkl.append("Unknown")
        
#         # Match predicted peaks to Miller indices
#         pred_peak_hkl = []
#         for i, peak in enumerate(pred_peaks):
#             distances = np.sqrt(np.sum((vs - peak)**2, axis=1))
#             min_dist_idx = np.argmin(distances)
#             min_dist = distances[min_dist_idx]
#             if min_dist < 0.02:  # Threshold for matching to Miller indices
#                 pred_peak_hkl.append(hkl_list[min_dist_idx])
#             else:
#                 pred_peak_hkl.append("Unknown")
        
#         # Print matched peaks information
#         print("\n=== PEAK MATCHING RESULTS ===")
#         print(f"True peaks matched to Miller indices:")
#         for i, (peak, hkl) in enumerate(zip(true_peaks, true_peak_hkl)):
#             print(f"  Peak {i+1}: {peak} -> {hkl}")
        
#         print(f"\nPredicted peaks matched to Miller indices:")
#         for i, (peak, hkl) in enumerate(zip(pred_peaks, pred_peak_hkl)):
#             print(f"  Peak {i+1}: {peak} -> {hkl}")
        
#         # Print matched pairs with their hkl
#         print(f"\nMatched peak pairs (True -> Predicted):")
#         for (i, j) in matches:
#             print(f"  {true_peak_hkl[i]} -> {pred_peak_hkl[j]}")
        
#         matched_hkl_info = {
#             'true_peak_hkl': true_peak_hkl,
#             'pred_peak_hkl': pred_peak_hkl,
#             'matched_pairs_hkl': [(true_peak_hkl[i], pred_peak_hkl[j]) for (i, j) in matches]
#         }

#     results = {
#         'TP': TP,
#         'FP': FP,
#         'FN': FN,
#         'n_true': n_true,
#         'n_pred': n_pred,
#         'matched_hkl_info': matched_hkl_info
#     }

#     if plot:
#         fig = go.Figure(layout=dict(width=1000, height=800))
#         # Plot all true peaks
#         if n_true > 0:
#             fig.add_trace(go.Scatter3d(
#                 x=true_peaks[:,0], y=true_peaks[:,1], z=true_peaks[:,2],
#                 mode='markers',
#                 marker=dict(size=19, color='blue', opacity=0.5, symbol='circle'),
#                 name='True Peaks'
#             ))
#         # Plot all predicted peaks
#         if n_pred > 0:
#             fig.add_trace(go.Scatter3d(
#                 x=pred_peaks[:,0], y=pred_peaks[:,1], z=pred_peaks[:,2],
#                 mode='markers',
#                 marker=dict(size=10, color='orange', opacity=0.5, symbol='diamond'),
#                 name='Predicted Peaks'
#             ))
#         # Plot matched pairs with lines
#         for (i, j) in matches:
#             fig.add_trace(go.Scatter3d(
#                 x=[true_peaks[i,0], pred_peaks[j,0]],
#                 y=[true_peaks[i,1], pred_peaks[j,1]],
#                 z=[true_peaks[i,2], pred_peaks[j,2]],
#                 mode='lines',
#                 line=dict(color='green', width=8),
#                 name='Matched Pair',
#                 showlegend=False
#             ))
#         fig.update_layout(
#             title='3D Peaks: True (blue), Predicted (orange), Matches (green lines)',
#             scene=dict(
#                 xaxis_title='Qx',
#                 yaxis_title='Qy',
#                 zaxis_title='Qz',
#                 aspectmode='cube'
#             ),
#             legend=dict(itemsizing='constant')
#         )
#         fig.show()
#         results['fig'] = fig

#     if overlay:
#         # Overlay for true dataset
#         kx_t, ky_t, kz_t, mag_t = get_thresholded_points(true_dataset, true_threshold, true_q_cutoff)
#         fig_true = go.Figure()
#         fig_true.add_trace(go.Scatter3d(
#             x=kx_t, y=ky_t, z=kz_t,
#             mode='markers',
#             marker=dict(size=3, color=mag_t, colorscale='Viridis', opacity=0.2),
#             name='Thresholded Points'
#         ))
#         if n_true > 0:
#             fig_true.add_trace(go.Scatter3d(
#                 x=true_peaks[:,0], y=true_peaks[:,1], z=true_peaks[:,2],
#                 mode='markers',
#                 marker=dict(size=10, color='blue', opacity=0.4, symbol='circle'),
#                 name='True Peaks'
#             ))
#         fig_true.update_layout(
#             title='True Peaks Overlayed on Reciprocal Space',
#             scene=dict(xaxis_title='Qx', yaxis_title='Qy', zaxis_title='Qz', aspectmode='cube')
#         )

#         # Overlay for predicted dataset
#         kx_p, ky_p, kz_p, mag_p = get_thresholded_points(pred_dataset, pred_threshold, pred_q_cutoff)
#         fig_pred = go.Figure()
#         fig_pred.add_trace(go.Scatter3d(
#             x=kx_p, y=ky_p, z=kz_p,
#             mode='markers',
#             marker=dict(size=3, color=mag_p, colorscale='Plasma', opacity=0.2),
#             name='Thresholded Points'
#         ))
#         if n_pred > 0:
#             fig_pred.add_trace(go.Scatter3d(
#                 x=pred_peaks[:,0], y=pred_peaks[:,1], z=pred_peaks[:,2],
#                 mode='markers',
#                 marker=dict(size=10, color='orange', opacity=0.4, symbol='diamond'),
#                 name='Predicted Peaks'
#             ))
#         fig_pred.update_layout(
#             title='Predicted Peaks Overlayed on Reciprocal Space',
#             scene=dict(xaxis_title='Qx', yaxis_title='Qy', zaxis_title='Qz', aspectmode='cube')
#         )

#         fig_true.show()
#         fig_pred.show()
#         results['overlay_true_fig'] = fig_true
#         results['overlay_pred_fig'] = fig_pred

#     return results

# from scipy.ndimage import gaussian_filter


# basedir='/scratch/2025_Feb/temp/'
# basedir='/scratch/2025_Feb/'

# # tomo_data_RS_file=sio.loadmat(f'{basedir}/FFT_RS_128.mat')
# # tomo_data_RS=tomo_data_RS_file['DATA_m']
# # tomo_data_RS_qs=tomo_data_RS_file['Qv_m']
# # tomo_data_RS[np.isnan(tomo_data_RS)]=0
# # tomo_data_RS_qs[np.isnan(tomo_data_RS_qs)]=0
# # tomo_data_RS = gaussian_filter(tomo_data_RS, sigma=0.)#.75) 

# tomo_data_RS_file_256=sio.loadmat(f'{basedir}/FFT_RS_256_NEW.mat')
# tomo_data_RS_256=tomo_data_RS_file_256['DATA256']
# tomo_data_RS_qs_256=tomo_data_RS_file_256['Qv256']
# tomo_data_RS_256[np.isnan(tomo_data_RS_256)]=0
# tomo_data_RS_qs_256[np.isnan(tomo_data_RS_qs_256)]=1
# tomo_data_RS_256 = gaussian_filter(tomo_data_RS_256, sigma=0.) 

# #tomo_data_RS_file_DECONV=sio.loadmat(f'{basedir}/DECONV_RS_256_NEW.mat')
# tomo_data_RS_file_DECONV=sio.loadmat(f'{basedir}/DECONV_RS_256_PROCESSED.mat')
# #tomo_data_RS_file_DECONV=sio.loadmat(f'{basedir}/DECONV_RS_256_PROCESSED_PREVIOUS_NN.mat')
# #tomo_data_RS_file_DECONV=sio.loadmat(f'{basedir}/DECONV_RS_256_VARIANCE.mat')
# #tomo_data_RS_file_DECONV=sio.loadmat(f'{basedir}/DECONV_RS_256_VARIANCE_THRESHOLD.mat')
# tomo_data_RS_DECONV=tomo_data_RS_file_DECONV['DATA_d']
# tomo_data_RS_qs_DECONV=tomo_data_RS_file_DECONV['Qv_d']
# tomo_data_RS_DECONV[np.isnan(tomo_data_RS_DECONV)]=0
# tomo_data_RS_qs_DECONV[np.isnan(tomo_data_RS_qs_DECONV)]=0
# tomo_data_RS_DECONV = gaussian_filter(tomo_data_RS_DECONV, sigma=0.0) 


# #cellinfo_data = load_cellinfo_data(f'{basedir}/cellinfo_ZCB9_3D.mat')
# cellinfo_data = load_cellinfo_data(f'{basedir}/cellinfo_FCC_forFFTs.mat')
# #cellinfo_data = load_cellinfo_data(f'{basedir}/cellinfo_ZCB_9_20251022.mat')


# def rotate_tomo(data, angle_deg, axis='z'):
#     """
#     Rotate a 3D tomogram by a given angle (in degrees) about the specified axis.
#     Args:
#         data: 3D numpy array
#         angle_deg: rotation angle in degrees
#         axis: 'x', 'y', or 'z' (axis about which to rotate)
#     Returns:
#         Rotated 3D numpy array (same shape as input, with reshape=False)
#     """
#     if axis == 'x':
#         return rotate(data, angle_deg, axes=(1, 2), reshape=False)
#     elif axis == 'y':
#         return rotate(data, angle_deg, axes=(0, 2), reshape=False)
#     elif axis == 'z':
#         return rotate(data, angle_deg, axes=(0, 1), reshape=False)
#     else:
#         raise ValueError("axis must be 'x', 'y', or 'z'")

# # Example usage:
# rotated_data = rotate_tomo(tomo_data_RS_DECONV, 0, axis='z')

# rs_datasets=[
#    #{'DATA':tomo_data_RS, 'Qv':tomo_data_RS_qs, 'label':'RS 128'},
#     {'DATA':tomo_data_RS_256, 'Qv':tomo_data_RS_qs_256, 'label':'RS 256'},
#     {'DATA':rotated_data, 'Qv':tomo_data_RS_qs_DECONV, 'label':'RS 256 DECONV'}
# ]


# # rs_datasets=[
# #     {'DATA':tomo_data_RS, 'Qv':tomo_data_RS_qs, 'label':'RS 128'}
# # ]





# # Generate indices up to 3rd order
# miller_indices = generate_miller_indices(8)
# hs = miller_indices[:, 0]
# ks = miller_indices[:, 1]
# ls = miller_indices[:, 2]
# # fig, close_peaks, close_peaks_hkl, filtered_datasets = plot_multi_reciprocal_space(
# #     rs_datasets,  # List of dicts: [{'magnitude': 3D array, 'Q': 4D array (shape: (nx,ny,nz,3)), 'label': str}, ...]
# #     cellinfo_data,
# #     hs, ks, ls,
# #     #thresholds=[0.00016, 0.0002, 0.0145],  # List of thresholds for each dataset
# #     thresholds=[0.00014+0.00006],#, 0.00014+0.00006, 0.0145],  # List of thresholds for each dataset
# #     q_cutoffs=[0.02],#, 0.07, 0.07],
# #     peak_distance_threshold=0.0085,
# #     colormaps=['inferno'], #'viridis', 'jet'],
# #     alphas=[0.1]#, 0.3, 1.0]
# # )
# # fig.show()

# parula = sio.loadmat('/home/beams0/PTYCHOSAXS/NN/ptychosaxsNN/utils/parula.mat')['cmap']
# parula_colors = ['rgb({:.0f},{:.0f},{:.0f})'.format(r*255, g*255, b*255) for r, g, b in parula]

# parula_blue = parula_colors[:20]  # slice lower values (bluer)
# parula_orange = parula_colors[-50:-30]  # slice higher values (yellows/oranges)


# jet_green_colorscale = [
#     (0.0, 'rgb(0, 255, 128)'),
#     (0.25, 'rgb(0, 255, 96)'),
#     (0.5, 'rgb(0, 255, 64)'),
#     (0.75, 'rgb(0, 255, 32)'),
#     (1.0, 'rgb(0, 255, 0)')
# ]

# jet_red_colorscale = [
#     (0.0, 'rgb(255, 128, 0)'),
#     (0.25, 'rgb(255, 64, 0)'),
#     (0.5, 'rgb(255, 32, 0)'),
#     (0.75, 'rgb(255, 16, 0)'),
#     (1.0, 'rgb(255, 0, 0)')
# ]


# ## BEST SETTINGS SO FAR
# # thresholds=[0.0001, 0.0134]
# # q_cutoffs=[0.073,0.073]
# # peak_distance_threshold=0.0105
# # dbscan_eps=0.08
# # dbscan_min_samples=12

# thresholds=[0.0001, 0.1]
# q_cutoffs=[0.074,0.074]
# # thresholds=[0.0001, 0.1]
# # q_cutoffs=[0.02,0.02]
# peak_distance_threshold=0.01#0.0105
# dbscan_eps=0.08#0.08
# dbscan_min_samples=7

# fig, close_peaks, close_peaks_hkl, filtered_datasets = plot_multi_reciprocal_space(
#     rs_datasets,  # List of dicts: [{'magnitude': 3D array, 'Q': 4D array (shape: (nx,ny,nz,3)), 'label': str}, ...]
#     cellinfo_data,
#     hs, ks, ls,
#     #thresholds=[0.0003, 0.00015, 0.0130],  # List of thresholds for each dataset
#     #q_cutoffs=[0.02,0.07,0.07],
#     thresholds=thresholds,  # List of thresholds for each dataset
#     q_cutoffs=q_cutoffs,
#     peak_distance_threshold=peak_distance_threshold,
#     #colormaps=['inferno','viridis', 'jet'],
#     colormaps=[parula_colors,parula_blue, parula_orange],
#     alphas=[0.6,0.5],#[0.3,0.6,0.6]
#     dbscan_eps=dbscan_eps, dbscan_min_samples=dbscan_min_samples
# ) 
# fig.show()
# n_peaks1, n_peaks2, n_overlapping = count_overlapping_peaks(
#     rs_datasets[0], rs_datasets[1],
#     threshold1=thresholds[0], threshold2=thresholds[1],
#     q_cutoff1=q_cutoffs[0], q_cutoff2=q_cutoffs[1],
#     overlap_distance=peak_distance_threshold,
#     dbscan_eps=dbscan_eps, dbscan_min_samples=dbscan_min_samples,
#     q_axes=[1,2,0], q_signs=[1,1,1], flatten_order='C'
# )

# print(f"Number of overlapping peaks: {n_overlapping}")
# print(f"Number of peaks in dataset 1: {n_peaks1}")
# print(f"Number of peaks in dataset 2: {n_peaks2}")


# results = peak_confusion_matrix(
#     true_dataset=rs_datasets[0], pred_dataset=rs_datasets[1],
#     true_threshold=thresholds[0], pred_threshold=thresholds[1],
#     true_q_cutoff=q_cutoffs[0], pred_q_cutoff=q_cutoffs[1],
#     overlap_distance=peak_distance_threshold,
#     dbscan_eps=dbscan_eps, dbscan_min_samples=dbscan_min_samples,
#     q_axes=[1,2,0], q_signs=[1,1,1], flatten_order='C',
#     plot=True,
#     overlay=True,
#     cellinfo_data=cellinfo_data,
#     miller_indices=miller_indices
# )

# print("TP: ", results['TP'])
# print("FP: ", results['FP'])
# print("FN: ", results['FN'])
# print("n_true: ", results['n_true'])
# print("n_pred: ", results['n_pred'])