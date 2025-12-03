#%%
import os
import argparse
import random
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
import numpy as np

import torch
from tqdm import tqdm
from scipy.ndimage import rotate, zoom
from scipy.ndimage import gaussian_filter
import tifffile
import h5py
from matplotlib import colors
from ptycho_tomo_funcs import *

sample_dir = 'ZCB_9_3D_'
base_directory = '/net/micdata/data2/12IDC/2025_Feb/results/'
recon_path = 'roi0_Ndp256/MLc_L1_p10_gInf_Ndp128_mom0.5_pc0_maxPosError500nm_bg0.1_vi_mm/MLc_L1_p10_g100_Ndp256_mom0.5_pc800_maxPosError500nm_bg0.1_vp4_vi_mm/'
Niter = 1000
scan_number = 5065#5102
mask_path = '/home/beams/PTYCHOSAXS/deconvolutionNN/data/mask/mask_ZCB_9_3D.npy'
mask_np = np.load(mask_path)

# #1280x1280
# sample_dir = 'RC02_R3D_'
# base_directory = '/net/micdata/data2/12IDC/2024_Dec/results/'
# recon_path = 'roi0_Ndp1280/MLc_L1_p10_g50_Ndp1280_mom0.5_pc0_noModelCon_bg0.1_vi_mm/MLc_L1_p10_g50_Ndp1280_mom0.5_bg0.1_vp4_vi_mm/'
# Niter = 200
# scan_number = 888
# mask_path = '/home/beams/PTYCHOSAXS/deconvolutionNN/data/mask/mask_RC02_R3D_1280.npy'
# mask_np = np.load(mask_path)


# #128x128
# sample_dir = 'RC02_R3D_'
# base_directory = '/net/micdata/data2/12IDC/2024_Dec/results/'
# recon_path = 'roi0_Ndp512/MLc_L1_p10_gInf_Ndp128_mom0.5_pc200_model_scale_rotation_shear_asymmetry_noModelCon_bg0.1_vi_mm/'
# Niter = 1000
# scan_number = 888
# mask_path = '/home/beams/PTYCHOSAXS/deconvolutionNN/data/mask/mask_RC02_R3D_1280.npy'
# mask_np = np.load(mask_path)

result = run_tomo_simulation(
    #probe_mat="/net/micdata/data2/12IDC/2024_Dec/results/RC02_R3D_/fly888/roi0_Ndp512/MLc_L1_p10_gInf_Ndp128_mom0.5_pc200_model_scale_rotation_shear_asymmetry_noModelCon_bg0.1_vi_mm/Niter1000.mat",
    probe_mat=f"{base_directory}/{sample_dir}/fly{scan_number}/{recon_path}/Niter{Niter}.mat",
    probe_key="probe",
    probe_slice=0,
    target_size=1280,
    lattice_path=lattice_path,
    num_phi=1,
    scan_step_div_min=8,
    scan_step_div_max=8,
    phi_start_random=False,
    apply_initial_random_orientation=True,
    apply_random_rotation_during_simulation=False,
    batch_size=32,
    segment_size=256,
    add_lattice_noise=False,
    lattice_noise_std_fraction=0.3,
    lattice_noise_lowpass_sigma=3.0,
    add_projection_noise=False,
    projection_noise_std_fraction=0.0, #0.3
    projection_noise_lowpass_sigma=0.0,
    zoom_lattice=False,
    zoom_lattice_factor=3.0,
    #mask_path="/home/beams/PTYCHOSAXS/deconvolutionNN/data/mask/mask_sum_RC02_R3D_1280.npy",
    mask_path=mask_path, #THIS DOES NOT MATTER FOR THE SIMULATION
    output_dir=h5_out_dir,
    h5_suffix='TEST_TOMO_SIM',
    Nc_avg=1e4,
    #output_dir='/scratch/tomo_output',
    #seed=1234,  # optional for reproducibility
)

print(result)  # {'dp_count': ..., 'phi_start': ..., 'scan_step': ...}




#%%
plot_example=False
if plot_example:
    import matplotlib.pyplot as plt
    import h5py
    import numpy as np
    from matplotlib import colors
    import random

    h5_path='/home/beams/PTYCHOSAXS/NN/ptychosaxsNN/data/tomo_output/tomo_phi_000TESTNEW.h5'
    with h5py.File(h5_path, 'r') as f:
        print(f.keys())

        convDPs=f['convDPs'][()]
        idealDPs=f['idealDPs'][()]
        num_patterns=f['convDPs'][()].shape[0]
        print(f'{convDPs.shape=}')
        overlay_rgb=f['overlay_rgb'][()]
    #%%
    ri=random.randint(0, num_patterns-1)
    ri=0
    plt.imshow(convDPs[ri], norm=colors.LogNorm())
    plt.show()
    plt.imshow(idealDPs[ri], norm=colors.LogNorm())
    plt.show()

 
    # Reload mask for 256x256 patterns
    mask_path = '/home/beams/PTYCHOSAXS/deconvolutionNN/data/mask/mask_ZCB_9_3D.npy'
    mask_np = np.load(mask_path)

    plt.figure(figsize=(20,15))
    plt.subplot(1, 2, 1)
    plt.imshow(idealDPs[ri][1280//2-128:1280//2+128,1280//2-128:1280//2+128]*mask_np, norm=colors.LogNorm())
    plt.title('Sum of Ideal DPs')
    plt.subplot(1, 2, 2)
    plt.imshow(convDPs[ri][1280//2-128:1280//2+128,1280//2-128:1280//2+128]*mask_np, norm=colors.LogNorm())
    plt.title('Sum of Convoluted DPs')
    plt.show()

    plt.figure(figsize=(20,15))
    plt.subplot(1, 2, 1)
    plt.imshow(np.sum(idealDPs, axis=0), norm=colors.LogNorm())
    plt.title('Sum of Ideal DPs')
    plt.subplot(1, 2, 2)
    plt.imshow(np.sum(convDPs, axis=0), norm=colors.LogNorm())
    plt.title('Sum of Convoluted DPs')
    plt.show()

    plt.figure(figsize=(20,15))
    plt.subplot(1, 2, 1)
    plt.imshow(np.sum(idealDPs, axis=0)[1280//2-128:1280//2+128,1280//2-128:1280//2+128]*mask_np, norm=colors.LogNorm())
    plt.title('Sum of Ideal DPs')
    plt.subplot(1, 2, 2)
    plt.imshow(np.sum(convDPs, axis=0)[1280//2-128:1280//2+128,1280//2-128:1280//2+128]*mask_np, norm=colors.LogNorm())
    plt.title('Sum of Convoluted DPs')
    plt.show()
   # Randomly select n patterns and sum them
    n = 2  # Number of patterns to sum
    random_indices = random.sample(range(num_patterns), n)
    random_indices = [0,1,2]
    print(random_indices)
    random_sum_ideal = np.sum(idealDPs[random_indices], axis=0)
    random_sum_conv = np.sum(convDPs[random_indices], axis=0)

    # Crop to central 256x256 region
    center = 1280//2
    crop_size = 128  # Half of 256
    random_sum_ideal_crop = random_sum_ideal[center-crop_size:center+crop_size, center-crop_size:center+crop_size]
    random_sum_conv_crop = random_sum_conv[center-crop_size:center+crop_size, center-crop_size:center+crop_size]

    plt.figure(figsize=(20,15))
    plt.subplot(1, 2, 1)
    plt.imshow(random_sum_ideal_crop*mask_np, norm=colors.LogNorm())
    plt.title(f'Sum of {n} Random Ideal DPs (256x256)')
    plt.subplot(1, 2, 2)
    plt.imshow(random_sum_conv_crop*mask_np, norm=colors.LogNorm())
    plt.title(f'Sum of {n} Random Convoluted DPs (256x256)')
    plt.show()

    plt.figure()
    plt.imshow(overlay_rgb[:,:,0])
    plt.show()
    # print(f['scan_positions'][()].shape)
    # print(f['object_projection'][()].shape)
    # print(f['probe_amplitude'][()].shape)
    # print(f['probe_phase'][()].shape)
    # print(f['overlay_rgb'][()].shape)
    # %%
