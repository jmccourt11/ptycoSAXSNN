#%%
import sys
from tqdm import tqdm 
import numpy as np
from skimage.transform import resize
from numpy.fft import fftn, fftshift
import matplotlib.pyplot as plt
import os
from matplotlib import colors
import matplotlib.pyplot as plt
import random
from multiprocessing import Pool, get_context
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from joblib import Parallel, delayed
from pathlib import Path
import h5py
import torch
import cupy as cp
from cupyx.scipy.signal import convolve2d as conv2
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../')))
from src.utils.preprocessing_functions import *
plt.rcParams["image.cmap"] = "jet"
#%%
#save processed data
save=True

#shuffle data
shuffle_data=False

#resize data
red=False 

# Setting path
# path = Path("Y:/ptychosaxs")  # /net/micdata/data2/12IDC mounted windows drive
path = Path("/net/micdata/data2/12IDC/ptychosaxs")  # on remote server
print(path)

lattice_list=['ClathII','SC']
noise_list=['Noise']
probe_size_list=[256]
N=180

#%%
for lattice in lattice_list:
    for noise in noise_list:
        for probe_size in probe_size_list:
            h5file_data=f'/net/micdata/data2/12IDC/ptychosaxs/batch_mode_250/tomo_tests/Lattice{lattice}_Probe{probe_size}x{probe_size}_{noise}/sim_ZCB_9_3D_S5065_N{N}_steps4_dp256.h5'
            save_string=h5file_data.split('/')[-2].split('.')[0]+'_'+h5file_data.split('/')[-1].split('.')[0]
            print(save_string)
            
            # Load data directly from the HDF5 file
            print(f"Loading data from: {h5file_data}")

            with h5py.File(h5file_data, "r") as h5f:
                # Load convDP and pinholeDP_raw_FFT as conv and ideal diffraction patterns
                conv_DPs = h5f['convDP'][:]  # Shape: (10800, 256, 256)
                ideal_DPs = h5f['pinholeDP_raw_FFT'][:]  # Shape: (10800, 256, 256)
                
                num_patterns = len(conv_DPs)
                print(f"Loaded {num_patterns} diffraction patterns")
                print(f"Pattern shapes - conv_DPs: {conv_DPs.shape}, ideal_DPs: {ideal_DPs.shape}")

            # Create dummy probe array (as before)
            probe_DPs = np.ones(conv_DPs.shape)  # dummy array for testing network with a probe

            # Display a random pattern to verify the data
            ri = np.random.randint(0, len(conv_DPs))
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            im1 = ax[0].imshow(conv_DPs[ri], norm=colors.LogNorm())
            ax[0].set_title("Sample Convoluted DP")
            plt.colorbar(im1, ax=ax[0])

            im2 = ax[1].imshow(ideal_DPs[ri], norm=colors.LogNorm())
            ax[1].set_title("Sample Ideal DP")
            plt.colorbar(im2, ax=ax[1])

            plt.tight_layout()
            plt.show()
            
            # Load your detector mask as before
            detector_mask = np.load(os.path.abspath(os.path.join(os.getcwd(), '../../data/mask/mask_ZCB_9_3D.npy')))

            # Apply both masks to each image
            processed_conv_DPs = []
            processed_ideal_DPs = []

            # Process each image
            for i in tqdm(range(len(conv_DPs))):
                # First apply the detector mask
                masked_conv = apply_mask(conv_DPs[i], detector_mask)
                #masked_ideal = apply_mask(ideal_DPs[i], detector_mask)
                masked_ideal=ideal_DPs[i]
                
                processed_conv_DPs.append(masked_conv)
                processed_ideal_DPs.append(masked_ideal)
            # Convert to numpy arrays
            processed_conv_DPs = np.array(processed_conv_DPs)
            processed_ideal_DPs = np.array(processed_ideal_DPs)
            
            #shuffle data correspondingly
            if shuffle_data:
                indices = np.arange(processed_conv_DPs.shape[0])
                np.random.shuffle(indices)
                conv_DPs_shuff = processed_conv_DPs[indices]
                ideal_DPs_shuff = processed_ideal_DPs[indices]
                probe_DPs_shuff=probe_DPs[indices]
            else:
                conv_DPs_shuff = processed_conv_DPs
                ideal_DPs_shuff = processed_ideal_DPs
                probe_DPs_shuff=probe_DPs
            
            # separate amplitude and phase
            # thus network used the amplitude of the intensity patterns
            amp_conv = log10_custom(conv_DPs_shuff)
            amp_ideal = log10_custom(ideal_DPs_shuff)
            amp_probe = np.abs(probe_DPs_shuff)
            phase_conv = np.angle(conv_DPs_shuff)
            phase_ideal = np.angle(ideal_DPs_shuff)
            phase_probe = np.angle(probe_DPs_shuff)
            
            # resize data
            if red:
                print("Resizing...")
                amp_ideal_red=np.asarray([resize(d[:,:],(256,256),preserve_range=True,anti_aliasing=True) for d in tqdm(amp_ideal)])
                amp_conv_red=np.asarray([resize(d[:,:],(256,256),preserve_range=True,anti_aliasing=True) for d in tqdm(amp_conv)])
                amp_probe_red=np.asarray([resize(d[:,:],(256,256),preserve_range=True,anti_aliasing=True) for d in tqdm(amp_probe)])
            else:
                print("No resizing...")
                amp_ideal_red=amp_ideal
                amp_conv_red=amp_conv
                amp_probe_red=amp_probe
            
            ri=random.randint(0,len(amp_conv))
            print(f'maximum in ideal patterns: {np.max(amp_ideal_red[ri])}')
            print(f'maximum in conv patterns: {np.max(amp_conv_red[ri])}')
            
            # Scale factors
            #NORMALIZE OUTPUT FROM 0 to 1
            ideal_scale_factors=np.asarray([(np.max(a)-np.min(a)) for a in amp_ideal_red])
            ideal_constants=np.asarray([(np.min(a)) for a in amp_ideal_red])
            amp_ideal_red=np.asarray([(a-np.min(a))/(np.max(a)-np.min(a)) for a in amp_ideal_red])

            # #NORMALIZE INPUT FROM 0 to 1
            conv_scale_factors=np.asarray([(np.max(a)-np.min(a)) for a in amp_conv_red])
            conv_constants=np.asarray([(np.min(a)) for a in amp_conv_red])
            amp_conv_red=np.asarray([(a-np.min(a))/(np.max(a)-np.min(a)) for a in amp_conv_red])

            # #NORMALIZE INPUT FROM 0 to 1
            probe_scale_factors=np.asarray([(np.max(a)-np.min(a)) for a in amp_probe_red])
            probe_constants=np.asarray([(np.min(a)) for a in amp_probe_red])
            amp_probe_red=np.asarray([(a-np.min(a))/(np.max(a)-np.min(a)) for a in amp_probe_red])

            print(f'normalized maximum in ideal/output patterns: {np.max(amp_ideal_red[0])}')
            print(f'normalized maximum in conv/input patterns: {np.max(amp_conv_red[0])}')

            fig,ax=plt.subplots(1,2,layout='constrained')
            im1=ax[1].imshow(amp_ideal_red[0])#,norm=colors.LogNorm())
            plt.colorbar(im1, ax=ax[1], format='%.2f')
            im2=ax[0].imshow(amp_conv_red[0])#,norm=colors.LogNorm())
            plt.colorbar(im2, ax=ax[0], format='%.2f')
            plt.show()

            if save:
                np.savez(os.path.abspath(os.path.join(os.getcwd(), f'/net/micdata/data2/12IDC/ptychosaxs/batch_mode_250/tomo_tests/Lattice{lattice}_Probe{probe_size}x{probe_size}_{noise}/preprocessed_sim_{save_string}.npz')),amp_conv_red=amp_conv_red,amp_ideal_red=amp_ideal_red,amp_probe_red=amp_probe_red)
                print('save string:', save_string)
                #print(f'saved preprocessed data to {os.path.abspath(os.path.join(os.getcwd(), f'/scratch/preprocessed_sim_{save_string}.npz'))}')


# %%
# Load saved data
lattice='ClathII'
noise='Noise'
probe_size=256
N=180
data = np.load(os.path.abspath(os.path.join(os.getcwd(), f'/net/micdata/data2/12IDC/ptychosaxs/batch_mode_250/tomo_tests/Lattice{lattice}_Probe{probe_size}x{probe_size}_{noise}/preprocessed_sim_Lattice{lattice}_Probe{probe_size}x{probe_size}_{noise}_sim_ZCB_9_3D_S5065_N{N}_steps4_dp256.npz')))
conv_data = data['amp_conv_red']
ideal_data = data['amp_ideal_red']
#probe_data = data['amp_probe_red']
# %%
# import matplotlib.pyplot as plt
# from IPython.display import display, clear_output

# plt.ion()  # Enable interactive mode

# # Plot saved data
# fig, ax = plt.subplots(1, 2, figsize=(15, 5))

# # Initial plots, assuming `amp_conv_red` and `amp_ideal_red` are arrays
# img1 = ax[0].imshow(amp_conv_red[0])
# ax[0].set_title('Convoluted DP')
# img2 = ax[1].imshow(amp_ideal_red[0])
# ax[1].set_title('Ideal DP')

# for i in range(10):
#     ri = i * 16 + 1
    
#     # Update data in the existing plots
#     img1.set_data(amp_conv_red[ri])
#     img2.set_data(amp_ideal_red[ri])
    
#     fig.canvas.flush_events()
#     display(fig)
#     plt.pause(0.5)
#     clear_output(wait=True)

#%%


import sys
import os
import importlib

# Add the models directory to the path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../src/models/')))

# # First, try to import the module
# try:
#     import encoder1_no_Unet
#     # Force reload the module
#     importlib.reload(encoder1_no_Unet)
#     # Now import the class from the freshly reloaded module
#     from encoder1_no_Unet import recon_model
#     print("Successfully imported recon_model")
#     unet_status = "no_Unet"
#     loss_function = "L2"
# except Exception as e:
#     print(f"Import error: {e}")
    
#First, try to import the module
try:
    import encoder1
    importlib.reload(encoder1)
    # Now import the class from the freshly reloaded module
    from encoder1 import recon_model
    print("Successfully imported recon_model")
    unet_status = "Unet"
    loss_function = "pearson_loss"
    #loss_function = "L2"
except Exception as e:
    print(f"Import error: {e}")

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
#convert to tensor
selected_index=np.random.randint(0, len(conv_data))

for i in range(0,180):
    for j in range(0,16):
        #selected_index=1297
        print(f"selected_index: {selected_index}")
        dp_size=256
        conv_data_test=torch.tensor(conv_data[selected_index].reshape(1,1,dp_size,dp_size))
        ideal_data_test=torch.tensor(ideal_data[selected_index].reshape(1,1,dp_size,dp_size))
        print(conv_data_test.shape, ideal_data_test.shape)

        #
        model_path='/net/micdata/data2/12IDC/ptychosaxs/batch_mode_250/trained_model/best_model_LatticeClathII_Probe256x256_ZCB_9_3D__Noise_sim_ZCB_9_3D_S5065_N600_steps4_dp256_Unet_epoch_25_pearson_loss_symmetry_0.0.pth'
        #load model
        model = recon_model()

        model.load_state_dict(torch.load(model_path))
        model.to(device=device, dtype=torch.float)
        model.eval()
        #forward pass
        result_test = model(conv_data_test.to(device=device, dtype=torch.float))
        #plot results
        fig,ax=plt.subplots(1,3,figsize=(20,5))
        #plot predicted data
        result_numpy = result_test.detach().to("cpu").numpy()[0][0]

        def norm_0to1(array):
            return (array-np.min(array))/(np.max(array)-np.min(array))

        im1=ax[0].imshow(conv_data_test[0][0],cmap='jet')
        im2=ax[1].imshow(ideal_data_test[0][0],cmap='jet')
        im3=ax[2].imshow(norm_0to1(result_numpy),cmap='jet')
        # im4=ax[3].imshow(masked_result,cmap='jet')
        ax[0].set_title('Conv DP')
        ax[1].set_title('Ideal DP')
        ax[2].set_title('Predicted DP (0.0)')# (0.0)')
        plt.colorbar(im1)
        plt.colorbar(im2)
        plt.colorbar(im3)
        plt.show()
# %%
