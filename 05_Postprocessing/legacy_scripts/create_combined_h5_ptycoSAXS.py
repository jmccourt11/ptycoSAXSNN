#%%
# Remove unused imports
import os
import matplotlib.pyplot as plt
from matplotlib import colors
import sys
import time
import pandas as pd
# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../../NN/ptychosaxsNN/')))
import utils.ptychosaxsNN_utils as ptNN_U
import importlib
import tqdm 
import h5py
import torch
import numpy as np
import random
#%%
# Add the models directory to the path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../src/models/')))

# Remove retry_h5_operation function
def retry_h5_operation(operation, max_retries=5, initial_delay=1):
    pass

# Remove get_processed_scans function
def get_processed_scans(h5_file):
    return set()

# Remove save_to_h5 function
def save_to_h5(output_file, result):
    pass

# Remove load_indices_from_h5 function
def load_indices_from_h5(indices_file, scan_number):
    pass

# Remove process_scan function
def process_scan(scan_number, base_path, model, device, center, dpsize, mask, indices_file=None):
    pass

# Remove save_verification_plot function
def save_verification_plot(results, output_dir='plots'):
    pass

# Remove remove_scans_from_h5 function
def remove_scans_from_h5(h5_file, scan_numbers_to_remove, create_backup=True):
    pass

# Remove add_metadata_to_h5 function
def add_metadata_to_h5(h5_file, df):
    pass

# Single Main Function for Deconvolution

def main():
    npzpath = '/scratch/preprocessed_sim_LatticeSC_Probe256x256_ZCB_9_3D__TOMO_TEST_sim_ZCB_9_3D_S5065_N180_steps4_dp256.npz'
    data = np.load(npzpath)

    deconv_output_file = '/scratch/deconvolved_patterns_from_sim.h5'

    convoluted_patterns = data['amp_conv_red']
    ideal_patterns = data['amp_ideal_red']

    model_path = "/scratch/trained_model/best_model_LatticeSC_Probe256x256_ZCB_9_3D__sim_ZCB_9_3D_S5065_N600_steps4_dp256_Unet_epoch_25_pearson_loss_symmetry_0.1.pth"
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    from encoder1 import recon_model as recon_model2
    model = recon_model2()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    with h5py.File(deconv_output_file, 'w') as f:
        f.create_group('ideal')
        f.create_group('convoluted')
        f.create_group('deconvolved')

        for i in tqdm(range(convoluted_patterns.shape[0])):
            convoluted_pattern = convoluted_patterns[i]
            ideal_pattern = ideal_patterns[i]

            convoluted_tensor = torch.tensor(convoluted_pattern).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
            with torch.no_grad():
                deconvolved_output = model(convoluted_tensor).cpu().numpy().squeeze()

            f['ideal'].create_dataset(f'scan_{i}', data=ideal_pattern)
            f['convoluted'].create_dataset(f'scan_{i}', data=convoluted_pattern)
            f['deconvolved'].create_dataset(f'scan_{i}', data=deconvolved_output)

    print("Deconvolution complete. Data saved to", deconv_output_file)
    

    

if __name__ == "__main__":
    main()
# %%

h5_path = 'deconvolved_patterns_from_sim.h5'
with h5py.File(h5_path, 'r') as f:
    ideal = f['ideal']
    convoluted = f['convoluted']
    deconvolved = f['deconvolved']

    ri = random.randint(0, 180)
    ideal_pattern = ideal['scan_{}'.format(ri)]
    convoluted_pattern = convoluted['scan_{}'.format(ri)]
    deconvolved_pattern = deconvolved['scan_{}'.format(ri)]

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(ideal_pattern, cmap='jet')
    ax[0].set_title('Ideal Pattern')
    ax[1].imshow(convoluted_pattern, cmap='jet')
    ax[1].set_title('Convoluted Pattern')
    ax[2].imshow(deconvolved_pattern, cmap='jet')
    ax[2].set_title('Deconvolved Pattern')
    plt.show()
# %%
