#%%
import torch
import torch.nn as nn
import sys
import os
from pathlib import Path
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../'))) 
import importlib
#from utils.ptychosaxsNN_utils import * # run into issues with reimporting
import utils.ptychosaxsNN_utils as ptNN
importlib.reload(ptNN)
# from models.UNet import recon_model as recon_model_old

#from models.UNet512x512 import recon_model
import sys
sys.path.append('/home/beams/PTYCHOSAXS/deconvolutionNN/')
from src.models.encoder1 import recon_model

class ptychosaxsNN:
    def __init__(self):
        self.model=None
        self.nconv=64 #same number for convolutions for all networks so far
        self.probe=None
        self.device=None
        self.full_data=None
        self.sum_data=None
        self.ptycho_hdf5=None
        self.scan=None
        
    def __repr__(self):
        return f'ptychosaxsNN (model: {self.model!r}, probe: {self.probe!r}, device: {self.device!r})'
           
    #def load_probe(self,probefile):
    #    self.probe=np.load(probefile)
    
    def load_probe(self,probe_file,file_format='mat'):
        if file_format=='mat':
            # Load reconstructed probe from ptychoshelves recon and take only the first mode
            probe = sio.loadmat(probe_file)['probe'][:,:,0,0]
            self.probe = probe
        else:
            print('Need *.mat formatted probe')
            
    def load_model(self,state_dict_pth=None,load_state_dict=True):
        #self.model= recon_model(self.nconv)
        self.model= recon_model()

        if load_state_dict and state_dict_pth!=None:
            # Load the state_dict on the CPU first to avoid memory issues
            state_dict=torch.load(state_dict_pth, map_location=torch.device('cpu'))
            self.model.load_state_dict(state_dict)
        
    # def set_device(self,gpu_index=None):
    #     self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     if gpu_index is not None:
    #         self.device = torch.device(f"cuda:{gpu_index}")
    #     else:
    #         self.device = torch.device("cpu")
    #     self.model = self.model.to(self.device)
        
    #     if torch.cuda.device_count() > 1:
    #         print("Let's use", torch.cuda.device_count(), "GPUs!")
    #         #self.model = nn.parallel.DistributedDataParallel(self.model)
    #         self.model = nn.DataParallel(self.model)

    #     self.model = self.model.to(self.device)
    
    def set_device(self, gpu_index=None):
        if torch.cuda.is_available():
            if gpu_index is not None:
                # Use only the specified GPU
                self.device = torch.device(f"cuda:{gpu_index}")
                self.model = self.model.to(self.device)
                print(f"Using GPU {gpu_index}")
                # Do NOT wrap with DataParallel
            else:
                # Use all GPUs (default)
                self.device = torch.device("cuda")
                self.model = self.model.to(self.device)
                if torch.cuda.device_count() > 1:
                    print("Let's use", torch.cuda.device_count(), "GPUs!")
                    self.model = nn.DataParallel(self.model)
        else:
            self.device = torch.device("cpu")
            self.model = self.model.to(self.device)
            print("No GPU available")
        
    def model_size_in_megabytes(self):
        param_size = 0
        for param in x.model.parameters():
            param_size += param.numel() * param.element_size()  # numel() gives number of elements, element_size() gives size in bytes
        param_size /= (1024 ** 2)  # Convert bytes to megabytes
        print(f"Model size: {param_size:.2f} MB")
        
    def load_h5(self,exp_dir,scan):
        #e.g. 2024_Dec , 630
        file_path=Path(f'/net/micdata/data2/12IDC/{exp_dir}/ptycho/')
        res=ptNN.load_h5_scan_to_npy(file_path=file_path,scan=scan,plot=False)
        self.full_data=res
        self.sum_data=np.sum(res,axis=0)
        self.scan=scan

    def load_hdf5(self,exp_dir,sample_name,scan):
        # e.g. 2024_Dec , JM02_3D_ , 630
        file_path=Path(f'/net/micdata/data2/12IDC/{exp_dir}/results/{sample_name}/')
        self.ptycho_hdf5=ptNN.load_hdf5_scan_to_npy(file_path=file_path,scan=scan,plot=False)
        self.scan=scan
        
#%%
outputs=[]
inputs=[]         
if __name__ == "__main__":
    x=ptychosaxsNN()
    #path = os.path.abspath(os.path.join(os.getcwd(), '../'))
    local=True
    if local:
        path = Path("Y:/ptychosaxs")
    else:
        path = Path('/net/micdata/data2/12IDC/ptychosaxs/')
    #%%    
    x.load_model(state_dict_pth=path / 'trained_model/best_model.pth')   
    x.set_device()
    x.model.to(x.device)
    x.model.eval()
    


#    date_dir = '2024_Dec'
#    exp_dir='JM03_3D_'
#    scans=np.arange(706,720,1)
    
    date_dir = '2024_Dec'
    exp_dir='RC_01_'
    #scans=np.arange(706,315,1)
    scans=[315]


#    date_dir = '2024_Dec'
#    exp_dir='JM02_3D_'
#    scans=np.arange(445,635,1)
#    #scans=[500]
    #%%
    directory='results' #'test'
    Ndp=1408
    #Ndp=512
    for scan in scans:
        try:
            #filepath=Path("Y:/") / f'{date_dir}/{directory}/{exp_dir}/fly{scans[0]:03d}/data_roi0_Ndp{Ndp}_dp.hdf5'
            #filepath=Path("Y:/") / f'{date_dir}/{directory}/{exp_dir}/fly{scan:03d}/data_roi0_Ndp{Ndp}_dp.hdf5'
            filepath=Path(f'/net/micdata/data2/12IDC/{date_dir}/{directory}/{exp_dir}/fly{scans[0]:03d}/data_roi0_Ndp{Ndp}_dp.hdf5')
            data=ptNN.read_hdf5_file(filepath)
            dps_copy = np.sum(data['dp'],axis=0)
            
            #dps_copy=np.load(Path("Y:/ptychosaxs/data/diff_sim/14/output_hanning_conv_00001.npz"))['convDP']
            
            
            # # Load data
            # scan=1125 #1115,1083,1098
            # filename = path / f'data/cindy_scan{scan}_diffraction_patterns.npy'
            # full_dps_orig=np.load(filename)
            # full_dps=full_dps_orig.copy()
            # for dp_pp in full_dps:
            #     dp_pp[dp_pp >= 2**16-1] = np.min(dp_pp) #get rid of hot pixel
            
            # # Plot and return a full scan
            # # inputs,outputs,sfs,bkgs=plot_and_save_scan(full_dps,x,scanx=20,scany=15)
            
            # # Summed scan 
            # dps_copy=np.sum(full_dps[:,1:513,259:771],axis=0)
            
            # # Specific frame
            # index=230
            # dps_copy=full_dps[index,1:513,259:771]

            # Preprocess and run data through NN
            #mask=np.load(f'/net/micdata/data2/12IDC/{date_dir}/mask1408.npy')
            #d=ptNN.read_hdf5_file('/net/micdata/data2/12IDC/2024_Dec/results/JM02_3D_/fly446/data_roi0_Ndp512_dp.hdf5')
            
            #%%
            d=ptNN.read_hdf5_file(Path("Y:/") / f'{date_dir}/{directory}/{exp_dir}/fly482/data_roi0_Ndp{Ndp}_dp.hdf5')
            mask = np.sum(d['dp'],axis=0)<=0
            #%%
            #mask=np.load(Path("Y:/") / f'{date_dir}/mask1408.npy')
            #mask=np.load(Path(f'/net/micdata/data2/12IDC/{date_dir}/mask1408.npy'))
            mask=np.ones(mask.shape)-mask
            
            plt.figure()
            plt.imshow(mask,cmap='jet')
            plt.show()
            #%%
            waxs_mask=np.load("/home/beams/PTYCHOSAXS/NN/waxs_mask.npy")
           
            
            center_decays=[0]#,0.01,0.1,0.3,0.5,0.8,1,1.5,2,3,4]
            for c in center_decays:
                #resultT,sfT,bkgT=ptNN.preprocess_zhihua(dps_copy,mask,center_decay=c) # preprocess
                resultT,sfT,bkgT=ptNN.preprocess_zhihua2(dps_copy,mask,waxs_mask) # preprocess
                resultTa=resultT.to(device=x.device, dtype=torch.float) #convert to tensor and send to device
                final=x.model(resultTa).detach().to("cpu").numpy()[0][0] #pass through model and convert to np.array
                
                # Plot
                fig,ax=plt.subplots(1,3)
                im1=ax[0].imshow(dps_copy,norm=colors.LogNorm(),cmap='jet')
                im2=ax[1].imshow(resultT[0][0],clim=(np.max(resultT[0][0].numpy())-0.2,np.max(resultT[0][0].numpy())),cmap='jet')
                im3=ax[2].imshow(final,clim=(np.max(final)-0.2,np.max(final)),cmap='jet')
                plt.colorbar(im1)
                plt.colorbar(im2)
                plt.colorbar(im3)
                plt.show()
                outputs.append(final)
                inputs.append(resultT[0][0])
        except:
            print('error reading file')
            continue
#outputs=np.asarray(outputs)
#inputs=np.asarray(inputs)
#np.savez("JM02_3D_deconv.npz",deconv=outputs,conv=inputs)
# %%
