# coding: utf-8
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from skimage.transform import resize
from ptychosaxsNN_utils import *
from UNet import recon_model
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter
from ptychosaxsNN_class import *
import cmasher as cmr
#from matplotlib.gridspec import GridSpec


# Load model
m=ptychosaxsNN()
path='/net/micdata/data2/12IDC/ptychosaxs/'
#path='/mnt/micdata2/12IDC/ptychosaxs/'


#model_path='models/best_model_diff_sim12_20000_512epochs.pth'
model_path='models/best_model_Unet_cindy.pth'


if model_path=='models/best_model_Unet_cindy.pth':
    # peak_params
    center_cut=75
    n=10
    threshold=0.25
elif model_path=='models/best_model_diff_sim12_20000_512epochs.pth':
    # peak_params
    center_cut=25
    n=5
    threshold=3e-2

m.load_model(state_dict_pth=path+model_path)
m.set_device()
m.model.to(m.device)
m.model.eval()

# For radius of circular averages
r=28

plot=True
save=False
all_peaks=[]
save_peaks=False
for scan in np.arange(1053,1287,1):#1287,1):
#for scan in np.arange(1125,1126,1):
    print(scan)
    # Load h5 data
    full_dps_orig=load_hdf5_scan_to_npy('/net/micdata/data2/12IDC/2021_Nov/ptycho/',scan=scan,plot=False)

    # Crop original data to 512x512 and remove hot pixel
    full_dps=full_dps_orig.copy()[:,1:513,259:771]
    for i,dp_pp in enumerate(full_dps):
        dp_pp[dp_pp >= 2**16-1] = np.min(dp_pp) #get rid of hot pixel


    # Summed scan 
    dps_copy=np.sum(full_dps,axis=0)

    # Preprocess and run data through NN
    resultT,sfT,bkgT=preprocess_cindy(dps_copy) # preprocess
    resultTa=resultT.to(device=m.device, dtype=torch.float) #convert to tensor and send to device
    final=m.model(resultTa).detach().to("cpu").numpy()[0][0] #pass through model and convert to np.array

    # Plot
    if plot:
        fig,ax=plt.subplots(1,3)
        im1=ax[0].imshow(dps_copy,norm=colors.LogNorm())
        im2=ax[1].imshow(resultT[0][0])
        im3=ax[2].imshow(final)
        plt.colorbar(im1)
        plt.colorbar(im2)
        plt.colorbar(im3)
        plt.show()
        
        
    

    # Find and plot peaks in NN result deconvolution, plot over frame
    # peaks=find_peaks_2d_filter(final,center_cut=center_cut,n=n,threshold=threshold,plot=False)
    peaks=find_peaks2d(final,center_cut=center_cut,n=n,threshold=threshold,plot=False)
    center_x,center_y=134,124
    adjusted=[]
    peaks_shifted=[(p[0]-center_x,p[1]-center_y) for p in peaks] 
    updated_peaks=ensure_inverse_peaks(peaks_shifted)
    updated_peaks_unshifted=[(p[0]+center_x,p[1]+center_y) for p in updated_peaks]
    peaks=updated_peaks_unshifted
    all_peaks.append({"scan":scan,"peaks":peaks})
    

    print(peaks)
    if len(peaks)==0:    
        print(f'no identified peaks in {scan}, continuing to next scan')
    else:
        peak_y,peak_x=zip(*peaks);
        # Caclulate peak intensities for summed frames (subtracting out first frame for background)
        full_dps_sub=full_dps.copy()
        dps_sum_sub=[]
        for i in range(0,len(full_dps_sub)):
            test=np.subtract(full_dps_sub[i],full_dps[0],dtype=float)
            dps_sum_sub.append(test)
        dps_sum_sub=np.array(dps_sum_sub)
        dps_sum_sub=np.sum(dps_sum_sub,axis=0)


        #Calculated integrated intensity of summed peaks
        radius = r  # Define the neighborhood radius
        intensities_sum=[]
        for peak in peaks:
            x, y = peak
            intensity = circular_neighborhood_intensity(resize_dp(dps_sum_sub), x, y, radius=radius,plot=False)#-bkg
            intensities_sum.append(intensity)
            print(f"Peak at ({x}, {y}) has neighborhood integrated intensity: {intensity}")


        fig,ax=plt.subplots(1,2);
        ax[0].imshow(resize_dp(dps_copy),norm=colors.LogNorm(),cmap='jet');
        ax[0].scatter(peak_x,peak_y,color='red',marker='x',s=100);
        ax[1].imshow(final,cmap='jet');
        ax[1].scatter(peak_x,peak_y,color='red',marker='x',s=100);
        if save:
            plt.savefig(f'/net/micdata/data2/12IDC/ptychosaxs/figures/sum{scan}.pdf')
            plt.close()
        else:
            plt.show()


        ref=full_dps[0]

        count=0
        ss=[]
        for i in tqdm(range(0,15)):#15)):
            for j in range(0,20):#20):
                #print(count)
                # Pick specific frame
                dps_index=np.asarray(full_dps[count])

                # Calculate normalized difference map between summed and frame
                ref_index=ref.copy()
                #ref_index[ref_index<=0]=np.min(ref_index[ref_index>0])
                test_index=dps_index.copy()
                #test_index[test_index<=0]=np.min(test_index[test_index>0])    
                test_copy=dps_copy.copy()
                #test_copy[test_copy<=0]=np.min(test_copy[test_copy>0])
                norm_diff=np.subtract(norm_0to1(test_copy),norm_0to1(test_index),dtype=float)

                # Subtracted from reference value
                # sub=np.subtract(norm_0to1(test_index),norm_0to1(ref_index),dtype=float)
                sub=np.subtract(test_index,ref_index,dtype=float)
                
                # Example: Calculate the integrated intensity for each peak's neighborhood from the normalized difference map
                radius = r  # Define the neighborhood radius
                intensities=[]
                    
                for peak in peaks:
                    x, y = peak
                    #intensity = neighborhood_intensity(resize_dp(norm_diff_copy), x, y, radius=radius)
                    #intensity = circular_neighborhood_intensity(resize_dp(test_copy), x, y, radius=radius)
                    intensity = circular_neighborhood_intensity(resize_dp(sub), x, y, radius=radius,plot=False)#-bkg
                    intensities.append(intensity)

                # Normalize intensities
                offset=1e-4 # so that all peaks (including intensity =0) are plotted
                #intensities=[(i-np.min(intensities))/(np.max(intensities)-np.min(intensities)) for i in intensities]#+ offset for i in intensities]
                #intensities=[i if i > 0.1 else 0 for i in intensities]
                # Plot the image with peaks
                #s=np.array(np.ones(np.array(intensities).shape)*offset+np.ones(np.array(intensities).shape)-np.array(intensities))
                #s=np.array(intensities)
                #s=np.array([intensities[i]/intensities_sum[i] for i in range(0,len(intensities))])*1e5
                s=np.array([intensities[i]/intensities_sum[i]/np.max(intensities_sum) for i in range(0,len(intensities))])#*1e10
                #print(s)
                alphas=[(i-np.min(s))/(np.max(s)-np.min(s)) for i in s]
                #print(alphas)
                ss.append(s)
                count+=1

        ss=np.asarray(ss)
        test_ss=np.array([(s-np.min(ss))/(np.max(ss)-np.min(ss)) for s in ss])
        count=0


        #cmap = cmr.rainforest                   
        #cmap = plt.get_cmap('cmr.rainforest')

        cmap = cmr.get_sub_cmap('jet', 0.0, 0.5)
        #cmap = 'jet'

        fig, axs = plt.subplots(15,20, sharex=True,sharey=True)#,figsize=(20-0.2,15))
        # Remove vertical space between Axes
        fig.subplots_adjust(hspace=0,wspace=0)
        for i in tqdm(range(0,15)):#15)):
            for j in range(0,20):#20):

                axs[i][j].imshow(resize_dp(full_dps[count]),norm=colors.LogNorm(),cmap=cmap)#;,clim=(1,1000));
                axs[i][j].axis("off")
                if count==0:
                    axs[i][j].scatter(peak_x,peak_y,color='red',marker='o',s=5)#,s=s);
                else:
                    axs[i][j].scatter(peak_x,peak_y,color='red',marker='o',alpha=test_ss[count],s=5);
                count+=1

        #plt.show()
        if save:
            plt.savefig(f'/net/micdata/data2/12IDC/ptychosaxs/figures/full{scan}.pdf', bbox_inches='tight')
            plt.close()
        else:
            plt.show()
if save_peaks:
    np.save('cindy_3_tomo_peaks',all_peaks)
