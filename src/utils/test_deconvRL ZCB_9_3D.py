#%%
import numpy as np
import sys
import os
from pathlib import Path
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
from time import perf_counter
import cupy as cp
from matplotlib import colors
plt.rcParams['image.cmap']='jet'

from skimage.restoration import richardson_lucy

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../src/')))
import utils.ptychosaxsNN_utils as ptNN_U
from utils.deconvolutionRL import *
import importlib
importlib.reload(ptNN_U)

def interpolate(dp,cmap='jet'):
    size=dp.shape
    grid_x,grid_y=np.meshgrid(np.arange(0,size[0],1),np.arange(0,size[1],1))
    grid_x=grid_x.T
    grid_y=grid_y.T
    points=np.asarray([(x,y) for x in range(0,size[0],1) for y in range(0,size[1],1)])
    d=dp
    
    #create mask of detector image
    mask=list(zip(np.where(d<=0)[0],np.where(d<=0)[1]))
    mask=np.asarray([list(m) for m in mask])

    plt.plot(mask[:,1],mask[:,0],color='k',alpha=0.5)

    plt.imshow(d,norm=colors.LogNorm(),cmap=cmap);plt.clim(1,1000);plt.plot(mask[:,1],mask[:,0],color='k',alpha=0.5); plt.show()
    plt.show()
    
    
    #filter out mask points
    out=points
    for m in mask:
        row = np.where( (out == m).all(axis=1))
        out=np.delete(out, row, axis=0)
    
    
    #interpolate
    #d=mat['dt'].T[0]
    values=d[out[:,0], out[:,1]]
    method='linear'
    grid_z2 = griddata(out, values, (grid_x, grid_y), method=method)
    plt.imshow(grid_z2, norm=colors.LogNorm(),cmap=cmap)
    plt.title(method+' (griddata method)')
    plt.show()
    return grid_z2
            

#%%
#set paths
basepath1=Path("/net/micdata/data2/12IDC/2025_Feb/ptycho/")

scan_num=5065
dps=ptNN_U.load_h5_scan_to_npy(basepath1, scan_num, plot=False,point_data=True)

#%%
ri=666
fig,ax=plt.subplots(1,2)
ax[0].imshow(dps[ri],norm=colors.LogNorm(),cmap='jet')
ax[0].set_title('Diffraction Pattern',fontsize=12)
ax[1].imshow(dps[ri],norm=colors.LogNorm(),cmap='jet',clim=(1,1000))
ax[1].set_title('Rescaled (RL)',fontsize=12)
plt.show()
#%%
probe_scan=5065
probe=loadmat(f'/net/micdata/data2/12IDC/2025_Feb/results/ZCB_9_3D_/fly{probe_scan}/roi0_Ndp256/MLc_L1_p10_gInf_Ndp128_mom0.5_pc0_maxPosError500nm_bg0.1_vi_mm/MLc_L1_p10_g100_Ndp256_mom0.5_pc800_maxPosError500nm_bg0.1_vp4_vi_mm/Niter1000.mat')['probe']
print(probe.shape)

#%%
#########################################################################################################
#probe_tests_new=object_tests_new['probe'].T[0][0].T #take first mode
probe_tests_new=probe.T[0][0].T #take first mode
probe_tests_new_FT=np.abs(np.fft.fftshift(np.fft.fft2(probe_tests_new)))

size=64
psf=probe_tests_new_FT[probe_tests_new_FT.shape[0]//2-size:probe_tests_new_FT.shape[0]//2+size,probe_tests_new_FT.shape[1]//2-size:probe_tests_new_FT.shape[1]//2+size] #zhihua

#crop probe
probe_gray=(psf*255/np.max(psf)).astype(np.uint8)
bgr = cv2.cvtColor(probe_gray, cv2.COLOR_GRAY2BGR)
img = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
img = cv2.resize(img,(256,256))
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
#_,thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
_,thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
edges = cv2.dilate(cv2.Canny(thresh,0,255),None)
cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
mask = np.zeros((256,256), np.uint8)
masked = cv2.drawContours(mask, [cnt],-1, 255, -1)
mask_2=cv2.resize(mask,psf.shape)
psf_masked=cv2.bitwise_and(psf,psf,mask=mask_2)
plt.imshow(psf_masked,norm=colors.LogNorm())
plt.clim(1,1000)
plt.show()
psf=psf_masked
#normalize psf
psf_norm=psf/np.max(psf)
psf=psf_norm
plt.imshow(psf,norm=colors.LogNorm())
plt.colorbar()
plt.show()

#%%
device=1
count=0
result_test=[]
probes=[]

full_dps=np.sum(dps,axis=0)
print(full_dps.shape)


#%%

#scan informtion
center=(517,575)

#crop diffraction patterns
dpsize=256
dp=full_dps[center[0]-dpsize//2:center[0]+dpsize//2,
    center[1]-dpsize//2:center[1]+dpsize//2]

plt.imshow(dp,norm=colors.LogNorm())
plt.show()


dp=interpolate(dp)


dp_norm=dp.copy()
dp_norm=np.where(dp_norm<=0,np.min(dp_norm[dp_norm>0]),dp_norm)
dp_norm=np.where(np.isnan(dp_norm),np.min(dp_norm[dp_norm>0]),dp_norm)
sf=np.max(dp_norm)-np.min(dp_norm)
bkg=np.min(dp_norm)
dp_norm=np.asarray((dp_norm-bkg)/(sf)) 
plt.imshow(dp_norm,norm=colors.LogNorm())
plt.colorbar()
plt.show()

#%%

SNRs=[]
ITERATIONS=[]
TIMES=[]
figs=[]
PSNRs=[]

def psnr_iterative(prev, curr, max_val=1.0):
    mse = np.mean((curr - prev)**2)
    return float('inf') if mse == 0 else 10 * np.log10((max_val**2) / mse)
mask = np.load('/home/beams/PTYCHOSAXS/deconvolutionNN/data/mask/mask_ZCB_9_3D.npy')

def preprocess_dp(dp,mask):
    size=256
    dp_pp=dp
    dp_pp=dp_pp*mask
    #dp_pp=ptNN_U.log10_custom(dp_pp)
    sf=np.max(dp_pp)-np.min(dp_pp)
    bkg=np.min(dp_pp)
    dp_pp=np.asarray((dp_pp-bkg)/(sf))
    return dp_pp






iterations=25
    
psf=cp.asarray(psf_norm) 
dp=cp.asarray(dp_norm)

#initialize GPU timer
start_time=perf_counter()        
result = RL_deconvblind(dp, psf, iterations,TV=False)
result_cpu=result.get()
dp_cpu=dp.get()
psf_cpu=psf.get()

snr=np.mean(result_cpu)/np.std(result_cpu)
print("SNR: ",snr)

#calculate time of deconvolution on GPU
cp.cuda.Device(device).synchronize()
stop_time = perf_counter()
time=str(round(stop_time-start_time,4))

#remove nan and zero
result_cpu=np.where(result_cpu<=0,np.min(result_cpu[result_cpu>0]),result_cpu)
result_cpu=np.where(np.isnan(result_cpu),np.min(result_cpu[result_cpu>0]),result_cpu)

print("time: ",time)

#%%
fig=plotter([psf_cpu,dp_cpu,result_cpu],['psf','dp','recovered'],log=True)
#plt.savefig(f'TEMP/iterTEST{iterations}.png')
#plt.close(fig)

fig,ax=plt.subplots(1,3,figsize=(15*2,5*2))
plt.subplots_adjust(wspace=0.3)  # Add horizontal padding between subplots

im1=ax[1].imshow(dp_cpu,norm=colors.LogNorm())
normalized_result=result_cpu
normalized_result=(result_cpu-np.min(result_cpu))/(np.max(result_cpu)-np.min(result_cpu))
#normalized_result = np.nan_to_num(normalized_result, nan=0.0)
normalized_result = np.clip(normalized_result, 1e-6, 1)
im2=ax[2].imshow(normalized_result,norm=colors.LogNorm())
im3=ax[0].imshow(psf_cpu,norm=colors.LogNorm())#,clim=(.5e3,1.4e8))
ax[1].set_title('Diffraction Pattern',fontsize=26)
ax[2].set_title('RL Reconstructed Image',fontsize=26)
ax[0].set_title('PSF',fontsize=26)
for axis in ax:
    axis.tick_params(axis='both', which='major', labelsize=20)
    axis.tick_params(axis='both', which='minor', labelsize=20)

cbar1 = plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
cbar2 = plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
cbar3 = plt.colorbar(im3, ax=ax[0], fraction=0.046, pad=0.04)

# Set font sizes for labels and ticks
count=0
for cbar in [cbar1, cbar2, cbar3]:
    if count==0 or count==2:
        cbar.set_label(label='Intensity',fontsize=20)
    else:
        cbar.set_label(label='Normalized Intensity',fontsize=20)
    cbar.ax.tick_params(labelsize=16)
    count+=1
plt.show()












#%%


from skimage.restoration import estimate_sigma
from skimage.filters import sobel


tv_list=[]
iters_list = list(range(1, 200, 5))  # Adjust step size as needed (e.g., 5)

for iters in iters_list:
    # initialize GPU timer
    start_time = perf_counter()

    # Run RL deconvolution
    result = RL_deconvblind(dp, psf, iters, TV=False)
    result_cpu = result.get()
    # Optional: clip or floor to avoid nan/inf
    result_cpu = np.clip(result_cpu, 1e-8, np.max(result_cpu))

    grad = np.sqrt(np.square(np.diff(result_cpu, axis=0, append=0)) +
                   np.square(np.diff(result_cpu, axis=1, append=0)))
    tv = np.sum(np.abs(grad))
    tv_list.append(tv)

    cp.cuda.Device(device).synchronize()
    print(f"Iter: {iters}, TV: {tv:.4f}, Time: {round(perf_counter() - start_time, 2)}s")

# Plot SNR vs iterations
plt.figure(figsize=(6,4))
plt.plot(iters_list, tv_list, marker='o')
plt.xlabel('Iterations')
plt.ylabel('TV')
plt.title('TV vs Iterations')
plt.grid(True)
plt.tight_layout()
#plt.savefig('TEMP/TV_vs_iterations.png')
plt.show()

















#%%

for i in tqdm(range(20,21)):
    #deconvolute dp and PSF
    iterations=i+1
    prev_iteration=i
    
    # #normalize dp and psf
    # dp_norm=dp/np.max(dp)
    # psf_norm=psf/np.max(psf)

   
    print(psf.shape)
    print(dp.shape)
    #preprocess dp
    # dp_norm=dp*mask
    # #remove nan and zero
    dp_norm=dp.copy()
    dp_norm=np.where(dp_norm<=0,np.min(dp_norm[dp_norm>0]),dp_norm)
    dp_norm=np.where(np.isnan(dp_norm),np.min(dp_norm[dp_norm>0]),dp_norm)
    sf=np.max(dp_norm)-np.min(dp_norm)
    bkg=np.min(dp_norm)
    dp_norm=np.asarray((dp_norm-bkg)/(sf))
    
    #normalize psf
    psf_norm=psf/np.max(psf)
    
    # psf=cp.asarray(psf_norm) 
    # dp=cp.asarray(dp_norm)
        
    psf=cp.asarray(psf) 
    dp=cp.asarray(dp)

    #initialize GPU timer
    start_time=perf_counter()        

    result = RL_deconvblind(dp, psf, iterations,TV=False)
    result_prev = RL_deconvblind(dp, psf, prev_iteration,TV=False)
    result_cpu=result.get()
    result_prev_cpu=result_prev.get()
    dp_cpu=dp.get()
    psf_cpu=psf.get()

    snr=np.mean(result_cpu)/np.std(result_cpu)
    # print("SNR: ",snr)
    SNRs.append(snr)
    psnr=psnr_iterative(result_prev_cpu,result_cpu)
    PSNRs.append(psnr)
    print(f"PSNR: {psnr}")
    #calculate time of deconvolution on GPU
    cp.cuda.Device(device).synchronize()
    stop_time = perf_counter( )
    time=str(round(stop_time-start_time,4))
    
    #remove nan and zero
    result_cpu=np.where(result_cpu<=0,np.min(result_cpu[result_cpu>0]),result_cpu)
    result_cpu=np.where(np.isnan(result_cpu),np.min(result_cpu[result_cpu>0]),result_cpu)
    

    ITERATIONS.append(iterations)
    # print("time: ",time)
    TIMES.append(time)

    fig=plotter([psf_cpu,dp_cpu,result_cpu],['psf','dp','recovered'],log=True)
    #plt.savefig(f'TEMP/iterTEST{i}.png')
    #plt.close(fig)
fig,ax=plt.subplots(1,3,figsize=(15,5))
im1=ax[0].imshow(dp_cpu,norm=colors.LogNorm())
im2=ax[1].imshow(result_cpu,norm=colors.LogNorm())
im3=ax[2].imshow(psf_cpu,norm=colors.LogNorm())
plt.colorbar(im1,ax=ax[0])
plt.colorbar(im2,ax=ax[1])
plt.colorbar(im3,ax=ax[2])
plt.show()
#%%

plt.xlabel('Iterations')
plt.ylabel('SNR')
plt.plot(ITERATIONS,SNRs)
plt.show()
# %%
fig,ax=plt.subplots(1,3,figsize=(10,5))
ax[0].imshow(ptNN_U.log10_custom(dp_cpu))
ax[1].imshow(ptNN_U.log10_custom(result_cpu))#,norm=colors.PowerNorm(gamma=.1))
ax[2].imshow(ptNN_U.log10_custom(psf_cpu))
plt.show()
# %%
