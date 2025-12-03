import numpy as np
import pyFAI, pyFAI.detectors
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
import matplotlib.pyplot as plt
from tqdm import tqdm



class DP_processor:
    def __init__(self,dp_file,index=0):
        self.dp=np.load(dp_file)['convDP']#[index]
        self.detector=pyFAI.detectors.Detector(pixel1=150e-6, pixel2=150e-6)
        self.sdd=0.1
        self.wavelength=1.24e-10
        self.center=(256,256)#(250,270)
        self.ai = AzimuthalIntegrator(dist=self.sdd, detector=self.detector)
        self.ai.set_wavelength(self.wavelength)
        self.ai.setFit2D(100, self.center[0]//2,self.center[1]//2) #center of (250, 270) is offset from true center
        
    def __repr__(self):
        return f'DP_processor (DP: {self.dp!r}, detector: {self.detector!r})'
        
    
    def azimuthal_and_plot(self,plot=True):
        res2d1 = self.ai.integrate2d(self.dp, 300, 360, unit="2th_deg")
        I, tth, chi = res2d1
        if plot:
            fig,ax=plt.subplots()
            ax.imshow(I, origin="lower", extent=[tth.min(), tth.max(), chi.min(), chi.max()], aspect="auto",cmap='jet')
            ax.set_xlabel("2 theta (deg)")
            ax.set_ylabel("Azimuthal angle (deg)")
            plt.show()
        return res2d1
        
    def integrate1d_and_plot(self,plot=True):
        res1 = self.ai.integrate1d(self.dp, 300, unit="2th_deg")
        tth1 = res1[0]
        I1 = res1[1]
        if plot:
            fig,ax=plt.subplots()
            ax.plot(tth1,I1)
            ax.set_xlabel("2 theta (deg)")
            ax.set_ylabel("Integrated Intensity")
            plt.show()
        return res1    
               
    def find_peaks2d(self,n=25,threshold=0.19,center_cut=48,plot=True):
        peaks = []
        
        # Define the shape of the image
        rows, cols = self.dp.shape
        
        # Define center
        center_x,center_y = rows//2,cols//2
        
        # Iterate over each pixel, excluding the border pixels for simplicity
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                # Extract the nxn neighborhood of the current pixel
                neighborhood = self.dp[i-1:i+n, j-1:j+n]
                
                # Check if the center pixel is greater than all its neighbors and above the threshold
                if self.dp[i, j] > threshold and self.dp[i, j] == np.max(neighborhood) and np.count_nonzero(self.dp[i, j] == neighborhood) == 1 and (i-center_x)**2+(j-center_y)**2>center_cut**2:
                    peaks.append((i, j))
        if plot:
            fig,ax=plt.subplots()
            im=ax.imshow(self.dp, cmap='jet', interpolation='nearest')
            peak_y, peak_x = zip(*peaks)
            ax.scatter(peak_x, peak_y, color='red', marker='x', s=100, label='Peaks')
            plt.colorbar(im)
            plt.show()
        return peaks
        
if __name__=='__main__':
    #filename='results_cindy3_scan1125_diff_sim11_conv_deconvML.npz'
    ri=21000
    filename=f'output_hanning_conv_{ri:05d}.npz'
    #path='/net/micdata/data2/12IDC/ptychosaxs/results/'
    path='/net/micdata/data2/12IDC/ptychosaxs/data/diff_sim/12/'
    x=DP_processor(path+filename,index=0)#230)
    x.dp=np.log10(x.dp)
    #x=DP_processor('/mnt/micdata2/12IDC/ptychosaxs/results/'+filename)
    x.azimuthal_and_plot()
    x.integrate1d_and_plot()
    
    #x.find_peaks2d(n=25,threshold=5)#0.19)
    plt.imshow(x.dp,cmap='jet')
    plt.show()
    
#    scanx=20
#    scany=15
#    fig, axs = plt.subplots(scany,scanx, sharex=True,sharey=True,figsize=(scanx,scany))

#    # Remove vertical space between Axes
#    fig.subplots_adjust(hspace=0,wspace=0)
#    count=0
#    for i in tqdm(range(0,scany)):
#        for j in range(0,scanx):
#            dp_count=DP_processor('/net/micdata/data2/12IDC/ptychosaxs/results/'+filename,index=count)
#            result=dp_count.azimuthal_and_plot(plot=False)[0]
#            
##            result=dp_count.dp
##            peaks=dp_count.find_peaks2d(n=25,threshold=0.19,plot=False)
##            peak_y, peak_x = zip(*peaks)            
##            
#            im=axs[i][j].imshow(result)
#            axs[i][j].imshow(result)
#            #axs[i][j].scatter(peak_x, peak_y, color='red', marker='x', s=10, label='Peaks')
#            axs[i][j].axis("off")
#            count+=1
#    cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
#    fig.colorbar(im, cax=cbar_ax)
#    plt.show()
    
    
