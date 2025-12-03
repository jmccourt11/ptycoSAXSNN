# coding: utf-8
from cupyx.scipy.signal import convolve2d as conv2
from scipy.signal import convolve2d as conv2np
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mask_probe import *
from skimage.filters import window
from scipy import signal
import random
from scipy.special import j1
from skimage.transform import resize

cmap='jet'

def gkern(kernlen=21, std=4):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d


save=False
save_complex=False
complex_valued=True
real_valued=False

# Given parameters
xray_wavelength_AA = 1.24
pinhole_diameter_um = 200
detector_pixel_size_um = 75
detector_pixels = 512
distance_pinhole_detector_m =1e-2

# Pinhole parameters
pinhole_radius_m = pinhole_diameter_um * 1e-6 / 2

# Detector parameters
detector_size_m = detector_pixels * detector_pixel_size_um * 1e-6
pixel_size_m = detector_pixel_size_um * 1e-6

# Create a grid for the detector
x = np.linspace(-detector_size_m / 2, detector_size_m / 2, detector_pixels)
y = np.linspace(-detector_size_m / 2, detector_size_m / 2, detector_pixels)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)

# Calculate the wave number
k = 2 * np.pi / (xray_wavelength_AA*10**(-10))



# Calculate the intensity pattern (Airy pattern)
if real_valued:
    I = (2 * j1(k * pinhole_radius_m * R / distance_pinhole_detector_m) / 
         (k * pinhole_radius_m * R / distance_pinhole_detector_m))**2
    I[R == 0] = 1  # Handle the singularity at R = 0

elif complex_valued:
    # Generate a random phase distribution
    phase = np.random.uniform(0, 2*np.pi, size=R.shape)
    complex_amplitude = (2 * j1(k * pinhole_radius_m * R / distance_pinhole_detector_m) / 
                         (k * pinhole_radius_m * R / distance_pinhole_detector_m)) * np.exp(1j * phase)
    complex_amplitude[R == 0] = 1
    I= np.abs(complex_amplitude)**2

if save_complex:
    np.save('probe_pinhole_complex.npy',complex_amplitude)
    
# Normalize the intensity
I /= np.max(I)

# Plot the intensity pattern
plt.figure(figsize=(8, 8))
plt.imshow(I, extent=[-detector_size_m/2*1e6, detector_size_m/2*1e6, -detector_size_m/2*1e6, detector_size_m/2*1e6], cmap='jet')
plt.colorbar(label='Normalized Intensity')
plt.xlabel('X (µm)')
plt.ylabel('Y (µm)')
plt.title('Pinhole X-ray Beam Diffraction Pattern on 2D Detector')
plt.show()



# create probes
psf=np.abs(np.load('/home/beams/B304014/ptychosaxs/NN/probe_FT.npy'))
psf=np.abs(np.load('/mnt/micdata2/12IDC/ptychosaxs/data/probe_FT_cindy_zheng.npy')) #cindy
psf=resize(psf[:,:],(256,256),preserve_range=True,anti_aliasing=True) #cindy

#image=gkern(256,std=12)
#image_fft=np.abs(np.fft.fftshift(np.fft.fft2(image)))
image_fft=I

fig,ax=plt.subplots(1,2)
im3=ax[0].imshow(psf,norm=colors.LogNorm(),cmap=cmap);
im4=ax[1].imshow(image_fft,norm=colors.LogNorm(),cmap=cmap);
plt.colorbar(im3, ax=ax[0])#, format='%.2f')
plt.colorbar(im4, ax=ax[1])#, format='%.2f')
plt.show()


index=random.randint(1,13000)
ideal_DP=np.load('/mnt/micdata2/12IDC/ptychosaxs/data/diff_sim/2/output_ideal_{:05d}.npz'.format(index))['idealDP']

psf=cp.asarray(psf)
image_fft=cp.asarray(image_fft)
ideal_DP=cp.asarray(ideal_DP)


conv_DP=conv2(ideal_DP,psf,'same', boundary='symm')
gauss_DP=conv2(ideal_DP,image_fft,'same', boundary='symm')


psf=psf.get()
image_fft=image_fft.get()
conv_DP=conv_DP.get()
gauss_DP=gauss_DP.get()


fig,ax=plt.subplots(2,2,layout='constrained');
im1=ax[0][0].imshow(conv_DP,norm=colors.LogNorm(),cmap=cmap);
im2=ax[0][1].imshow(gauss_DP,norm=colors.LogNorm(),cmap=cmap);
im3=ax[1][0].imshow(psf,norm=colors.LogNorm(),cmap=cmap);
im4=ax[1][1].imshow(image_fft,norm=colors.LogNorm(),cmap=cmap);
plt.colorbar(im1, ax=ax[1][0])#, format='%.2f')
plt.colorbar(im1, ax=ax[1][1])#, format='%.2f')
plt.show()




if save:
    np.save('probe_pinhole.npy',image_fft)
