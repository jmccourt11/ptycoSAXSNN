import numpy as np
from scipy import special 
import scipy.ndimage
from numpy.random import randint
from typing import Any, Final
from scipy.io import loadmat
import cv2
import numpy.typing
import matplotlib.pyplot as plt
from matplotlib import colors
from math import ceil
from PIL import Image
from IPython.display import clear_output
from tqdm import tqdm
import os
from skimage.transform import resize
from skimage.restoration import unwrap_phase
'''EXAMPLE ZONE PLATE PARAMETERS
zonePlateDiameterInMeters=160e-6  #zone plate diameter, m
outerMostZoneWidthInMeters=70e-9 #outermost zone width, m 
centralBeamstopDiameterInMeters= 80e-6 #cenral beamstop diamater, m
centralWavelengthInMeters=1.24e-10 #x-ray wavelength in meters
defocusDistanceInMeters = 800e-6 # from sample to the focal plane
sddInMeters = 2 # sample to detector distance

#detector coordinates are x and y
pixelSizeXInMeters=75e-6 #pixel size x, m
pixelSizeYInMeters=75e-6 #pixel size y, m

#detector size
DPsizeX,DPsizeY=(512,512) #shape of diffraction pattern, pixels
probeSize=DPsizeX #probe size in X direction, this currently only supports square probes'''


def focalLengthInMeters(zonePlateDiameterInMeters: float, outermostZoneWidthInMeters: float, centralWavelengthInMeters: float) -> float:
        return zonePlateDiameterInMeters * outermostZoneWidthInMeters \
                / centralWavelengthInMeters

def getObjectPlanePixelGeometry(pixelSizeXInMeters,pixelSizeYInMeters,DPsizeX,DPsizeY,sddInMeters,centralWavelengthInMeters) -> tuple:
    lambdaZInSquareMeters = centralWavelengthInMeters*sddInMeters
    extentXInMeters=pixelSizeXInMeters*DPsizeX
    extentYInMeters=pixelSizeYInMeters*DPsizeY
    return (lambdaZInSquareMeters / extentXInMeters, lambdaZInSquareMeters / extentYInMeters) #width and height, m


'''
HOW TO USE
# pixel size on sample plane (TODO non-square pixels are unsupported)
dx = getObjectPlanePixelGeometry()[0] #only x pixel size, square probes supported
print(dx)
'''


def fzp_calculate(wavelength: float, FL: float, dis_defocus: float, probeSize: int, dx: float,zonePlateDiameterInMeters: float, centralBeamstopDiameterInMeters: float) -> tuple[np.typing.NDArray[Any], float, float]:
    """
    this function can calculate the transfer function of zone plate
    return the transfer function, and the pixel sizes
    
    transfer function is a fourier space probe pattern
    """

    #FL = focalLengthInMeters(zonePlateDiameterInMeters,outerMostZoneWidthInMeters,wavelength)

    
    # pixel size on FZP plane
    dx_fzp = wavelength * (FL + dis_defocus) / probeSize / dx
    
    # coordinate on FZP plane
    lx_fzp = -float(dx_fzp) * np.arange(-1 * np.floor(probeSize / 2),
                                           np.ceil(probeSize / 2))

    XX_FZP, YY_FZP = np.meshgrid(lx_fzp, lx_fzp)

    # transmission function of FZP
    T = np.exp(-1j * 2 * np.pi / float(wavelength) * (XX_FZP**2 + YY_FZP**2) / 2 / float(FL))
    C = np.sqrt(XX_FZP**2 + YY_FZP**2) <= zonePlateDiameterInMeters / 2
    H = np.sqrt(XX_FZP**2 + YY_FZP**2) >= centralBeamstopDiameterInMeters / 2

    
    return T * C * H, dx_fzp, FL

'''
HOW TO USE
lambda0=centralWavelengthInMeters
T, dx_fzp, FL0 = fzp_calculate(lambda0, defocusDistanceInMeters, probeSize,
                               dx)
print(dx_fzp,FL0)
fig,ax=plt.subplots(1,2)
ax[0].imshow(np.abs(T))
ax[1].imshow(np.angle(T))
plt.show()
'''


def fresnel_propagation(input: np.typing.NDArray[Any], dxy: float, z: float,
                        wavelength: float) -> np.typing.NDArray[Any]:
    """
    This is the python version code for fresnel propagation
    Summary of this function goes here
    Parameters:    dx,dy  -> the pixel pitch of the object
                z      -> the distance of the propagation
                lambda -> the wave length
                X,Y    -> meshgrid of coordinate
                input     -> input object
    """

    (M, N) = input.shape
    k = 2 * np.pi / wavelength
    # the coordinate grid
    M_grid = np.arange(-1 * np.floor(M / 2), np.ceil(M / 2))
    N_grid = np.arange(-1 * np.floor(N / 2), np.ceil(N / 2))
    lx = M_grid * dxy
    ly = N_grid * dxy

    XX, YY = np.meshgrid(lx, ly)

    # the coordinate grid on the output plane
    fc = 1 / dxy
    fu = wavelength * z * fc
    lu = M_grid * fu / M
    lv = N_grid * fu / N
    Fx, Fy = np.meshgrid(lu, lv)

    if z > 0:
        pf = np.exp(1j * k * z) * np.exp(1j * k * (Fx**2 + Fy**2) / 2 / z)
        kern = input * np.exp(1j * k * (XX**2 + YY**2) / 2 / z)
        cgh = np.fft.fft2(np.fft.fftshift(kern))
        OUT = np.fft.fftshift(cgh * np.fft.fftshift(pf))
    else:
        pf = np.exp(1j * k * z) * np.exp(1j * k * (XX**2 + YY**2) / 2 / z)
        cgh = np.fft.ifft2(
            np.fft.fftshift(input * np.exp(1j * k * (Fx**2 + Fy**2) / 2 / z)))
        OUT = np.fft.fftshift(cgh) * pf
    return OUT


'''
HOW TO USE
#propogate probe
nprobe = fresnel_propagation(T, float(dx_fzp),
                             (float(FL0) + float(defocusDistanceInMeters)),
                             float(lambda0))
'''


def plot_slice(images,first=1,last=10,log=False,cmap='jet'):
    #images = images[first:last]
    fig=plt.figure(figsize=(10,10),layout='constrained')
    for ii in range(len(images)):
        sub = fig.add_subplot(1, len(images), ii+1)
        image = images[ii]
        if log:
            plt.imshow(image,norm=colors.LogNorm(),cmap=cmap)
        else:
            plt.imshow(image,cmap=cmap)


def simulate_DP(objects,probe,coordx,coordy,probe_size,plot=False):
    #PLOT
    if plot:
        fig,ax=plt.subplots(1,2,layout='constrained')
        ax[0].set_title('Probe (Real Space)')
        ax[0].imshow(np.abs(probe))
        ax[1].set_title('Objects[0]')
        ax[1].imshow(objects[0],cmap='gray')
    #initial crop of the images to get rid of scale bar and reduce size
    x=coordx
    y=coordy
    size=probe_size
    objects=np.array([l[x:x+size,y:y+size] for l in objects]) #selected ROI from tif images)
    #objects=np.array([(p-p.min())/(p.max()-p.min()) for p in objects])
    
    dps=[np.fft.fftshift(np.fft.fft2(o*probe)) for o in objects]
    
    return (objects,dps)


def simulate_ptycho_scan(objects,probe,startx,starty,sizeO,sizeP,pixel_P,pixel_O):
    #origin
    x=startx
    y=starty

    #final full lists
    obj_list=[]
    dp_list=[]

    #pixel_P=14 #nm/pixel, probe
    #pixel_O=8 #nm/pixel, object
    step=ceil(pixel_O/pixel_P)
    pixel_D=pixel_P*step
    pixel_step=int(np.max([pixel_D,pixel_P])/pixel_P)
    
    #objs=np.zeros(len(objects)*(sizeO[0]-sizeP-startx)*(sizeO[1]-sizeP-starty))
    #dps=np.zeros(len(objects)*(sizeO[0]-sizeP-startx)*(sizeO[1]-sizeP-starty))
    
    for p in tqdm(objects):
        for i in np.arange(startx,sizeO[0]-sizeP,pixel_step):
            for j in np.arange(starty,sizeO[1]-sizeP,pixel_step):
                roi=np.array(p[i:i+sizeP,j:j+sizeP])
                o=roi
                d=np.fft.fftshift(np.fft.fft2(o*probe))
                obj_list.append(o)
                dp_list.append(d)
                
    return {'Objects':np.asarray(obj_list),'DPs':np.asarray(dp_list)}


def circ_mask(image,radius):
    #circular mask of radius=radius over image 
    xs=np.arange(0,image.shape[0])
    ys=np.arange(0,image.shape[1])
    xx,yy=np.meshgrid(xs, ys)
    temp=image
    temp[(xx-radius/2)**2+(yy-radius/2)**2>(radius/2)**2]=0
    return temp


def live_plot_2images(images,images2):
    #live plot
    plt.ion()
    fig,ax=plt.subplots(1,2,layout='constrained')
    for i in range(0,len(images)):
        try:
            ax[0].imshow(images[i],cmap='gray')
            ax[1].imshow(images2[i],norm=colors.LogNorm())
            plt.pause(.2)
            plt.draw()
            clear_output(wait=True)
        except KeyboardInterrupt:
            break
    #plt.show()

def downsample_probe(probe,dfactor=2):
    probe_shape=probe.shape
    real=np.real(probe)
    imag=np.imag(probe)
    real_resize=resize(real[probe_shape[0]//(2*dfactor):-probe_shape[0]//(2*dfactor),probe_shape[0]//(2*dfactor):-probe_shape[0]//(2*dfactor)],(probe_shape[0]//dfactor,probe_shape[1]//dfactor),preserve_range=True, anti_aliasing=True)
    imag_resize=resize(imag[probe_shape[0]//(2*dfactor):-probe_shape[0]//(2*dfactor),probe_shape[0]//(2*dfactor):-probe_shape[0]//(2*dfactor)],(probe_shape[0]//dfactor,probe_shape[1]//dfactor),preserve_range=True, anti_aliasing=True)
    #real_resize=resize(real[:,:],(probe_shape[0]//dfactor,probe_shape[1]//dfactor),preserve_range=True, anti_aliasing=True)
    #imag_resize=resize(imag[:,:],(probe_shape[0]//dfactor,probe_shape[1]//dfactor),preserve_range=True, anti_aliasing=True)
    probe_resize=real_resize+1j*imag_resize
    probe=probe_resize
    return probe


def hanning(image):
    #circular mask of radius=radius over image 
    xs=np.hanning(image.shape[0])
    ys=np.hanning(image.shape[1])
    temp=np.outer(xs,ys)
    return temp
