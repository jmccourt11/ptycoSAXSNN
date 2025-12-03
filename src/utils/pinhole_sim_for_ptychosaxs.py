#%%
import numpy as np
import matplotlib.pyplot as plt

save=False

# Parameters for the Gaussian beam
beam_width = 0.5*2 # Beam width (standard deviation) in micrometers
wavelength = 1.24e-10  # X-ray wavelength in meters (e.g., 10 keV X-rays)
pixel_size = 0.150  # Pixel size in micrometers
grid_size = 256  # Number of pixels along one dimension

# #512x512 probe
# Parameters for the Gaussian beam
beam_width = 0.5  # Beam width (standard deviation) in micrometers
wavelength = 1.24e-10  # X-ray wavelength in meters (e.g., 10 keV X-rays)
pixel_size = 0.150  # Pixel size in micrometers
grid_size = 512  # Number of pixels along one dimension








#NEW CODE
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d

save=True

# Parameters for the Gaussian beam
beam_width = 0.5*3 # Beam width (standard deviation) in micrometers
wavelength = 1.24e-10  # X-ray wavelength in meters (e.g., 10 keV X-rays)
pixel_size = 0.150  # Pixel size in micrometers
grid_size = 256  # Number of pixels along one dimension

# #512x512 probe
# Parameters for the Gaussian beam
beam_width = 0.5/2.5  # Beam width (standard deviation) in micrometers
wavelength = 1.24e-10  # X-ray wavelength in meters (e.g., 10 keV X-rays)
pixel_size = 0.150  # Pixel size in micrometers
grid_size_large = 512  # Number of pixels for the larger grid
grid_size_target = 256  # Target grid size after resizing

# // ... existing code ...
#NEW CODE
# Generate the 512x512 probe
x_large = np.linspace(-grid_size_large//2, grid_size_large//2, grid_size_large) * pixel_size
y_large = np.linspace(-grid_size_large//2, grid_size_large//2, grid_size_large) * pixel_size
X_large, Y_large = np.meshgrid(x_large, y_large)

# Gaussian beam amplitude (real part)
amplitude = np.exp(- (X_large**2 + Y_large**2) / (2 * beam_width**2))

# Phase shift due to curvature of wavefront (spherical wave approximation)
radius_of_curvature = 10.0  # Radius of curvature of the wavefront in micrometers
phase = np.exp(-1j * np.pi * (X_large**2 + Y_large**2) / (wavelength * radius_of_curvature))

# Complex-valued illumination function
illumination_function_large = amplitude * phase

# Now resize to 256x256 using interpolation
# Create interpolation functions for real and imaginary parts separately
real_interp = interp2d(x_large, y_large, np.real(illumination_function_large), kind='cubic')
imag_interp = interp2d(x_large, y_large, np.imag(illumination_function_large), kind='cubic')

# Create the target grid
x_target = np.linspace(-grid_size_target//2, grid_size_target//2, grid_size_target) * pixel_size
y_target = np.linspace(-grid_size_target//2, grid_size_target//2, grid_size_target) * pixel_size

# Interpolate onto the target grid
real_part = real_interp(x_target, y_target)
imag_part = imag_interp(x_target, y_target)

# Combine to get the resized complex illumination function
illumination_function = real_part + 1j * imag_part

# Compute the Fourier transform of the illumination function
fourier_transform = np.fft.fftshift(np.fft.fft2(illumination_function))
print(fourier_transform.shape)

# Calculate the spatial frequency axes directly
kx = np.fft.fftshift(np.fft.fftfreq(grid_size_target, d=pixel_size))
ky = np.fft.fftshift(np.fft.fftfreq(grid_size_target, d=pixel_size))
#NEW CODE
# // ... existing code ...


# # NEWCODE
# # Create a 2D grid of points
# x = np.linspace(-grid_size//2, grid_size//2, grid_size) * pixel_size
# y = np.linspace(-grid_size//2, grid_size//2, grid_size) * pixel_size
# X, Y = np.meshgrid(x, y)

# # Ensure the beam is well-contained within the window
# # Make sure beam_width is small enough compared to the grid size
# effective_window_size = grid_size * pixel_size
# if beam_width > effective_window_size / 10:
#     print(f"Warning: Beam width ({beam_width} μm) is large compared to window size ({effective_window_size} μm)")
#     print("This may cause aliasing artifacts. Consider increasing grid_size or reducing beam_width.")

# # Apply window function to reduce edge effects
# window = np.hanning(grid_size).reshape(-1, 1) * np.hanning(grid_size).reshape(1, -1)

# # Gaussian beam amplitude (real part)
# amplitude = np.exp(- (X**2 + Y**2) / (2 * beam_width**2))

# # Apply window function to reduce edge effects
# amplitude = amplitude * window

# # Phase shift due to curvature of wavefront (spherical wave approximation)
# radius_of_curvature = 10.0  # Radius of curvature of the wavefront in micrometers
# phase = np.exp(-1j * np.pi * (X**2 + Y**2) / (wavelength * radius_of_curvature))

# # Complex-valued illumination function
# illumination_function = amplitude * phase

# # Zero-pad the illumination function to reduce aliasing
# padded_size = grid_size * 2
# padded_illumination = np.zeros((padded_size, padded_size), dtype=complex)
# pad = (padded_size - grid_size) // 2
# padded_illumination[pad:pad+grid_size, pad:pad+grid_size] = illumination_function

# # Compute the Fourier transform of the padded illumination function
# fourier_transform = np.fft.fftshift(np.fft.fft2(padded_illumination))
# print(fourier_transform.shape)

# # Crop back to original size (center portion)
# fourier_transform = fourier_transform[pad:pad+grid_size, pad:pad+grid_size]
# print("After cropping:", fourier_transform.shape)

# # Calculate the spatial frequency axes for the cropped transform
# kx = np.fft.fftshift(np.fft.fftfreq(grid_size, d=pixel_size))
# ky = np.fft.fftshift(np.fft.fftfreq(grid_size, d=pixel_size))
# #NEW CODE









# # #512x512 probe
# x = np.linspace(-grid_size//2, grid_size//2, grid_size) * pixel_size
# y = np.linspace(-grid_size//2, grid_size//2, grid_size) * pixel_size
# X, Y = np.meshgrid(x, y)

# # Gaussian beam amplitude (real part)
# amplitude = np.exp(- (X**2 + Y**2) / (2 * beam_width**2))

# # Phase shift due to curvature of wavefront (spherical wave approximation)
# radius_of_curvature = 10.0  # Radius of curvature of the wavefront in micrometers
# phase = np.exp(-1j * np.pi * (X**2 + Y**2) / (wavelength * radius_of_curvature))

# # Complex-valued illumination function
# illumination_function = amplitude * phase

# # Compute the Fourier transform of the illumination function
# fourier_transform = np.fft.fftshift(np.fft.fft2(illumination_function))
# print(fourier_transform.shape)

# # No conversion to detector pixel sizes
# # Calculate the spatial frequency axes directly
# kx = np.fft.fftshift(np.fft.fftfreq(grid_size, d=pixel_size))
# ky = np.fft.fftshift(np.fft.fftfreq(grid_size, d=pixel_size))





# Visualization of the Fourier transform amplitude and phase
plt.figure(figsize=(10, 5))

# Plot amplitude of the Fourier transform
plt.subplot(1, 2, 1)
plt.imshow(np.abs(fourier_transform), extent=[kx.min(), kx.max(), ky.min(), ky.max()])
plt.title('Fourier Transform Amplitude')
plt.xlabel('kx (1/micrometers)')
plt.ylabel('ky (1/micrometers)')
plt.colorbar()

# Plot phase of the Fourier transform
plt.subplot(1, 2, 2)
plt.imshow(np.angle(fourier_transform), extent=[kx.min(), kx.max(), ky.min(), ky.max()])
plt.title('Fourier Transform Phase')
plt.xlabel('kx (1/micrometers)')
plt.ylabel('ky (1/micrometers)')
plt.colorbar()

plt.tight_layout()
plt.show()

# Visualization of the Fourier transform amplitude and phase
plt.figure(figsize=(10, 5))

# Plot amplitude of the Fourier transform
plt.subplot(1, 2, 1)
plt.imshow(np.abs(illumination_function), extent=[kx.min(), kx.max(), ky.min(), ky.max()])
plt.title('Amplitude')
plt.xlabel('kx (1/micrometers)')
plt.ylabel('ky (1/micrometers)')
plt.colorbar()

# Plot phase of the Fourier transform
plt.subplot(1, 2, 2)
plt.imshow(np.angle(illumination_function), extent=[kx.min(), kx.max(), ky.min(), ky.max()])
plt.title('Phase')
plt.xlabel('kx (1/micrometers)')
plt.ylabel('ky (1/micrometers)')
plt.colorbar()

plt.tight_layout()
plt.show()


if save:
    filename=f'probe_pinhole_bw{beam_width}_wl{wavelength}_ps{pixel_size}_gs{grid_size_target}x{grid_size_target}.npy'
    np.save(filename,fourier_transform)
    print(filename)


# %%
