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
from cupyx.scipy.ndimage import rotate as rotate_gpu
from scipy.ndimage import gaussian_filter
import tifffile
import h5py
from matplotlib import colors
import cupy as cp

# def main():
#     parser = argparse.ArgumentParser(description="Ptycho-tomography simulation with 180 phi projections.")
#     parser.add_argument('--probe_mat', type=str, required=True, help="Path to .mat probe file (contains 'probe').")
#     parser.add_argument('--probe_key', type=str, default='probe', help="MAT key for probe.")
#     parser.add_argument('--probe_slice', type=int, default=0, help="Slice index for 3D probe in MAT file.")
#     parser.add_argument('--target_size', type=int, default=1280, help="Target size for probe/object grid.")
#     parser.add_argument('--lattice_path', type=str, required=True, help="Path to 3D lattice (.npy or .tif).")
#     parser.add_argument('--phi_axis', type=str, choices=['x', 'y', 'z'], default='y', help="Rotation axis for phi scan.")
#     parser.add_argument('--num_phi', type=int, default=180, help="Number of phi projections.")
#     parser.add_argument('--phi_start_random', action='store_true', help="Use random starting phi angle [0, 360).")
#     parser.add_argument('--phi_start_deg', type=float, default=0.0, help="Starting phi angle (ignored if --phi_start_random).")
#     parser.add_argument('--apply_initial_random_orientation', action='store_true', help="Apply random X/Y/Z rotation before phi scan.")
#     parser.add_argument('--scan_step_div_min', type=int, default=18, help="Min divisor for scan step (probe_size // div).")
#     parser.add_argument('--scan_step_div_max', type=int, default=36, help="Max divisor for scan step (probe_size // div).")
#     parser.add_argument('--batch_size', type=int, default=32, help="Batch size for FFTs.")
#     parser.add_argument('--segment_size', type=int, default=256, help="Segment size for saving DPs.")
#     parser.add_argument('--output_dir', type=str, default='/home/beams/PTYCHOSAXS/NN/ptychosaxsNN/data/tomo_output', help="Output directory for NPZ files.")
#     parser.add_argument('--mask_path', type=str, default=None, help="Optional .npy mask to apply to conv DPs.")
#     args = parser.parse_args()

#     _ = run_tomo_simulation(
#         probe_mat=args.probe_mat,
#         lattice_path=args.lattice_path,
#         probe_key=args.probe_key,
#         probe_slice=args.probe_slice,
#         target_size=args.target_size,
#         phi_axis=args.phi_axis,
#         num_phi=args.num_phi,
#         phi_start_random=args.phi_start_random,
#         phi_start_deg=args.phi_start_deg,
#         apply_initial_random_orientation=args.apply_initial_random_orientation,
#         scan_step_div_min=args.scan_step_div_min,
#         scan_step_div_max=args.scan_step_div_max,
#         batch_size=args.batch_size,
#         segment_size=args.segment_size,
#         output_dir=args.output_dir,
#         mask_path=args.mask_path,
#     )


# # if __name__ == '__main__':
# #     main()


def get_device() -> torch.device:
    return torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def load_probe_from_mat(mat_path: str, key: str = 'probe', slice_index: int = 0) -> np.ndarray:
    import scipy.io as sio
    data = sio.loadmat(mat_path)
    probe = data[key]
    if probe.ndim == 3:
        probe = probe[:, :, slice_index]
    if probe.ndim == 4:
        probe = probe[:, :, 0,slice_index]   
    fig,ax=plt.subplots(1,3,figsize=(10,5))
    ax[0].set_title('Original Probe Amplitude')
    ax[1].set_title('Original Probe Phase')
    ax[2].set_title('Original Probe FFT Amplitude')
    ax[0].imshow(np.abs(probe))
    ax[1].imshow(np.angle(probe))
    ax[2].imshow(np.abs(np.fft.fftshift(np.fft.fft2(probe)))**2,norm=colors.LogNorm())
    plt.show()
    return probe


def resize_probe_fourier(probe_np: np.ndarray, target_size: int, device: torch.device) -> torch.Tensor:
    probe = torch.tensor(probe_np, dtype=torch.cfloat, device=device)
    
    # Transform to Fourier space and resize to target size
    probe_fft = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(probe, dim=(-2, -1)), norm='ortho'), dim=(-2, -1))
    pad_size = (target_size - probe_fft.shape[0]) // 2
    # # Pad with zeros
    # # padded_fft = torch.zeros((target_size, target_size), dtype=torch.cfloat, device=device)
    # # Pad with the minimum absolute value of the probe, avoids upsampled probe edge artifacts
    # padded_fft = torch.ones((target_size, target_size), dtype=torch.cfloat, device=device)*(torch.min(torch.abs(probe_fft)))
    # padded_fft[pad_size:pad_size + probe_fft.shape[0], pad_size:pad_size + probe_fft.shape[1]] = probe_fft

    # probe_resized = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(padded_fft, dim=(-2, -1)), norm='ortho'), dim=(-2, -1))

    # 1) Zero-pad FFT and optionally blend tile edge
    hy, hx = probe_fft.shape
    padded_fft = torch.zeros((target_size, target_size), dtype=torch.cfloat, device=device)

    fig,ax=plt.subplots(1,3,figsize=(10,5))
    ax[0].imshow(np.abs(probe.cpu().numpy()))
    ax[1].imshow(np.abs(probe_fft.cpu().numpy()),norm=colors.LogNorm())
    ax[2].imshow(np.abs(padded_fft.cpu().numpy()))
    plt.show()

    # optional narrow Hann rim (3–8% of size)
    rim = max(4, min(hy, hx) // 32)
    #rim=16
    wy = torch.hann_window(hy, periodic=False, device=device)
    wx = torch.hann_window(hx, periodic=False, device=device)
    w2d = wy[:, None] * wx[None, :]
    padded_fft[pad_size:pad_size+hy, pad_size:pad_size+hx] = probe_fft*w2d

    probe_resized = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(padded_fft, dim=(-2,-1)), norm='ortho'), dim=(-2,-1))
    
    fig,ax=plt.subplots(1,3,figsize=(10,5))
    ax[0].imshow(np.abs(probe_resized.cpu().numpy()))
    ax[1].imshow(np.abs(w2d.cpu().numpy()))
    ax[2].imshow(np.abs(padded_fft.cpu().numpy()),norm=colors.LogNorm())
    plt.show()

    # 2) Spatial vignette on the complex field (do NOT taper phase separately)
    y = torch.linspace(-1, 1, probe_resized.shape[0], device=device)
    x = torch.linspace(-1, 1, probe_resized.shape[1], device=device)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    R = torch.sqrt(X**2 + Y**2)
    win = (1 - R**2).clamp(0, 1)
    probe_resized = probe_resized * win


    fig,ax=plt.subplots(1,3,figsize=(10,5))
    ax[0].imshow(np.abs(probe_resized.cpu().numpy()))
    ax[1].imshow(np.angle(probe_resized.cpu().numpy()))
    ax[2].imshow(np.abs(np.fft.fftshift(np.fft.fft2(probe_resized.cpu().numpy())))**2,norm=colors.LogNorm())
    plt.show()

    # 3) Energy renormalization (keeps probe strength comparable to original)
    orig_E = torch.sum(torch.abs(probe)**2)
    new_E  = torch.sum(torch.abs(probe_resized)**2) + 1e-12
    probe_resized = probe_resized * torch.sqrt(orig_E / new_E)
    
    fig,ax=plt.subplots(1,3,figsize=(10,5))
    ax[0].imshow(np.abs(probe_resized.cpu().numpy()))
    ax[1].imshow(np.angle(probe_resized.cpu().numpy()))
    ax[2].imshow(np.abs(np.fft.fftshift(np.fft.fft2(probe_resized.cpu().numpy())))**2,norm=colors.LogNorm())
    plt.show()

    # 4) Optional tiny pedestal to avoid exact zeros during multiplication
    eps = 1e-6 * torch.max(torch.abs(probe_resized))
    probe_resized = probe_resized + eps  # adds a tiny DC; harmless, no box

    fig,ax=plt.subplots(1,3,figsize=(10,5))
    ax[0].imshow(np.abs(probe_resized.cpu().numpy()))
    ax[1].imshow(np.angle(probe_resized.cpu().numpy()))
    ax[2].imshow(np.abs(np.fft.fftshift(np.fft.fft2(probe_resized.cpu().numpy())))**2,norm=colors.LogNorm())
    plt.show()

    # # Vignette the resized probe amplitude and phase separately
    # y = torch.linspace(-1, 1, probe_resized.shape[0], device=device)
    # x = torch.linspace(-1, 1, probe_resized.shape[1], device=device)
    # X, Y = torch.meshgrid(x, y, indexing='xy')
    # R = torch.sqrt(X ** 2 + Y ** 2)
    # vignette = (1 - R ** 2).clamp(0, 1)
    
    # # Split into amplitude and phase
    # amplitude = torch.abs(probe_resized)
    # phase = torch.angle(probe_resized)
    
    # # Apply vignette to both
    # amplitude = amplitude * vignette
    # phase = phase * vignette
    
    # # Recombine into complex probe
    # probe_resized = amplitude * torch.exp(1j * phase)

    
    fig,ax=plt.subplots(1,2,figsize=(10,5))
    ax[0].set_title('Resized Probe Amplitude')
    ax[1].set_title('Resized Probe Phase')
    im1=ax[0].imshow(np.abs(probe_resized.cpu().numpy()))
    im2=ax[1].imshow(np.angle(probe_resized.cpu().numpy()))
    plt.colorbar(im1)
    plt.colorbar(im2)
    plt.show()
    
    fig,ax=plt.subplots(1,2,figsize=(10,5))
    ax[0].set_title('Resized Probe FFT Amplitude')
    ax[1].set_title('Resized Probe FFT Phase')
    im1=ax[0].imshow(np.abs(np.fft.fftshift(np.fft.fft2(probe_resized.cpu().numpy()))),norm=colors.LogNorm())
    im2=ax[1].imshow(np.angle(np.fft.fftshift(np.fft.fft2(probe_resized.cpu().numpy()))),norm=colors.LogNorm())
    plt.colorbar(im1)
    plt.colorbar(im2)
    plt.show()
    return probe_resized


def load_lattice(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        return np.load(path)
    elif ext in ['.tif', '.tiff']:
        return tifffile.imread(path)
    else:
        raise ValueError(f"Unsupported lattice format: {ext}")


def apply_3d_vignette(volume: np.ndarray) -> np.ndarray:
    x, y, z = np.meshgrid(
        np.linspace(-1, 1, volume.shape[0]),
        np.linspace(-1, 1, volume.shape[1]),
        np.linspace(-1, 1, volume.shape[2]),
        indexing='ij'
    )
    distance = np.maximum.reduce([np.abs(x), np.abs(y), np.abs(z)])
    vignette_mask = np.clip(1 - distance, 0, 1)
    return volume * vignette_mask


def add_noise_3d(volume: np.ndarray,
                 std_fraction: float = 0.01,
                 lowpass_sigma: Optional[float] = None,
                 seed: Optional[int] = None) -> np.ndarray:
    """Add slight zero-mean noise to a 3D volume to reduce FFT artifacts.

    Args:
        volume: Input 3D array (np.ndarray, float32/float64).
        std_fraction: Noise std as a fraction of volume's std (default 1%).
        lowpass_sigma: If provided (>0), applies Gaussian smoothing to the
            generated noise to create low-frequency variations.
        seed: Optional RNG seed for reproducibility.

    Returns:
        Noisy volume as float32 with same shape.
    """
    rng = np.random.default_rng(seed)
    vol = volume.astype(np.float32, copy=False)
    vol_std = float(np.std(vol))
    scale = vol_std if vol_std > 0 else 1.0
    noise = rng.normal(loc=0.0, scale=std_fraction * scale, size=vol.shape).astype(np.float32)
    if lowpass_sigma is not None and lowpass_sigma > 0:
        noise = gaussian_filter(noise, sigma=float(lowpass_sigma), mode='nearest')
    return (vol + noise).astype(np.float32, copy=False)


def rotate_volume(volume: np.ndarray, angle_deg: float, axis: str) -> np.ndarray:
    if axis == 'x':
        axes = (1, 2)
    elif axis == 'y':
        axes = (0, 2)
    elif axis == 'z':
        axes = (0, 1)
    else:
        raise ValueError("axis must be one of 'x', 'y', 'z'")
    return rotate(volume, angle_deg, axes=axes, reshape=False, order=1)


def project_to_2d(volume: np.ndarray, projection_axis: int = 2) -> np.ndarray:
    return np.sum(volume, axis=projection_axis)


def apply_2d_vignette(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    y = torch.linspace(-1, 1, t.shape[0], device=device)
    x = torch.linspace(-1, 1, t.shape[1], device=device)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    R = torch.sqrt(X ** 2 + Y ** 2)
    return t * (1 - R ** 2).clamp(0, 1)


def add_noise_2d(image: np.ndarray,
                 std_fraction: float = 0.01,
                 lowpass_sigma: Optional[float] = None,
                 seed: Optional[int] = None) -> np.ndarray:
    """Add slight zero-mean noise to a 2D image prior to FFT.

    Args:
        image: Input 2D array.
        std_fraction: Noise std as fraction of image std.
        lowpass_sigma: Optional Gaussian sigma to smooth noise (pixels).
        seed: Optional RNG seed for reproducibility.

    Returns:
        Noisy image (float32).
    """
    rng = np.random.default_rng(seed)
    img = image.astype(np.float32, copy=False)
    img_std = float(np.std(img))
    scale = img_std if img_std > 0 else 1.0
    noise = rng.normal(0.0, std_fraction * scale, size=img.shape).astype(np.float32)
    if lowpass_sigma is not None and lowpass_sigma > 0:
        noise = gaussian_filter(noise, sigma=float(lowpass_sigma), mode='nearest')
    return (img + noise).astype(np.float32, copy=False)


def pad_to_center(canvas_size: int, small: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    canvas = torch.zeros((canvas_size, canvas_size), dtype=dtype, device=device)
    h, w = small.shape
    ys = canvas_size // 2 - h // 2
    xs = canvas_size // 2 - w // 2
    canvas[ys:ys + h, xs:xs + w] = small
    return canvas


def build_probe_vignette(size: int, device: torch.device) -> torch.Tensor:
    y = torch.linspace(-1, 1, size, device=device)
    x = torch.linspace(-1, 1, size, device=device)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    R = torch.sqrt(X ** 2 + Y ** 2)
    return (1 - R ** 2).clamp(0, 1)


def compute_scan_positions(image_h: int, image_w: int, window: int, step: int) -> torch.Tensor:
    positions: List[List[int]] = []
    for yy in range(0, image_h - window + 1, step):
        for xx in range(0, image_w - window + 1, step):
            positions.append([yy, xx])
    if not positions:
        positions = [[0, 0]]
    return torch.tensor(positions)


def segment_and_save(dp_conv: np.ndarray,
                     dp_ideal: np.ndarray,
                     segment_size: int,
                     dp_count_start: int,
                     output_dir: str,
                     scan_position: Tuple[int, int],
                     scan_index: int,
                     phi_deg: float,
                     phi_index: int,
                     mask: Optional[np.ndarray] = None) -> int:
    if mask is not None:
        dp_conv = dp_conv * mask
    segs_per_side = dp_conv.shape[0] // segment_size
    dp_count = dp_count_start
    for i in range(segs_per_side):
        for j in range(segs_per_side):
            dp_count += 1
            y_start = i * segment_size
            y_end = (i + 1) * segment_size
            x_start = j * segment_size
            x_end = (j + 1) * segment_size
            segment_conv = dp_conv[y_start:y_end, x_start:x_end]
            segment_ideal = dp_ideal[y_start:y_end, x_start:x_end]
            fname = os.path.join(output_dir, f'tomo_output_{dp_count:06d}.npz')
            np.savez(
                fname,
                convDP=segment_conv,
                pinholeDP=segment_ideal,
                scan_position=np.array(scan_position, dtype=np.int32),
                scan_index=np.int32(scan_index),
                segment_position=np.array([i, j], dtype=np.int32),
                phi_deg=np.float32(phi_deg),
                phi_index=np.int32(phi_index)
            )
    return dp_count


class H5ScanWriter:
    def __init__(self,
                 h5_path: str,
                 dp_height: int,
                 dp_width: int,
                 object_image: np.ndarray,
                 probe_complex: np.ndarray,
                 metadata: dict,
                 compression: str = 'gzip',
                 compression_opts: int = 4,
                 overlay_alpha: float = 0.5):
        self.h5_path = h5_path
        self.dp_h = int(dp_height)
        self.dp_w = int(dp_width)
        self.n = 0
        self.overlay_alpha = float(overlay_alpha)

        # Prepare display items
        obj = np.abs(object_image.astype(np.float32))
        if np.max(obj) > 0:
            obj_norm = (obj - obj.min()) / (obj.max() - obj.min() + 1e-12)
        else:
            obj_norm = obj
        self.overlay_rgb = np.stack([obj_norm, obj_norm, obj_norm], axis=-1)
        self.overlay_rgb = (self.overlay_rgb * 255.0).astype(np.uint8)
        probe_amp = np.abs(probe_complex)
        self.probe_amp = probe_amp.astype(np.float32)
        self.probe_phase = np.angle(probe_complex).astype(np.float32)
        self.probe_amp_norm_u8 = (np.clip(self.probe_amp / (self.probe_amp.max() + 1e-12), 0, 1) * 255.0 * self.overlay_alpha).astype(np.uint8)

        # Open file and create datasets
        self.f = h5py.File(self.h5_path, 'w')
        # Extendable datasets for DPs and positions
        self.ds_conv = self.f.create_dataset(
            'convDPs', shape=(0, self.dp_h, self.dp_w), maxshape=(None, self.dp_h, self.dp_w),
            dtype='float32', compression=compression, compression_opts=compression_opts, chunks=(1, self.dp_h, self.dp_w)
        )
        self.ds_ideal = self.f.create_dataset(
            'idealDPs', shape=(0, self.dp_h, self.dp_w), maxshape=(None, self.dp_h, self.dp_w),
            dtype='float32', compression=compression, compression_opts=compression_opts, chunks=(1, self.dp_h, self.dp_w)
        )
        self.ds_pos = self.f.create_dataset(
            'scan_positions', shape=(0, 2), maxshape=(None, 2),
            dtype='int32', compression=compression, compression_opts=compression_opts, chunks=(1024, 2)
        )

        # Static datasets
        self.f.create_dataset('object_projection', data=obj.astype(np.float32), compression=compression, compression_opts=compression_opts)
        self.f.create_dataset('probe_amplitude', data=self.probe_amp, compression=compression, compression_opts=compression_opts)
        self.f.create_dataset('probe_phase', data=self.probe_phase, compression=compression, compression_opts=compression_opts)

        # Metadata
        for k, v in metadata.items():
            try:
                self.f.attrs[k] = v
            except Exception:
                self.f.attrs[k] = str(v)

    def append(self, conv_batch: np.ndarray, ideal_batch: np.ndarray, positions_batch: np.ndarray):
        b = conv_batch.shape[0]
        # Resize and write
        self.ds_conv.resize(self.n + b, axis=0)
        self.ds_conv[self.n:self.n + b, :, :] = conv_batch.astype(np.float32)
        self.ds_ideal.resize(self.n + b, axis=0)
        self.ds_ideal[self.n:self.n + b, :, :] = ideal_batch.astype(np.float32)
        self.ds_pos.resize(self.n + b, axis=0)
        self.ds_pos[self.n:self.n + b, :] = positions_batch.astype(np.int32)

        # Update overlay with probe outlines per position
        # Red channel index 0
        H, W, _ = self.overlay_rgb.shape
        Ph, Pw = self.probe_amp_norm_u8.shape
        for k in range(b):
            y, x = int(positions_batch[k, 0]), int(positions_batch[k, 1])
            y2 = min(y + Ph, H)
            x2 = min(x + Pw, W)
            sy = max(0, y)
            sx = max(0, x)
            ph_s = 0 if y >= 0 else -y
            pw_s = 0 if x >= 0 else -x
            oh = y2 - sy
            ow = x2 - sx
            if oh > 0 and ow > 0:
                patch = self.probe_amp_norm_u8[ph_s:ph_s + oh, pw_s:pw_s + ow]
                self.overlay_rgb[sy:y2, sx:x2, 0] = np.maximum(self.overlay_rgb[sy:y2, sx:x2, 0], patch)

        self.n += b

    def finalize(self):
        # Save overlay
        self.f.create_dataset('overlay_rgb', data=self.overlay_rgb, compression='gzip', compression_opts=4)
        self.f.close()


def simulate_projection_scans(object_img_c: torch.Tensor,
                              probe_c: torch.Tensor,
                              probe_vignette: torch.Tensor,
                              scan_step: int,
                              batch_size: int,
                              segment_size: int,
                              output_dir: str,
                              phi_deg: float,
                              phi_index: int,
                              mask: Optional[np.ndarray],
                              device: torch.device,
                              dp_counter_start: int = 0) -> int:
    probe_size = probe_c.shape[0]
    positions = compute_scan_positions(object_img_c.shape[0], object_img_c.shape[1], probe_size, scan_step)
    positions = positions.to(device)
    dp_count = dp_counter_start

    for i in tqdm(range(0, len(positions), batch_size), desc=f"Phi {phi_index:03d} scans"):
        batch = positions[i:i + batch_size]
        bs = len(batch)
        object_patches = torch.zeros((bs, probe_size, probe_size), dtype=torch.cfloat, device=device)
        for j, pos in enumerate(batch):
            yy, xx = int(pos[0].item()), int(pos[1].item())
            object_patches[j] = object_img_c[yy:yy + probe_size, xx:xx + probe_size]

        exit_waves = object_patches * probe_c * probe_vignette
        exit_waves_ideal = object_patches * probe_vignette

        dp = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(exit_waves, dim=(-2, -1)), norm='ortho'), dim=(-2, -1))
        dp_ideal = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(exit_waves_ideal, dim=(-2, -1)), norm='ortho'), dim=(-2, -1))
        intensity = torch.abs(dp) ** 2
        intensity_ideal = torch.abs(dp_ideal) ** 2

        dps_conv_np = intensity.detach().cpu().numpy()
        dps_ideal_np = intensity_ideal.detach().cpu().numpy()

        for k in range(bs):
            scan_index = i + k
            yy = int(batch[k][0].item())
            xx = int(batch[k][1].item())
            dp_count = segment_and_save(
                dps_conv_np[k],
                dps_ideal_np[k],
                segment_size,
                dp_count,
                output_dir,
                scan_position=(yy, xx),
                scan_index=scan_index,
                phi_deg=phi_deg,
                phi_index=phi_index,
                mask=mask
            )

        del object_patches, exit_waves, exit_waves_ideal, dp, dp_ideal, intensity, intensity_ideal
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return dp_count


def convolve2d_batch_fft(input_batch: torch.Tensor, kernel: torch.Tensor, mode: str = 'same') -> torch.Tensor:
    """
    Perform batched 2D convolution using FFT for efficiency.
    Similar to scipy.signal.convolve2d with mode='same' and boundary='symm'.
    
    Args:
        input_batch: (N, H, W) tensor to convolve
        kernel: (kH, kW) convolution kernel
        mode: 'same' to match input size (default)
    
    Returns:
        Convolved tensor of shape (N, H, W)
    """
    N, H, W = input_batch.shape
    kH, kW = kernel.shape
    
    # For 'same' mode, pad the input
    if mode == 'same':
        pad_h = kH // 2
        pad_w = kW // 2
        # Symmetric padding to match boundary='symm'
        input_padded = torch.nn.functional.pad(input_batch, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
    else:
        input_padded = input_batch
    
    # Compute FFT size (next power of 2 for efficiency)
    fft_h = input_padded.shape[1]
    fft_w = input_padded.shape[2]
    
    # FFT of input batch
    input_fft = torch.fft.rfft2(input_padded, s=(fft_h, fft_w))
    
    # Pad kernel to match input size and FFT
    kernel_padded = torch.zeros((fft_h, fft_w), dtype=kernel.dtype, device=kernel.device)
    kernel_padded[:kH, :kW] = kernel
    kernel_fft = torch.fft.rfft2(kernel_padded, s=(fft_h, fft_w))
    
    # Multiply in frequency domain (convolution in spatial domain)
    output_fft = input_fft * kernel_fft.unsqueeze(0)
    
    # Inverse FFT
    output = torch.fft.irfft2(output_fft, s=(fft_h, fft_w))
    
    # Crop to match original input size for 'same' mode
    if mode == 'same':
        output = output[:, pad_h:pad_h+H, pad_w:pad_w+W]
    
    return output


def simulate_projection_scans_to_h5(object_img_c: torch.Tensor,
                                    probe_c: torch.Tensor,
                                    probe_vignette: torch.Tensor,
                                    scan_step: int,
                                    batch_size: int,
                                    h5_writer: H5ScanWriter,
                                    device: torch.device,
                                    Nc_avg: float = 1e4) -> int:
    probe_size = probe_c.shape[0]
    positions = compute_scan_positions(object_img_c.shape[0], object_img_c.shape[1], probe_size, scan_step)
    positions = positions.to(device)

    # Load pbp (pinhole probe) kernel once outside the loop
    #pbp_256_np = np.abs(np.load('/home/beams0/PTYCHOSAXS/NN/probe_pinhole_complex_256x256_bw0.75.npy'))
    pbp_256_np = np.fft.fftshift(np.abs(np.load('/home/beams0/PTYCHOSAXS/NN/probe_pinhole.npy')))
    pbp_256_np = zoom(pbp_256_np, (5, 5), order=1)  # 256 * 5 = 1280
    pbp_kernel = torch.tensor(pbp_256_np, dtype=torch.float32, device=device)

    total = 0
    for i in tqdm(range(0, len(positions), batch_size), desc="Saving to H5"):
        batch = positions[i:i + batch_size]
        bs = len(batch)
        object_patches = torch.zeros((bs, probe_size, probe_size), dtype=torch.cfloat, device=device)
        for j, pos in enumerate(batch):
            yy, xx = int(pos[0].item()), int(pos[1].item())
            object_patches[j] = object_img_c[yy:yy + probe_size, xx:xx + probe_size]
            object_patches[j] = apply_2d_vignette(object_patches[j], device=device) # Vignette patch
        exit_waves = object_patches * probe_c * probe_vignette
        
        # Apply pbp to exit waves
        exit_waves_ideal = object_patches * probe_vignette
        
        dp = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(exit_waves, dim=(-2, -1))), dim=(-2, -1))
        dp_ideal = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(exit_waves_ideal, dim=(-2, -1))), dim=(-2, -1))
        
        # Add Poisson noise to diffraction patterns
        dp_true_noise, dp_noise, snr_noise = add_poisson_noise_to_dp_torch(dp, Nc_avg=Nc_avg, N=1280)
        dp_true_ideal_noise, dp_ideal_noise, snr_ideal_noise = add_poisson_noise_to_dp_torch(dp_ideal, Nc_avg=Nc_avg, N=1280)
        
        # Convolve dp_ideal amplitude with pbp kernel (mimicking cupy conv2)
        dp_ideal_amplitude = torch.abs(dp_ideal_noise)  # Get amplitude
        dp_ideal_convolved = convolve2d_batch_fft(dp_ideal_amplitude, pbp_kernel, mode='same')
        
        intensity = torch.abs(dp_noise) ** 2
        intensity_ideal = torch.abs(dp_ideal_convolved) ** 2

        dps_conv_np = intensity.detach().cpu().numpy()
        dps_ideal_np = intensity_ideal.detach().cpu().numpy()
        pos_np = batch.detach().cpu().numpy()

        h5_writer.append(dps_conv_np, dps_ideal_np, pos_np)
        total += bs

        del object_patches, exit_waves, exit_waves_ideal, dp, dp_ideal, dp_ideal_amplitude, dp_ideal_convolved, intensity, intensity_ideal
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return total


def add_poisson_noise_to_dp_torch(psi: torch.Tensor, Nc_avg: float = float('inf'), N: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Add Poisson noise to diffraction patterns using PyTorch.
    
    Parameters:
    -----------
    psi : torch.Tensor
        Complex exit wave (probe * object)
    Nc_avg : float, optional
        Average number of counts per pixel. If inf, no noise is added.
        Default is float('inf') (no noise)
    N : int, optional
        Grid size for normalization. If None, uses the size of psi.
        Default is None
        
    Returns:
    --------
    dp_true : torch.Tensor
        True diffraction pattern (no noise)
    dp : torch.Tensor
        Diffraction pattern with Poisson noise added
    snr : float
        Signal-to-noise ratio
    """
    # Calculate true diffraction pattern
    dp_true = torch.abs(psi)**2
    dp = dp_true.clone()
    
    # Initialize SNR
    snr = float('inf')
    
    # Add Poisson noise if Nc_avg is finite
    if Nc_avg < float('inf'):
        dp_true_temp = dp_true.clone()
        
        # Normalize to get total counts
        if N is None:
            N = int(torch.sqrt(torch.tensor(psi.numel())))
        
        # Scale to desired average counts per pixel
        dp_temp = dp_true_temp / torch.sum(dp_true_temp) * (N**2 * Nc_avg)
        
        # Add Poisson noise using PyTorch's Poisson distribution
        dp_temp = torch.poisson(dp_temp)
        
        # Scale back to original intensity
        dp_temp = dp_temp * torch.sum(dp_true_temp) / (N**2 * Nc_avg)
        
        # Calculate SNR
        snr = torch.mean(dp_true_temp) / torch.std(dp_true_temp - dp_temp)
        snr = float(snr.cpu())  # Convert to Python float
        
        dp = dp_temp
    
    return dp_true, dp, snr


def run_tomo_simulation(
    probe_mat: str,
    lattice_path: str,
    probe_key: str = 'probe',
    probe_slice: int = 0,
    target_size: int = 1280,
    phi_axis: str = 'y',
    num_phi: int = 180,
    phi_start_random: bool = False,
    phi_start_deg: float = 0.0,
    apply_initial_random_orientation: bool = False,
    apply_random_rotation_during_simulation: bool = False,
    scan_step_div_min: int = 18,
    scan_step_div_max: int = 36,
    batch_size: int = 32,
    segment_size: int = 256,
    #output_dir: str = '/home/beams/PTYCHOSAXS/NN/ptychosaxsNN/data/tomo_output',
    output_dir: str = '/scratch/tomo_output',
    mask_path: Optional[str] = None,
    seed: Optional[int] = None,
    add_lattice_noise: bool = False,
    lattice_noise_std_fraction: float = 0.01,
    lattice_noise_lowpass_sigma: Optional[float] = None,
    add_projection_noise: bool = False,
    projection_noise_std_fraction: float = 0.01,
    projection_noise_lowpass_sigma: Optional[float] = None,
    zoom_lattice: bool = False,
    zoom_lattice_factor: float = 1.5,
    h5_suffix: str = 'TEST',
    Nc_avg: float = 1e4,
) -> dict:
    """Run ptycho-tomography simulation programmatically (Jupyter-friendly).

    Returns a dict with simple run metadata: dp_count, phi_start, scan_step.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    device = get_device()

    # Load and resize probe
    probe_np = load_probe_from_mat(probe_mat, key=probe_key, slice_index=probe_slice)
    if probe_np.shape[0] != target_size:
        print(f"Resizing probe from {probe_np.shape[0]} to {target_size}")
        probe_c = resize_probe_fourier(probe_np, target_size, device=device)
    else:
        print(f"Probe already resized to {target_size}")
        probe_c = torch.tensor(probe_np, dtype=torch.cfloat, device=device)
    probe_vignette = build_probe_vignette(target_size, device=device)

    # Load lattice and prepare initial orientation
    lattice = load_lattice(lattice_path).astype(np.float32)
    lattice = apply_3d_vignette(lattice)
    if add_lattice_noise:
        print("Adding lattice noise...")
        lattice = add_noise_3d(
            lattice,
            std_fraction=float(lattice_noise_std_fraction),
            lowpass_sigma=lattice_noise_lowpass_sigma,
            seed=seed,
        )
    
    # # Plot lattice
    # plt.figure()
    # plt.imshow(np.sum(lattice,axis=2))
    # plt.show()
    # plt.figure()
    # plt.imshow(np.abs(np.fft.fftshift(np.fft.fft2(np.sum(lattice,axis=2))))**2,norm=colors.LogNorm())
    # plt.show()

    if apply_initial_random_orientation:
        print("Applying initial 3D random orientation...")
        angle_x0 = random.uniform(0, 360)
        angle_y0 = random.uniform(0, 360)
        angle_z0 = random.uniform(0, 360)
        # lattice = rotate(lattice, angle_x0, axes=(1, 2), reshape=False, order=1)
        # lattice = rotate(lattice, angle_y0, axes=(0, 2), reshape=False, order=1)
        # lattice = rotate(lattice, angle_z0, axes=(0, 1), reshape=False, order=1)
        lattice = rotate_gpu(cp.array(lattice), angle_x0, axes=(1, 2), reshape=False, order=1)
        lattice = rotate_gpu(cp.array(lattice), angle_y0, axes=(0, 2), reshape=False, order=1)
        lattice = rotate_gpu(cp.array(lattice), angle_z0, axes=(0, 1), reshape=False, order=1)

    lattice = cp.asnumpy(lattice)
    
    phi_start = random.uniform(0, 360) if phi_start_random else float(phi_start_deg)
    phi_step = 360.0 / float(num_phi)

    # Optional mask
    mask_np: Optional[np.ndarray] = None
    if mask_path is not None and os.path.isfile(mask_path):
        mask_np = np.load(mask_path).astype(np.float32)

    # Prepare constants
    probe_size = target_size
    scan_step_div = random.randint(scan_step_div_min, scan_step_div_max)
    scan_step = probe_size // scan_step_div


    # Toggle to plot projections
    plot_projections=True
    
    dp_counter = 0
    for phi_idx in tqdm(range(num_phi),desc="Phi projections"):
        phi_deg = phi_start + phi_step * phi_idx
        print(f"Phi {phi_idx:03d} angle: {phi_deg:.1f}°")
        vol_rot = rotate_volume(lattice, angle_deg=phi_deg, axis=phi_axis)
        
        if apply_random_rotation_during_simulation & (phi_idx % 10 == 0):
            angle_x = random.uniform(0, 360)
            angle_y = random.uniform(0, 360)
            angle_z = random.uniform(0, 360)
            vol_rot = rotate_volume(vol_rot, angle_deg=angle_x, axis='x')
            vol_rot = rotate_volume(vol_rot, angle_deg=angle_y, axis='y')
            vol_rot = rotate_volume(vol_rot, angle_deg=angle_z, axis='z')

        # Apply zoom and rotation to the projected lattice
        proj2d = project_to_2d(vol_rot, projection_axis=2)
        if zoom_lattice:
            zoomed = zoom(proj2d, (zoom_lattice_factor, zoom_lattice_factor), order=1)
            proj2d = zoomed

        if add_projection_noise:
            print("Adding 2D projection noise...")
            proj2d = add_noise_2d(
                proj2d,
                std_fraction=float(projection_noise_std_fraction),
                lowpass_sigma=projection_noise_lowpass_sigma,
                seed=seed,
            )
        
        if plot_projections:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle('Projected Lattice')
            ax1.imshow(proj2d)
            ax1.set_title('Real Space')
            ax2.imshow(np.abs(np.fft.fftshift(np.fft.fft2(proj2d)))**2, norm=colors.LogNorm())
            ax2.set_title('Fourier Space')
            plt.tight_layout()
            plt.show()
        else:
            print('Not plotting projections')

        proj2d = proj2d/np.max(proj2d)
        proj2d = np.exp(1j * proj2d)
        
        if plot_projections:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle('Projected Lattice (phase object)')
            ax1.imshow(np.abs(proj2d))
            ax1.set_title('Amplitude')
            ax2.imshow(np.angle(proj2d))
            ax2.set_title('Phase')
            plt.tight_layout()
            plt.show()

        #lattice_2d = torch.tensor(proj2d, dtype=torch.float32, device=device)
        lattice_2d = torch.tensor(proj2d, dtype=torch.cfloat, device=device)
        lattice_2d = apply_2d_vignette(lattice_2d, device=device)

        # Margin for scanning (pad to 1.5x probe size if needed)
        pad_target = int(1.5 * probe_size)
        if lattice_2d.shape[0] < pad_target or lattice_2d.shape[1] < pad_target:
            lattice_2d = apply_2d_vignette(lattice_2d, device=device)
            pad_h = max(0, (pad_target - lattice_2d.shape[0]) // 2)
            pad_w = max(0, (pad_target - lattice_2d.shape[1]) // 2)
            # Background handling
            pad_value = 0.0
            bkg = pad_value
            lattice_2d = torch.nn.functional.pad(lattice_2d+bkg, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=pad_value)
            #lattice_2d = apply_2d_vignette(lattice_2d, device=device)

        if plot_projections:
            fig, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(12, 5))
            fig.suptitle('Projected Lattice (Padded and Vignetted)')
            ax1.imshow(np.abs(lattice_2d.cpu().numpy()))
            ax1.set_title('Real Space (abs)')
            ax2.imshow(np.angle(lattice_2d.cpu().numpy()))
            ax2.set_title('Real Space (angle)')
            ax3.imshow(np.abs(np.fft.fftshift(np.fft.fft2(lattice_2d.cpu().numpy())))**2, norm=colors.LogNorm())
            ax3.set_title('Fourier Space')
            plt.tight_layout()
            plt.show()

        object_img_c = lattice_2d 

        # Create H5 writer for this projection
        h5_name = f"tomo_phi_{phi_idx:03d}{h5_suffix}.h5"
        h5_path = os.path.join(output_dir, h5_name)
        h5_writer = H5ScanWriter(
            h5_path=h5_path,
            dp_height=probe_size,
            dp_width=probe_size,
            object_image=object_img_c.abs().detach().cpu().numpy(),
            probe_complex=probe_c.detach().cpu().numpy(),
            metadata={
                'phi_deg': float(phi_deg),
                'phi_index': int(phi_idx),
                'phi_axis': str(phi_axis),
                'target_size': int(target_size),
                'scan_step': int(scan_step),
            },
            compression='gzip',
            compression_opts=4,
            overlay_alpha=0.5,
        )

        _ = simulate_projection_scans_to_h5(
            object_img_c=object_img_c,
            probe_c=probe_c,
            probe_vignette=probe_vignette,
            scan_step=scan_step,
            batch_size=batch_size,
            h5_writer=h5_writer,
            device=device,
            Nc_avg=Nc_avg,
        )
        h5_writer.finalize()

        del vol_rot, proj2d, lattice_2d, object_img_c
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return {
        'dp_count': int(dp_counter),
        'phi_start': float(phi_start),
        'scan_step': int(scan_step),
    }
# %%
