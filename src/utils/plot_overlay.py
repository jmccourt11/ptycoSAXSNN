#%%
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Circle
import scipy.io as sio
from scipy.interpolate import RegularGridInterpolator
import matplotlib.ticker as ticker

import matplotlib as mpl

# ----------------------------
# 0) Load your data (you already have this)
# ----------------------------
base_path = r"Z:\12IDC\\"
path = base_path + r"ptychosaxs\batch_mode_250\RSM\best_model_LatticeClathII_Probe256x256_ZCB_9_3D__Noise_sim_ZCB_9_3D_S5065_N600_steps4_dp256_Unet_epoch_250_pearson_loss_symmetry_0.0\combined_diffraction_patterns.h5"
scan_number = 5065

with h5py.File(path, 'r') as f:
    data = f['deconvolved'][f'scan_{scan_number}'][()]   # diffraction (intensity) [Ndety x Ndetx] or [Ndetx x Ndety]

print("data shape:", data.shape)
obj_path = base_path + rf"2025_Feb\results\ZCB_9_3D_\fly{scan_number}\roi0_Ndp256\MLc_L1_p10_gInf_Ndp128_mom0.5_pc0_maxPosError500nm_bg0.1_vi_mm\MLc_L1_p10_g100_Ndp256_mom0.5_pc800_maxPosError500nm_bg0.1_vp4_vi_mm\Niter1000.mat"
obj = sio.loadmat(obj_path)["object_roi"]      # complex object in real space
print("obj shape:", obj.shape, "dtype:", obj.dtype)

# ----------------------------
# 1) Experiment constants (FILL THESE)
# ----------------------------
# wavelength [m], sample-detector distance [m], detector pixel size(s) [m]
lam = 1.239e-10  # e.g., 1.23984e-10 for 10 keV
D   = 10  # e.g., 2.0
p_x = 172e-6  # e.g., 75e-6  (pixel pitch in x on detector, after binning)
p_y = 172e-6  # e.g., 75e-6  (pixel pitch in y on detector, after binning)

# Optional: beam center (in detector pixel indices). Use data center if None.
cx = None  # e.g., 128.3
cy = None  # e.g., 127.7

# Conversion constant: 1 Angstrom = 1e-10 m (for q-space units)
ANGSTROM_TO_M = 1e10

# ----------------------------
# 2) Ensure arrays are 2D and consistent orientation
# ----------------------------
# We will treat axis 1 as x and axis 0 as y throughout (numpy "image" convention).
data = np.asarray(data)
if data.ndim != 2:
    raise ValueError("Expected 2D diffraction pattern.")

Ndety, Ndetx = data.shape   # rows (y), cols (x)

# Real-space object: accept complex 2D; if 3D or transposed, adapt here
obj = np.asarray(obj)
# If obj has trailing singleton dims, squeeze:
obj = np.squeeze(obj)
if obj.ndim != 2:
    # Try to pick a 2D slice if it is 3D (user can adjust)
    raise ValueError("Expected 2D real-space object array (obj).")

Ny, Nx = obj.shape  # rows (y), cols (x)

# ----------------------------
# 3) Link ptychography geometry to real-space pixel size
#     Δx = λ D / (Ndetx * p_x)
#     Δy = λ D / (Ndety * p_y)
# ----------------------------
dx = lam * D / (Ndetx * p_x)
dy = lam * D / (Ndety * p_y)

Lx = Nx * dx
Ly = Ny * dy

dq_x_obj = 2.0 * np.pi / Lx   # spacing of FFT(obj) along qx
dq_y_obj = 2.0 * np.pi / Ly   # spacing of FFT(obj) along qy

# q-axes for the object's FFT (centered)
qx_obj = (np.arange(Nx) - Nx//2) * dq_x_obj
qy_obj = (np.arange(Ny) - Ny//2) * dq_y_obj

# Convert from 1/m to 1/Å
qx_obj = qx_obj / ANGSTROM_TO_M
qy_obj = qy_obj / ANGSTROM_TO_M

# ----------------------------
# 4) Detector q-grid from geometry
#     Δq_det_x = (2π/λ) * (p_x / D)
#     Δq_det_y = (2π/λ) * (p_y / D)
# ----------------------------
dq_det_x = (2.0 * np.pi / lam) * (p_x / D)
dq_det_y = (2.0 * np.pi / lam) * (p_y / D)

# Beam center: default to middle of the detector array if not provided
if cx is None: cx = (Ndetx - 1) / 2.0
if cy is None: cy = (Ndety - 1) / 2.0

ix = np.arange(Ndetx)  # x indices (cols)
iy = np.arange(Ndety)  # y indices (rows)

qx_det = (ix - cx) * dq_det_x
qy_det = (iy - cy) * dq_det_y

# Convert from 1/m to 1/Å
qx_det = qx_det / ANGSTROM_TO_M
qy_det = qy_det / ANGSTROM_TO_M
Qx_det, Qy_det = np.meshgrid(qx_det, qy_det, indexing='xy')

# ----------------------------
# 5) FFT of object and interpolate onto detector q-grid
# ----------------------------
# Use centered FFT; magnitude if comparing to intensity (phase ignored here).
Fobj = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(obj)))
Amp_obj = np.abs(Fobj)  # amplitude; for intensity model you'd square (but compare with scaling)

# Build interpolator on (qx_obj, qy_obj). Note RegularGridInterpolator expects increasing axes.
# Our qx_obj, qy_obj are already monotonically increasing because of the way we built them.
interp = RegularGridInterpolator(
    (qy_obj, qx_obj),   # order: (y, x) because Amp_obj is indexed [y, x]
    Amp_obj,
    bounds_error=False,
    fill_value=0.0
)

# Points to sample: pairs of (qy, qx) from detector grid
pts = np.column_stack([Qy_det.ravel(), Qx_det.ravel()])
Amp_on_det = interp(pts).reshape(Ndety, Ndetx)

# If your measured data is intensity, compare to |FFT(obj)|^2 (Fraunhofer intensity proxy).
# Try both and choose what matches your pipeline best:
I_theory = Amp_on_det**2

# ----------------------------
# 6) Intensity scaling to match measured data (simple least-squares scale)
# ----------------------------
meas = np.asarray(data, dtype=float)
meas = np.where(np.isfinite(meas), meas, 0.0)

theory = I_theory
theory = np.where(np.isfinite(theory), theory, 0.0)

# Optional: mask central beamstop or saturated center to avoid bias
# Example: circular mask of r < r0 pixels (edit r0 if needed)
r0 = 3
Y, X = np.indices((Ndety, Ndetx))
mask_center = ((X - cx)**2 + (Y - cy)**2) >= r0**2

# Compute scalar 'a' minimizing || a*theory - meas || over masked region
num = np.sum(theory[mask_center] * meas[mask_center])
den = np.sum(theory[mask_center] * theory[mask_center]) + 1e-30
a = num / den
I_theory_scaled = a * theory

print(f"LS scale factor a = {a:.6g}")

# ----------------------------
# 7) Plots in q-space with overlay
# ----------------------------
# q-space extent for imshow: [qx_min, qx_max, qy_min, qy_max]
# Note: imshow uses extent in the order [left, right, bottom, top]
qx_min, qx_max = qx_det[0], qx_det[-1]
qy_min, qy_max = qy_det[0], qy_det[-1]
extent_q = [qx_min, qx_max, qy_min, qy_max]



# Set font sizes even larger for publication-quality
mpl.rcParams.update({
    'font.size': 32,
    'axes.titlesize': 36,
    'axes.labelsize': 34,
    'xtick.labelsize': 30,
    'ytick.labelsize': 30,
    'legend.fontsize': 28,
    'axes.linewidth': 3,
    'xtick.major.width': 2.5,
    'ytick.major.width': 2.5,
    'figure.titlesize': 36,  # For figure-level titles if used
})

fig, axs = plt.subplots(2, 2, figsize=(32, 24), constrained_layout=True)

# Circle radius in q-space (1/Å)
circle_radius = 0.0065

tick_step = 0.005
# Font sizes are now controlled by rcParams above

# Determine the tick locations within the plotting limits only
qx_tick_min = np.ceil(qx_min / tick_step) * tick_step if qx_min % tick_step else qx_min
qx_tick_max = np.floor(qx_max / tick_step) * tick_step if qx_max % tick_step else qx_max
qy_tick_min = np.ceil(qy_min / tick_step) * tick_step if qy_min % tick_step else qy_min
qy_tick_max = np.floor(qy_max / tick_step) * tick_step if qy_max % tick_step else qy_max

qx_ticks = np.arange(qx_tick_min, qx_tick_max + tick_step/2, tick_step)
qy_ticks = np.arange(qy_tick_min, qy_tick_max + tick_step/2, tick_step)

# --- Helper to add black rectangle border to an axis ---
def add_black_border(ax, linewidth=4.0):
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(linewidth)

# Top row: individual plots in q-space
im0 = axs[0, 0].imshow(
    meas, extent=extent_q, origin='lower',
    norm=colors.LogNorm(vmin=max(meas[mask_center].min()+1e-12, 1e-12), vmax=meas[mask_center].max()),
    cmap='jet', aspect='auto'
)
circle0 = Circle((0, 0), circle_radius, fill=False, edgecolor='white', linewidth=5, linestyle='--')
axs[0, 0].add_patch(circle0)
axs[0, 0].set_title("Measured diffraction (data)", pad=16)
axs[0, 0].set_xlabel("$q_{x,\perp}$ (1/Å)", labelpad=12)
axs[0, 0].set_ylabel("$q_{y,\perp}$ (1/Å)", labelpad=10)
axs[0, 0].set_xticks(qx_ticks)
axs[0, 0].set_yticks(qy_ticks)
axs[0, 0].set_xlim(qx_min, qx_max)
axs[0, 0].set_ylim(qy_min, qy_max)
axs[0, 0].tick_params(axis='both', which='major', width=2.5, length=9)
add_black_border(axs[0, 0], linewidth=4.0)
cb0 = plt.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)
cb0.ax.tick_params()
cb0.outline.set_linewidth(3)

im1 = axs[0, 1].imshow(
    I_theory_scaled, extent=extent_q, origin='lower',
    norm=colors.LogNorm(vmin=max(I_theory_scaled[mask_center].min()+1e-12, 1e-12), vmax=I_theory_scaled[mask_center].max()),
    cmap='jet', aspect='auto'
)
circle1 = Circle((0, 0), circle_radius, fill=False, edgecolor='white', linewidth=5, linestyle='--')
axs[0, 1].add_patch(circle1)
axs[0, 1].set_title("|FFT(obj)|² resampled → detector (scaled)", pad=16)
axs[0, 1].set_xlabel("$q_{x,\perp}$ (1/Å)", labelpad=12)
axs[0, 1].set_ylabel("$q_{y,\perp}$ (1/Å)", labelpad=10)
axs[0, 1].set_xticks(qx_ticks)
axs[0, 1].set_yticks(qy_ticks)
axs[0, 1].set_xlim(qx_min, qx_max)
axs[0, 1].set_ylim(qy_min, qy_max)
axs[0, 1].tick_params(axis='both', which='major', width=2.5, length=9)
add_black_border(axs[0, 1], linewidth=4.0)
cb1 = plt.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)
cb1.ax.tick_params()
cb1.outline.set_linewidth(3)

# Bottom left: Ratio in q-space
eps = 1e-30
res = (meas + eps) / (I_theory_scaled + eps)
im2 = axs[1, 0].imshow(
    res, extent=extent_q, origin='lower',
    norm=colors.LogNorm(vmin=0.5, vmax=2.0),
    cmap='jet', aspect='auto'
)
circle2 = Circle((0, 0), circle_radius, fill=False, edgecolor='white', linewidth=5, linestyle='--')
axs[1, 0].add_patch(circle2)
axs[1, 0].set_title("Ratio: data / theory", pad=16)
axs[1, 0].set_xlabel("$q_{x,\perp}$ (1/Å)", labelpad=12)
axs[1, 0].set_ylabel("$q_{y,\perp}$ (1/Å)", labelpad=10)
axs[1, 0].set_xticks(qx_ticks)
axs[1, 0].set_yticks(qy_ticks)
axs[1, 0].set_xlim(qx_min, qx_max)
axs[1, 0].set_ylim(qy_min, qy_max)
axs[1, 0].tick_params(axis='both', which='major', width=2.5, length=9)
add_black_border(axs[1, 0], linewidth=4.0)
cb2 = plt.colorbar(im2, ax=axs[1, 0], fraction=0.046, pad=0.04)
cb2.ax.tick_params()
cb2.outline.set_linewidth(3)

# Bottom right: Overlay plot
# Show measured data as background (grayscale) and theory as colored overlay
axs[1, 1].imshow(
    meas, extent=extent_q, origin='lower',
    norm=colors.LogNorm(vmin=max(meas[mask_center].min()+1e-12, 1e-12), vmax=meas[mask_center].max()),
    cmap='Reds', aspect='auto', alpha=0.8
)
im_overlay = axs[1, 1].imshow(
    I_theory_scaled, extent=extent_q, origin='lower',
    norm=colors.LogNorm(vmin=max(I_theory_scaled[mask_center].min()+1e-12, 1e-12), vmax=I_theory_scaled[mask_center].max()),
    cmap='Blues', aspect='auto', alpha=0.3
)
circle3 = Circle((0, 0), circle_radius, fill=False, edgecolor='white', linewidth=5, linestyle='--')
axs[1, 1].add_patch(circle3)
axs[1, 1].set_title("Overlay: Measured (gray) + Theory (color)", pad=16)
axs[1, 1].set_xlabel("$q_{x,\perp}$ (1/Å)", labelpad=12)
axs[1, 1].set_ylabel("$q_{y,\perp}$ (1/Å)", labelpad=10)
axs[1, 1].set_xticks(qx_ticks)
axs[1, 1].set_yticks(qy_ticks)
axs[1, 1].set_xlim(qx_min, qx_max)
axs[1, 1].set_ylim(qy_min, qy_max)
axs[1, 1].tick_params(axis='both', which='major', width=2.5, length=9)
add_black_border(axs[1, 1], linewidth=4.0)
cb_overlay = plt.colorbar(im_overlay, ax=axs[1, 1], fraction=0.046, pad=0.04, label='Theory intensity')
cb_overlay.ax.tick_params()
cb_overlay.outline.set_linewidth(3)
cb_overlay.set_label('Theory intensity', labelpad=10)

# Extra: Make sure the figure background is white and tight
fig.patch.set_facecolor('white')
plt.show()


# ----------------------------
# 8) Sanity checks printed
# ----------------------------
print(f"Ndetx, Ndety = {Ndetx}, {Ndety}")
print(f"Nx, Ny (object) = {Nx}, {Ny}")
print(f"dx, dy [m]: {dx:.3e}, {dy:.3e}")
print(f"dq_obj_x, dq_obj_y [1/Å]: {dq_x_obj/ANGSTROM_TO_M:.3e}, {dq_y_obj/ANGSTROM_TO_M:.3e}")
print(f"dq_det_x, dq_det_y [1/Å]: {dq_det_x/ANGSTROM_TO_M:.3e}, {dq_det_y/ANGSTROM_TO_M:.3e}")
print(f"Center (cx, cy) = ({cx:.3f}, {cy:.3f})")

# %%
