#%%
from tomo_funcs import *
import h5py
import os
from pathlib import Path
from datetime import datetime
import matplotlib.patches as patches
#%%
# Load the data
# Clathrate I

# Clathrate II
#tomogram="/net/micdata/data2/12IDC/2025_Feb/misc/ZCB_9_3D_/tomogram_alignment_recon_56nm.tiff"
tomogram="/net/micdata/data2/12IDC/2025_Feb/misc/ZCB_9_3D_/tomogram_alignment_recon_28nm.tiff"

# Clathrate IV


# RBP
#tomogram="/net/micdata/data2/12IDC/2025_Jul/results/analysis_tomo/ZC6_3D/p20nm/tomogram_alignment_recon.tiff"
tomo_data = tifffile.imread(tomogram)

# Get current shape
shape = tomo_data.shape

# Calculate start indices to center the crop
start_z = (shape[0] - 384) // 2
start_y = (shape[1] - 412) // 2
start_x = (shape[2] - 506) // 2
# Crop to specified size
tomo_data = tomo_data[start_z:start_z+384, start_y:start_y+412, start_x:start_x+506]

#Rotate the tomogram if necessary
axis='z'
angle=0
if axis == 'x':
    rotated_data = rotate(tomo_data, angle, axes=(1, 2), reshape=False)
elif axis == 'y':
    rotated_data = rotate(tomo_data, angle, axes=(0, 2), reshape=False)
elif axis == 'z':
    rotated_data = rotate(tomo_data, angle, axes=(0, 1), reshape=False)
else:
    rotated_data = tomo_data
 
tomo_data = rotated_data

# Create and display the plot
intensity_threshold=0.8#0.6
fig = plot_3D_tomogram(tomo_data, intensity_threshold=intensity_threshold)
fig.show()

# Print dimensions
print(f"Tomogram shape: {tomo_data.shape}")

# Set pixel size
pixel_size=56
print(f"Pixel size: {pixel_size}")

#%%
magnitude, KX, KY, KZ=compute_fft_q(tomo_data, use_vignette=True, pixel_size=pixel_size,scale=1)

# Define a threshold for the magnitude
threshold_factor = 0.0005 #0.0005
threshold = threshold_factor* np.max(magnitude)  # Example: 1% of the max magnitude

# Flatten the arrays
kx_flat = KX.flatten()
ky_flat = KY.flatten()
kz_flat = KZ.flatten()
magnitude_flat = magnitude.flatten()

# Calculate radial distance from center for each point
radial_distance = np.sqrt(kx_flat**2 + ky_flat**2 + kz_flat**2)

# Apply both magnitude threshold and center cutoff
center_cutoff_radius=0.015#0.015
mask = (magnitude_flat > threshold) & (radial_distance > center_cutoff_radius)

# Apply the threshold
kx_filtered = kx_flat[mask]
ky_filtered = ky_flat[mask]
kz_filtered = kz_flat[mask]
magnitude_filtered = magnitude_flat[mask]

# Create a 3D scatter plot of the FFT magnitude
fig_fft = go.Figure(data=go.Scatter3d(
    x=kx_filtered,
    y=ky_filtered,
    z=kz_filtered,
    mode='markers',
    marker=dict(
        size=4,
        color=magnitude_filtered,
        colorscale='Viridis',
        opacity=0.8,
        colorbar=dict(title='Magnitude')
    )
))
fig_fft.show()


#%%

center_cutoff_radius=1e-2
# Add a semi-transparent sphere to visualize the cutoff region (optional)
u = np.linspace(0, 2 * np.pi, 20)
v = np.linspace(0, np.pi, 20)
x_sphere = center_cutoff_radius * np.outer(np.cos(u), np.sin(v))
y_sphere = center_cutoff_radius * np.outer(np.sin(u), np.sin(v))
z_sphere = center_cutoff_radius * np.outer(np.ones(np.size(u)), np.cos(v))

fig_fft.add_trace(go.Surface(
    x=x_sphere,
    y=y_sphere,
    z=z_sphere,
    opacity=0.2,
    showscale=False,
    name='Cutoff Region'
))

fig_fft.update_layout(
    title="3D FFT Magnitude with Threshold",
    scene=dict(
        xaxis_title="KX",
        yaxis_title="KY",
        zaxis_title="KZ",
        aspectmode='cube'
    ),
    width=800, height=800
)

#%%
# Run test
# Load and plot cell info
cellinfo_data = load_cellinfo_data("/home/beams/PTYCHOSAXS/cellinfo_ZCB_9.mat")
#cellinfo_data = load_cellinfo_data("/net/micdata/data2/12IDC/2025_Jul/results/analysis_tomo/ZC6_3D/cellinfo.mat")

# Generate all combinations for h, k, l in [-1, 0, 1]
hkl = np.array(list(product([-2, -1, 0, 1, 2], repeat=3)))

# Separate into hs, ks, ls arrays
hs = hkl[:, 0]
ks = hkl[:, 1]
ls = hkl[:, 2]

print("hs:", hs)
print("ks:", ks)
print("ls:", ls)
vs=[]

# Voxel size
limiting_axes=np.min(tomo_data.shape) #pixels
tomo_nm_size=pixel_size*limiting_axes #nm
n_unit_cells=tomo_nm_size//(cellinfo_data['Vol'][0][0]**(1/3)) #n unit cells / tomogram

print(f'~n unit cells per tomogram: {n_unit_cells}')

voxel_size = (25,25,25)  # cubic voxel size, pixels

print(f'~m unit cells per voxel: {n_unit_cells*voxel_size[0]/limiting_axes}')

voxel_results = analyze_tomogram_voxels(tomo_data, voxel_size=voxel_size)

# Print number of voxels in each dimension
print(f"Number of voxels (z, y, x): {voxel_results['n_voxels']}")

show_plots = False

# Define a threshold for the magnitude
threshold = 0.2 # Example: 5% of the max magnitude

# Peak finding threshold and sigma
peak_threshold=0.1
sigma=.5


for i,h in enumerate(hs):
    v=hs[i]*cellinfo_data['recilatticevectors'][0]+ks[i]*cellinfo_data['recilatticevectors'][1]+ls[i]*cellinfo_data['recilatticevectors'][2]
    vs.append(v)

vs=np.array(vs)
fig_fft.add_trace(go.Scatter3d(
    x=vs.T[0],
    y=vs.T[1],
    z=vs.T[2],
    mode='markers',
    marker=dict(size=5, color='red', opacity=0.1),
    name='Cell Info'
))


fig_fft.update_layout(
    title="3D FFT Magnitude with Threshold",
    scene=dict(
        xaxis_title="KX",
        yaxis_title="KY",
        zaxis_title="KZ",
        aspectmode='cube'
    ),
    width=800, height=800
)

fig_fft.show()


#%%
# Apply magnitude threshold
threshold_factor=0.0001#0.001#0.0002
magnitude_test = magnitude > threshold_factor*np.max(magnitude)

# Create spherical mask to remove central region
center = np.array(magnitude.shape) // 2
R = 48# Radius of sphere to mask out
x, y, z = np.ogrid[:magnitude.shape[0], :magnitude.shape[1], :magnitude.shape[2]]
sphere_mask = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 >= R**2

# Apply both filters
magnitude_test = magnitude_test & sphere_mask
h,k,l=1,0,1
pixel_size=pixel_size

# Have to do this because the tomogram is transposed with 
# respect to the cellinfo_data (cellinfo.mat) defined in MATLAB
tomogram_test=tomo_data.T
magnitude_test=magnitude_test.T

# Project and plot along hkl
# Real space
projection_test, rotated_tomo_test = project_and_plot_along_hkl(tomogram_test-np.mean(tomogram_test), cellinfo_data, h, k, l, \
    title_prefix="Test", is_reciprocal=False, q_vectors=None, pixel_size=pixel_size)
#%%
# Reciprocal space
projection_test_reciprocal, rotated_tomo_test_reciprocal = project_and_plot_along_hkl(magnitude_test-np.mean(magnitude_test), cellinfo_data, h, k, l, \
    title_prefix="Test", is_reciprocal=True, q_vectors=None, pixel_size=pixel_size)


#%%
plot_hkl_vector_in_tomogram(tomogram_test, cellinfo_data, h, k, l, \
    is_reciprocal=False, scale=0.1, plot_tomogram=True)
plot_unit_cell_in_tomogram(tomogram_test, cellinfo_data, plot_tomogram=True, \
    intensity_threshold=0.8, pixel_size=pixel_size)


#%%
# Plot real space and FFT of real space (diffraction pattern)
fig,ax=plt.subplots(1,2,figsize=(20,10))
ax[0].imshow(projection_test,cmap='gray')
ax[1].imshow(np.abs(np.fft.fftshift(np.fft.fft2(projection_test)))**2,\
    cmap='jet',norm=colors.LogNorm())
plt.show()


#%%
# Plot projections for all hkl [-1,0,1]
plot_hkl_projection_grid(tomo_data.T, magnitude_test.T, \
    cellinfo_data, pixel_size=None)








############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################

    










############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################











#%%
from tomo_funcs import *
import h5py
import os
from pathlib import Path
from datetime import datetime
import matplotlib.patches as patches
def load_and_analyze_tomogram(tomogram_path, cellinfo_path, pixel_size=20, intensity_threshold=0.8, 
                            rotation_axis='z', rotation_angle=0, fft_threshold=0.0005, 
                            center_cutoff=0.015, voxel_size=(25,25,25), output_dir=None, 
                            save_outputs=True, crop_tomogram=True, crop_size=None, hkl_projection=[1,0,1]):
    """
    Load and analyze tomogram data with FFT analysis and cell info visualization.
    
    Parameters:
    -----------
    tomogram_path : str
        Path to the tomogram TIFF file
    cellinfo_path : str 
        Path to the cellinfo.mat file
    pixel_size : float
        Size of pixels in nm
    intensity_threshold : float
        Threshold for 3D tomogram visualization
    rotation_axis : str
        Axis for rotation ('x', 'y', or 'z')
    rotation_angle : float
        Angle to rotate tomogram in degrees
    fft_threshold : float
        Threshold factor for FFT magnitude
    center_cutoff : float
        Radius for center cutoff in FFT
    voxel_size : tuple
        Size of voxels for analysis (z,y,x)
    output_dir : str, optional
        Base directory for saving outputs. If None, creates one based on tomogram filename
    save_outputs : bool
        Whether to save outputs to disk (default: True)
    
    Returns:
    --------
    dict containing:
        tomo_data : ndarray
            The loaded and processed tomogram data
        magnitude : ndarray
            FFT magnitude data
        voxel_results : dict
            Results of voxel analysis
        cellinfo_data : dict
            Loaded cell information
        figures : dict
            Generated matplotlib and plotly figures:
            - tomogram: 3D tomogram visualization
            - fft: 3D FFT visualization
            - hkl_vector: HKL vector plot
            - unit_cell: Unit cell visualization
            - projection_combined: Combined projection analysis
            - hkl_grid: HKL projection grid
        projections : dict
            Projection data in real and reciprocal space
        output_paths : dict (if save_outputs=True)
            Paths to all saved files including plots, HDF5 file, and summary
        output_dir : str (if save_outputs=True)
            Base directory containing all outputs
    
    Output Directory Structure:
    ---------------------------
    analysis_<tomogram_name>_<timestamp>/
    ├── plots/
    │   ├── 3d_visualizations/
    │   │   ├── tomogram_3d.html (interactive)
    │   │   ├── fft_3d.html (interactive)
    │   │   └── hkl_vector_in_tomogram.html (interactive)
    │   ├── projections/
    │   │   ├── projection_real_space.png
    │   │   ├── projection_real_space_fft.png
    │   │   ├── projection_reciprocal_space.png
    │   │   ├── projection_combined_analysis.png
    │   │   └── hkl_projection_grid.png
    │   └── fft_analysis/
    ├── data/
    │   └── analysis_complete.h5
    ├── unit_cell_info/
    │   └── unit_cell_in_tomogram.html (interactive)
    └── analysis_summary.txt
    """
    
    # Setup output directory structure
    if save_outputs:
        if output_dir is None:
            # Create output directory based on tomogram filename and timestamp
            tomo_name = Path(tomogram_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(tomogram_path).parent / f"analysis_{tomo_name}_{timestamp}"
        else:
            output_dir = Path(output_dir)
        
        # Create subdirectories
        dirs = {
            'base': output_dir,
            'plots': output_dir / 'plots',
            'plots_3d': output_dir / 'plots' / '3d_visualizations',
            'plots_projections': output_dir / 'plots' / 'projections',
            'plots_fft': output_dir / 'plots' / 'fft_analysis',
            'data': output_dir / 'data',
            'unit_cell': output_dir / 'unit_cell_info',
        }
        
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Output directory created: {output_dir}")
    else:
        dirs = None
    # Load tomogram
    tomo_data = tifffile.imread(tomogram_path)
    
    print(f"Tomogram shape: {tomo_data.shape}")
    print(f"Pixel size: {pixel_size} nm")
    
    # Crop tomogram from the center
    if crop_tomogram:
        shape = tomo_data.shape
        center = np.array(shape) // 2
        start = center - crop_size // 2
        end = start + crop_size
        
        tomo_data = tomo_data[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        print(f"Tomogram shape after crop: {tomo_data.shape}")
    else:
        print("Tomogram not cropped")
    # Rotate if needed
    if rotation_axis == 'x':
        tomo_data = rotate(tomo_data, rotation_angle, axes=(1, 2), reshape=False)
    elif rotation_axis == 'y':
        tomo_data = rotate(tomo_data, rotation_angle, axes=(0, 2), reshape=False)
    elif rotation_axis == 'z':
        tomo_data = rotate(tomo_data, rotation_angle, axes=(0, 1), reshape=False)
    
    # Create 3D tomogram plot
    fig_tomo = plot_3D_tomogram(tomo_data, intensity_threshold=intensity_threshold)
    
    # Compute FFT
    magnitude, KX, KY, KZ = compute_fft_q(tomo_data, use_vignette=True, pixel_size=pixel_size, scale=1)
    
    # Process FFT data
    threshold = fft_threshold * np.max(magnitude)
    kx_flat = KX.flatten()
    ky_flat = KY.flatten()
    kz_flat = KZ.flatten()
    magnitude_flat = magnitude.flatten()
    
    radial_distance = np.sqrt(kx_flat**2 + ky_flat**2 + kz_flat**2)
    mask = (magnitude_flat > threshold) & (radial_distance > center_cutoff)
    
    kx_filtered = kx_flat[mask]
    ky_filtered = ky_flat[mask]
    kz_filtered = kz_flat[mask]
    magnitude_filtered = magnitude_flat[mask]
    
    # Create FFT plot
    fig_fft = go.Figure(data=go.Scatter3d(
        x=kx_filtered, y=ky_filtered, z=kz_filtered,
        mode='markers',
        marker=dict(size=4, color=magnitude_filtered, colorscale='Viridis', opacity=0.8)
    ))
    
    # Add cutoff sphere visualization
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x_sphere = center_cutoff * np.outer(np.cos(u), np.sin(v))
    y_sphere = center_cutoff * np.outer(np.sin(u), np.sin(v))
    z_sphere = center_cutoff * np.outer(np.ones(np.size(u)), np.cos(v))
    
    #fig_fft.add_trace(go.Surface(x=x_sphere, y=y_sphere, z=z_sphere, opacity=0.2))
    fig_fft.update_layout(scene=dict(aspectmode='cube'), width=800, height=800)
    
    # Load cell info and analyze
    cellinfo_data = load_cellinfo_data(cellinfo_path)
    
    # Calculate unit cells info
    limiting_axes = np.min(tomo_data.shape)
    tomo_nm_size = pixel_size * limiting_axes
    n_unit_cells = tomo_nm_size // (cellinfo_data['Vol'][0][0]**(1/3))
    
    print(f'~n unit cells per tomogram: {n_unit_cells}')
    
    # Perform voxel analysis
    voxel_results = analyze_tomogram_voxels(tomo_data, voxel_size=voxel_size)
    print(f"Number of voxels (z, y, x): {voxel_results['n_voxels']}")
    print(f'~m unit cells per voxel: {n_unit_cells*voxel_size[0]/limiting_axes}')
    
    # Generate reciprocal lattice points
    hkl = np.array(list(product([-2, -1, 0, 1, 2], repeat=3)))
    hs, ks, ls = hkl[:, 0], hkl[:, 1], hkl[:, 2]
    
    print("hs:", hs)
    print("ks:", ks)
    print("ls:", ls)
    
    # Calculate reciprocal lattice vectors
    vs = []
    for i, h in enumerate(hs):
        v = (hs[i]*cellinfo_data['recilatticevectors'][0] + 
             ks[i]*cellinfo_data['recilatticevectors'][1] + 
             ls[i]*cellinfo_data['recilatticevectors'][2])
        vs.append(v)
    vs = np.array(vs)
    
    # Add reciprocal lattice points to FFT plot
    fig_fft.add_trace(go.Scatter3d(
        x=vs.T[0], y=vs.T[1], z=vs.T[2],
        mode='markers',
        marker=dict(size=5, color='red', opacity=0.1),
        name='Cell Info'
    ))

    fig_fft.update_layout(
        title="3D FFT Magnitude with Threshold",
        scene=dict(
            xaxis_title="KX",
            yaxis_title="KY", 
            zaxis_title="KZ",
            aspectmode='cube'
        ),
        width=800, height=800
    )

    # Create magnitude threshold mask
    threshold_factor = 1e-3
    magnitude_test = magnitude > threshold_factor*np.max(magnitude)
    

    center = np.array(magnitude.shape) // 2
    R = 48
    x, y, z = np.ogrid[:magnitude.shape[0], :magnitude.shape[1], :magnitude.shape[2]]
    sphere_mask = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 >= R**2
    magnitude_test = magnitude_test & sphere_mask
    
    # Set projection direction
    h, k, l = hkl_projection[0], hkl_projection[1], hkl_projection[2]
    
    # Transpose data to match MATLAB orientation
    tomogram_test = tomo_data.T
    magnitude_test = magnitude_test.T
    
    project_and_plot_along_hkl_2(
        tomogram_test-np.mean(tomogram_test),
        cellinfo_data,
        h, k, l,                        # (h,k,l) or (h,k,i,l) for hex planes; or (u,v,w) for directions
        title_prefix="Test",
        mode="plane",                    # 'plane' → (hkl)/(hkil) normal; 'direction' → [uvw]
        q_vectors=None,                  # for reciprocal-space labeling
        voxel_size=None,                 # (dx, dy, dz) in nm for real-space labeling; if None → pixels
        show_plot=True,
        return_vector=False
    )
        
    # Generate projections
    projection_test, rotated_tomo_test = project_and_plot_along_hkl(
        tomogram_test-np.mean(tomogram_test), cellinfo_data, h, k, l,
        title_prefix="Test", is_reciprocal=False, q_vectors=None, pixel_size=pixel_size
    )
    
    projection_test_reciprocal, rotated_tomo_test_reciprocal = project_and_plot_along_hkl(
        magnitude_test-np.mean(magnitude_test), cellinfo_data, h, k, l,
        title_prefix="Test", is_reciprocal=True, q_vectors=None, pixel_size=pixel_size
    )
    
    # Plot HKL vector and unit cell
    fig_hkl_vector = plot_hkl_vector_in_tomogram(tomogram_test, cellinfo_data, h, k, l,
        is_reciprocal=False, scale=0.1, plot_tomogram=True)
    
    fig_unit_cell = plot_unit_cell_in_tomogram(tomogram_test, cellinfo_data, plot_tomogram=True,
        intensity_threshold=intensity_threshold, pixel_size=pixel_size)
    
    # Plot real space projection and its FFT
    fig_proj_combined, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(projection_test.T, cmap='gray')
    ax[0].set_title('Real Space Projection')
    ax[0].axis('off')
    
    # Add scale bar (positioned in bottom-left corner with padding)
    scalebar_length_nm = 2000  # Length in nm
    scalebar_length_pixels = scalebar_length_nm / pixel_size
    scalebar_width_pixels = 10  # Fixed width in pixels for better visibility
    scalebar_color = 'white'
    
    # Position scale bar in bottom-left corner with padding
    padding = 30  # Padding from edges in pixels
    scalebar_x_start = padding
    scalebar_y_pos = projection_test.T.shape[0] - padding - scalebar_width_pixels
    
    # Ensure scale bar doesn't go outside image bounds
    if scalebar_x_start + scalebar_length_pixels > projection_test.T.shape[1]:
        scalebar_length_pixels = projection_test.T.shape[1] - scalebar_x_start - 10
        scalebar_length_nm = scalebar_length_pixels * pixel_size
    
    rect = patches.Rectangle(
        (scalebar_x_start, scalebar_y_pos),
        scalebar_length_pixels,
        scalebar_width_pixels,
        facecolor=scalebar_color,
        edgecolor='black',
        linewidth=1
    )
    ax[0].add_patch(rect)
    
    # Add scale bar label (positioned above the scale bar)
    label_y_pos = scalebar_y_pos - 15  # Position label above scale bar
    ax[0].text(
        scalebar_x_start + scalebar_length_pixels/2,
        label_y_pos,
        f'{int(scalebar_length_nm/1000)} μm',
        color=scalebar_color,
        ha='center',
        va='top',
        fontsize=20,
        weight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5, edgecolor='white', linewidth=1)
    )
    ax[1].imshow((np.abs(np.fft.fftshift(np.fft.fft2(projection_test.T)))**2),
        cmap='jet', norm=colors.LogNorm())
    ax[1].set_title('FFT of Real Space Projection')
    ax[1].axis('off')
    fig_proj_combined.suptitle(f'Projection Analysis (h={h}, k={k}, l={l})', fontsize=14)
    plt.show()
    
    # Plot HKL projection grid
    fig_grid = plot_hkl_projection_grid(tomo_data.T, magnitude_test.T,
        cellinfo_data, pixel_size=pixel_size)
    
    # Save all outputs if requested
    output_paths = {}
    if save_outputs:
        print("\nSaving outputs...")
        
        # Save Plotly figures (HTML format)
        print("Saving 3D visualizations...")
        fig_tomo_path = dirs['plots_3d'] / 'tomogram_3d.html'
        fig_tomo.write_html(str(fig_tomo_path))
        output_paths['tomogram_3d_html'] = str(fig_tomo_path)
        
        fig_fft_path = dirs['plots_fft'] / 'fft_3d.html'
        fig_fft.write_html(str(fig_fft_path))
        output_paths['fft_3d_html'] = str(fig_fft_path)
        
        # Save projection images
        print("Saving projections...")
        proj_real_path = dirs['plots_projections'] / 'projection_real_space.png'
        plt.figure(figsize=(10, 10))
        plt.imshow(projection_test, cmap='gray')
        plt.title(f'Real Space Projection (h={h}, k={k}, l={l})')
        plt.colorbar()
        plt.savefig(proj_real_path, dpi=300, bbox_inches='tight')
        plt.close()
        output_paths['projection_real'] = str(proj_real_path)
        
        proj_real_fft_path = dirs['plots_projections'] / 'projection_real_space_fft.png'
        plt.figure(figsize=(10, 10))
        plt.imshow(np.abs(np.fft.fftshift(np.fft.fft2(projection_test)))**2,
                   cmap='jet', norm=colors.LogNorm())
        plt.title(f'FFT of Real Space Projection (h={h}, k={k}, l={l})')
        plt.colorbar()
        plt.savefig(proj_real_fft_path, dpi=300, bbox_inches='tight')
        plt.close()
        output_paths['projection_real_fft'] = str(proj_real_fft_path)
        
        proj_reciprocal_path = dirs['plots_projections'] / 'projection_reciprocal_space.png'
        plt.figure(figsize=(10, 10))
        plt.imshow(projection_test_reciprocal, cmap='gray')
        plt.title(f'Reciprocal Space Projection (h={h}, k={k}, l={l})')
        plt.colorbar()
        plt.savefig(proj_reciprocal_path, dpi=300, bbox_inches='tight')
        plt.close()
        output_paths['projection_reciprocal'] = str(proj_reciprocal_path)
        
        # Save combined projection plot
        proj_combined_path = dirs['plots_projections'] / 'projection_combined_analysis.png'
        fig_proj_combined.savefig(proj_combined_path, dpi=300, bbox_inches='tight')
        output_paths['projection_combined'] = str(proj_combined_path)
        
        # Save HKL projection grid (matplotlib figure)
        grid_path = dirs['plots_projections'] / 'hkl_projection_grid.png'
        fig_grid.savefig(grid_path, dpi=300, bbox_inches='tight')
        output_paths['hkl_projection_grid'] = str(grid_path)
        
        # Save HKL vector plot (Plotly figure - save as HTML)
        hkl_vector_path = dirs['plots_3d'] / 'hkl_vector_in_tomogram.html'
        fig_hkl_vector.write_html(str(hkl_vector_path))
        output_paths['hkl_vector_plot'] = str(hkl_vector_path)
        
        # Save unit cell plot (Plotly figure - save as HTML)
        unit_cell_path = dirs['unit_cell'] / 'unit_cell_in_tomogram.html'
        fig_unit_cell.write_html(str(unit_cell_path))
        output_paths['unit_cell_plot'] = str(unit_cell_path)
        
        print(f"Saved {len([k for k in output_paths.keys() if 'plot' in k or 'grid' in k or 'html' in k])} plot files")
        
        # Save HDF5 file with all data
        print("Saving HDF5 file with all data...")
        h5_path = dirs['data'] / 'analysis_complete.h5'
        with h5py.File(h5_path, 'w') as h5f:
            # Create main groups
            grp_tomo = h5f.create_group('tomogram')
            grp_fft = h5f.create_group('fft')
            grp_projections = h5f.create_group('projections')
            grp_voxels = h5f.create_group('voxel_analysis')
            grp_cell = h5f.create_group('cell_info')
            grp_params = h5f.create_group('parameters')
            
            # Save tomogram data
            grp_tomo.create_dataset('data', data=tomo_data, compression='gzip')
            grp_tomo.create_dataset('rotated_real_space', data=rotated_tomo_test, compression='gzip')
            grp_tomo.create_dataset('rotated_reciprocal_space', data=rotated_tomo_test_reciprocal, compression='gzip')
            grp_tomo.attrs['shape'] = tomo_data.shape
            grp_tomo.attrs['pixel_size_nm'] = pixel_size
            
            # Save FFT data
            grp_fft.create_dataset('magnitude', data=magnitude, compression='gzip')
            grp_fft.create_dataset('KX', data=KX, compression='gzip')
            grp_fft.create_dataset('KY', data=KY, compression='gzip')
            grp_fft.create_dataset('KZ', data=KZ, compression='gzip')
            grp_fft.create_dataset('magnitude_mask', data=magnitude_test, compression='gzip')
            grp_fft.attrs['threshold_factor'] = fft_threshold
            grp_fft.attrs['center_cutoff'] = center_cutoff
            
            # Save projection data
            grp_projections.create_dataset('real_space', data=projection_test, compression='gzip')
            grp_projections.create_dataset('reciprocal_space', data=projection_test_reciprocal, compression='gzip')
            grp_projections.attrs['h'] = h
            grp_projections.attrs['k'] = k
            grp_projections.attrs['l'] = l
            
            # Save voxel analysis results
            for key, value in voxel_results.items():
                if isinstance(value, np.ndarray):
                    grp_voxels.create_dataset(key, data=value, compression='gzip')
                elif isinstance(value, (tuple, list)):
                    grp_voxels.attrs[key] = value
                else:
                    grp_voxels.attrs[key] = value
            
            # Save cell info data
            for key, value in cellinfo_data.items():
                if isinstance(value, np.ndarray):
                    grp_cell.create_dataset(key, data=value, compression='gzip')
                else:
                    try:
                        grp_cell.attrs[key] = value
                    except:
                        # Some MATLAB types might not convert directly
                        grp_cell.attrs[key] = str(value)
            
            # Save parameters
            grp_params.attrs['tomogram_path'] = tomogram_path
            grp_params.attrs['cellinfo_path'] = cellinfo_path
            grp_params.attrs['pixel_size'] = pixel_size
            grp_params.attrs['intensity_threshold'] = intensity_threshold
            grp_params.attrs['rotation_axis'] = rotation_axis
            grp_params.attrs['rotation_angle'] = rotation_angle
            grp_params.attrs['fft_threshold'] = fft_threshold
            grp_params.attrs['center_cutoff'] = center_cutoff
            grp_params.attrs['voxel_size'] = voxel_size
            grp_params.attrs['analysis_timestamp'] = datetime.now().isoformat()
            
        output_paths['h5_file'] = str(h5_path)
        
        # Save analysis summary as text file
        print("Saving analysis summary...")
        summary_path = dirs['base'] / 'analysis_summary.txt'
        limiting_axes = np.min(tomo_data.shape)
        tomo_nm_size = pixel_size * limiting_axes
        n_unit_cells = tomo_nm_size // (cellinfo_data['Vol'][0][0]**(1/3))
        m_unit_cells_per_voxel = n_unit_cells * voxel_size[0] / limiting_axes
        
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TOMOGRAM ANALYSIS SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("INPUT FILES:\n")
            f.write(f"  Tomogram: {tomogram_path}\n")
            f.write(f"  Cell Info: {cellinfo_path}\n\n")
            
            f.write("TOMOGRAM PROPERTIES:\n")
            f.write(f"  Shape: {tomo_data.shape}\n")
            f.write(f"  Pixel Size: {pixel_size} nm\n")
            f.write(f"  Rotation: {rotation_angle}° around {rotation_axis}-axis\n")
            f.write(f"  Intensity Threshold: {intensity_threshold}\n\n")
            
            f.write("FFT ANALYSIS:\n")
            f.write(f"  FFT Threshold Factor: {fft_threshold}\n")
            f.write(f"  Center Cutoff Radius: {center_cutoff}\n")
            f.write(f"  Max Magnitude: {np.max(magnitude):.6e}\n\n")
            
            f.write("VOXEL ANALYSIS:\n")
            f.write(f"  Voxel Size: {voxel_size}\n")
            f.write(f"  Number of Voxels (z, y, x): {voxel_results['n_voxels']}\n\n")
            
            f.write("UNIT CELL INFORMATION:\n")
            f.write(f"  ~n unit cells per tomogram: {n_unit_cells:.2f}\n")
            f.write(f"  ~m unit cells per voxel: {m_unit_cells_per_voxel:.2f}\n\n")
            
            f.write("PROJECTION:\n")
            f.write(f"  HKL indices: ({h}, {k}, {l})\n\n")
            
            f.write("OUTPUT FILES:\n")
            for key, path in output_paths.items():
                f.write(f"  {key}: {path}\n")
            f.write("\n" + "=" * 80 + "\n")
        
        output_paths['summary'] = str(summary_path)
        
        print(f"\nAll outputs saved to: {output_dir}")
        print(f"HDF5 file: {h5_path}")
        print(f"Summary file: {summary_path}")
    
    result = {
        'tomo_data': tomo_data,
        'magnitude': magnitude,
        'voxel_results': voxel_results,
        'cellinfo_data': cellinfo_data,
        'figures': {
            'tomogram': fig_tomo,
            'fft': fig_fft,
            'hkl_vector': fig_hkl_vector,
            'unit_cell': fig_unit_cell,
            'projection_combined': fig_proj_combined,
            'hkl_grid': fig_grid
        },
        'projections': {
            'real_space': projection_test,
            'reciprocal_space': projection_test_reciprocal,
            'rotated_tomo_real': rotated_tomo_test,
            'rotated_tomo_reciprocal': rotated_tomo_test_reciprocal
        }
    }
    
    if save_outputs:
        result['output_paths'] = output_paths
        result['output_dir'] = str(output_dir)
    
    return result


"""
USAGE EXAMPLE:
==============

The load_and_analyze_tomogram function now automatically saves all outputs to a 
structured directory hierarchy and an HDF5 file.

Basic Usage:
-----------
"""

# 2025_Feb: ZC9
#tomogram_path = "/net/micdata/data2/12IDC/2025_Feb/misc/ZCB_9_3D_/tomogram_alignment_recon_56nm.tiff"
tomogram_path = "/net/micdata/data2/12IDC/2025_Feb/misc/ZCB_9_3D_/180_projections_tomogram_alignment_recon_28nm.tiff"
cellinfo_path = "/home/beams/PTYCHOSAXS/cellinfo_ZCB_9.mat"
cellinfo_path = "/home/beams/PTYCHOSAXS/cellinfo_ZCB_9_20251113.mat"
zc9_params={
    'pixel_size': 28,
    'intensity_threshold': 0.75,
    'fft_threshold': 0.00007,
    'center_cutoff': 0.025,
    'hkl_projection': [1,0,1],
    'crop_size': np.array([384, 372, 506]) 
}


# # 2025_Feb: ZC4
# tomogram_path="/net/micdata/data2/12IDC/2025_Feb/misc/ZC4_3D_/tomogram_alignment_recon_28nm.tif"
# #tomogram_path="/net/micdata/data2/12IDC/2025_Feb/misc/ZC4_/new_tomogram_alignment_recon_56nm.tiff"
# cellinfo_path = "/home/beams/PTYCHOSAXS/cellinfo_ZC4.mat"
# zc4_params={
#     'pixel_size': 28,
#     'intensity_threshold': 0.75,
#     'fft_threshold': 0.00007,
#     'center_cutoff': 0.025,
#     'hkl_projection': [1,1,-1], #[0,1,0]
#     'crop_size': np.array([300, 280, 300])
# }

# # 2024_Dec: JM01
# #tomogram_path = "/net/micdata/data2/12IDC/2024_Dec/results/analysis_tomo/JM01/tomogram_alignment_recon_FBP_28nm_2.tiff"
# #tomogram_path = "/net/micdata/data2/12IDC/2024_Dec/results/analysis_tomo/JM01/tomogram_alignment_recon_FBP_14nm.tif"
# tomogram_path = "/net/micdata/data2/12IDC/2024_Dec/misc/JM01_3D/Ndp512_MLc_p10_vp4_gInf_Iter400/recons/tomogram_alignment_recon_FBP_14nm.tif"
# #cellinfo_path = "/net/micdata/data2/12IDC/2024_Dec/results/analysis_tomo/JM01/cellinfo_JM01.mat"
# #cellinfo_path = "/net/micdata/data2/12IDC/2024_Dec/misc/JM01_3D/Ndp512_MLc_p10_vp4_gInf_Iter400/recons/jm01_3d.mat"
# cellinfo_path="/home/beams/PTYCHOSAXS/cellinfo_JM01_hex.mat"
# jm01_params={
#     'pixel_size': 14,
#     'intensity_threshold': 0.75,
#     'fft_threshold': 0.00002,
#     'center_cutoff': 0.025,
#     'hkl_projection': [0,0,1,0],
#     'crop_size': np.array([431, 512, 512]) 
# }

# # 2025_Jul: ZC6
# tomogram_path = "/net/micdata/data2/12IDC/2025_Jul/results/analysis_tomo/ZC6_3D/p20nm/tomogram_alignment_recon.tiff"
# cellinfo_path = "/net/micdata/data2/12IDC/2025_Jul/results/analysis_tomo/ZC6_3D/cellinfo.mat"
# zc6_params={
#     'pixel_size': 20,
#     'intensity_threshold': 0.65,
#     'fft_threshold': 0.000002,
#     'center_cutoff': 0.01,
#     'hkl_projection': [0,1,-1],
#     'crop_size': np.array([512, 550, 650]) 
# }

results = load_and_analyze_tomogram(
    tomogram_path=tomogram_path,
    cellinfo_path=cellinfo_path,
    pixel_size=zc9_params['pixel_size'],#14,#14,#20,#28, #56, #28,
    intensity_threshold=zc9_params['intensity_threshold'], #0.72, #0.8,
    fft_threshold=zc9_params['fft_threshold'], #0.001, #0.0001,
    output_dir=None,  # Auto-creates: analysis_<name>_<timestamp>
    save_outputs=False,  # Set to False to skip saving
    center_cutoff=zc9_params['center_cutoff'], #0.01, #0.025,
    crop_tomogram=True,
    hkl_projection=zc9_params['hkl_projection'],
    crop_size=zc9_params['crop_size']
)

# Access results programmatically:
# - results['tomo_data']: tomogram data array
# - results['magnitude']: FFT magnitude
# - results['voxel_results']: voxel analysis
# - results['cellinfo_data']: cell information
# - results['figures']: dictionary of all figures (tomogram, fft, hkl_vector, unit_cell, etc.)
# - results['projections']: projection data
# - results['output_dir']: path to output directory
# - results['output_paths']: dictionary of all saved file paths

# Display figures if needed
# results['figures']['tomogram'].show()
# results['figures']['fft'].show()

# Saved files include:
# - HTML files for interactive 3D plots (tomogram, FFT, HKL vector, unit cell)
# - PNG files for all 2D plots (projections, grids)
# - HDF5 file with all numerical data
# - Text summary of the analysis

# Note: Plotly figures (tomogram, FFT, HKL vector, unit cell) are saved as interactive HTML
#       Matplotlib figures (projections, grid) are saved as PNG images



############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################

    










############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################










#%%
# Plot line profile for a specific hkl
start_x=124
start_y=160
end_x=150
end_y=180
line = plot_arbitrary_line_profile(projection_test, \
    src=(start_x, start_y), dst=(end_x, end_y))
#%%
# Interactive line selector
selector = LineSelector(projection_test, n_lines=20)
# After selection, your lines are in:
lines_points=selector.lines

#%%
lines=[]
lines_bg_subtracted=[]
num_lines=1
for i in range(0,num_lines):
    start_x=lines_points[i][0][0]
    start_y=lines_points[i][0][1]
    end_x=lines_points[i][1][0]
    end_y=lines_points[i][1][1]
    if i==num_lines-1 or i==0 or i==num_lines//2:
        line3 = plot_arbitrary_line_profile(projection_test, \
            src=(start_x, start_y), dst=(end_x, end_y), show_plot=True)
        # Fit linear background
        x = np.arange(len(line3))
        coeffs = np.polyfit(x, line3, 1)
        background = coeffs[0] * x + coeffs[1]
        # Subtract background
        line3_bg_subtracted = line3 - background
        lines.append(line3)
        lines_bg_subtracted.append(line3_bg_subtracted)
    else:
        line3 = plot_arbitrary_line_profile(projection_test, \
            src=(start_x, start_y), dst=(end_x, end_y), show_plot=False)
        # Fit linear background
        x = np.arange(len(line3))
        coeffs = np.polyfit(x, line3, 1)
        background = coeffs[0] * x + coeffs[1]
        # Subtract background
        line3_bg_subtracted = line3 - background
        lines.append(line3)
        lines_bg_subtracted.append(line3_bg_subtracted)

#%%
switch_backend('inline')
plt.figure()
fft_freqs_list=[]
amplitude_list=[]
phase_list=[]
for i,line in enumerate(lines_bg_subtracted):
    plt.plot(line)
    fft_freqs, amplitude, phase = plot_1d_fft(line-np.mean(line), pixel_size=None, \
        title="FFT of Line Profile", show_plot=False)
    fft_freqs_list.append(fft_freqs)
    amplitude_list.append(amplitude)
    phase_list.append(phase)
    print(amplitude.shape,fft_freqs.shape)
plt.show()


#%%
plt.figure()
for i in range(0,num_lines):
    pos_mask = fft_freqs_list[i] >= 0
    plt.plot(fft_freqs_list[i][pos_mask], amplitude_list[i][pos_mask], label='Amplitude')
plt.legend()
plt.show()
#%%
plt.figure()
for i in range(0,num_lines):
    pos_mask = fft_freqs_list[i] >= 0
    # Convert frequencies to length scales (pixels)
    length_scales = 1/fft_freqs_list[i][pos_mask]
    plt.plot(length_scales, amplitude_list[i][pos_mask], label=f'Line {i+1}')
plt.xlabel('Length Scale (pixels)')
plt.ylabel('Amplitude')
plt.title('FFT Amplitude vs Length Scale')
plt.legend()
# Only show reasonable length scales (exclude very large values near zero frequency)
plt.xlim(0, 100)  # Adjust this limit based on your data
plt.show()

















hkl_list=[(1,0,0),(0,1,0),(0,0,1)]
hkl_list=[(-1,0,-1),(1,0,1)]
cluster_k=4
upsample_factor=4
patch_size=11*upsample_factor+1

#%%
#need to select the patch manually, requires upsampled projection
#(100) patch
h,k,l=hkl_list[0]
projection_test, rotated_tomo_test = project_and_plot_along_hkl(tomogram_test, cellinfo_data, h, k, l, \
    title_prefix="test", is_reciprocal=False, q_vectors=None, pixel_size=None)
projection_test_reciprocal, rotated_tomo_test_reciprocal = project_and_plot_along_hkl(magnitude_test, cellinfo_data, h, k, l, \
    title_prefix="test", is_reciprocal=True, q_vectors=None, pixel_size=None)

# # --- UPSAMPLING STEP ---
# #ZC4
# projection_test_upsampled = zoom(projection_test[100:375, 100:350], upsample_factor, order=3)
# reference_patch_ZC4_100 = crop_patch(projection_test_upsampled, (607,551), patch_size)

#ZCB9
projection_test_upsampled = zoom(projection_test, upsample_factor, order=3)
reference_patch_ZCB9_1bar01bar = crop_patch(projection_test_upsampled, (1106,1126), patch_size)

#%%
#(010) patch
h,k,l=hkl_list[1]
projection_test, rotated_tomo_test = project_and_plot_along_hkl(tomogram_test, cellinfo_data, h, k, l, \
    title_prefix="test", is_reciprocal=False, q_vectors=None, pixel_size=None)
projection_test_reciprocal, rotated_tomo_test_reciprocal = project_and_plot_along_hkl(magnitude_test, cellinfo_data, h, k, l, \
    title_prefix="test", is_reciprocal=True, q_vectors=None, pixel_size=None)

# # --- UPSAMPLING STEP ---
# #ZC4
# projection_test_upsampled = zoom(projection_test[100:375, 100:350], upsample_factor, order=3)
# reference_patch_ZC4_010 = crop_patch(projection_test_upsampled, (531,611), patch_size)

#ZCB9
projection_test_upsampled = zoom(projection_test, upsample_factor, order=3)
reference_patch_ZCB9_101 = crop_patch(projection_test_upsampled, (917,1127), patch_size)

#%%
#(001) patch
h,k,l=hkl_list[2]
projection_test, rotated_tomo_test = project_and_plot_along_hkl(tomogram_test, cellinfo_data, h, k, l, \
    title_prefix="test", is_reciprocal=False, q_vectors=None, pixel_size=None)
projection_test_reciprocal, rotated_tomo_test_reciprocal = project_and_plot_along_hkl(magnitude_test, cellinfo_data, h, k, l, \
    title_prefix="test", is_reciprocal=True, q_vectors=None, pixel_size=None)

# --- UPSAMPLING STEP ---
projection_test_upsampled = zoom(projection_test[100:375, 100:350], upsample_factor, order=3)
reference_patch_ZC4_001 = crop_patch(projection_test_upsampled, (537,542), patch_size)

#%%

clusters_hkl=[]

for h,k,l in hkl_list:
    pixel_size=pixel_size

    # Have to do this because the tomogram is transposed with 
    # respect to the cellinfo_data defined in MATLAB
    tomogram_test=tomo_data.T
    magnitude_test=magnitude_test.T
    projection_test, rotated_tomo_test = project_and_plot_along_hkl(tomogram_test, cellinfo_data, h, k, l, \
        title_prefix="test", is_reciprocal=False, q_vectors=None, pixel_size=None)
    projection_test_reciprocal, rotated_tomo_test_reciprocal = project_and_plot_along_hkl(magnitude_test, cellinfo_data, h, k, l, \
        title_prefix="test", is_reciprocal=True, q_vectors=None, pixel_size=None)

    # --- UPSAMPLING STEP ---
    # projection_test_upsampled = zoom(projection_test[100:375, 100:350], upsample_factor, order=3)
    projection_test_upsampled = zoom(projection_test, upsample_factor, order=3)
    print("projection_test_upsampled.shape: ", projection_test_upsampled.shape)
    fig,ax=plt.subplots(1,2,figsize=(10,5))
    #ax[0].imshow(projection_test[100:375, 100:350], cmap='jet')
    ax[0].imshow(projection_test, cmap='jet')
    ax[0].set_title("Original")
    ax[1].imshow(projection_test_upsampled, cmap='jet')
    ax[1].set_title("Upsampled")
    plt.tight_layout()
    plt.show()

    # if h==1:
    #     reference_patch = reference_patch_ZC4_100
    # elif k==1:
    #     reference_patch = reference_patch_ZC4_010
    # elif l==1:
    #     reference_patch = reference_patch_ZC4_001

    if h==1 and k==0 and l==1:
        reference_patch = reference_patch_ZCB9_101
    else:
        reference_patch = reference_patch_ZCB9_1bar01bar

    avg, coords, patches, mean_cluster_list = single_particle_analysis_rotation_scan(
        image=projection_test_upsampled,
        patch_size=patch_size,
        center_dip=True,
        angle_range=np.arange(0, 60, 1),
        threshold_abs=0.4,
        score_thresh=0.5,
        cluster_k=cluster_k,
        show_gallery=True,
        shape=6,  # Pentagon, 5; Hexagon, 6
        cluster_by_angle_only=False,  # <--- Only use angle for clustering
        random_seed=None,
        reference_template=reference_patch  # <-- new argument
    )


    # fig,ax=plt.subplots(1,cluster_k,figsize=(10,10))
    # for i,mean_cluster in enumerate(mean_cluster_list):
    #     ax[i].imshow(mean_cluster-np.mean(mean_cluster), cmap='jet')
    #     ax[i].axis('off')
    # plt.tight_layout()
    # plt.show()
    
    clusters_hkl.append(mean_cluster_list)

clusters_hkl=np.array(clusters_hkl)


#%%


if cluster_k>1:
    fig,ax=plt.subplots(len(hkl_list),cluster_k,figsize=(10,10))
    for i in range(len(hkl_list)):
        for j in range(cluster_k):
            im1=ax[i,j].imshow(clusters_hkl[i][j], cmap='jet')
            ax[i,j].axis('off')
            ax[i,0].set_title(f"hkl = {hkl_list[i]}")
            plt.colorbar(im1)
    plt.tight_layout()
    plt.show()
else:
    fig,ax=plt.subplots(len(hkl_list),1,figsize=(10,10))
    for i in range(len(hkl_list)):
        im1=ax[i].imshow(clusters_hkl[i][0], cmap='jet')#,norm=colors.PowerNorm(gamma=10.0))
        ax[i].axis('off')
        ax[i].set_title(f"hkl = {hkl_list[i]}")
        plt.colorbar(im1)
    plt.tight_layout()

    plt.show()











#%%


def max_rotational_correlation(img1, img2, angles=np.arange(0, 360, 1)):
    best_corr = -1
    best_angle = 0
    for angle in angles:
        rotated = rotate(img2, angle, reshape=False)
        corr = np.corrcoef(img1.flatten(), rotated.flatten())[0, 1]
        if corr > best_corr:
            best_corr = corr
            best_angle = angle
    return best_corr, best_angle

def max_positional_correlation(img1, img2, max_shift=10):
    """Find best x,y shift to align img2 with img1"""
    best_corr = -1
    best_shift = (0,0)
    
    for dy in range(-max_shift, max_shift+1):
        for dx in range(-max_shift, max_shift+1):
            shifted = shift(img2, (dy,dx))
            corr = np.corrcoef(img1.flatten(), shifted.flatten())[0,1]
            if corr > best_corr:
                best_corr = corr
                best_shift = (dy,dx)
                
    return best_corr, best_shift

# Example usage:
img1 = clusters_hkl[0][0]
img2 = clusters_hkl[1][0]
img3 = clusters_hkl[2][0]

# First align positions
corr12_pos, shift12 = max_positional_correlation(img1, img2)
corr13_pos, shift13 = max_positional_correlation(img1, img3)
corr23_pos, shift23 = max_positional_correlation(img2, img3)

# Shift images to align centers
img2_shifted = shift(img2, shift12)
img3_shifted = shift(img3, shift13)

# Then find rotational alignment
corr12, angle12 = max_rotational_correlation(img1, img2_shifted)
corr13, angle13 = max_rotational_correlation(img1, img3_shifted)
corr23, angle23 = max_rotational_correlation(img2_shifted, img3_shifted)

fig,ax=plt.subplots(1,3,figsize=(10,10))
ax[0].imshow(img1, cmap='jet')
ax[0].axis('off')
ax[1].imshow(img2, cmap='jet')
ax[1].axis('off')
ax[2].imshow(img3, cmap='jet')
ax[2].axis('off')

print(f"Image 1-2: Max correlation: {corr12:.3f} at angle {angle12}° and shift {shift12}")
print(f"Image 1-3: Max correlation: {corr13:.3f} at angle {angle13}° and shift {shift13}")
print(f"Image 2-3: Max correlation: {corr23:.3f} at angle {angle23}° and shift {shift23}")

# Apply both position and rotation corrections
img2_aligned = rotate(img2_shifted, angle12, reshape=False)
img3_aligned = rotate(img3_shifted, angle13, reshape=False)

# Create a new figure showing original and aligned images
fig2, ax2 = plt.subplots(2, 3, figsize=(12, 8))
plt.suptitle('Original vs Aligned Images')

# First row - original images
ax2[0,0].imshow(img1, cmap='jet')
ax2[0,0].set_title('Image 1 (Reference)')
ax2[0,0].axis('off')

ax2[0,1].imshow(img2, cmap='jet')
ax2[0,1].set_title('Image 2 (Original)')
ax2[0,1].axis('off')

ax2[0,2].imshow(img3, cmap='jet')
ax2[0,2].set_title('Image 3 (Original)')
ax2[0,2].axis('off')

# Second row - reference and aligned images
ax2[1,0].imshow(img1, cmap='jet')
ax2[1,0].set_title('Image 1 (Reference)')
ax2[1,0].axis('off')

ax2[1,1].imshow(img2_aligned, cmap='jet')
ax2[1,1].set_title(f'Image 2 (Shifted {shift12}, Rotated {angle12:.1f}°)')
ax2[1,1].axis('off')

ax2[1,2].imshow(img3_aligned, cmap='jet')
ax2[1,2].set_title(f'Image 3 (Shifted {shift13}, Rotated {angle13:.1f}°)')
ax2[1,2].axis('off')

plt.tight_layout()
plt.show()

# Show sum of aligned images with background subtraction
sum_aligned = img1 + img2_aligned + img3_aligned
background = np.mean(sum_aligned)  # Calculate mean background
sum_aligned_bg = sum_aligned - background  # Subtract background
sum_aligned_bg[sum_aligned_bg<0]=0
plt.figure(figsize=(6,6))
plt.imshow(sum_aligned_bg, cmap='jet')
plt.title('Sum of Aligned Images (Background Subtracted)')
plt.colorbar()
plt.axis('off')
plt.show()

#%%

import numpy as np

def pentagon_template_fit(shape, center, scale, angle_deg, amplitude=1.0, background=0.0):
    # shape: (height, width) of output image
    # center: (y, x) center of pentagon
    # scale: size of pentagon (relative to image) 
    # angle_deg: rotation angle in degrees
    # amplitude: scaling factor
    # background: offset
    size = int(scale)
    template = polygon_template(size=size, center_dip=True, angle_deg=angle_deg, shape=5)
    # Place the template in a blank image at the specified center
    img = np.ones(shape) * background
    y0, x0 = int(center[0]), int(center[1])
    half = size // 2
    y1, y2 = max(0, y0-half), min(shape[0], y0+half)
    x1, x2 = max(0, x0-half), min(shape[1], x0+half)
    t_y1, t_y2 = half-(y0-y1), half+(y2-y0)
    t_x1, t_x2 = half-(x0-x1), half+(x2-x0)
    # Add the scaled template
    img[y1:y2, x1:x2] += amplitude * template[t_y1:t_y2, t_x1:t_x2]
    return img



def hexagon_template_fit(shape, center, scale, angle_deg, amplitude=1.0, background=0.0):
    # shape: (height, width) of output image
    # center: (y, x) center of pentagon
    # scale: size of pentagon (relative to image)
    # angle_deg: rotation angle in degrees
    # amplitude: scaling factor
    # background: offset
    size = int(scale)
    template = polygon_template(size=size, center_dip=True, angle_deg=angle_deg, shape=6)
    # Place the template in a blank image at the specified center
    img = np.ones(shape) * background
    y0, x0 = int(center[0]), int(center[1])
    half = size // 2
    y1, y2 = max(0, y0-half), min(shape[0], y0+half)
    x1, x2 = max(0, x0-half), min(shape[1], x0+half)
    t_y1, t_y2 = half-(y0-y1), half+(y2-y0)
    t_x1, t_x2 = half-(x0-x1), half+(x2-x0)
    # Add the scaled template
    img[y1:y2, x1:x2] += amplitude * template[t_y1:t_y2, t_x1:t_x2]
    return img

def fit_objective(params, data):
    center_y, center_x, scale, angle, amplitude, background = params
    template_img = pentagon_template_fit(
        data.shape, (center_y, center_x), scale, angle, amplitude, background
    )
    # Use sum of squared differences
    return np.sum((data - template_img)**2)
from scipy.optimize import minimize

# Initial guess: center in the middle, scale ~image size/2, angle=0, amplitude=1, background=0
init = [
    sum_aligned_bg.shape[0] // 2 ,  # center_y
    sum_aligned_bg.shape[1] // 2 ,  # center_x
    45,  # scale
    10,  # angle
    np.max(sum_aligned_bg),  # amplitude
    10   # background
]

bounds = [
    (0, sum_aligned_bg.shape[0]),  # center_y
    (0, sum_aligned_bg.shape[1]),  # center_x
    (5, min(sum_aligned_bg.shape)),  # scale
    (0, 360),  # angle
    (0, None),  # amplitude
    (0, None)  # background
]

result = minimize(fit_objective, init, args=(sum_aligned_bg,), bounds=bounds, method='L-BFGS-B')
print("Fit result:", result)

best_params = result.x
fitted_template = hexagon_template_fit(
    sum_aligned_bg.shape,
    (best_params[0], best_params[1]),
    best_params[2],
    best_params[3],
    best_params[4],
    best_params[5]
)

import matplotlib.pyplot as plt
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(sum_aligned_bg, cmap='jet')
plt.title('Data')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(fitted_template, cmap='jet')
plt.title('Fitted Hexagon Template')
plt.axis('off')
plt.show()

# Overlay
plt.figure(figsize=(6,6))
plt.imshow(sum_aligned_bg, cmap='jet', alpha=0.7)
plt.imshow(fitted_template, cmap='gray', alpha=0.3)
plt.title('Overlay: Data + Fitted Hexagon')
plt.axis('off')
plt.show()












#%%
# Get selected cluster images and sum them
test = sum_aligned_bg

# Create a 3x3 grid of particles
grid_size = (3, 3)
grid_image = np.zeros((test.shape[0]*grid_size[0], test.shape[1]*grid_size[1]))

# Fill the grid with the particle
for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        y_start = i * test.shape[0]
        y_end = (i + 1) * test.shape[0]
        x_start = j * test.shape[1]
        x_end = (j + 1) * test.shape[1]
        grid_image[y_start:y_end, x_start:x_end] = test

plt.figure(figsize=(10,10))
plt.imshow(grid_image, cmap='jet')
plt.colorbar()  
plt.title(f'{grid_size[0]}x{grid_size[1]} Grid of Summed Cluster Particles')
plt.axis('off')
plt.show()

plt.figure(figsize=(10,10))
plt.imshow(np.abs(np.fft.fftshift(np.fft.fft2(grid_image)))**2, cmap='jet', norm=colors.PowerNorm(gamma=0.25))
plt.colorbar()  
plt.title(f'{grid_size[0]}x{grid_size[1]} Grid of Summed Cluster Particles')
plt.axis('off')
plt.show()












#%%
import scipy.io as sio
#tomo_data_RS_file=sio.loadmat('ZCB_9_3D_deconvolved_RS.mat')
#tomo_data_RS_file=sio.loadmat('reciprocal_space_binned.mat')
tomo_data_RS_file=sio.loadmat('/scratch/2025_Feb/temp/FFT_RS_128.mat')
tomo_data_RS=tomo_data_RS_file['DATA_m']
tomo_data_RS_qs=tomo_data_RS_file['Qv_m']
    
# nx, ny, nz = tomo_data_RS.shape
# QX = tomo_data_RS_qs[:, 0].reshape(nx, ny, nz)
# QY = tomo_data_RS_qs[:, 1].reshape(nx, ny, nz)
# QZ = tomo_data_RS_qs[:, 2].reshape(nx, ny, nz)

tomo_data_RS[np.isnan(tomo_data_RS)]=0
tomo_data_RS_qs[np.isnan(tomo_data_RS_qs)]=0
center = np.array(tomo_data_RS.shape) // 2
radius = 70  # Radius of sphere to mask
x, y, z = np.ogrid[:tomo_data_RS.shape[0], :tomo_data_RS.shape[1], :tomo_data_RS.shape[2]]

mask = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 >= radius**2
#mask = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 <= radius**2

# Apply mask to tomogram
tomo_data_RS_masked = tomo_data_RS.copy()
tomo_data_RS_masked[mask] = 0

# # Plot the masked tomogram
# fig = plot_3D_tomogram(tomo_data_RS_masked, intensity_threshold=0.5)
# fig.show()


magnitude = tomo_data_RS_masked
KX,KY,KZ=-tomo_data_RS_qs[:,1],tomo_data_RS_qs[:,2],tomo_data_RS_qs[:,0]
# Define a threshold for the magnitude
threshold = 0.0005* np.max(magnitude)  # Example: 10% of the max magnitude

# Flatten the arrays
kx_flat = KX.flatten()
ky_flat = KY.flatten()
kz_flat = KZ.flatten()
magnitude_flat = magnitude.flatten()

mask = (magnitude_flat > threshold)

kx_filtered = kx_flat[mask]
ky_filtered = ky_flat[mask]
kz_filtered = kz_flat[mask]
magnitude_filtered = magnitude_flat[mask]

# Create a 3D scatter plot of the FFT magnitude
fig_fft = go.Figure(data=go.Scatter3d(
    x=kx_filtered,
    y=ky_filtered,
    z=kz_filtered,
    mode='markers',
    marker=dict(
        size=10,
        color=magnitude_filtered,
        colorscale='Viridis',
        opacity=0.4,
        colorbar=dict(title='Magnitude')
    )
))
# Run test
# Load and plot cell info
cellinfo_data = load_cellinfo_data("/scratch/2025_Feb/temp/cellinfo_256_BL_2.mat")
hs=np.array([1,1,0,0,-1,-1,0,0,1,-1,1,-1])
ks=np.array([1,-1,1,1,-1,1,-1,-1,0,0,0,0])
ls=np.array([0,0,1,-1,0,0,-1,1,0,1,-1,-1])



# Generate Miller indices up to 3rd order
def generate_miller_indices(max_order=3):
    """
    Generate Miller indices (h,k,l) up to specified order.
    Excludes (0,0,0) and includes all combinations where |h|,|k|,|l| ≤ max_order.
    """
    indices = []
    for h in range(-max_order, max_order + 1):
        for k in range(-max_order, max_order + 1):
            for l in range(-max_order, max_order + 1):
                # Skip the origin
                if h == 0 and k == 0 and l == 0:
                    continue
                indices.append([h, k, l])
    return np.array(indices)

# Generate indices up to 3rd order
miller_indices = generate_miller_indices(1)
hs = miller_indices[:, 0]
ks = miller_indices[:, 1]
ls = miller_indices[:, 2]
# hs=np.array([4,4,0,0,-4,-4,0,0,4,-4,4,-4])
# ks=np.array([4,-4,4,4,-4,4,-4,-4,0,0,0,0])
# ls=np.array([0,0,4,-4,0,0,-4,4,0,4,-4,-4])
vs=[]

for i,h in enumerate(hs):
    v=hs[i]*cellinfo_data['recilatticevectors'][0]+ks[i]*cellinfo_data['recilatticevectors'][1]+ls[i]*cellinfo_data['recilatticevectors'][2]
    vs.append(v)

vs=np.array(vs)


fig_fft.add_trace(go.Scatter3d(
    x=vs.T[0],
    y=vs.T[1],
    z=vs.T[2],
    mode='markers',
    marker=dict(size=10, color='red', opacity=0.8),
    name='Cell Info'
))


fig_fft.update_layout(
    title="3D FFT Magnitude with Threshold",
    scene=dict(
        xaxis_title="KX",
        yaxis_title="KY",
        zaxis_title="KZ",
        aspectmode='cube'
    ),
    width=800, height=800
)

# # Example: set the same range for all axes
# qmin = -0.05  # set to your desired min
# qmax = 0.05   # set to your desired max

# fig_fft.update_layout(
#     scene=dict(
#         xaxis=dict(title='Qx', range=[qmin, qmax]),
#         yaxis=dict(title='Qy', range=[qmin, qmax]),
#         zaxis=dict(title='Qz', range=[qmin, qmax]),
#         aspectmode='cube'
#     )
# )
fig_fft.show()















#%%
#tomo_data_RS_file=sio.loadmat('ZCB_9_3D_deconvolved_RS.mat')
tomo_data_RS_file=sio.loadmat('reciprocal_space_binned.mat')
tomo_data_RS_file=sio.loadmat('/net/micdata/data2/12IDC/2025_Feb/misc/ZCB_9_3D_/reciprocal_space_binned_fft_256_NEW_complex_object_roi_TEST.mat')

tomo_data_RS=tomo_data_RS_file['DATA']#.swapaxes(1,2)
tomo_data_RS_qs=tomo_data_RS_file['Qv']
tomo_data_RS[np.isnan(tomo_data_RS)]=0
tomo_data_RS_qs[np.isnan(tomo_data_RS_qs)]=0


# Generate indices up to 3rd order
miller_indices = generate_miller_indices(8)
hs = miller_indices[:, 0]
ks = miller_indices[:, 1]
ls = miller_indices[:, 2]

# Create the combined plot
fig = plot_combined_reciprocal_space(tomo_data, tomo_data_RS, tomo_data_RS_qs, 
                                     cellinfo_data, hs, ks, ls, threshold_D=0.05,
                                     threshold_tomo_FFT=0.005, q_cutoff=0.065, 
                                     peak_distance_threshold=0.008)
fig.show()










#%%


basedir='/scratch/2025_Feb/temp/'

tomo_data_RS_file=sio.loadmat(f'{basedir}/FFT_RS_128.mat')
tomo_data_RS=tomo_data_RS_file['DATA_m']
tomo_data_RS_qs=tomo_data_RS_file['Qv_m']
tomo_data_RS[np.isnan(tomo_data_RS)]=0
tomo_data_RS_qs[np.isnan(tomo_data_RS_qs)]=0
tomo_data_RS = gaussian_filter(tomo_data_RS, sigma=0) 

tomo_data_RS_file_256=sio.loadmat(f'{basedir}/FFT_RS_256.mat')
tomo_data_RS_256=tomo_data_RS_file_256['DATA']
tomo_data_RS_qs_256=tomo_data_RS_file_256['Qv']
tomo_data_RS_256[np.isnan(tomo_data_RS_256)]=0
tomo_data_RS_qs_256[np.isnan(tomo_data_RS_qs_256)]=0
tomo_data_RS_256 = gaussian_filter(tomo_data_RS_256, sigma=1) 

tomo_data_RS_file_DECONV=sio.loadmat(f'{basedir}/DECONV_RS_256.mat')
tomo_data_RS_DECONV=tomo_data_RS_file_DECONV['DATA_m']
tomo_data_RS_qs_DECONV=tomo_data_RS_file_DECONV['Qv_m']
tomo_data_RS_DECONV[np.isnan(tomo_data_RS_DECONV)]=0
tomo_data_RS_qs_DECONV[np.isnan(tomo_data_RS_qs_DECONV)]=0
tomo_data_RS_DECONV = gaussian_filter(tomo_data_RS_DECONV, sigma=1) 


cellinfo_data = load_cellinfo_data(f'{basedir}/cellinfo_256_BL.mat')


rs_datasets=[
    {'DATA':tomo_data_RS, 'Qv':tomo_data_RS_qs, 'label':'RS 128'},
    {'DATA':tomo_data_RS_256, 'Qv':tomo_data_RS_qs_256, 'label':'RS 256'},
    {'DATA':tomo_data_RS_DECONV, 'Qv':tomo_data_RS_qs_DECONV, 'label':'RS 256 DECONV'}
]


rs_datasets=[
    {'DATA':tomo_data_RS, 'Qv':tomo_data_RS_qs, 'label':'RS 128'}
]

# Generate indices up to 3rd order
miller_indices = generate_miller_indices(8)
hs = miller_indices[:, 0]
ks = miller_indices[:, 1]
ls = miller_indices[:, 2]
plot_multi_reciprocal_space(
    rs_datasets,  # List of dicts: [{'magnitude': 3D array, 'Q': 4D array (shape: (nx,ny,nz,3)), 'label': str}, ...]
    cellinfo_data,
    hs, ks, ls,
    #thresholds=[0.00016, 0.0001, 0.0115],  # List of thresholds for each dataset
    thresholds=[0.0002],#, 0.00012, 0.0135],  # List of thresholds for each dataset
    q_cutoffs=[0.02],#, 0.07, 0.07],
    peak_distance_threshold=0.01,
    colormaps=['inferno'],#, 'viridis', 'jet'],
    alphas=[0.1]#, 0.3, 1.0]
)








































#%%

crystal_peaks = np.array(vs)  # Your existing vs array from cellinfo
z_idx, y_idx, x_idx=voxel_results['n_voxels'][0]//2, voxel_results['n_voxels'][1]//2, voxel_results['n_voxels'][2]//2
voxel_data = extract_voxel_region(tomo_data, voxel_results, z_idx, y_idx, x_idx)
fig, fig_local = plot_voxel_region(tomo_data, voxel_results, z_idx, y_idx, x_idx, 0.8)
fig.show()



#%%
fig, angle, rmsd = test_orientation_analysis(rotate(voxel_data, 0, axes=(1,2), reshape=False), 
                                          crystal_peaks, z_idx, y_idx, x_idx, hs, ks, ls,threshold=0.005, sigma=0.5, cutoff=3, pixel_size=56, visualize=True)
print(f"\nResults:")
print(f"Rotation angle around (110): {angle:.1f}°")
print(f"Final RMSD: {rmsd:.3e} nm⁻¹")

# Calculate angles between reciprocal lattice vectors
a_star = cellinfo_data['recilatticevectors'][0]
b_star = cellinfo_data['recilatticevectors'][1] 
c_star = cellinfo_data['recilatticevectors'][2]

# Calculate magnitudes
a_mag = np.linalg.norm(a_star)
b_mag = np.linalg.norm(b_star)
c_mag = np.linalg.norm(c_star)

# Calculate angles (in degrees)
ab_angle = np.arccos(np.dot(a_star, b_star)/(a_mag * b_mag)) * 180/np.pi
bc_angle = np.arccos(np.dot(b_star, c_star)/(b_mag * c_mag)) * 180/np.pi
ac_angle = np.arccos(np.dot(a_star, c_star)/(a_mag * c_mag)) * 180/np.pi

print("\nReciprocal Lattice Vector Magnitudes:")
print(f"||a*|| = {a_mag:.3f} nm⁻¹")
print(f"||b*|| = {b_mag:.3f} nm⁻¹")
print(f"||c*|| = {c_mag:.3f} nm⁻¹")

print("\nAngles between Reciprocal Lattice Vectors:")
print(f"a*^b* = {ab_angle:.1f}°")
print(f"b*^c* = {bc_angle:.1f}°")
print(f"a*^c* = {ac_angle:.1f}°")

# Plot reciprocal lattice vectors
scale = 1  # Scale factor for better visualization
fig.add_trace(go.Scatter3d(
    x=[0, a_star[0] * scale],
    y=[0, a_star[1] * scale],
    z=[0, a_star[2] * scale],
    mode='lines+text',
    line=dict(color='red', width=5),
    text=['', 'a*'],
    name='a* vector'
))

fig.add_trace(go.Scatter3d(
    x=[0, b_star[0] * scale],
    y=[0, b_star[1] * scale],
    z=[0, b_star[2] * scale],
    mode='lines+text',
    line=dict(color='green', width=5),
    text=['', 'b*'],
    name='b* vector'
))

fig.add_trace(go.Scatter3d(
    x=[0, c_star[0] * scale],
    y=[0, c_star[1] * scale],
    z=[0, c_star[2] * scale],
    mode='lines+text',
    line=dict(color='blue', width=5),
    text=['', 'c*'],
    name='c* vector'
))

fig.show()




#%%


# # Example usage:
# z_idx, y_idx, x_idx = 3, 3, 4  # Example voxel coordinates
# fig = visualize_single_voxel_orientation(tomo_data, voxel_results, crystal_peaks, 
#                                        hs, ks, ls, z_idx, y_idx, x_idx)
# fig.show()

# # Example usage:
# z_idx = 2
# x_idx = 4
# y_range = np.arange(0, 9)  # Analyze voxels y=1 through y=4
# fig = visualize_line_orientation(tomo_data, voxel_results, crystal_peaks, 
#                                hs, ks, ls, z_idx, y_range, x_idx)
# fig.show()



# Example usage:

# #12x12x12 pixel voxels
# z_range = np.arange(2, 10)
# y_range = np.arange(2, 10)
# x_range = np.arange(5, 12)

#15x15x15 pixel voxels
z_range = np.arange(0, voxel_results['n_voxels'][0]-1)
y_range = np.arange(0, voxel_results['n_voxels'][1]-1)
x_range = np.arange(0, voxel_results['n_voxels'][2]-1)


# With cyclic period
cyclic_period=120
fig = visualize_section_orientation(tomo_data, voxel_results, crystal_peaks, 
                                  hs, ks, ls, z_range, y_range, x_range,threshold=0.1, sigma=0.5, cutoff=3, pixel_size=18, 
                                  cyclic_period=cyclic_period)
# Save as HTML file
fig.write_html(f"/home/beams/PTYCHOSAXS/NN/ptychosaxsNN/data/orientation_analysis_{voxel_size[0]}x{voxel_size[1]}x{voxel_size[2]}_cyclic_period_{cyclic_period}.html")
#fig.show()

# # Without cyclic period (raw angles)
# fig = visualize_section_orientation(tomo_data, voxel_results, crystal_peaks, 
#                                   hs, ks, ls, z_range, y_range, x_range,
#                                   cyclic_period=None)
# fig.show()
























#%%
'''
TEST PEAK ANALYSISFOR SINGLE VOXEL
'''
vz, vy, vx = voxel_results['voxel_size']
figE = go.Figure()

# Process only the first voxel
z_idx, y_idx, x_idx = voxel_results['n_voxels'][0]//2, voxel_results['n_voxels'][1]//2, voxel_results['n_voxels'][2]//2
intensity_threshold_tomo = 0.5

# Compute orientation tensor and eigenvalues/eigenvectors
region = extract_voxel_region(tomo_data, voxel_results, z_idx, y_idx, x_idx)
magnitude, KX, KY, KZ = compute_fft_q(region, use_vignette=True, pixel_size=18)
fig, fig_local = plot_voxel_region(tomo_data, voxel_results, z_idx, y_idx, x_idx, intensity_threshold_tomo)
fig.show()

# Flatten the arrays
kx_flat = KX.flatten()
ky_flat = KY.flatten()
kz_flat = KZ.flatten()
magnitude_flat = magnitude.flatten()

# Apply the threshold
mask = magnitude_flat > threshold*np.max(magnitude)
kx_filtered = kx_flat[mask]
ky_filtered = ky_flat[mask]
kz_filtered = kz_flat[mask]
magnitude_filtered = magnitude_flat[mask]

#Find peaks in 3D
peak_positions, peak_values = find_peaks_3d_cutoff(magnitude,threshold = peak_threshold ,sigma=sigma, center_cutoff_radius=3)
voxel_peaks,voxel_values=peak_positions,peak_values

for pos, val in zip(peak_positions, peak_values):
    print(f"Peak at position {pos} with value {val}")

# Extract peak coordinates
voxel_peak_kx = KX[voxel_peaks[:, 0], voxel_peaks[:, 1], voxel_peaks[:, 2]]
voxel_peak_ky = KY[voxel_peaks[:, 0], voxel_peaks[:, 1], voxel_peaks[:, 2]]
voxel_peak_kz = KZ[voxel_peaks[:, 0], voxel_peaks[:, 1], voxel_peaks[:, 2]]

# Create a 3D scatter plot of the FFT magnitude
fig_fft_ex = go.Figure(data=go.Scatter3d(
    x=kx_filtered,
    y=ky_filtered,
    z=kz_filtered,
    mode='markers',
    marker=dict(
        size=10,
        color=magnitude_filtered,
        colorscale='Viridis',
        opacity=0.8,
        colorbar=dict(title='Magnitude')
    )
))

fig_fft_ex.update_layout(
    title="3D FFT Magnitude with Threshold",
    scene=dict(
        xaxis_title="KX",
        yaxis_title="KY",
        zaxis_title="KZ",
        aspectmode='cube'
    ),
    width=800, height=800
)

# Assuming peak_positions and peak_values are already obtained
peak_kx = KX[peak_positions[:, 0], peak_positions[:, 1], peak_positions[:, 2]]
peak_ky = KY[peak_positions[:, 0], peak_positions[:, 1], peak_positions[:, 2]]
peak_kz = KZ[peak_positions[:, 0], peak_positions[:, 1], peak_positions[:, 2]]

# Create a 3D scatter plot of the peaks
fig_peaks_ex = go.Figure(data=go.Scatter3d(
    x=peak_kx,
    y=peak_ky,
    z=peak_kz,
    mode='markers',
    marker=dict(
        size=10,
        color=peak_values,
        colorscale='Plasma',
        opacity=0.8,
        #colorbar=dict(title='Peak Magnitude')
        showscale=False
    )
))

fig_peaks_ex.add_trace(go.Scatter3d(
    x=crystal_peaks.T[0],
    y=crystal_peaks.T[1],
    z=crystal_peaks.T[2],
    mode='markers',
    marker=dict(size=5, color='red', opacity=0.5),
    name='Cell Info'
))


fig_peaks_ex.update_layout(
    title="3D FFT Peaks",
    scene=dict(
        xaxis_title="KX",
        yaxis_title="KY",
        zaxis_title="KZ",
        aspectmode='cube'
    ),
    width=800, height=800
)

fig_peaks_ex.show()






















#%%
'''
VISUALIZING TOMOGRAM WITH RECIPROCAL SPACE PEAKS FOR MULTIPLE VOXELS
'''
# Calculate maximum magnitudes for each voxel for normalization
z_indices_all = range(0,10)
all_magnitudes = []
for plot_idx, z_idx in enumerate(z_indices_all):
    region = extract_voxel_region(tomo_data, voxel_results, z_idx, y_idx, x_idx)
    magnitude, KX, KY, KZ = compute_fft_q(region, use_vignette=True, pixel_size=18)
    all_magnitudes.append(np.max(magnitude))

# Define the range of z indices to analyze
# Plots can only handle so many subplots, have to break up for memory sake

#z_indices = z_indices_all[:len(z_indices_all)//2]# Example range along the z-axis
z_indices = z_indices_all[len(z_indices_all)//2:]# Example range along the z-axis

# Store peak data for each voxel
voxel_peaks = {}

# Intialize combined figure
fig_combined=initialize_combined_figure(len(z_indices))

    
for plot_idx, z_idx in enumerate(z_indices):
    # Extract the voxel region
    region = extract_voxel_region(tomo_data, voxel_results, z_idx, y_idx, x_idx)
    magnitude, KX, KY, KZ = compute_fft_q(region, use_vignette=True,pixel_size=18)
    fig, fig_local = plot_voxel_region(tomo_data, voxel_results, z_idx, y_idx, x_idx, intensity_threshold_tomo)
    fig_combined.add_trace(fig.data[0], row=plot_idx+1, col=1)
    fig_combined.add_trace(fig.data[1], row=plot_idx+1, col=1)
    fig_combined.add_trace(fig.data[2], row=plot_idx+1, col=1)
        
    # Find peaks in the 3D FFT magnitude
    peak_positions, peak_values = find_peaks_3d(magnitude, threshold=peak_threshold, sigma=sigma)
    
    # Extract peak coordinates
    peak_kx = KX[peak_positions[:, 0], peak_positions[:, 1], peak_positions[:, 2]]# + x_idx
    peak_ky = KY[peak_positions[:, 0], peak_positions[:, 1], peak_positions[:, 2]]# + y_idx
    peak_kz = KZ[peak_positions[:, 0], peak_positions[:, 1], peak_positions[:, 2]]# + z_idx
    
    # Store peaks for this voxel
    voxel_peaks[z_idx] = {
        'positions': np.column_stack((peak_kx, peak_ky, peak_kz)),
        'values': peak_values
    }
    
    # Create a 3D scatter plot of the peaks
    fig_peaks = go.Figure(data=go.Scatter3d(
        x=peak_kx,
        y=peak_ky,
        z=peak_kz,
        mode='markers',
        marker=dict(
            size=5,
            color=peak_values,
            colorscale='Plasma',
            opacity=np.max(peak_values)/np.max(all_magnitudes),
            colorbar=dict(title='Peak Magnitude'),
            showscale=False
            )
        )
    )
    fig_combined.add_trace(fig_peaks.data[0], row=plot_idx+1, col=2)


    # Create a 3D scatter plot of the FFT magnitude

    # Flatten the arrays
    kx_flat = KX.flatten()
    ky_flat = KY.flatten()
    kz_flat = KZ.flatten()
    magnitude_flat = magnitude.flatten()

    # Apply the threshold
    mask = magnitude_flat > threshold*np.max(all_magnitudes)
    kx_filtered = kx_flat[mask]
    ky_filtered = ky_flat[mask]
    kz_filtered = kz_flat[mask]
    magnitude_filtered = magnitude_flat[mask]

    # Create a 3D scatter plot of the FFT magnitude
    fig_fft = go.Figure(data=go.Scatter3d(
        x=kx_filtered,
        y=ky_filtered,
        z=kz_filtered,
        mode='markers',
        marker=dict(
            size=2,
            color=magnitude_filtered,
            colorscale='Viridis',
            opacity=0.3,
            showscale=False
            #colorbar=dict(title='Magnitude')
        )
    ))
    fig_combined.add_trace(fig_fft.data[0], row=plot_idx+1, col=2)


# Compare peaks between neighboring voxels
for z_idx in z_indices[:-1]:
    current_peaks = voxel_peaks[z_idx]
    next_peaks = voxel_peaks[z_idx + 1]
    
    # Compare positions and intensities
    for i, (pos, val) in enumerate(zip(current_peaks['positions'], current_peaks['values'])):
        # Find matching peaks in the next voxel
        distances = np.linalg.norm(next_peaks['positions'] - pos, axis=1)
        match_idx = np.argmin(distances)
        if distances[match_idx] < 1:#some_threshold:  # Define a suitable threshold
            matched_pos = next_peaks['positions'][match_idx]
            matched_val = next_peaks['values'][match_idx]
            
            print(f"Voxel {z_idx} Peak {i} at {pos} with value {val}")
            print(f"Matches Voxel {z_idx + 1} Peak at {matched_pos} with value {matched_val}")
            print(f"Position Difference: {distances[match_idx]}")
            print(f"Intensity Difference: {abs(val - matched_val)}\n")


fig_combined.show()






























#%%

'''
Plot FFT peaks for all voxels in the tomogram
'''
fig_RS = go.Figure()
voxel_results['n_voxels'][0]
# Define the range for all dimensions
z_indices = range(3, voxel_results['n_voxels'][0]-5)  # Adjust range as needed
y_indices = range(3, voxel_results['n_voxels'][1]-5)  # Adjust range as needed
x_indices = range(3, voxel_results['n_voxels'][2]-5)  # Adjust range as needed



# First pass to find maximum peak value across all voxels
max_peak_value = 0
for z_idx in tqdm(z_indices):
    for y_idx in y_indices:
        for x_idx in x_indices:
            region = extract_voxel_region(tomo_data, voxel_results, z_idx, y_idx, x_idx)
            magnitude, KX, KY, KZ = compute_fft_q(region, use_vignette=True,pixel_size=18)
            peak_positions, peak_values = find_peaks_3d(magnitude, threshold=peak_threshold, sigma=sigma)
            
            if len(peak_values) > 0:
                # Filter out central peaks
                non_central_peaks = []
                for i, pos in enumerate(peak_positions):
                    peak_kx = KX[pos[0], pos[1], pos[2]]
                    peak_ky = KY[pos[0], pos[1], pos[2]]
                    peak_kz = KZ[pos[0], pos[1], pos[2]]
                    
                    # Check if peak is not at center (allowing for small numerical errors)
                    if not (abs(peak_kx) < 0.02 and abs(peak_ky) < 0.02 and abs(peak_kz) < 0.02):
                        non_central_peaks.append(i)
                
                if non_central_peaks:  # If there are non-central peaks
                    max_peak_value = max(max_peak_value, np.max(peak_values[non_central_peaks]))

# Second pass to plot peaks
for z_idx in tqdm(z_indices):
    for y_idx in y_indices:
        for x_idx in x_indices:
            # Extract the voxel region
            region = extract_voxel_region(tomo_data, voxel_results, z_idx, y_idx, x_idx)
            magnitude, KX, KY, KZ = compute_fft_q(region, use_vignette=True,pixel_size=18)
            
            # Find peaks in the 3D FFT magnitude
            peak_threshold = 0.2
            peak_positions, peak_values = find_peaks_3d(magnitude, threshold=peak_threshold, sigma=0.5)
            
            if len(peak_positions) > 0:  # Only add traces if peaks were found
                for i in range(len(peak_positions)):
                    # Extract single peak coordinates
                    peak_kx = KX[peak_positions[i, 0], peak_positions[i, 1], peak_positions[i, 2]]
                    peak_ky = KY[peak_positions[i, 0], peak_positions[i, 1], peak_positions[i, 2]]
                    peak_kz = KZ[peak_positions[i, 0], peak_positions[i, 1], peak_positions[i, 2]]
                    
                    # Skip central peaks
                    if abs(peak_kx) < 0.01 and abs(peak_ky) < 0.01 and abs(peak_kz) < 0.01:
                        continue
                        
                    # Shift to voxel position
                    peak_kx += x_idx
                    peak_ky += y_idx
                    peak_kz += z_idx
                    
                    # Normalize peak value for opacity
                    normalized_opacity = peak_values[i] / max_peak_value
                    
                    # Add individual peak to the figure
                    fig_RS.add_trace(
                        go.Scatter3d(
                            x=[peak_kx],
                            y=[peak_ky],
                            z=[peak_kz],
                            mode='markers',
                            marker=dict(
                                size=5,
                                color=[peak_values[i]],
                                colorscale='Plasma',
                                opacity=normalized_opacity,
                                showscale=False
                            ),
                            name=f'x={x_idx},y={y_idx},z={z_idx},p={i}',
                            showlegend=False
                        )
                    )

# Update layout
fig_RS.update_layout(
    title=f"FFT Peaks in All Voxel Positions (excluding central peaks, max value: {max_peak_value:.2f})",
    scene=dict(
        aspectmode='cube',
        xaxis_title="KX + voxel_x",
        yaxis_title="KY + voxel_y",
        zaxis_title="KZ + voxel_z",
        camera=dict(
            eye=dict(x=2, y=2, z=2)
        )
    ),
    width=1000,
    height=1000
)

fig_RS.show()

# # Plot peak-based orientations with magnitude-based scaling
# for i, y_pos in enumerate(y_positions):
#     fig_all.add_trace(
#         go.Scatter3d(
#             x=[fixed_x, fixed_x + all_orientations_peak[i,0] * norm_magnitudes_peak[i]],
#             y=[y_pos, y_pos + all_orientations_peak[i,1] * norm_magnitudes_peak[i]],
#             z=[fixed_z, fixed_z + all_orientations_peak[i,2] * norm_magnitudes_peak[i]],
#             mode='lines',
#             line=dict(
#                 color='blue', 
#                 width=3,
#                 #opacity=0.7
#             ),
#             name='Peak orientation'
#         )
#     )

# Update layout
fig_all.update_layout(
    title=f"Orientation Vectors Along Y-axis (x={fixed_x}, z={fixed_z})<br>Arrow length scaled by intensity",
    scene=dict(
        aspectmode='cube',
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",
        camera=dict(eye=dict(x=2, y=2, z=2))
    ),
    width=1000,
    height=1000,
    showlegend=True
)

fig_all.show()

# Print average orientations and magnitudes
print("\nFFT Magnitude Method:")
print(f"Average orientation: ({np.mean(all_orientations_fft[:,0]):.3f}, {np.mean(all_orientations_fft[:,1]):.3f}, {np.mean(all_orientations_fft[:,2]):.3f})")
print(f"Average magnitude: {np.mean(all_magnitudes_fft):.3f}")

print("\nPeak-based Method:")
print(f"Average orientation: ({np.mean(all_orientations_peak[:,0]):.3f}, {np.mean(all_orientations_peak[:,1]):.3f}, {np.mean(all_orientations_peak[:,2]):.3f})")
print(f"Average magnitude: {np.mean(all_magnitudes_peak):.3f}")






#%%
'''
Plot orientation vectors for all voxels in the 3D volume
'''
# Create figure for all orientation vectors
fig_all = go.Figure()

# Store orientations and magnitudes for comparison
all_orientations = []
all_magnitudes = []
x_positions = []
y_positions = []
z_positions = []

# Analyze each voxel in the volume
for x_idx in range(0, voxel_results['n_voxels'][2]):
    for y_idx in range(0, voxel_results['n_voxels'][1]):
        for z_idx in range(0, voxel_results['n_voxels'][0]):
            # Extract and process voxel
            region = extract_voxel_region(tomo_data, voxel_results, z_idx, y_idx, x_idx)
            magnitude, KX, KY, KZ = compute_fft_q(region, use_vignette=True,pixel_size=18)
            
            # ---- FFT Magnitude Method ----
            center_x = magnitude.shape[0] // 2
            center_y = magnitude.shape[1] // 2
            center_z = magnitude.shape[2] // 2
            
            z_coords, y_coords, x_coords = np.meshgrid(
                np.arange(magnitude.shape[2]) - center_z,
                np.arange(magnitude.shape[1]) - center_y,
                np.arange(magnitude.shape[0]) - center_x,
                indexing='ij'
            )
            
            # Create masks
            radius = 5
            r = np.sqrt(x_coords**2 + y_coords**2 + z_coords**2)
            central_mask = r > radius
            upper_mask = KZ > 0
            
            max_val = np.max(magnitude)
            threshold = max_val * 0
            threshold_mask = magnitude > threshold
            combined_mask = central_mask & threshold_mask & upper_mask
            
            total_intensity = np.sum(magnitude * combined_mask)
            
            if total_intensity > 0:
                com_x = np.sum(KX * magnitude * combined_mask) / total_intensity
                com_y = np.sum(KY * magnitude * combined_mask) / total_intensity
                com_z = np.sum(KZ * magnitude * combined_mask) / total_intensity
                
                # Store orientation and magnitude
                all_orientations.append([com_x, com_y, com_z])
                all_magnitudes.append(total_intensity)
                x_positions.append(x_idx)
                y_positions.append(y_idx)
                z_positions.append(z_idx)

# Convert to numpy arrays
all_orientations = np.array(all_orientations)
all_magnitudes = np.array(all_magnitudes)
x_positions = np.array(x_positions)
y_positions = np.array(y_positions)
z_positions = np.array(z_positions)

# Normalize magnitudes for scaling
max_mag = np.max(all_magnitudes)
norm_magnitudes = all_magnitudes / max_mag * 2.0  # Reduced scale factor for better visibility

# Plot orientation vectors
for i in range(len(x_positions)):
    # Add arrow
    fig_all.add_trace(
        go.Scatter3d(
            x=[x_positions[i], x_positions[i] + all_orientations[i,0] * norm_magnitudes[i]],
            y=[y_positions[i], y_positions[i] + all_orientations[i,1] * norm_magnitudes[i]],
            z=[z_positions[i], z_positions[i] + all_orientations[i,2] * norm_magnitudes[i]],
            mode='lines',
            line=dict(
                color='red', 
                width=1  # Reduced width for better visibility
                #opacity=0.6  # Added opacity for better visualization of overlapping arrows
            ),
            name='FFT orientation',
            showlegend=False
        )
    )

# Update layout
fig_all.update_layout(
    title="3D Orientation Vectors (All Voxels)<br>Arrow length scaled by intensity",
    scene=dict(
        aspectmode='cube',
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
    ),
    width=1000,
    height=1000,
    showlegend=False
)

fig_all.show()

# Print average orientation and magnitude
print("\nFFT Magnitude Method:")
print(f"Average orientation: ({np.mean(all_orientations[:,0]):.3f}, {np.mean(all_orientations[:,1]):.3f}, {np.mean(all_orientations[:,2]):.3f})")
print(f"Average magnitude: {np.mean(all_magnitudes):.3f}")

# Optional: Save orientations and positions to file
np.savez('orientation_data.npz', 
         orientations=all_orientations,
         magnitudes=all_magnitudes,
         x_pos=x_positions,
         y_pos=y_positions,
         z_pos=z_positions)





























