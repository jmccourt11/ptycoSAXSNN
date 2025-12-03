import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import colors
import sys
import os
from skimage import measure
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '/home/beams/PTYCHOSAXS/NN/ptychosaxsNN/utils/')))
import ptychosaxsNN_utils as ptNN_U
import importlib
importlib.reload(ptNN_U)

def calculate_psnr(img1, img2, max_val=None):
    """
    Calculate Peak Signal-to-Noise Ratio between two images.
    
    Parameters:
    -----------
    img1, img2 : array_like
        The images between which to calculate PSNR
    max_val : float, optional
        Maximum possible pixel value. If None, uses the maximum value in img1
        
    Returns:
    --------
    psnr : float
        The PSNR value in decibels
    """
    if max_val is None:
        max_val = np.max(img1)
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    return 20 * np.log10(max_val) - 10 * np.log10(mse)

class Scan_Plotter:
    """
    A class to handle various plotting methods for diffraction pattern scans.
    
    This class provides a unified interface for different visualization techniques
    including full patterns, absorption maps, azimuthal segment analysis, and more.
    It handles both raw diffraction patterns and model-processed (deconvolved) patterns.
    """
    
    def __init__(self, dps, preprocess_func=None, mask=None, model=None, 
                 scanx=36, scany=29, dpsize=256, center=(517,575)):
        """
        Initialize the Scan_Plotter with diffraction patterns and optional processing parameters.
        
        Parameters:
        -----------
        dps : array
            Array of diffraction patterns
        preprocess_func : function, optional
            Function to preprocess diffraction patterns
        mask : array, optional
            Mask for diffraction patterns
        model : model, optional
            Neural network model for deconvolution
        scanx, scany : int
            Dimensions of the scan grid
        dpsize : int
            Size of the diffraction pattern (assumed square)
        center : tuple
            Center coordinates for cropping diffraction patterns
        """
        self.dps = dps
        self.preprocess_func = preprocess_func
        self.mask = mask
        self.model = model
        self.scanx = scanx
        self.scany = scany
        self.dpsize = dpsize
        self.center = center
        
        # Pre-calculate indices for cropping
        self.y_start = center[0] - dpsize//2
        self.y_end = center[0] + dpsize//2
        self.x_start = center[1] - dpsize//2
        self.x_end = center[1] + dpsize//2
        
        # Cache for processed patterns
        self.processed_cache = {}
    
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def normalize_patterns(self, mode='model', method='sum'):
        """
        Normalize all diffraction patterns in the scan.
        
        Parameters:
        -----------
        mode : str or list
            Processing mode(s): 'raw', 'preprocessed', or 'model', or list of modes
        method : str
            Normalization method:
                'sum': Normalize by total intensity (sum)
                'mean': Normalize by mean intensity
                'max': Normalize by maximum intensity
                'subtract_first': Subtract the first pattern (straight-through beam) from all others
                
        Returns:
        --------
        normalization_factors : dict
            Dictionary of normalization factors for each mode
        """
        print("Calculating normalization factors...")
        
        # Convert single mode to list
        if isinstance(mode, str):
            modes = [mode]
        else:
            modes = mode
        
        # Initialize normalization factors based on method
        if method == 'subtract_first':
            # For subtract_first, we'll store the reference patterns and differences for each mode
            if not hasattr(self, 'reference_patterns'):
                self.reference_patterns = {}
            if not hasattr(self, 'difference_patterns'):
                self.difference_patterns = {}
            
            # Initialize dictionaries for each mode
            for m in modes:
                if m not in self.reference_patterns:
                    self.reference_patterns[m] = None
                if m not in self.difference_patterns:
                    self.difference_patterns[m] = {}
        else:
            normalization_factors = {m: np.zeros(len(self.dps)) for m in modes}
        
        # First pass: calculate normalization factors
        try:
            pbar = tqdm(total=len(self.dps))
            for idx in range(len(self.dps)):
                for m in modes:
                    pattern = self.get_pattern(idx, m)
                    
                    if method == 'sum':
                        factor = np.sum(pattern)
                    elif method == 'mean':
                        factor = np.mean(pattern)
                    elif method == 'max':
                        factor = np.max(pattern)
                    elif method == 'subtract_first':
                        if idx == 0:
                            # Store the first pattern as the reference for this mode
                            self.reference_patterns[m] = pattern
                            # Store the difference pattern (zeros for first pattern)
                            self.difference_patterns[m][idx] = np.zeros_like(pattern)
                        else:
                            # Calculate and store the difference pattern
                            self.difference_patterns[m][idx] = pattern - self.reference_patterns[m]
                        continue  # Skip storing in normalization_factors
                    else:
                        raise ValueError(f"Unknown normalization method: {method}")
                    
                    normalization_factors[m][idx] = factor
                pbar.update(1)
            pbar.close()
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user (Ctrl+C)")
        
        # Store normalization factors and method info
        if method != 'subtract_first':
            self.normalization_factors = normalization_factors
        self.normalization_method = method
        
        # Store which modes have been normalized
        if not hasattr(self, 'normalized_modes'):
            self.normalized_modes = set()
        self.normalized_modes.update(modes)
        
        return normalization_factors if method != 'subtract_first' else None

    def _get_base_pattern(self, idx, mode='model'):
        """
        Get a diffraction pattern in various processing stages without normalization.
        This is the base method that actually retrieves the pattern.
        
        Parameters:
        -----------
        idx : int
            Index of the diffraction pattern
        mode : str
            Processing mode: 'raw', 'preprocessed', or 'model'
            
        Returns:
        --------
        pattern : array
            The processed or raw diffraction pattern
        """
        # Check if we've already processed this pattern
        cache_key = (idx, mode, False)  # False for not normalized
        if cache_key in self.processed_cache:
            return self.processed_cache[cache_key]
        
        # Crop the diffraction pattern
        dp = self.dps[idx][self.y_start:self.y_end, self.x_start:self.x_end]
        
        if mode == 'raw':
            # Return the raw pattern
            pattern = dp
        elif mode in ['preprocessed', 'model'] and self.preprocess_func is not None:
            # Process the diffraction pattern
            resultT, _, _ = self.preprocess_func(dp, self.mask)
            
            if mode == 'preprocessed':
                # Return the preprocessed pattern (model input)
                pattern = resultT.detach().to("cpu").numpy()[0][0]
            elif mode == 'model' and self.model is not None:
                # Apply the model and return the output
                pattern = self.model(resultT.to(device=self.device, dtype=torch.float)).detach().to("cpu").numpy()[0][0]
            else:
                # Fallback to raw if model is None
                pattern = dp
                print(f"Warning: Model not available, returning raw pattern for idx {idx}")
        else:
            # Fallback to raw for unknown mode
            pattern = dp
            print(f"Warning: Invalid mode '{mode}' or preprocessing function not available, returning raw pattern for idx {idx}")
        
        # Cache the result
        self.processed_cache[cache_key] = pattern
        return pattern

    def get_pattern(self, idx, mode='model', normalized=False):
        """
        Get a diffraction pattern in various processing stages.
        
        Parameters:
        -----------
        idx : int
            Index of the diffraction pattern
        mode : str
            Processing mode: 'raw', 'preprocessed', or 'model'
        normalized : bool
            Whether to return the normalized pattern
            
        Returns:
        --------
        pattern : array
            The processed or raw diffraction pattern
        """
        # Check if we've already processed this pattern
        cache_key = (idx, mode, normalized)
        if cache_key in self.processed_cache:
            return self.processed_cache[cache_key]
        
        # Get the base pattern
        pattern = self._get_base_pattern(idx, mode)
        
        # Apply normalization if requested
        if normalized:
            if not hasattr(self, 'normalization_method'):
                raise ValueError("Patterns must be normalized first using normalize_patterns()")
            if mode not in self.normalized_modes:
                raise ValueError(f"Patterns for mode '{mode}' have not been normalized. Please normalize this mode first.")
            
            if self.normalization_method == 'subtract_first':
                # For subtract_first method, return the pre-calculated difference pattern for this mode
                pattern = self.difference_patterns[mode][idx]
            else:
                factor = self.normalization_factors[mode][idx]
                if factor > 0:  # Avoid division by zero
                    pattern = pattern / factor
                else:
                    print(f"Warning: Zero normalization factor for pattern {idx}")
        
        # Cache the result
        self.processed_cache[cache_key] = pattern
        return pattern
    
    def create_azimuthal_segments(self, num_segments=8, inner_radius=0, outer_radius=None):
        """
        Create masks dividing a diffraction pattern into azimuthal segments.
        
        Parameters:
        -----------
        num_segments : int
            Number of azimuthal segments to create
        inner_radius : float
            Inner radius of the annular segments
        outer_radius : float
            Outer radius of the annular segments
            
        Returns:
        --------
        segment_masks : list of arrays
            List of boolean masks for each segment
        """
        return ptNN_U.create_azimuthal_segments((self.dpsize//4, self.dpsize//4), center=None, 
                                        num_segments=num_segments,
                                        inner_radius=inner_radius, 
                                        outer_radius=outer_radius)
    
    def plot_full_scan(self, mode='model', log_scale=False):
        """
        Plot all diffraction patterns in a grid.
        
        Parameters:
        -----------
        mode : str
            Processing mode: 'raw', 'preprocessed', or 'model'
        log_scale : bool
            Whether to use logarithmic color scale
            
        Returns:
        --------
        fig : matplotlib figure
            The figure containing the plots
        """
        # Create figure and axes
        fig, axs = plt.subplots(self.scany, self.scanx, figsize=(self.scanx, self.scany))
        fig.subplots_adjust(hspace=0, wspace=0)
        
        # Handle different dimensions of axs
        if self.scany == 1 and self.scanx == 1:
            axs = np.array([[axs]])
        elif self.scany == 1:
            axs = np.array([axs])
        elif self.scanx == 1:
            axs = np.array([[ax] for ax in axs])
        
        # Turn off all axes at once
        for ax_row in axs:
            for ax in ax_row:
                ax.axis('off')
        
        # Process and plot patterns
        count = 0
        try:
            pbar = tqdm(total=min(self.scanx*self.scany, len(self.dps)))
            
            for i in range(self.scany):
                # Handle serpentine scan pattern
                if i % 2 == 0:
                    j_range = range(self.scanx)  # Left to right
                else:
                    j_range = range(self.scanx-1, -1, -1)  # Right to left
                
                for j in j_range:
                    if count < len(self.dps):
                        # Get the pattern
                        pattern = self.get_pattern(count, mode)
                        
                        # Plot the pattern with optional log scale
                        if log_scale:
                            # Add a small value to avoid log(0)
                            pattern_for_log = pattern.copy()
                            pattern_for_log[pattern_for_log <= 0] = pattern_for_log[pattern_for_log > 0].min() / 10
                            im = axs[i][j].imshow(pattern_for_log, cmap='jet', norm=colors.LogNorm())
                        else:
                            im = axs[i][j].imshow(pattern, cmap='jet')
                        
                        count += 1
                        pbar.update(1)
                    else:
                        axs[i][j].text(0.5, 0.5, 'No data', ha='center', va='center')
            
            pbar.close()
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user (Ctrl+C)")
        
        # # Add colorbar
        # if count > 0:
        #     cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
        #     fig.colorbar(im, cax=cbar_ax)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_absorption_map(self, mode='model', log_scale=False, normalized=False):
        """
        Plot a map of total absorption/intensity across the scan.
        
        Parameters:
        -----------
        mode : str
            Processing mode: 'raw', 'preprocessed', or 'model'
        log_scale : bool
            Whether to use logarithmic color scale
        normalized : bool
            Whether to use normalized patterns
            
        Returns:
        --------
        absorption_map : array
            2D array of absorption values
        """
        # Create figure and axes
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Create a 2D array to store absorption values
        absorption_map = np.zeros((self.scany, self.scanx))
        
        # Process patterns and calculate absorption
        count = 0
        try:
            pbar = tqdm(total=min(self.scanx*self.scany, len(self.dps)))
            
            for i in range(self.scany):
                # Handle serpentine scan pattern
                if i % 2 == 0:
                    j_range = range(self.scanx)  # Left to right
                else:
                    j_range = range(self.scanx-1, -1, -1)  # Right to left
                
                for j in j_range:
                    if count < len(self.dps):
                        # Get the pattern
                        pattern = self.get_pattern(count, mode, normalized=normalized)
                        
                        # Calculate absorption (mean intensity)
                        #absorption = np.mean(pattern)
                        absorption = np.sum(pattern)
                        
                        # Store the absorption value
                        absorption_map[i, j] = absorption
                        
                        count += 1
                        pbar.update(1)
            
            pbar.close()
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user (Ctrl+C)")
        
        # Plot the absorption map with optional log scale
        if log_scale:
            im = ax.imshow(absorption_map, cmap='jet', norm=colors.LogNorm())
        else:
            im = ax.imshow(absorption_map, cmap='jet')
        
        #ax.set_title('Absorption Map')
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Integrated Intensity (a.u.)')
        
        plt.tight_layout()
        plt.show()
        
        return fig, absorption_map
    
    def plot_azimuthal_segment(self, segment_idx=0, num_segments=8, 
                              inner_radius=0, outer_radius=None, mode='model', log_scale=False,
                              normalized=False):
        """
        Plot integrated intensity of a specific azimuthal segment.
        
        Parameters:
        -----------
        segment_idx : int
            Index of the segment to plot
        num_segments : int
            Total number of segments
        inner_radius, outer_radius : float
            Radial constraints for the segment
        mode : str
            Processing mode: 'raw', 'preprocessed', or 'model'
        log_scale : bool
            Whether to use logarithmic color scale
        normalized : bool
            Whether to use normalized patterns
            
        Returns:
        --------
        segment_map : array
            2D array of segment intensity values
        """
        # Create figure and axes
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Create segment masks
        segment_masks = self.create_azimuthal_segments(num_segments, inner_radius, outer_radius)
        segment_mask = segment_masks[segment_idx]
        
        # Create a 2D array to store segment intensity values
        segment_map = np.zeros((self.scany, self.scanx))
        
        # Process patterns and calculate segment intensity
        count = 0
        try:
            pbar = tqdm(total=min(self.scanx*self.scany, len(self.dps)))
            
            for i in range(self.scany):
                # Handle serpentine scan pattern
                if i % 2 == 0:
                    j_range = range(self.scanx)  # Left to right
                else:
                    j_range = range(self.scanx-1, -1, -1)  # Right to left
                
                for j in j_range:
                    if count < len(self.dps):
                        # Get the pattern
                        pattern = self.get_pattern(count, mode, normalized=normalized)
                        
                        # Calculate segment intensity
                        segment_intensity = np.sum(pattern * segment_mask)
                        
                        # Store the segment intensity value
                        segment_map[i, j] = segment_intensity
                        
                        count += 1
                        pbar.update(1)
            
            pbar.close()
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user (Ctrl+C)")
        
        # Plot the segment map with optional log scale
        if log_scale:
            im = ax.imshow(segment_map, cmap='jet', norm=colors.LogNorm())
        else:
            im = ax.imshow(segment_map, cmap='jet')
        
        ax.set_title(f'Azimuthal Segment {segment_idx} Intensity Map')
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Integrated Intensity (a.u.)')
        
        plt.tight_layout()
        plt.show()
        
        return segment_map
    
    def plot_all_azimuthal_segments(self, num_segments=8, inner_radius=0, 
                                   outer_radius=None, example_idx=None, 
                                   mode='model', log_scale=False, normalized=False):
        """
        Plot all azimuthal segments with optional insets.
        
        Parameters:
        -----------
        num_segments : int
            Number of segments to create
        inner_radius, outer_radius : float
            Radial constraints for the segments
        example_idx : int, optional
            Index of an example pattern to show in insets
        mode : str
            Processing mode: 'raw', 'preprocessed', or 'model'
        log_scale : bool
            Whether to use logarithmic color scale
        normalized : bool
            Whether to use normalized patterns
            
        Returns:
        --------
        fig : matplotlib figure
            The figure containing the plots
        """
        # Calculate grid dimensions
        grid_size = int(np.ceil(np.sqrt(num_segments)))
        
        # Create figure and axes
        fig, axs = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        axs = axs.flatten()
        
        # Create segment masks
        segment_masks = self.create_azimuthal_segments(num_segments, inner_radius, outer_radius)
        
        # Process example pattern for insets if requested
        if example_idx is not None:
            example_pattern = self.get_pattern(example_idx, mode, normalized=normalized)
        
        # Process all patterns once
        processed_patterns = []
        try:
            print("Processing diffraction patterns...")
            pbar = tqdm(total=min(self.scanx*self.scany, len(self.dps)))
            
            count = 0
            for i in range(self.scany):
                # Handle serpentine scan pattern
                if i % 2 == 0:
                    j_range = range(self.scanx)  # Left to right
                else:
                    j_range = range(self.scanx-1, -1, -1)  # Right to left
                
                for j in j_range:
                    if count < len(self.dps):
                        # Get the pattern
                        pattern = self.get_pattern(count, mode, normalized=normalized)
                        
                        # Store the pattern with its position
                        processed_patterns.append((i, j, pattern))
                        
                        count += 1
                        pbar.update(1)
            
            pbar.close()
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user (Ctrl+C)")
        
        # Create and calculate all segment maps first to determine global min/max
        print("Creating segment maps...")
        segment_maps = []
        vmin, vmax = float('inf'), float('-inf')
        
        for segment_idx in range(num_segments):
            if segment_idx >= len(axs):
                break
                
            # Get the segment mask
            segment_mask = segment_masks[segment_idx]
            
            # Create segment map
            segment_map = np.zeros((self.scany, self.scanx))
            for i, j, pattern in processed_patterns:
                segment_map[i, j] = np.sum(pattern * segment_mask)
            
            # Update global min/max
            vmin = min(vmin, np.min(segment_map))
            vmax = max(vmax, np.max(segment_map))
            
            segment_maps.append(segment_map)
        
        # Now plot all segments with the same color scale
        for segment_idx, segment_map in enumerate(segment_maps):
            if segment_idx >= len(axs):
                break
            
            # Plot segment map with common scale
            if log_scale:
                # Ensure positive values for log scale
                if vmin <= 0:
                    vmin = np.min(segment_map[segment_map > 0])
                norm = colors.LogNorm(vmin=vmin, vmax=vmax)
                im = axs[segment_idx].imshow(segment_map, cmap='jet', norm=norm)
            else:
                im = axs[segment_idx].imshow(segment_map, cmap='jet', vmin=vmin, vmax=vmax)
            
            axs[segment_idx].set_title(f'Segment {segment_idx}')
            
            # Add inset if example_idx was provided
            if example_idx is not None:
                # Create inset axes
                inset_ax = axs[segment_idx].inset_axes([0.65, 0.65, 0.3, 0.3])
                
                # # Create masked example
                # masked_example = example_pattern.copy()
                # masked_example[~segment_masks[segment_idx]] = 0
                
                # Plot masked example
                if log_scale:
                    inset_ax.imshow(example_pattern, cmap='jet', norm=colors.LogNorm())
                else:
                    inset_ax.imshow(example_pattern, cmap='jet', clim=(0,1))
                
                # Find the boundary of the segment mask
                contours = measure.find_contours(segment_masks[segment_idx].astype(float), 0.5)
                for contour in contours:
                    # Plot the contour with a dashed red line
                    inset_ax.plot(contour[:, 1], contour[:, 0], '--r', linewidth=1)
                
                inset_ax.set_xticks([])
                inset_ax.set_yticks([])
                #inset_ax.set_title(f'Frame {example_idx}', fontsize=8)
        
        # Hide unused subplots
        for idx in range(num_segments, len(axs)):
            axs[idx].axis('off')
        
        # Add a single colorbar for all segments
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='Integrated Intensity (a.u.)')
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.9)  # Make room for the colorbar
        plt.show()
        
        return fig, segment_maps
    
    def plot_radial_profile(self, idx=None, mode='model', num_bins=100, max_radius=None, 
                           log_scale=True, azimuthal_range=None):
        """
        Plot the radial intensity profile of a diffraction pattern.
        
        Parameters:
        -----------
        idx : int or None
            Index of the diffraction pattern to analyze. If None, averages all patterns.
        mode : str
            Processing mode: 'raw', 'preprocessed', or 'model'
        num_bins : int
            Number of radial bins
        max_radius : float or None
            Maximum radius to include. If None, uses the maximum possible radius.
        log_scale : bool
            Whether to use logarithmic scale for intensity
        azimuthal_range : tuple or None
            Optional (min_angle, max_angle) in radians to restrict the azimuthal range
            
        Returns:
        --------
        fig : matplotlib figure
            The figure containing the plot
        radial_profile : array
            The calculated radial profile
        """
        # Create figure and axes
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Set up radial bins
        if max_radius is None:
            # Calculate maximum radius that fits within the pattern
            center = (self.dpsize // 2, self.dpsize // 2)
            max_radius = min(center[0], center[1], 
                             self.dpsize - center[0], 
                             self.dpsize - center[1])
        
        radial_bins = np.linspace(0, max_radius, num_bins + 1)
        bin_centers = 0.5 * (radial_bins[1:] + radial_bins[:-1])
        
        # Create coordinate arrays
        y, x = np.ogrid[:self.dpsize, :self.dpsize]
        y = y - self.dpsize // 2
        x = x - self.dpsize // 2
        
        # Calculate radius for each pixel
        radius_map = np.sqrt(y**2 + x**2)
        
        # Create azimuthal mask if needed
        if azimuthal_range is not None:
            min_angle, max_angle = azimuthal_range
            angles = np.arctan2(y, x) % (2 * np.pi)
            azimuthal_mask = (angles >= min_angle) & (angles <= max_angle)
        else:
            azimuthal_mask = np.ones((self.dpsize, self.dpsize), dtype=bool)
        
        # Process patterns and calculate radial profile
        if idx is not None:
            # Single pattern mode
            pattern = self.get_pattern(idx, mode)
            
            # Calculate radial profile
            radial_profile = np.zeros(num_bins)
            for i in range(num_bins):
                # Create mask for this radial bin
                bin_mask = (radius_map >= radial_bins[i]) & (radius_map < radial_bins[i+1]) & azimuthal_mask
                
                # Calculate mean intensity in this bin
                if np.any(bin_mask):
                    radial_profile[i] = np.mean(pattern[bin_mask])
                else:
                    radial_profile[i] = 0
            
            title = f"Radial Profile - {'Processed' if mode in ['preprocessed', 'model'] else 'Raw'} Pattern {idx}"
        
        else:
            # Average all patterns mode
            print("Calculating average radial profile across all patterns...")
            radial_profile = np.zeros(num_bins)
            count = 0
            
            try:
                pbar = tqdm(total=min(self.scanx*self.scany, len(self.dps)))
                
                for i in range(self.scany):
                    # Handle serpentine scan pattern
                    if i % 2 == 0:
                        j_range = range(self.scanx)  # Left to right
                    else:
                        j_range = range(self.scanx-1, -1, -1)  # Right to left
                    
                    for j in j_range:
                        if count < len(self.dps):
                            # Get the pattern
                            pattern = self.get_pattern(count, mode)
                            
                            # Accumulate radial profile
                            for i in range(num_bins):
                                # Create mask for this radial bin
                                bin_mask = (radius_map >= radial_bins[i]) & (radius_map < radial_bins[i+1]) & azimuthal_mask
                                
                                # Accumulate mean intensity in this bin
                                if np.any(bin_mask):
                                    radial_profile[i] += np.mean(pattern[bin_mask])
                        
                            count += 1
                            pbar.update(1)
                
                pbar.close()
                
                # Calculate average
                if count > 0:
                    radial_profile /= count
                
                title = f"Average Radial Profile - {'Processed' if mode in ['preprocessed', 'model'] else 'Raw'} Patterns"
                
            except KeyboardInterrupt:
                print("\nProcessing interrupted by user (Ctrl+C)")
                if count > 0:
                    radial_profile /= count
        
        # Plot the radial profile
        if log_scale and np.all(radial_profile > 0):
            ax.semilogy(bin_centers, radial_profile)
        else:
            ax.plot(bin_centers, radial_profile)
        
        ax.set_title(title)
        ax.set_xlabel('Radius (pixels)')
        ax.set_ylabel('Intensity (a.u.)')
        ax.grid(True, alpha=0.3)
        
        # Add azimuthal range info if used
        if azimuthal_range is not None:
            min_deg = np.degrees(azimuthal_range[0])
            max_deg = np.degrees(azimuthal_range[1])
            ax.text(0.02, 0.95, f"Azimuthal range: {min_deg:.1f}° - {max_deg:.1f}°", 
                    transform=ax.transAxes, fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
        return fig, radial_profile, bin_centers
    
    def plot_radial_profile_map(self, radius_range, mode='model', num_bins=10, log_scale=False):
        """
        Plot a map of integrated intensity within a specific radius range.
        
        Parameters:
        -----------
        radius_range : tuple
            (min_radius, max_radius) range to integrate
        mode : str
            Processing mode: 'raw', 'preprocessed', or 'model'
        num_bins : int
            Number of radial bins within the range
        log_scale : bool
            Whether to use logarithmic color scale
            
        Returns:
        --------
        fig : matplotlib figure
            The figure containing the plots
        radial_maps : list of arrays
            List of 2D arrays for each radial bin
        """
        min_radius, max_radius = radius_range
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, (num_bins+1)//2, figsize=(15, 8))
        axs = axs.flatten()
        
        # Create coordinate arrays
        y, x = np.ogrid[:self.dpsize, :self.dpsize]
        y = y - self.dpsize // 2
        x = x - self.dpsize // 2
        
        # Calculate radius for each pixel
        radius_map = np.sqrt(y**2 + x**2)
        
        # Create radial bins
        radial_bins = np.linspace(min_radius, max_radius, num_bins + 1)
        
        # Create maps for each radial bin
        radial_maps = [np.zeros((self.scany, self.scanx)) for _ in range(num_bins)]
        
        # Process patterns and calculate radial bin intensities
        count = 0
        try:
            print("Processing diffraction patterns...")
            pbar = tqdm(total=min(self.scanx*self.scany, len(self.dps)))
            
            for i in range(self.scany):
                # Handle serpentine scan pattern
                if i % 2 == 0:
                    j_range = range(self.scanx)  # Left to right
                else:
                    j_range = range(self.scanx-1, -1, -1)  # Right to left
                
                for j in j_range:
                    if count < len(self.dps):
                        # Get the pattern
                        pattern = self.get_pattern(count, mode)
                        
                        # Calculate intensity in each radial bin
                        for bin_idx in range(num_bins):
                            # Create mask for this radial bin
                            bin_mask = (radius_map >= radial_bins[bin_idx]) & (radius_map < radial_bins[bin_idx+1])
                            
                            # Calculate integrated intensity in this bin
                            if np.any(bin_mask):
                                radial_maps[bin_idx][i, j] = np.sum(pattern[bin_mask])
                        
                        count += 1
                        pbar.update(1)
            
            pbar.close()
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user (Ctrl+C)")
        
        # Plot each radial bin map with optional log scale
        for bin_idx in range(num_bins):
            if bin_idx < len(axs):
                # Calculate bin range for title
                min_r = radial_bins[bin_idx]
                max_r = radial_bins[bin_idx+1]
                
                # Plot the map with optional log scale
                if log_scale:
                    im = axs[bin_idx].imshow(radial_maps[bin_idx], cmap='jet', norm=colors.LogNorm())
                else:
                    im = axs[bin_idx].imshow(radial_maps[bin_idx], cmap='jet')
                
                axs[bin_idx].set_title(f'r = {min_r:.1f}-{max_r:.1f} px')
                fig.colorbar(im, ax=axs[bin_idx])
        
        # Hide unused subplots
        for idx in range(num_bins, len(axs)):
            axs[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return fig, radial_maps
    
    def calculate_pattern_psnr(self, idx=None, normalized=False):
        """
        Calculate PSNR between preprocessed (convolved) and model (deconvolved) patterns.
        
        Parameters:
        -----------
        idx : int or None
            Index of the pattern to analyze. If None, calculates PSNR for all patterns.
        normalized : bool
            Whether to use normalized patterns
            
        Returns:
        --------
        psnr_values : float or array
            PSNR value(s) in decibels. Single float if idx is provided, array otherwise.
        """
        if idx is not None:
            # Calculate PSNR for a single pattern
            preprocessed = self.get_pattern(idx, mode='preprocessed', normalized=normalized)
            model = self.get_pattern(idx, mode='model', normalized=normalized)
            
            return calculate_psnr(preprocessed, model)
        else:
            # Calculate PSNR for all patterns
            psnr_values = np.zeros(len(self.dps))
            
            try:
                print("Calculating PSNR for all patterns...")
                pbar = tqdm(total=len(self.dps))
                
                for i in range(len(self.dps)):
                    preprocessed = self.get_pattern(i, mode='preprocessed', normalized=normalized)
                    model = self.get_pattern(i, mode='model', normalized=normalized)
                    
                    psnr_values[i] = calculate_psnr(preprocessed, model)
                    pbar.update(1)
                
                pbar.close()
            except KeyboardInterrupt:
                print("\nProcessing interrupted by user (Ctrl+C)")
            
            return psnr_values
    
    def plot_psnr_map(self, normalized=False):
        """
        Plot a map of PSNR values across the scan.
        
        Parameters:
        -----------
        normalized : bool
            Whether to use normalized patterns
            
        Returns:
        --------
        fig : matplotlib figure
            The figure containing the plot
        psnr_map : array
            2D array of PSNR values
        """
        # Calculate PSNR for all patterns
        psnr_values = self.calculate_pattern_psnr(normalized=normalized)
        
        # Create figure and axes
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Reshape PSNR values into 2D map
        psnr_map = np.zeros((self.scany, self.scanx))
        count = 0
        
        for i in range(self.scany):
            # Handle serpentine scan pattern
            if i % 2 == 0:
                j_range = range(self.scanx)  # Left to right
            else:
                j_range = range(self.scanx-1, -1, -1)  # Right to left
            
            for j in j_range:
                if count < len(psnr_values):
                    psnr_map[i, j] = psnr_values[count]
                    count += 1
        
        # Plot the PSNR map
        im = ax.imshow(psnr_map, cmap='viridis')
        
        ax.set_title('PSNR Map')
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('PSNR (dB)')
        
        # Add mean PSNR value as text
        mean_psnr = np.mean(psnr_values)
        ax.text(0.02, 0.98, f'Mean PSNR: {mean_psnr:.2f} dB',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        return fig, psnr_map
    
    def plot_model_comparison(self, model_paths, pattern_idx=664, normalized=False, 
                             figsize=(20, 15), cmap='viridis'):
        """
        Plot comparison of different models side by side.
        
        Parameters:
        -----------
        model_paths : dict
            Dictionary with structure:
            {
                'iterations': [2, 10, 25, 50, 100, 150, 200, 250, 300, 400, 500],
                'models': {
                    'L2_no_Unet': 'best_model_ZCB_9_32_no_Unet_epoch_{}_L2.pth',
                    'L2_Unet': 'best_model_ZCB_9_32_Unet_epoch_{}_L2.pth',
                    'pearson_no_Unet': 'best_model_ZCB_9_31_no_Unet_epoch_{}_pearson_loss.pth',
                    'pearson_Unet': 'best_model_ZCB_9_31_Unet_epoch_{}_pearson_loss.pth'
                }
            }
        pattern_idx : int
            Index of the pattern to compare
        normalized : bool
            Whether to use normalized patterns
        figsize : tuple
            Figure size (width, height)
        cmap : str
            Colormap to use for plotting
            
        Returns:
        --------
        fig : matplotlib figure
            The figure containing the comparison plots
        """
        import torch
        from pathlib import Path
        
        # Create figure and axes
        n_rows = len(model_paths['iterations'])
        n_cols = len(model_paths['models'])
        fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # Get the base pattern for comparison
        base_pattern = self.get_pattern(pattern_idx, mode='model', normalized=normalized)
        
        # Load and process each model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for i, iteration in enumerate(model_paths['iterations']):
            for j, (model_name, model_template) in enumerate(model_paths['models'].items()):
                try:
                    # Load model
                    model_path = Path("/net/micdata/data2/12IDC/ptychosaxs/trained_model") / model_template.format(iteration)
                    if not model_path.exists():
                        print(f"Model not found: {model_path}")
                        continue
                    
                    # Initialize appropriate model class based on name
                    if 'no_Unet' in model_name:
                        from encoder1_no_Unet import recon_model
                        model = recon_model()
                    else:
                        from encoder1 import recon_model
                        model = recon_model()
                    
                    # Load model weights and move to device
                    model.load_state_dict(torch.load(model_path))
                    model.to(device)
                    model.eval()
                    
                    # Get pattern using this model
                    with torch.no_grad():
                        pattern = self.get_pattern(pattern_idx, mode='preprocessed', normalized=normalized)
                        pattern_tensor = torch.from_numpy(pattern).unsqueeze(0).unsqueeze(0).to(device)
                        output = model(pattern_tensor)
                        output = output.cpu().numpy()[0, 0]
                    
                    # Plot the pattern
                    im = axs[i, j].imshow(output, cmap=cmap)
                    axs[i, j].set_title(f'{model_name}\nIteration {iteration}')
                    
                    # Calculate and display PSNR
                    psnr = calculate_psnr(base_pattern, output)
                    axs[i, j].text(0.02, 0.98, f'PSNR: {psnr:.2f} dB',
                                 transform=axs[i, j].transAxes,
                                 verticalalignment='top',
                                 bbox=dict(facecolor='white', alpha=0.8))
                    
                except Exception as e:
                    print(f"Error processing {model_name} iteration {iteration}: {e}")
                    axs[i, j].text(0.5, 0.5, 'Error',
                                 ha='center', va='center',
                                 transform=axs[i, j].transAxes)
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.9)  # Make room for the colorbar
        plt.show()
        
        return fig
    
    # Additional methods could include:
    # - plot_radial_profile
    # - plot_q_resolved_map
    # - plot_anisotropy_map
    # - plot_correlation_map
    # - etc.
    
# # Example usage
# # Create the plotter
# plotter = Scan_Plotter(dps, ptNN_U.preprocess_ZCB_9, mask, model_new, 
#                        scanx=36, scany=29)

# # Normalize all patterns (using sum normalization)
# normalization_factors = plotter.normalize_patterns(mode='model', method='sum')

# # Plot normalized patterns
# plotter.plot_full_scan(mode='model', normalized=True)

# # Plot normalized absorption map
# absorption_map = plotter.plot_absorption_map(mode='model', normalized=True)

# # Plot normalized azimuthal segments
# plotter.plot_all_azimuthal_segments(num_segments=8, inner_radius=10, 
#                                    outer_radius=100, example_idx=664,
#                                    mode='model', normalized=True)

# # Plot radial profile for a specific diffraction pattern
# fig, profile, bins = plotter.plot_radial_profile(idx=664, mode='model', num_bins=100)

# # Plot average radial profile across all patterns
# fig, avg_profile, bins = plotter.plot_radial_profile(idx=None, mode='model', num_bins=100)

# # Plot radial profile maps for specific q-ranges
# fig, radial_maps = plotter.plot_radial_profile_map(radius_range=(10, 100), mode='model', num_bins=8)