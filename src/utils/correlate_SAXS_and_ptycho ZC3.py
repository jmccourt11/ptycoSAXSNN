#%%
#!/usr/bin/env python3
"""
Script to correlate SAXS and ptychography reconstructions based on phi angles.

This script parses the log file to find ptychography and SAXS data entries,
extracts phi angles, and matches them for a given scan number.

The script looks for:
1. Ptychography entries: Lines with "#I detector_filename SZC3_3D_XXX_XXXXX_XXXXX.tif"
   followed by "#I phi = YYY.YYYY" lines
2. SAXS entries: Lines with "#I detector_filename SZC3SAXSXXX_XXXXX_XXXXX.tif"
   preceded by data lines where the first column contains the phi angle

Usage:
    python correlate_SAXS_and_ptycho.py --scan 323 --log Jul22_2025.log
    python correlate_SAXS_and_ptycho.py --scan 323 --tolerance 0.1
    python correlate_SAXS_and_ptycho.py --scan 344 --tolerance 0.05 --verbose

Examples:
    # Basic correlation for scan 323
    python correlate_SAXS_and_ptycho.py --scan 323
    
    # With custom tolerance and verbose output
    python correlate_SAXS_and_ptycho.py --scan 323 --tolerance 0.1 --verbose
    
    # Using a different log file
    python correlate_SAXS_and_ptycho.py --scan 344 --log /path/to/other.log
"""

import argparse
import re
import sys
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import colors
from PIL import Image
from scipy import ndimage
from scipy.signal import find_peaks
from scipy.ndimage import maximum_filter, label
import scipy.constants as const


@dataclass
class PtychoEntry:
    """Data structure for ptychography entries."""
    scan_number: int
    line_number: str
    scan_point: str
    phi_angle: float
    detector_filename: str
    log_line: int


@dataclass
class SAXSEntry:
    """Data structure for SAXS entries."""
    scan_number: int
    phi_angles: List[float]  # All phi angles in the scan
    detector_filename: str
    log_line: int
    header_line: int  # Line number of the "#H phi QDS1 QDS2 QDS3" header


@dataclass
class SAXSMeasurement:
    """Data structure for individual SAXS measurements at specific phi angles."""
    scan_number: int
    phi_angle: float
    phi_index: int  # Index in the phi_angles list
    detector_filename: str
    log_line: int


class LogParser:
    """Parser for the experimental log file."""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.ptycho_entries: List[PtychoEntry] = []
        self.saxs_entries: List[SAXSEntry] = []
    
    def parse_log(self) -> None:
        """Parse the entire log file to extract SAXS and ptychography entries."""
        print(f"Parsing log file: {self.log_file}")
        
        with open(self.log_file, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for ptychography detector filename entries (support SZC3_3D_ and ZC8_3D_, etc.)
            if line.startswith('#I detector_filename') and re.search(r'\bS?ZC\d+_3D_', line):
                ptycho_entry = self._parse_ptycho_entry(lines, i)
                if ptycho_entry:
                    self.ptycho_entries.append(ptycho_entry)
            
            # Look for SAXS detector filename entries (keep for legacy logs)
            elif line.startswith('#I detector_filename') and ('SZC3SAXS' in line or 'SZC8SAXS' in line):
                saxs_entry = self._parse_saxs_entry(lines, i)
                if saxs_entry:
                    self.saxs_entries.append(saxs_entry)
            
            i += 1
        
        print(f"Found {len(self.ptycho_entries)} ptychography entries")
        print(f"Found {len(self.saxs_entries)} SAXS entries")
    
    def _parse_ptycho_entry(self, lines: List[str], start_idx: int) -> Optional[PtychoEntry]:
        """Parse a ptychography entry starting from the detector_filename line."""
        detector_line = lines[start_idx].strip()
        
        # Extract filename components: e.g., SZC3_3D_323_00031_00071.tif or ZC8_3D_566_00001_00721.tif
        filename_match = re.search(r'(?:S?ZC\d+)_3D_(\d+)_(\d+)_(\d+)\.tif', detector_line)
        if not filename_match:
            return None
        
        scan_number = int(filename_match.group(1))
        line_number = filename_match.group(2)
        scan_point = filename_match.group(3)
        detector_filename = filename_match.group(0)
        
        # Look for phi angle in the next few lines
        phi_angle = None
        for i in range(start_idx + 1, min(start_idx + 10, len(lines))):
            line = lines[i].strip()
            if line.startswith('#I phi ='):
                phi_match = re.search(r'#I phi =\s+([-+]?\d*\.?\d+)', line)
                if phi_match:
                    phi_angle = float(phi_match.group(1))
                    break
        
        if phi_angle is None:
            print(f"Warning: Could not find phi angle for {detector_filename}")
            return None
        
        return PtychoEntry(
            scan_number=scan_number,
            line_number=line_number,
            scan_point=scan_point,
            phi_angle=phi_angle,
            detector_filename=detector_filename,
            log_line=start_idx + 1
        )
    
    def _parse_saxs_entry(self, lines: List[str], start_idx: int) -> Optional[SAXSEntry]:
        """Parse a SAXS entry starting from the detector_filename line."""
        detector_line = lines[start_idx].strip()
        
        # Extract filename components: SZC3SAXS344_00001_00721.tif
        filename_match = re.search(r'SZC3SAXS(\d+)_(\d+)_(\d+)\.tif', detector_line)
        if not filename_match:
            return None
        
        scan_number = int(filename_match.group(1))
        detector_filename = filename_match.group(0)
        
        # Look backwards for the header line "#H phi QDS1 QDS2 QDS3"
        header_line_idx = None
        phi_angles = []
        
        for i in range(start_idx - 1, max(start_idx - 2000, 0), -1):  # Look further back
            line = lines[i].strip()
            
            # Found the header line
            if line.startswith('#H') and 'phi' in line and 'QDS1' in line:
                header_line_idx = i
                break
        
        if header_line_idx is None:
            print(f"Warning: Could not find header line for {detector_filename}")
            return None
        
        # Extract all phi angles from header line to detector filename line
        for i in range(header_line_idx + 1, start_idx):
            line = lines[i].strip()
            
            # Skip empty lines and comment lines
            if not line or line.startswith('#'):
                continue
            
            # Look for data lines with 4 columns (phi, QDS1, QDS2, QDS3)
            parts = line.split()
            if len(parts) == 4:
                try:
                    phi_angle = float(parts[0])
                    phi_angles.append(phi_angle)
                except ValueError:
                    continue
        
        if not phi_angles:
            print(f"Warning: Could not find any phi angles for {detector_filename}")
            return None
        
        return SAXSEntry(
            scan_number=scan_number,
            phi_angles=phi_angles,
            detector_filename=detector_filename,
            log_line=start_idx + 1,
            header_line=header_line_idx + 1
        )
    
    def parse_saxs_by_line_range(self, start_line: int, end_line: int, saxs_scan: int) -> List[SAXSEntry]:
        """Parse SAXS phi angles by explicit raw line number range.

        Parameters:
            start_line: 1-based starting line number in the log file (inclusive)
            end_line: 1-based ending line number in the log file (inclusive)
            saxs_scan: SAXS scan number to associate with these angles

        Returns:
            List containing one SAXSEntry with collected phi angles, or empty if none found.
        """
        print(f"Parsing SAXS by line range: {start_line}-{end_line} for scan {saxs_scan}")
        with open(self.log_file, 'r') as f:
            lines = f.readlines()
        n_lines = len(lines)
        if start_line < 1 or end_line < start_line or start_line > n_lines:
            print("Warning: Invalid SAXS line range specified")
            return []
        end_line = min(end_line, n_lines)
        segment = [ln.strip() for ln in lines[start_line-1:end_line]]

        # Find header line within the segment
        header_rel_idx = None
        for j in range(len(segment)):
            ln = segment[j]
            if ln.startswith('#H') and 'phi' in ln and 'QDS1' in ln:
                header_rel_idx = j
                break
        if header_rel_idx is None:
            print("Warning: Could not find SAXS header ('#H phi QDS1 QDS2 QDS3') in the specified range")
            return []

        # Collect phi angles from data lines after header until comments or end of segment
        phi_angles: List[float] = []
        for ln in segment[header_rel_idx+1:]:
            if not ln or ln.startswith('#'):
                continue
            parts = ln.split()
            if len(parts) == 4:
                try:
                    phi_angles.append(float(parts[0]))
                except ValueError:
                    continue

        if not phi_angles:
            print("Warning: No SAXS phi angles found in the specified range")
            return []

        entry = SAXSEntry(
            scan_number=saxs_scan,
            phi_angles=phi_angles,
            detector_filename=f"combined_range_lines_{start_line}_{end_line}.tif",
            log_line=start_line,
            header_line=start_line + header_rel_idx
        )
        print(f"Collected {len(phi_angles)} SAXS phi angles from lines {start_line}-{end_line}")
        return [entry]
    
    def filter_by_scan(self, scan_number: int) -> Tuple[List[PtychoEntry], List[SAXSEntry]]:
        """Filter entries by scan number (same scan for both SAXS and ptycho)."""
        ptycho_filtered = [entry for entry in self.ptycho_entries if entry.scan_number == scan_number]
        saxs_filtered = [entry for entry in self.saxs_entries if entry.scan_number == scan_number]
        
        return ptycho_filtered, saxs_filtered
    
    def filter_by_separate_scans(self, ptycho_scan: int, saxs_scan: int) -> Tuple[List[PtychoEntry], List[SAXSEntry]]:
        """Filter entries by separate scan numbers for ptycho and SAXS."""
        ptycho_filtered = [entry for entry in self.ptycho_entries if entry.scan_number == ptycho_scan]
        saxs_filtered = [entry for entry in self.saxs_entries if entry.scan_number == saxs_scan]
        
        return ptycho_filtered, saxs_filtered


class CorrelationMatcher:
    """Matches SAXS and ptychography entries based on phi angles."""
    
    def __init__(self, tolerance: float = 0.01):
        self.tolerance = tolerance
    
    def create_saxs_measurements(self, saxs_entries: List[SAXSEntry]) -> List[SAXSMeasurement]:
        """Convert SAXS entries to individual measurements for each phi angle."""
        measurements = []
        
        for saxs_entry in saxs_entries:
            for idx, phi_angle in enumerate(saxs_entry.phi_angles):
                measurement = SAXSMeasurement(
                    scan_number=saxs_entry.scan_number,
                    phi_angle=phi_angle,
                    phi_index=idx,
                    detector_filename=saxs_entry.detector_filename,
                    log_line=saxs_entry.log_line
                )
                measurements.append(measurement)
        
        return measurements
    
    def find_matches(self, ptycho_entries: List[PtychoEntry], 
                    saxs_entries: List[SAXSEntry]) -> List[Tuple[PtychoEntry, SAXSMeasurement, float]]:
        """
        Find matching pairs based on phi angles.
        
        Returns:
            List of tuples (ptycho_entry, saxs_measurement, phi_difference)
        """
        # Convert SAXS entries to individual measurements
        saxs_measurements = self.create_saxs_measurements(saxs_entries)
        
        matches = []
        
        for ptycho in ptycho_entries:
            best_match = None
            best_diff = float('inf')
            
            for saxs_measurement in saxs_measurements:
                diff = abs(ptycho.phi_angle - saxs_measurement.phi_angle)
                if diff <= self.tolerance and diff < best_diff:
                    best_match = saxs_measurement
                    best_diff = diff
            
            if best_match:
                matches.append((ptycho, best_match, best_diff))
        
        # Sort by phi angle difference
        matches.sort(key=lambda x: x[2])
        
        return matches


def print_results(matches: List[Tuple[PtychoEntry, SAXSEntry, float]], 
                 scan_number: int) -> None:
    """Print the correlation results in a formatted table."""
    print(f"\n=== Correlation Results for Scan {scan_number} ===")
    print(f"Found {len(matches)} matching pairs")
    print()
    
    if not matches:
        print("No matches found.")
        return
    
    # Print header
    print(f"{'Phi Diff':<10} {'Ptycho Phi':<12} {'SAXS Phi':<12} {'Ptycho File':<35} {'SAXS File':<30}")
    print("-" * 110)
    
    # Print matches
    for ptycho, saxs, diff in matches:
        ptycho_file = ptycho.detector_filename
        saxs_file = saxs.detector_filename
        
        print(f"{diff:<10.4f} {ptycho.phi_angle:<12.4f} {saxs.phi_angle:<12.4f} {ptycho_file:<35} {saxs_file:<30}")
    
    print()
    print(f"Statistics:")
    print(f"  Total matches: {len(matches)}")
    print(f"  Average phi difference: {sum(m[2] for m in matches) / len(matches):.4f}")
    print(f"  Max phi difference: {max(m[2] for m in matches):.4f}")
    print(f"  Min phi difference: {min(m[2] for m in matches):.4f}")


def print_results_cross_scan(matches: List[Tuple[PtychoEntry, SAXSMeasurement, float]], 
                            ptycho_scan: int, saxs_scan: int) -> None:
    """Print the correlation results for cross-scan matching."""
    print(f"\n=== Correlation Results: Ptychography Scan {ptycho_scan} vs SAXS Scan {saxs_scan} ===")
    print(f"Found {len(matches)} matching pairs")
    print()
    
    if not matches:
        print("No matches found.")
        return
    
    # Print header
    print(f"{'Phi Diff':<10} {'Ptycho Phi':<12} {'SAXS Phi':<12} {'SAXS Index':<12} {'Ptycho File':<35} {'SAXS File':<30}")
    print("-" * 130)
    
    # Print matches
    for ptycho, saxs_measurement, diff in matches:
        ptycho_file = ptycho.detector_filename
        saxs_file = saxs_measurement.detector_filename
        
        print(f"{diff:<10.4f} {ptycho.phi_angle:<12.4f} {saxs_measurement.phi_angle:<12.4f} {saxs_measurement.phi_index:<12} {ptycho_file:<35} {saxs_file:<30}")
    
    print()
    print(f"Statistics:")
    print(f"  Total matches: {len(matches)}")
    print(f"  Average phi difference: {sum(m[2] for m in matches) / len(matches):.4f}")
    print(f"  Max phi difference: {max(m[2] for m in matches):.4f}")
    print(f"  Min phi difference: {min(m[2] for m in matches):.4f}")


def export_results(matches: List[Tuple[PtychoEntry, SAXSEntry, float]], 
                  output_file: str, scan_number: int) -> None:
    """Export correlation results to a CSV file."""
    import csv
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow([
            'Scan_Number', 'Phi_Difference', 'Ptycho_Phi', 'SAXS_Phi',
            'Ptycho_Filename', 'SAXS_Filename', 'Ptycho_Line_Number', 
            'Ptycho_Scan_Point', 'Ptycho_Log_Line', 'SAXS_Log_Line'
        ])
        
        # Write data
        for ptycho, saxs, diff in matches:
            writer.writerow([
                scan_number, diff, ptycho.phi_angle, saxs.phi_angle,
                ptycho.detector_filename, saxs.detector_filename,
                ptycho.line_number, ptycho.scan_point,
                ptycho.log_line, saxs.log_line
            ])
    
    print(f"Results exported to: {output_file}")


def export_results_cross_scan(matches: List[Tuple[PtychoEntry, SAXSMeasurement, float]], 
                             output_file: str, ptycho_scan: int, saxs_scan: int) -> None:
    """Export cross-scan correlation results to a CSV file."""
    import csv
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow([
            'Ptycho_Scan', 'SAXS_Scan', 'Phi_Difference', 'Ptycho_Phi', 'SAXS_Phi', 'SAXS_Phi_Index',
            'Ptycho_Filename', 'SAXS_Filename', 'Ptycho_Line_Number', 
            'Ptycho_Scan_Point', 'Ptycho_Log_Line', 'SAXS_Log_Line'
        ])
        
        # Write data
        for ptycho, saxs_measurement, diff in matches:
            writer.writerow([
                ptycho_scan, saxs_scan, diff, ptycho.phi_angle, saxs_measurement.phi_angle, saxs_measurement.phi_index,
                ptycho.detector_filename, saxs_measurement.detector_filename,
                ptycho.line_number, ptycho.scan_point,
                ptycho.log_line, saxs_measurement.log_line
            ])
    
    print(f"Results exported to: {output_file}")


def list_available_scans(log_file: str) -> None:
    """List all available scan numbers in the log file."""
    parser = LogParser(log_file)
    parser.parse_log()
    
    ptycho_scans = set(entry.scan_number for entry in parser.ptycho_entries)
    saxs_scans = set(entry.scan_number for entry in parser.saxs_entries)
    
    print(f"\nAvailable scans in {log_file}:")
    print(f"Ptychography scans: {sorted(ptycho_scans)}")
    print(f"SAXS scans: {sorted(saxs_scans)}")
    print(f"Common scans: {sorted(ptycho_scans.intersection(saxs_scans))}")


def load_ptycho_reconstruction(scan_number: int) -> Optional[np.ndarray]:
    """Load ptychography reconstruction TIFF file."""
    ptycho_path = f"/net/micdata/data2/12IDC/2025_Jul/ptychi_recons/S{scan_number:04d}/Ndp128_LSQML_s1000_gaussian_p5_cp_mm_opr2_ic_pc1_g_ul2/object_ph/object_ph_Niter1000.tiff"
    
    try:
        # Load TIFF image
        img = Image.open(ptycho_path)
        ptycho_data = np.array(img)
        print(f"Loaded ptychography reconstruction: {ptycho_path}")
        print(f"Shape: {ptycho_data.shape}, dtype: {ptycho_data.dtype}")
        return ptycho_data
    except FileNotFoundError:
        print(f"Warning: Ptychography reconstruction not found at {ptycho_path}")
        return None
    except Exception as e:
        print(f"Error loading ptychography reconstruction: {e}")
        return None


def load_ptycho_h5_reconstruction(scan_number: int) -> Optional[Tuple[np.ndarray, float]]:
    """
    Load ptychography reconstruction from HDF5 file.
    
    Parameters:
    -----------
    scan_number : int
        Scan number for the ptychography reconstruction
        
    Returns:
    --------
    ptycho_data : np.ndarray or None
        Complex-valued ptychography object
    pixel_size_nm : float or None
        Pixel size in nanometers from the HDF5 file
    """
    ptycho_path = f"/net/micdata/data2/12IDC/2025_Jul/ptychi_recons/S{scan_number:04d}/Ndp128_LSQML_s1000_gaussian_p5_cp_mm_opr2_ic_pc1_g_ul2/recon_Niter1000.h5"
    
    try:
        with h5py.File(ptycho_path, 'r') as f:
            print(f"Loading ptychography HDF5 file: {ptycho_path}")
            
            # Print available keys for debugging
            print(f"Available keys: {list(f.keys())}")
            
            # Load the complex object
            if 'object' not in f:
                print(f"Warning: 'object' key not found in {ptycho_path}")
                return None, None
            
            ptycho_object = f['object'][:]
            print(f"Loaded complex object shape: {ptycho_object.shape}, dtype: {ptycho_object.dtype}")
            
            # Handle multi-dimensional objects - take the first slice if it's 3D
            if ptycho_object.ndim == 3:
                print(f"3D object detected, taking first slice: shape {ptycho_object.shape} -> {ptycho_object[0].shape}")
                ptycho_object = ptycho_object[0]  # Take first slice
            elif ptycho_object.ndim > 3:
                print(f"Warning: {ptycho_object.ndim}D object detected, taking first 2D slice")
                # Flatten extra dimensions to get a 2D slice
                while ptycho_object.ndim > 2:
                    ptycho_object = ptycho_object[0]
            
            # Load pixel size
            if 'obj_pixel_size_m' not in f:
                print(f"Warning: 'obj_pixel_size_m' key not found in {ptycho_path}")
                print("Using default pixel size of 39.3 nm")
                pixel_size_nm = 39.3
            else:
                pixel_size_m = f['obj_pixel_size_m'][()]
                pixel_size_nm = pixel_size_m * 1e9  # Convert from meters to nanometers
                print(f"Loaded pixel size: {pixel_size_m} m = {pixel_size_nm:.3f} nm")
            
            return ptycho_object, pixel_size_nm
            
    except FileNotFoundError:
        print(f"Warning: Ptychography HDF5 file not found at {ptycho_path}")
        return None, None
    except Exception as e:
        print(f"Error loading ptychography HDF5 file: {e}")
        return None, None


def process_complex_ptycho_object(complex_object: np.ndarray, mode: str = 'phase') -> np.ndarray:
    """
    Process complex ptychography object for visualization and analysis.
    
    Parameters:
    -----------
    complex_object : np.ndarray
        Complex-valued ptychography object
    mode : str
        Processing mode: 'phase', 'amplitude', 'intensity', or 'complex_abs'
        'complex' returns the complex object
    Returns:
    --------
    processed_data : np.ndarray
        Processed real-valued data for visualization
    """
    if mode == 'phase':
        return np.angle(complex_object)
    elif mode == 'amplitude':
        return np.abs(complex_object)
    elif mode == 'intensity':
        return np.abs(complex_object)**2
    elif mode == 'complex_abs':
        return np.abs(complex_object)
    elif mode == 'complex':
        return complex_object
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'phase', 'amplitude', 'intensity', 'complex_abs', or 'complex'")


def pad_to_square_ptycho(image: np.ndarray, pad_mode: str = 'constant', pad_value: float = 0.0) -> np.ndarray:
    """
    Pad a rectangular image to make it square using the maximum dimension.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image (can be rectangular)
    pad_mode : str
        Padding mode: 'constant', 'edge', 'reflect', 'symmetric'
    pad_value : float
        Value to use for constant padding
        
    Returns:
    --------
    padded_image : np.ndarray
        Square padded image
    """
    h, w = image.shape
    max_dim = max(h, w)
    
    # Calculate padding needed for each dimension
    pad_h = max_dim - h
    pad_w = max_dim - w
    
    # Split padding symmetrically (center the original image)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    # Create padding specification
    if pad_mode == 'constant':
        padded = np.pad(image, 
                       ((pad_top, pad_bottom), (pad_left, pad_right)), 
                       mode=pad_mode, constant_values=pad_value)
    else:
        padded = np.pad(image, 
                       ((pad_top, pad_bottom), (pad_left, pad_right)), 
                       mode=pad_mode)
    
    print(f"Padded from {image.shape} to {padded.shape} using '{pad_mode}' mode")
    if pad_mode == 'constant':
        print(f"  Padding: top={pad_top}, bottom={pad_bottom}, left={pad_left}, right={pad_right}, value={pad_value}")
    else:
        print(f"  Padding: top={pad_top}, bottom={pad_bottom}, left={pad_left}, right={pad_right}")
    
    return padded


def get_ptycho_data_for_analysis(scan_number: int, use_h5: bool = True, 
                                complex_mode: str = 'amplitude', 
                                pad_to_square: bool = True,
                                padding_mode: str = 'constant') -> Tuple[Optional[np.ndarray], float]:
    """
    Get ptychography data for analysis, with option to use HDF5 or TIFF.
    
    Parameters:
    -----------
    scan_number : int
        Scan number
    use_h5 : bool
        If True, try to load from HDF5 first, fallback to TIFF
        If False, load from TIFF only
    complex_mode : str
        How to process complex data: 'phase', 'amplitude', 'intensity'
    pad_to_square : bool
        If True, pad rectangular images to square (preserves all data, recommended for FFT)
    padding_mode : str
        Padding mode: 'constant' (zeros), 'edge', 'reflect', 'symmetric'
        
    Returns:
    --------
    ptycho_data : np.ndarray or None
        Processed ptychography data
    pixel_size_nm : float
        Pixel size in nanometers
    """
    if use_h5:
        # Try loading from HDF5 first
        complex_object, pixel_size_nm = load_ptycho_h5_reconstruction(scan_number)
        
        if complex_object is not None:
            print(f"Processing complex object using '{complex_mode}' mode...")
            ptycho_data = process_complex_ptycho_object(complex_object, complex_mode)
            print(f"Processed data shape: {ptycho_data.shape}, dtype: {ptycho_data.dtype}")
            print(f"Data range: {ptycho_data.min():.3e} to {ptycho_data.max():.3e}")
            
            # Check for finite values
            finite_mask = np.isfinite(ptycho_data)
            print(f"Finite values: {np.sum(finite_mask)}/{ptycho_data.size} ({100*np.sum(finite_mask)/ptycho_data.size:.1f}%)")
            
            # Pad to square if requested and image is rectangular
            if pad_to_square and ptycho_data.shape[0] != ptycho_data.shape[1]:
                print(f"Rectangular image detected: {ptycho_data.shape}")
                # For phase data, use mean value padding to avoid sharp discontinuities
                if complex_mode == 'phase':
                    pad_value = np.mean(ptycho_data)
                    print(f"Using mean value padding for phase data: {pad_value:.3f}")
                else:
                    pad_value = 0.0
                ptycho_data = pad_to_square_ptycho(ptycho_data, pad_mode=padding_mode, pad_value=pad_value)
                print(f"Padded to square: {ptycho_data.shape}")
            
            return ptycho_data, pixel_size_nm
        else:
            print("HDF5 loading failed, falling back to TIFF...")
    
    # Fallback to TIFF loading
    ptycho_data = load_ptycho_reconstruction(scan_number)
    pixel_size_nm = 39.3  # Default pixel size
    
    if ptycho_data is not None:
        print(f"Using default pixel size: {pixel_size_nm} nm")
        
        # Pad to square if requested and image is rectangular
        if pad_to_square and ptycho_data.shape[0] != ptycho_data.shape[1]:
            print(f"Rectangular TIFF image detected: {ptycho_data.shape}")
            ptycho_data = pad_to_square_ptycho(ptycho_data, pad_mode=padding_mode, pad_value=0.0)
            print(f"Padded to square: {ptycho_data.shape}")
        
        return ptycho_data, pixel_size_nm
    else:
        return None, pixel_size_nm


def load_ptycho_raw_diffraction_sum(scan_number: int, crop_size: int = 256, 
                                   center: tuple = (773, 626)) -> Optional[np.ndarray]:
    """
    Load and sum all ptychography raw diffraction patterns for a given scan.
    
    Parameters:
    -----------
    scan_number : int
        Scan number for the ptychography data
    crop_size : int
        Size of the central crop (crop_size × crop_size)
    center : tuple
        Center coordinates (y, x) for cropping
        
    Returns:
    --------
    summed_diffraction : np.ndarray or None
        Sum of all cropped diffraction patterns
    """
    import glob
    
    # Pattern for finding all diffraction files for this scan
    pattern = f"/net/micdata/data2/12IDC/2025_Jul/ptycho/{scan_number}/SZC3_3D_{scan_number}_*.h5"
    h5_files = sorted(glob.glob(pattern))
    
    if not h5_files:
        print(f"No ptychography diffraction files found for scan {scan_number}")
        print(f"Searched pattern: {pattern}")
        return None
    
    print(f"Found {len(h5_files)} ptychography diffraction files for scan {scan_number}")
    print(f"Files range: {h5_files[0]} to {h5_files[-1]}")
    
    summed_pattern = None
    valid_files = 0
    
    for i, h5_file in enumerate(h5_files):
        try:
            with h5py.File(h5_file, 'r') as f:
                # Print available keys for the first file
                if i == 0:
                    print(f"Available keys in {h5_file}: {list(f.keys())}")
                
                # Common key names for diffraction data
                possible_keys = ['data', 'entry/data/data','detector_data', 'diffraction', 'image', 'measurement']
                data_key = None
                
                for key in possible_keys:
                    if key in f:
                        data_key = key
                        break
                
                if data_key is None:
                    # If none of the common keys found, use the first key
                    keys = list(f.keys())
                    if keys:
                        data_key = keys[0]
                        if i == 0:
                            print(f"Using key '{data_key}' for diffraction data")
                
                if data_key is None:
                    print(f"Warning: No data found in {h5_file}")
                    continue
                
                # Load the diffraction pattern
                pattern_data = f[data_key][:]
                
                if i == 0:
                    print(f"Original diffraction pattern shape: {pattern_data.shape}, dtype: {pattern_data.dtype}")
                    print(f"Original data range: {pattern_data.min():.2e} to {pattern_data.max():.2e}")
                    print(f"Cropping to {crop_size}×{crop_size} around center {center}")
                
                # Crop to central region
                center_y, center_x = center
                half_size = crop_size // 2
                
                y_start = center_y - half_size
                y_end = center_y + half_size
                x_start = center_x - half_size
                x_end = center_x + half_size
                
                # Ensure crop boundaries are within image bounds
                y_start = max(0, y_start)
                y_end = min(pattern_data.shape[0], y_end)
                x_start = max(0, x_start)
                x_end = min(pattern_data.shape[1], x_end)
                
                # Crop the pattern
                cropped_pattern = pattern_data[y_start:y_end, x_start:x_end]
                
                if i == 0:
                    print(f"Cropped pattern shape: {cropped_pattern.shape}")
                    print(f"Crop boundaries: y[{y_start}:{y_end}], x[{x_start}:{x_end}]")
                    print(f"Cropped data range: {cropped_pattern.min():.2e} to {cropped_pattern.max():.2e}")
                    summed_pattern = cropped_pattern.astype(np.float64)
                else:
                    summed_pattern += cropped_pattern.astype(np.float64)
                
                valid_files += 1
                
                # Progress indicator for large datasets
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(h5_files)} files...")
                    
        except Exception as e:
            print(f"Error loading {h5_file}: {e}")
            continue
    
    if summed_pattern is not None:
        print(f"Successfully summed {valid_files} diffraction patterns")
        print(f"Summed pattern shape: {summed_pattern.shape}")
        print(f"Summed data range: {summed_pattern.min():.2e} to {summed_pattern.max():.2e}")
        print(f"Average counts per pattern: {summed_pattern.sum() / valid_files:.2e}")
        return summed_pattern
    else:
        print(f"Failed to load any diffraction patterns for scan {scan_number}")
        return None


def load_saxs_data(scan_number: int, phi_index: int, combined_prefix: str = 'SZC3SAXS') -> Optional[np.ndarray]:
    """Load SAXS data from HDF5 file at specific phi index.

    Parameters:
        scan_number: SAXS scan number
        phi_index: index into combined_data (0-based)
        combined_prefix: prefix in combined filename (e.g., 'SZC3SAXS' or 'SZC8SAXS')
    """
    saxs_path = f"/net/micdata/data2/12IDC/2025_Jul/misc/combined_{combined_prefix}{scan_number}_00001.h5"
    
    try:
        with h5py.File(saxs_path, 'r') as f:
            if 'combined_data' not in f:
                print(f"Warning: 'combined_data' key not found in {saxs_path}")
                return None
            
            combined_data = f['combined_data'][:]
            print(f"Loaded SAXS data: {saxs_path}")
            print(f"Combined data shape: {combined_data.shape}")
            
            if phi_index >= combined_data.shape[0]:
                print(f"Warning: phi_index {phi_index} is out of range (max: {combined_data.shape[0]-1})")
                return None
            
            saxs_data = combined_data[phi_index]
            print(f"SAXS data at index {phi_index} shape: {saxs_data.shape}, dtype: {saxs_data.dtype}")
            return saxs_data
            
    except FileNotFoundError:
        print(f"Warning: SAXS data file not found at {saxs_path}")
        return None
    except Exception as e:
        print(f"Error loading SAXS data: {e}")
        return None


def pixel_to_q_ptycho(r_pixels: np.ndarray, pixel_size_nm: float = 39.3, 
                      image_size: int = 128) -> np.ndarray:
    """
    Convert ptychography FFT pixel coordinates to q-space.
    
    Parameters:
    -----------
    r_pixels : np.ndarray
        Radial distances in pixels from FFT center
    pixel_size_nm : float
        Real space pixel size in nanometers (default: 39.3 nm)
    image_size : int
        Size of the image in pixels (assumed square, default: 128)
        
    Returns:
    --------
    q : np.ndarray
        q values in nm^-1
    """
    # For FFT of real space data with pixel size 'a':
    # The reciprocal space pixel size is 2π/(N*a) where N is image size
    # For a pixel at distance r from center: q = r * 2π/(N*a)
    
    # Convert pixel size to meters
    pixel_size_m = pixel_size_nm * 1e-9
    
    # Calculate field of view
    field_of_view_m = image_size * pixel_size_m
    
    # Calculate q in m^-1
    # Use field of view for more accurate calibration
    q = (2 * np.pi * r_pixels) / field_of_view_m
    
    # Convert to nm^-1 for easier interpretation
    q_nm_inv = q * 1e-9  # Convert from m^-1 to nm^-1
    
    return q_nm_inv


def pixel_to_q_saxs(r_pixels: np.ndarray, energy_kev: float = 15.7, 
                   distance_m: float = 11.0, pixel_size_um: float = 172.0) -> np.ndarray:
    """
    Convert SAXS detector pixel coordinates to q-space.
    
    Parameters:
    -----------
    r_pixels : np.ndarray
        Radial distances in pixels from beam center
    energy_kev : float
        X-ray energy in keV (default: 15.7 keV)
    distance_m : float
        Sample to detector distance in meters (default: 11.0 m)
    pixel_size_um : float
        Detector pixel size in micrometers (default: 172.0 μm)
        
    Returns:
    --------
    q : np.ndarray
        q values in nm^-1
    """
    # Convert energy to wavelength
    # E = hc/λ, so λ = hc/E
    h = const.h  # Planck constant in J⋅s
    c = const.c  # Speed of light in m/s
    energy_j = energy_kev * 1000 * const.eV  # Convert keV to J
    
    wavelength_m = h * c / energy_j  # Wavelength in meters
    
    # Convert pixel size to meters
    pixel_size_m = pixel_size_um * 1e-6
    
    # Calculate scattering angle
    # tan(2θ) = (r_pixels * pixel_size) / distance
    # For small angles: 2θ ≈ (r_pixels * pixel_size) / distance
    two_theta = (r_pixels * pixel_size_m) / distance_m
    theta = two_theta / 2
    
    # Calculate q = 4π sin(θ) / λ
    q = 4 * np.pi * np.sin(theta) / wavelength_m  # q in m^-1
    
    # Convert to nm^-1 for easier interpretation
    q_nm_inv = q * 1e-9  # Convert from m^-1 to nm^-1
    
    return q_nm_inv


def get_experimental_parameters():
    """
    Get experimental parameters for q-space conversion.
    
    Returns:
    --------
    dict : Experimental parameters
    """
    return {
        'ptycho': {
            'pixel_size_nm': 39.4,  # Real space pixel size in nm
            #'description': 'Ptychography real space pixel size'
        },
        'saxs': {
            'energy_kev': 15.7,      # X-ray energy in keV
            'distance_m': 11.0,      # Sample to detector distance in m
            'pixel_size_um': 172.0,  # Detector pixel size in μm
            #'description': 'SAXS experimental parameters'
        }
    }


def diagnose_q_space_mismatch(ptycho_data: np.ndarray, saxs_data: np.ndarray, 
                             ptycho_pixel_size_nm: float, 
                             current_scaling: float = 1.0) -> dict:
    """
    Diagnose potential causes of q-space mismatch between ptychography and SAXS.
    
    Parameters:
    -----------
    ptycho_data : np.ndarray
        Ptychography reconstruction data
    saxs_data : np.ndarray  
        SAXS data
    ptycho_pixel_size_nm : float
        Current ptychography pixel size
    current_scaling : float
        Current scaling factor being used
        
    Returns:
    --------
    dict : Diagnostic information and suggestions
    """
    import scipy.constants as const
    
    exp_params = get_experimental_parameters()
    
    # Calculate theoretical q-ranges
    print(f"\n{'='*60}")
    print("Q-SPACE CALIBRATION DIAGNOSTIC")
    print(f"{'='*60}")
    
    # Wavelength calculation
    h = const.h
    c = const.c
    energy_j = exp_params['saxs']['energy_kev'] * 1000 * const.eV
    wavelength_m = h * c / energy_j
    wavelength_nm = wavelength_m * 1e9
    
    print(f"X-ray wavelength: {wavelength_nm:.4f} nm")
    print(f"Current ptycho pixel size: {ptycho_pixel_size_nm:.3f} nm")
    print(f"Current scaling factor: {current_scaling:.3f}")
    
    # Ptychography theoretical q_max
    pixel_size_m = ptycho_pixel_size_nm * 1e-9
    image_size = max(ptycho_data.shape)
    
    # Maximum meaningful q in FFT (Nyquist limit)
    q_max_nyquist = np.pi / pixel_size_m * 1e-9  # nm^-1
    
    # Current formula q_max
    q_max_current = (2 * np.pi * (image_size/2)) / (image_size * pixel_size_m) * 1e-9
    
    print(f"Ptycho image size: {image_size}")
    print(f"Theoretical q_max (Nyquist): {q_max_nyquist:.6f} nm⁻¹")
    print(f"Current formula q_max: {q_max_current:.6f} nm⁻¹")
    
    # SAXS theoretical q_max
    saxs_pixel_size_m = exp_params['saxs']['pixel_size_um'] * 1e-6
    distance_m = exp_params['saxs']['distance_m']
    detector_size = max(saxs_data.shape)
    
    max_detector_radius = (detector_size / 2) * saxs_pixel_size_m
    max_2theta = np.arctan(max_detector_radius / distance_m)
    max_theta = max_2theta / 2
    q_max_saxs = 4 * np.pi * np.sin(max_theta) / wavelength_m * 1e-9
    
    print(f"SAXS detector size: {detector_size}")
    print(f"SAXS q_max: {q_max_saxs:.6f} nm⁻¹")
    
    # Calculate ratios
    ratio_current = q_max_saxs / q_max_current
    ratio_nyquist = q_max_saxs / q_max_nyquist
    
    print(f"\nQ-RANGE RATIOS:")
    print(f"SAXS/Ptycho (current): {ratio_current:.3f}")
    print(f"SAXS/Ptycho (Nyquist): {ratio_nyquist:.3f}")
    
    # Analyze scaling factor
    if abs(current_scaling - 1.0) > 0.01:
        print(f"\nSCALING FACTOR ANALYSIS:")
        print(f"Required scaling: {current_scaling:.3f}")
        
        # What this suggests about pixel size
        corrected_pixel_size = ptycho_pixel_size_nm / current_scaling
        print(f"Suggested true pixel size: {corrected_pixel_size:.3f} nm")
        
        # What this suggests about other parameters
        print(f"Alternative explanations:")
        print(f"  - SAXS distance error: {distance_m * current_scaling:.2f} m instead of {distance_m} m")
        print(f"  - SAXS energy error: {exp_params['saxs']['energy_kev'] / current_scaling:.2f} keV instead of {exp_params['saxs']['energy_kev']} keV")
    
    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    recommendations = []
    
    if abs(current_scaling - 1.0) > 0.05:
        recommendations.append("1. Check the obj_pixel_size_m value in your HDF5 file")
        recommendations.append("2. Verify ptychography reconstruction parameters")
        recommendations.append("3. Cross-check with raw diffraction pattern scaling")
    
    if ratio_current > 2.5 or ratio_current < 1.5:
        recommendations.append("4. Verify SAXS experimental geometry (distance, pixel size)")
        recommendations.append("5. Check X-ray energy calibration")
    
    recommendations.append("6. Use a calibration standard with known d-spacings")
    recommendations.append("7. Compare peaks in both techniques for validation")
    
    for rec in recommendations:
        print(f"  {rec}")
    
    return {
        'wavelength_nm': wavelength_nm,
        'q_max_ptycho_current': q_max_current,
        'q_max_ptycho_nyquist': q_max_nyquist,
        'q_max_saxs': q_max_saxs,
        'ratio_current': ratio_current,
        'suggested_pixel_size': ptycho_pixel_size_nm / current_scaling if abs(current_scaling - 1.0) > 0.01 else ptycho_pixel_size_nm,
        'recommendations': recommendations
    }


def azimuthal_average(image: np.ndarray, center: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate azimuthal average of 2D image.
    
    Parameters:
    -----------
    image : np.ndarray
        2D image to be azimuthally averaged
    center : tuple, optional
        Center coordinates (y, x). If None, uses image center.
        
    Returns:
    --------
    r : np.ndarray
        Radial distances from center
    intensity : np.ndarray
        Azimuthally averaged intensities
    """
    # Get image dimensions
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    
    # Use image center if not provided
    if center is None:
        center = (image.shape[0] // 2, image.shape[1] // 2)
    
    # Calculate radial distance from center
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    
    # Get maximum radius
    r_max = int(np.max(r))
    
    # Initialize arrays for radial profile
    radial_profile = np.zeros(r_max)
    radial_counts = np.zeros(r_max)
    
    # Calculate azimuthal average
    for radius in range(r_max):
        # Create mask for current radius bin
        mask = (r >= radius) & (r < radius + 1)
        
        if np.any(mask):
            # Average intensity at this radius
            radial_profile[radius] = np.mean(image[mask])
            radial_counts[radius] = np.sum(mask)
    
    # Create radial distance array
    r_array = np.arange(r_max)
    
    return r_array, radial_profile


def find_beam_center(image: np.ndarray, method: str = 'center_of_mass') -> Tuple[int, int]:
    """
    Find the beam center in SAXS data.
    
    Parameters:
    -----------
    image : np.ndarray
        2D SAXS image
    method : str
        Method to find center ('center_of_mass', 'max_intensity', 'geometric')
        
    Returns:
    --------
    center : tuple
        Center coordinates (y, x)
    """
    if method == 'center_of_mass':
        # Use center of mass for high-intensity regions
        threshold = np.percentile(image, 95)  # Use top 5% of intensities
        binary_image = image > threshold
        center = ndimage.center_of_mass(binary_image)
        return (int(center[0]), int(center[1]))
    
    elif method == 'max_intensity':
        # Find maximum intensity position
        max_pos = np.unravel_index(np.argmax(image), image.shape)
        return max_pos
    
    elif method == 'geometric':
        # Use geometric center
        return (image.shape[0] // 2, image.shape[1] // 2)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def plot_radial_profile(r: np.ndarray, intensity: np.ndarray, 
                       scan_number: int, phi_index: int, phi_angle: float,
                       ax: plt.Axes = None) -> plt.Axes:
    """
    Plot radial profile of SAXS data.
    
    Parameters:
    -----------
    r : np.ndarray
        Radial distances
    intensity : np.ndarray
        Azimuthally averaged intensities
    scan_number : int
        SAXS scan number
    phi_index : int
        Phi angle index
    phi_angle : float
        Phi angle in degrees
    ax : plt.Axes, optional
        Matplotlib axes to plot on
        
    Returns:
    --------
    ax : plt.Axes
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Plot radial profile
    ax.plot(r, intensity, 'b-', linewidth=2, label='Linear scale')
    ax.set_xlabel('Radial distance (pixels)')
    ax.set_ylabel('Azimuthally averaged intensity')
    ax.set_title(f'SAXS Radial Profile\nScan {scan_number}, φ={phi_angle:.3f}°, Index {phi_index}')
    ax.grid(True, alpha=0.3)
    
    # Add log scale plot
    ax2 = ax.twinx()
    ax2.semilogy(r, intensity + 1, 'r--', linewidth=1.5, alpha=0.7, label='Log scale')
    ax2.set_ylabel('Log(Intensity + 1)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Add legends
    ax.legend(loc='upper right')
    ax2.legend(loc='upper left')
    
    return ax


def convert_to_q_space_image(image: np.ndarray, center: Tuple[int, int], 
                            conversion_func, **conversion_params) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert 2D image from pixel space to q-space coordinates.
    
    Parameters:
    -----------
    image : np.ndarray
        2D image in pixel space
    center : tuple
        Center coordinates (y, x)
    conversion_func : function
        Function to convert pixels to q-space
    **conversion_params : dict
        Parameters for the conversion function
        
    Returns:
    --------
    qx_grid : np.ndarray
        2D grid of qx values
    qy_grid : np.ndarray
        2D grid of qy values
    q_image : np.ndarray
        Image with q-space coordinates
    """
    # Create pixel coordinate grids relative to center
    y_pixels = np.arange(image.shape[0]) - center[0]
    x_pixels = np.arange(image.shape[1]) - center[1]
    X_pixels, Y_pixels = np.meshgrid(x_pixels, y_pixels)
    
    # Calculate radial distance for each pixel
    R_pixels = np.sqrt(X_pixels**2 + Y_pixels**2)
    
    # Convert radial distances to q-space
    q_values = conversion_func(R_pixels.flatten(), **conversion_params).reshape(R_pixels.shape)
    
    # Calculate qx and qy components
    # For small angles: qx = q * (x_pixels / r_pixels), qy = q * (y_pixels / r_pixels)
    # Handle division by zero at center
    qx_grid = np.zeros_like(q_values)
    qy_grid = np.zeros_like(q_values)
    
    mask = R_pixels > 0
    qx_grid[mask] = q_values[mask] * (X_pixels[mask] / R_pixels[mask])
    qy_grid[mask] = q_values[mask] * (Y_pixels[mask] / R_pixels[mask])
    
    return qx_grid, qy_grid, image


def interpolate_to_common_q_grid(qx1: np.ndarray, qy1: np.ndarray, image1: np.ndarray,
                                qx2: np.ndarray, qy2: np.ndarray, image2: np.ndarray,
                                q_range: Tuple[float, float], q_resolution: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate two images to a common q-space grid.
    
    Parameters:
    -----------
    qx1, qy1 : np.ndarray
        Q-space coordinates for first image
    image1 : np.ndarray
        First image
    qx2, qy2 : np.ndarray
        Q-space coordinates for second image  
    image2 : np.ndarray
        Second image
    q_range : tuple
        (q_min, q_max) range for the common grid
    q_resolution : int
        Resolution of the common grid
        
    Returns:
    --------
    qx_common : np.ndarray
        Common qx grid
    qy_common : np.ndarray
        Common qy grid
    image1_interp : np.ndarray
        First image interpolated to common grid
    image2_interp : np.ndarray
        Second image interpolated to common grid
    """
    from scipy.interpolate import griddata
    
    q_min, q_max = q_range
    q_axis = np.linspace(q_min, q_max, q_resolution)
    qx_common, qy_common = np.meshgrid(q_axis, q_axis)
    
    # Flatten coordinates for interpolation
    points1 = np.column_stack([qx1.flatten(), qy1.flatten()])
    values1 = image1.flatten()
    
    points2 = np.column_stack([qx2.flatten(), qy2.flatten()])
    values2 = image2.flatten()
    
    # Remove NaN and infinite values
    mask1 = np.isfinite(values1) & np.isfinite(points1[:, 0]) & np.isfinite(points1[:, 1])
    mask2 = np.isfinite(values2) & np.isfinite(points2[:, 0]) & np.isfinite(points2[:, 1])
    
    # Interpolate to common grid
    image1_interp = griddata(points1[mask1], values1[mask1], 
                            (qx_common, qy_common), method='linear', fill_value=0)
    image2_interp = griddata(points2[mask2], values2[mask2], 
                            (qx_common, qy_common), method='linear', fill_value=0)
    
    return qx_common, qy_common, image1_interp, image2_interp


def plot_correlation_results(ptycho_data: np.ndarray, saxs_data: np.ndarray, 
                           ptycho_scan: int, saxs_scan: int, phi_index: int,
                           ptycho_phi: float, saxs_phi: float, 
                           ptycho_pixel_size_nm: Optional[float] = None,
                           ptycho_q_scaling: float = 1.0) -> None:
    """Plot ptychography reconstruction, its FFT, corresponding SAXS data, and radial profiles."""
    
    # Get experimental parameters
    exp_params = get_experimental_parameters()
    
    # Use provided pixel size if available, otherwise use default
    if ptycho_pixel_size_nm is not None:
        actual_ptycho_pixel_size = ptycho_pixel_size_nm
        print(f"Using actual pixel size from HDF5: {actual_ptycho_pixel_size:.3f} nm")
    else:
        actual_ptycho_pixel_size = exp_params['ptycho']['pixel_size_nm']
        print(f"Using default pixel size: {actual_ptycho_pixel_size:.3f} nm")
    
    # Calculate FFT of ptychography data
    ptycho_fft = np.fft.fft2(ptycho_data)
    ptycho_fft_shifted = np.fft.fftshift(ptycho_fft)
    ptycho_fft_magnitude = np.abs(ptycho_fft_shifted)**2
    
    # Find beam center for SAXS data and calculate radial profile
    print(f"Calculating azimuthal average for SAXS data...")
    #saxs_center = find_beam_center(saxs_data, method='center_of_mass')
    saxs_center=(128,127)
    r_saxs, intensity_saxs = azimuthal_average(saxs_data, center=saxs_center)
    print(f"SAXS beam center found at: {saxs_center}")
    
    # Calculate radial profile for FFT as well
    print(f"Calculating azimuthal average for ptychography FFT...")
    ptycho_center = find_beam_center(ptycho_fft_magnitude, method='geometric')  # FFT center is usually geometric
    r_ptycho, intensity_ptycho = azimuthal_average(ptycho_fft_magnitude, center=ptycho_center)
    print(f"Ptycho FFT center: {ptycho_center}")
    
    # Convert to q-space with scaling applied at the fundamental level
    print(f"Converting to q-space...")
    q_saxs_original = pixel_to_q_saxs(r_saxs, **exp_params['saxs'])
    # For square images, just use the image dimension
    image_size = ptycho_data.shape[0]  # Should be square after cropping
    
    # Create scaled conversion functions that will be used for ALL analysis
    def pixel_to_q_ptycho_scaled(r_pixels, **kwargs):
        q_original = pixel_to_q_ptycho(r_pixels, **kwargs)
        return q_original * ptycho_q_scaling
    
    def pixel_to_q_saxs_scaled(r_pixels, **kwargs):
        return pixel_to_q_saxs(r_pixels, **kwargs)  # No SAXS scaling for now
    
    # Apply scaled conversions to 1D radial profiles
    q_ptycho = pixel_to_q_ptycho_scaled(r_ptycho, 
                                        pixel_size_nm=actual_ptycho_pixel_size,
                                        image_size=image_size)
    q_saxs = pixel_to_q_saxs_scaled(r_saxs, **exp_params['saxs'])
    
    print(f"Q-SPACE SCALING APPLIED:")
    print(f"  Ptycho scaling factor: {ptycho_q_scaling:.3f}")
    print(f"  Ptycho q_max (after scaling): {q_ptycho.max():.6f} nm⁻¹")
    print(f"  SAXS q_max: {q_saxs.max():.6f} nm⁻¹")
    print(f"  Ratio (SAXS/Ptycho): {q_saxs.max() / q_ptycho.max():.6f}")
    
    # Calculate wavelength for reference
    h = const.h
    c = const.c
    energy_j = exp_params['saxs']['energy_kev'] * 1000 * const.eV
    wavelength_nm = (h * c / energy_j) * 1e9  # Convert to nm
    
    print(f"X-ray wavelength: {wavelength_nm:.4f} nm")
    print(f"SAXS q-range: {q_saxs.min():.6f} to {q_saxs.max():.6f} nm^-1")
    print(f"Ptycho FFT q-range: {q_ptycho.min():.6f} to {q_ptycho.max():.6f} nm^-1")
    
    # Convert 2D images to q-space for overlay
    print(f"Converting 2D images to q-space...")
    print(f"Ptycho data shape: {ptycho_data.shape}")
    print(f"Ptycho FFT magnitude shape: {ptycho_fft_magnitude.shape}")
    print(f"SAXS data shape: {saxs_data.shape}")
    print(f"Using image_size: {image_size} for q-space conversion")
    
    # Use the same scaled conversion functions for 2D q-space grids
    qx_ptycho, qy_ptycho, _ = convert_to_q_space_image(
        ptycho_fft_magnitude, ptycho_center, pixel_to_q_ptycho_scaled,
        pixel_size_nm=actual_ptycho_pixel_size,
        image_size=image_size
    )
    
    qx_saxs, qy_saxs, _ = convert_to_q_space_image(
        saxs_data, saxs_center, pixel_to_q_saxs_scaled,
        **exp_params['saxs']
    )
    
    # Find overlapping q-range for interpolation
    q_min = max(np.min([qx_ptycho, qy_ptycho]), np.min([qx_saxs, qy_saxs]))
    q_max = min(np.max([qx_ptycho, qy_ptycho]), np.max([qx_saxs, qy_saxs]))
    #q_range = (q_min * 0.9, q_max * 0.9)  # Slightly reduce range to avoid edge effects
    q_range = (q_min, q_max)  # Slightly reduce range to avoid edge effects
    
    print(f"2D Q-space ranges after scaling:")
    print(f"  Ptycho qx range: {qx_ptycho.min():.6f} to {qx_ptycho.max():.6f} nm⁻¹")
    print(f"  Ptycho qy range: {qy_ptycho.min():.6f} to {qy_ptycho.max():.6f} nm⁻¹")
    print(f"  SAXS qx range: {qx_saxs.min():.6f} to {qx_saxs.max():.6f} nm⁻¹")
    print(f"  SAXS qy range: {qy_saxs.min():.6f} to {qy_saxs.max():.6f} nm⁻¹")
    
    print(f"Q-space overlay range: {q_range[0]:.6f} to {q_range[1]:.6f} nm⁻¹")
    
    # Interpolate both images to common q-grid
    qx_common, qy_common, ptycho_fft_interp, saxs_interp = interpolate_to_common_q_grid(
        qx_ptycho, qy_ptycho, ptycho_fft_magnitude,
        qx_saxs, qy_saxs, saxs_data,
        q_range, q_resolution=256
    )
    
    # Create figure with subplots (4x2 layout to include overlay)
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    fig.suptitle(f'Correlation Results: Ptycho Scan {ptycho_scan} (φ={ptycho_phi:.3f}°) vs SAXS Scan {saxs_scan} (φ={saxs_phi:.3f}°, index={phi_index})', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Ptychography reconstruction
    im1 = axes[0, 0].imshow(np.angle(ptycho_data), cmap='gray')
    axes[0, 0].set_title(f'Ptychography Reconstruction\nScan {ptycho_scan}')
    axes[0, 0].set_xlabel('X (pixels)')
    axes[0, 0].set_ylabel('Y (pixels)')
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
    
    # Plot 2: FFT of ptychography reconstruction (log scale)
    im2 = axes[0, 1].imshow(np.log10(ptycho_fft_magnitude + 1), cmap='hot')
    axes[0, 1].set_title(f'FFT of Ptycho Reconstruction\n(Log scale)')
    axes[0, 1].set_xlabel('kx (frequency)')
    axes[0, 1].set_ylabel('ky (frequency)')
    # Mark the center
    axes[0, 1].plot(ptycho_center[1], ptycho_center[0], 'w+', markersize=10, markeredgewidth=2)
    plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
    
    # Plot 3: SAXS data
    im3 = axes[1, 0].imshow(saxs_data, cmap='plasma')
    axes[1, 0].set_title(f'SAXS Data\nScan {saxs_scan}, Index {phi_index}')
    axes[1, 0].set_xlabel('Detector X (pixels)')
    axes[1, 0].set_ylabel('Detector Y (pixels)')
    # Mark the beam center
    axes[1, 0].plot(saxs_center[1], saxs_center[0], 'w+', markersize=10, markeredgewidth=2)
    plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)
    
    # Plot 4: SAXS data (log scale)
    saxs_log = np.log10(saxs_data + 1)
    im4 = axes[1, 1].imshow(saxs_log, cmap='plasma')
    axes[1, 1].set_title(f'SAXS Data (Log scale)\nScan {saxs_scan}, Index {phi_index}')
    axes[1, 1].set_xlabel('Detector X (pixels)')
    axes[1, 1].set_ylabel('Detector Y (pixels)')
    # Mark the beam center
    axes[1, 1].plot(saxs_center[1], saxs_center[0], 'w+', markersize=10, markeredgewidth=2)
    plt.colorbar(im4, ax=axes[1, 1], shrink=0.8)
    
    # Plot 5: SAXS radial profile in q-space
    ax5 = axes[2, 0]
    ax5.plot(q_saxs, intensity_saxs, 'b-', linewidth=2, label='Linear scale')
    ax5.set_xlabel('q (nm⁻¹)')
    ax5.set_ylabel('Azimuthally averaged intensity')
    ax5.set_title(f'SAXS Radial Profile (q-space)\nScan {saxs_scan}, φ={saxs_phi:.3f}°, Index {phi_index}')
    ax5.grid(True, alpha=0.3)
    
    # Add log scale plot for SAXS
    ax5_log = ax5.twinx()
    ax5_log.semilogy(q_saxs, intensity_saxs + 1, 'r--', linewidth=1.5, alpha=0.7, label='Log scale')
    ax5_log.set_ylabel('Log(Intensity + 1)', color='red')
    ax5_log.tick_params(axis='y', labelcolor='red')
    
    # Add legends
    ax5.legend(loc='upper right')
    ax5_log.legend(loc='upper left')
    
    # Plot 6: Ptychography FFT radial profile in q-space
    ax6 = axes[2, 1]
    ax6.plot(q_ptycho, intensity_ptycho, 'b-', linewidth=2, label='Linear scale')
    ax6.set_xlabel('q (nm⁻¹)')
    ax6.set_ylabel('FFT Magnitude')
    ax6.set_title(f'Ptycho FFT Radial Profile (q-space)\nScan {ptycho_scan}, φ={ptycho_phi:.3f}°')
    ax6.grid(True, alpha=0.3)
    
    # Add log scale plot for FFT
    ax6_log = ax6.twinx()
    ax6_log.semilogy(q_ptycho, intensity_ptycho + 1, 'r--', linewidth=1.5, alpha=0.7, label='Log scale')
    ax6_log.set_ylabel('Log(FFT Magnitude + 1)', color='red')
    ax6_log.tick_params(axis='y', labelcolor='red')
    
    # Add legends
    ax6.legend(loc='upper right')
    ax6_log.legend(loc='upper left')
    
    # Plot 7: Q-space overlay - FFT in common q-grid
    im7 = axes[3, 0].imshow(np.log10(ptycho_fft_interp + 1), 
                           extent=[qx_common.min(), qx_common.max(), qy_common.min(), qy_common.max()],
                           cmap='hot', origin='lower')
    axes[3, 0].set_title(f'Ptycho FFT in Q-space (log scale)\nCommon grid')
    axes[3, 0].set_xlabel('qx (nm⁻¹)')
    axes[3, 0].set_ylabel('qy (nm⁻¹)')
    plt.colorbar(im7, ax=axes[3, 0], shrink=0.8)
    
    # Plot 8: Q-space overlay - SAXS in common q-grid  
    im8 = axes[3, 1].imshow(np.log10(saxs_interp + 1),
                           extent=[qx_common.min(), qx_common.max(), qy_common.min(), qy_common.max()],
                           cmap='plasma', origin='lower')
    axes[3, 1].set_title(f'SAXS in Q-space (log scale)\nCommon grid')
    axes[3, 1].set_xlabel('qx (nm⁻¹)')
    axes[3, 1].set_ylabel('qy (nm⁻¹)')
    plt.colorbar(im8, ax=axes[3, 1], shrink=0.8)
    
    plt.tight_layout()
    plt.show()
    
    # Create a separate figure for direct q-space overlay comparison
    fig_overlay, axes_overlay = plt.subplots(2, 2, figsize=(16, 12))
    fig_overlay.suptitle(f'Q-space Overlay Analysis: Ptycho vs SAXS\nScan {ptycho_scan} (φ={ptycho_phi:.3f}°) vs Scan {saxs_scan} (φ={saxs_phi:.3f}°)', 
                        fontsize=14, fontweight='bold')
    
    # Use log scale for better visualization and normalize
    ptycho_log = np.log10(ptycho_fft_interp + 1)
    saxs_log = np.log10(saxs_interp + 1)
    
    # Normalize log-scaled data
    ptycho_norm = (ptycho_log - ptycho_log.min()) / (ptycho_log.max() - ptycho_log.min())
    saxs_norm = (saxs_log - saxs_log.min()) / (saxs_log.max() - saxs_log.min())
    
    # Plot 1: Ptychography FFT in q-space (log scale, normalized)
    im_o1 = axes_overlay[0, 0].imshow(ptycho_norm, 
                                     extent=[qx_common.min(), qx_common.max(), qy_common.min(), qy_common.max()],
                                     cmap='hot', origin='lower', vmin=0, vmax=1)
    axes_overlay[0, 0].set_title(f'Ptycho FFT (log scale)\nQ-space')
    axes_overlay[0, 0].set_xlabel('qx (nm⁻¹)')
    axes_overlay[0, 0].set_ylabel('qy (nm⁻¹)')
    plt.colorbar(im_o1, ax=axes_overlay[0, 0], shrink=0.8, label='Log(Intensity + 1)')
    
    # Plot 2: SAXS in q-space (log scale, normalized)
    im_o2 = axes_overlay[0, 1].imshow(saxs_norm,
                                     extent=[qx_common.min(), qx_common.max(), qy_common.min(), qy_common.max()],
                                     cmap='plasma', origin='lower', vmin=0, vmax=1)
    axes_overlay[0, 1].set_title(f'SAXS (log scale)\nQ-space')
    axes_overlay[0, 1].set_xlabel('qx (nm⁻¹)')
    axes_overlay[0, 1].set_ylabel('qy (nm⁻¹)')
    plt.colorbar(im_o2, ax=axes_overlay[0, 1], shrink=0.8, label='Log(Intensity + 1)')
    
    # Plot 3: RGB overlay (Red: SAXS, Blue: Ptycho FFT) - both in log scale
    # Create RGB image where red channel = SAXS, blue channel = Ptycho FFT
    rgb_overlay = np.zeros((saxs_norm.shape[0], saxs_norm.shape[1], 3))
    rgb_overlay[:, :, 0] = saxs_norm      # Red channel: SAXS (log scale)
    rgb_overlay[:, :, 2] = ptycho_norm    # Blue channel: Ptycho FFT (log scale)
    rgb_overlay[:, :, 1] = 0.3 * (saxs_norm + ptycho_norm)  # Green: blend
    
    axes_overlay[1, 0].imshow(rgb_overlay, 
                             extent=[qx_common.min(), qx_common.max(), qy_common.min(), qy_common.max()],
                             origin='lower')
    axes_overlay[1, 0].set_title('RGB Overlay (log scale)\nRed: SAXS, Blue: Ptycho FFT')
    axes_overlay[1, 0].set_xlabel('qx (nm⁻¹)')
    axes_overlay[1, 0].set_ylabel('qy (nm⁻¹)')
    
    # Plot 4: Difference map (SAXS - Ptycho FFT) in log scale
    diff_map = saxs_norm - ptycho_norm
    im_o4 = axes_overlay[1, 1].imshow(diff_map,
                                     extent=[qx_common.min(), qx_common.max(), qy_common.min(), qy_common.max()],
                                     cmap='RdBu_r', origin='lower', vmin=-1, vmax=1)
    axes_overlay[1, 1].set_title('Difference Map (log scale)\n(SAXS - Ptycho FFT)')
    axes_overlay[1, 1].set_xlabel('qx (nm⁻¹)')
    axes_overlay[1, 1].set_ylabel('qy (nm⁻¹)')
    plt.colorbar(im_o4, ax=axes_overlay[1, 1], shrink=0.8, label='Log Scale Difference')
    
    plt.tight_layout()
    plt.show()
    
    # Check if raw diffraction plotting is enabled
    raw_diffraction = None
    if hasattr(plot_correlation_results, '_plot_raw_diffraction') and plot_correlation_results._plot_raw_diffraction:
        print(f"\nLoading raw diffraction data for overlay...")
        raw_diffraction = load_ptycho_raw_diffraction_sum(ptycho_scan, crop_size=128, center=(773, 626))
        
        if raw_diffraction is not None:
            print(f"Adding raw diffraction to overlay analysis...")
            # Create 3-way overlay plot
            create_three_way_overlay(ptycho_norm, saxs_norm, raw_diffraction, qx_common, qy_common,
                                   ptycho_scan, saxs_scan, ptycho_phi, saxs_phi, actual_ptycho_pixel_size, ptycho_q_scaling)
    
    # Always create the 2-way FFT vs SAXS overlay
    create_two_way_overlay(ptycho_norm, saxs_norm, qx_common, qy_common,
                          ptycho_scan, saxs_scan, ptycho_phi, saxs_phi)
    
    # Perform common peak analysis
    print(f"\nPerforming common peak analysis...")
    plot_common_peaks(ptycho_norm, saxs_norm, qx_common, qy_common,
                     ptycho_scan, saxs_scan, ptycho_phi, saxs_phi)
    
    # Create a separate figure for q-space radial profile comparison
    fig2, (ax_comp1, ax_comp2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot 1: Direct q-space comparison (interpolated to common q-grid)
    q_min = max(q_saxs.min(), q_ptycho.min())
    q_max = min(q_saxs.max(), q_ptycho.max())
    q_common = np.linspace(q_min, q_max, 100)
    
    # Interpolate both profiles to common q-grid
    saxs_interp = np.interp(q_common, q_saxs, intensity_saxs)
    ptycho_interp = np.interp(q_common, q_ptycho, intensity_ptycho)
    
    # Normalize intensities to [0, 1] for comparison
    skip_points = 14 # for high intensity data
    saxs_norm = saxs_interp[skip_points:] / np.max(saxs_interp[skip_points:])
    ptycho_norm = ptycho_interp[skip_points:] / np.max(ptycho_interp[skip_points:])
    
    ax_comp1.semilogy(q_common[skip_points:], saxs_norm, 'b-', linewidth=2, label=f'SAXS (scan {saxs_scan}, φ={saxs_phi:.3f}°)')
    ax_comp1.semilogy(q_common[skip_points:], ptycho_norm, 'r-', linewidth=2, label=f'Ptycho FFT (scan {ptycho_scan}, φ={ptycho_phi:.3f}°)')
    ax_comp1.set_xlabel('q (nm⁻¹)')
    ax_comp1.set_ylabel('Normalized intensity (log scale)')
    ax_comp1.set_title('Q-space Radial Profile Comparison\n(Normalized, overlapping q-range)')
    ax_comp1.legend()
    ax_comp1.grid(True, alpha=0.3)
    
    # Plot 2: Full range comparison (separate y-axes)
    ax_comp2.semilogy(q_saxs, intensity_saxs, 'b-', linewidth=2, label=f'SAXS (scan {saxs_scan})')
    ax_comp2.set_xlabel('q (nm⁻¹)')
    ax_comp2.set_ylabel('SAXS Intensity (log scale)', color='blue')
    ax_comp2.tick_params(axis='y', labelcolor='blue')
    ax_comp2.grid(True, alpha=0.3)
    
    # Create second y-axis for ptycho FFT
    ax_comp2_twin = ax_comp2.twinx()
    ax_comp2_twin.semilogy(q_ptycho, intensity_ptycho, 'r-', linewidth=2, label=f'Ptycho FFT (scan {ptycho_scan})')
    ax_comp2_twin.set_ylabel('FFT Magnitude (log scale)', color='red')
    ax_comp2_twin.tick_params(axis='y', labelcolor='red')
    
    ax_comp2.set_title('Q-space Radial Profiles\n(Full ranges, separate scales)')
    
    # Add legends
    lines1, labels1 = ax_comp2.get_legend_handles_labels()
    lines2, labels2 = ax_comp2_twin.get_legend_handles_labels()
    ax_comp2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    # Run q-space calibration diagnostic
    diagnostic_info = diagnose_q_space_mismatch(ptycho_data, saxs_data, 
                                               actual_ptycho_pixel_size, 
                                               ptycho_q_scaling)
    
    # Print comprehensive statistics
    print(f"\n{'='*60}")
    print("COMPREHENSIVE ANALYSIS STATISTICS")
    print(f"{'='*60}")
    
    print(f"\nExperimental Parameters:")
    print(f"  X-ray energy: {exp_params['saxs']['energy_kev']} keV")
    print(f"  X-ray wavelength: {wavelength_nm:.4f} nm")
    print(f"  Sample-detector distance: {exp_params['saxs']['distance_m']} m")
    print(f"  SAXS detector pixel size: {exp_params['saxs']['pixel_size_um']} μm")
    print(f"  Ptycho real space pixel size: {actual_ptycho_pixel_size:.3f} nm {'(from HDF5)' if ptycho_pixel_size_nm is not None else '(default)'}")
    if abs(ptycho_q_scaling - 1.0) > 0.01:
        print(f"  Ptycho q-scaling applied: {ptycho_q_scaling:.3f}")
        print(f"  Suggested true pixel size: {diagnostic_info['suggested_pixel_size']:.3f} nm")
    
    print(f"\nData Statistics:")
    print(f"  Ptychography - Min: {ptycho_data.min():.3e}, Max: {ptycho_data.max():.3e}, Mean: {ptycho_data.mean():.3e}")
    print(f"  SAXS Data - Min: {saxs_data.min():.3e}, Max: {saxs_data.max():.3e}, Mean: {saxs_data.mean():.3e}")
    print(f"  FFT Magnitude - Min: {ptycho_fft_magnitude.min():.3e}, Max: {ptycho_fft_magnitude.max():.3e}, Mean: {ptycho_fft_magnitude.mean():.3e}")
    
    print(f"\nCenter Detection:")
    print(f"  SAXS beam center: {saxs_center}")
    print(f"  FFT center: {ptycho_center}")
    
    print(f"\nQ-space Ranges:")
    print(f"  SAXS q-range: {q_saxs.min():.6f} to {q_saxs.max():.6f} nm⁻¹")
    print(f"  Ptycho FFT q-range: {q_ptycho.min():.6f} to {q_ptycho.max():.6f} nm⁻¹")
    print(f"  Overlapping q-range: {max(q_saxs.min(), q_ptycho.min()):.6f} to {min(q_saxs.max(), q_ptycho.max()):.6f} nm⁻¹")
    print(f"  Q-range ratio (SAXS/Ptycho): {diagnostic_info['ratio_current']:.3f}")
    
    print(f"\nRadial Profile Points:")
    print(f"  SAXS profile: {len(r_saxs)} points")
    print(f"  FFT profile: {len(r_ptycho)} points")
    
    # Calculate some useful scattering parameters
    max_q_saxs = q_saxs.max()
    min_d_spacing = 2 * np.pi / max_q_saxs if max_q_saxs > 0 else np.inf
    
    print(f"\nScattering Information:")
    print(f"  Maximum q (SAXS): {max_q_saxs:.6f} nm⁻¹")
    print(f"  Minimum d-spacing (SAXS): {min_d_spacing:.2f} nm")
    
    if min_d_spacing < np.inf:
        print(f"  Resolution limit: {min_d_spacing:.2f} nm")


def find_saxs_peaks(saxs_image: np.ndarray, qx_grid: np.ndarray, qy_grid: np.ndarray,
                   min_distance: int = 10, threshold_rel: float = 0.10, 
                   max_peaks: int = 100, central_stop_radius: int = 30) -> List[Tuple[float, float, float, int, int]]:
    """
    Find peaks in SAXS image, excluding central beam stop region.
    
    Parameters:
    -----------
    saxs_image : np.ndarray
        Normalized SAXS image
    qx_grid, qy_grid : np.ndarray
        Q-space coordinate grids
    min_distance : int
        Minimum distance between peaks in pixels
    threshold_rel : float
        Relative threshold for peak detection (0-1)
    max_peaks : int
        Maximum number of peaks to find
    central_stop_radius : int
        Radius in pixels to exclude from center (beam stop region)
        
    Returns:
    --------
    peaks : List[Tuple[float, float, float, int, int]]
        List of (qx, qy, intensity, pixel_y, pixel_x) for each peak
    """
    # Create central stop mask to exclude beam stop region
    center_y, center_x = saxs_image.shape[0] // 2, saxs_image.shape[1] // 2
    y_coords, x_coords = np.ogrid[:saxs_image.shape[0], :saxs_image.shape[1]]
    central_mask = (x_coords - center_x)**2 + (y_coords - center_y)**2 <= central_stop_radius**2
    
    # Create a masked image excluding the central region
    masked_image = saxs_image.copy()
    masked_image[central_mask] = 0
    
    # Use maximum filter to find local maxima on masked image
    data_max = maximum_filter(masked_image, min_distance)
    maxima = (masked_image == data_max) & (masked_image > 0)  # Exclude zero regions
    
    # Apply threshold based on original image max
    threshold = threshold_rel * saxs_image.max()
    maxima = maxima & (masked_image > threshold)
    
# Removed verbose print statements for cleaner output
    
    # Get peak coordinates
    labeled, num_objects = label(maxima)
    peaks = []
    
    for i in range(1, num_objects + 1):
        coords = np.where(labeled == i)
        if len(coords[0]) > 0:
            # Take center of mass for sub-pixel accuracy
            y_center = np.mean(coords[0])
            x_center = np.mean(coords[1])
            y_pixel = int(y_center)
            x_pixel = int(x_center)
            
            intensity = saxs_image[y_pixel, x_pixel]  # Use original intensity (not masked)
            
            # Convert to q-space coordinates
            qx = qx_grid[y_pixel, x_pixel]
            qy = qy_grid[y_pixel, x_pixel]
            
            peaks.append((qx, qy, intensity, y_pixel, x_pixel))
    
    # Sort by intensity and take top peaks
    peaks.sort(key=lambda x: x[2], reverse=True)
    peaks = peaks[:max_peaks]
    
    print(f"Found {len(peaks)} peaks in SAXS data")
    return peaks


def find_corresponding_ptycho_peaks(saxs_peaks: List[Tuple[float, float, float, int, int]], 
                                   ptycho_image: np.ndarray, qx_grid: np.ndarray, qy_grid: np.ndarray,
                                   q_search_radius: float = 0.015, 
                                   min_ptycho_intensity: float = 0.03,
                                   central_stop_radius: int = 30) -> List[Tuple[Tuple[float, float, float, int, int], 
                                                                                 Tuple[float, float, float, int, int], float]]:
    """
    Find corresponding ptychography FFT peaks near SAXS peak positions with one-to-one matching.
    
    Parameters:
    -----------
    saxs_peaks : List[Tuple[float, float, float, int, int]]
        Peaks found in SAXS data
    ptycho_image : np.ndarray
        Normalized ptychography FFT image
    qx_grid, qy_grid : np.ndarray
        Q-space coordinate grids
    q_search_radius : float
        Search radius in q-space (nm⁻¹)
    min_ptycho_intensity : float
        Minimum relative intensity for ptycho peaks
    central_stop_radius : int
        Radius in pixels to exclude from center
        
    Returns:
    --------
    matches : List[Tuple[saxs_peak, ptycho_peak, distance]]
        Matched peak pairs with their q-space separation (one-to-one)
    """
    # Create central stop mask for ptychography
    center_y, center_x = ptycho_image.shape[0] // 2, ptycho_image.shape[1] // 2
    y_coords, x_coords = np.ogrid[:ptycho_image.shape[0], :ptycho_image.shape[1]]
    central_mask = (x_coords - center_x)**2 + (y_coords - center_y)**2 <= central_stop_radius**2
    
    # Find all potential matches first
    potential_matches = []
    
    for i, saxs_peak in enumerate(saxs_peaks):
        qx_saxs, qy_saxs, int_saxs, py_saxs, px_saxs = saxs_peak
        
        # Create a mask for the search region around this SAXS peak
        q_distances = np.sqrt((qx_grid - qx_saxs)**2 + (qy_grid - qy_saxs)**2)
        search_mask = q_distances <= q_search_radius
        
        if not np.any(search_mask):
            continue
        
        # Find all ptycho peaks within the search region that meet criteria
        ptycho_region = ptycho_image.copy()
        ptycho_region[~search_mask] = 0
        ptycho_region[central_mask] = 0  # Exclude central stop region
        
        # Apply minimum intensity threshold
        threshold = min_ptycho_intensity * ptycho_image.max()
        ptycho_region[ptycho_region < threshold] = 0
        
        # Find all local maxima in the search region
        from scipy.ndimage import maximum_filter
        local_maxima = maximum_filter(ptycho_region, size=3)
        peak_mask = (ptycho_region == local_maxima) & (ptycho_region > 0)
        
        # Get coordinates of all peaks in this region
        peak_coords = np.where(peak_mask)
        
        for py_ptycho, px_ptycho in zip(peak_coords[0], peak_coords[1]):
            qx_ptycho = qx_grid[py_ptycho, px_ptycho]
            qy_ptycho = qy_grid[py_ptycho, px_ptycho]
            int_ptycho = ptycho_image[py_ptycho, px_ptycho]
            
            # Calculate actual distance
            distance = np.sqrt((qx_saxs - qx_ptycho)**2 + (qy_saxs - qy_ptycho)**2)
            
            ptycho_peak = (qx_ptycho, qy_ptycho, int_ptycho, py_ptycho, px_ptycho)
            potential_matches.append((saxs_peak, ptycho_peak, distance, i))
    
    # Sort potential matches by distance (closest first)
    potential_matches.sort(key=lambda x: x[2])
    
    # Perform one-to-one matching using greedy algorithm
    used_ptycho_peaks = set()
    used_saxs_peaks = set()
    final_matches = []
    
    for saxs_peak, ptycho_peak, distance, saxs_idx in potential_matches:
        # Create unique identifiers for peaks
        ptycho_id = (ptycho_peak[3], ptycho_peak[4])  # (py, px)
        saxs_id = (saxs_peak[3], saxs_peak[4])  # (py, px)
        
        # Only add if neither peak has been used
        if ptycho_id not in used_ptycho_peaks and saxs_id not in used_saxs_peaks:
            final_matches.append((saxs_peak, ptycho_peak, distance))
            used_ptycho_peaks.add(ptycho_id)
            used_saxs_peaks.add(saxs_id)
    
    # Sort final matches by SAXS peak intensity (strongest first)
    final_matches.sort(key=lambda x: x[0][2], reverse=True)
    
    print(f"Found {len(potential_matches)} potential matches, {len(final_matches)} unique one-to-one matches")
    return final_matches


def create_three_way_overlay(ptycho_fft_norm: np.ndarray, saxs_norm: np.ndarray, raw_diffraction: np.ndarray,
                            qx_common: np.ndarray, qy_common: np.ndarray,
                            ptycho_scan: int, saxs_scan: int, ptycho_phi: float, saxs_phi: float,
                            ptycho_pixel_size_nm: float, ptycho_q_scaling: float = 1.0) -> None:
    """Create a 3-way overlay plot with raw diffraction, FFT, and SAXS data."""
    
    print(f"\n{'='*60}")
    print("CREATING 3-WAY OVERLAY WITH RAW DIFFRACTION")
    print(f"{'='*60}")
    
    # Convert raw diffraction to q-space on the same grid
    raw_center = (raw_diffraction.shape[0] // 2, raw_diffraction.shape[1] // 2)
    
    # Create scaled conversion function for raw diffraction
    def pixel_to_q_ptycho_scaled(r_pixels, **kwargs):
        q_original = pixel_to_q_ptycho(r_pixels, **kwargs)
        return q_original * ptycho_q_scaling
    
    qx_raw, qy_raw, _ = convert_to_q_space_image(
        raw_diffraction, raw_center, pixel_to_q_ptycho_scaled,
        pixel_size_nm=ptycho_pixel_size_nm,
        image_size=max(raw_diffraction.shape)
    )
    
    # Find overlapping q-range
    q_min = max(np.min([qx_common, qy_common]), np.min([qx_raw, qy_raw]))
    q_max = min(np.max([qx_common, qy_common]), np.max([qx_raw, qy_raw]))
    q_range = (q_min * 0.9, q_max * 0.9)
    
    print(f"3-way overlay q-range: {q_range[0]:.6f} to {q_range[1]:.6f} nm⁻¹")
    
    # Interpolate raw diffraction to the common grid
    _, _, raw_interp, _ = interpolate_to_common_q_grid(
        qx_raw, qy_raw, raw_diffraction,
        qx_common, qy_common, saxs_norm,  # Use existing common grid
        q_range, q_resolution=256
    )
    
    # Normalize raw diffraction (log scale then normalize)
    raw_log = np.log10(raw_interp + 1)
    raw_norm = (raw_log - raw_log.min()) / (raw_log.max() - raw_log.min())
    
    # Create 3-way overlay figure (2x3 layout)
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    fig.suptitle(f'3-Way Q-space Overlay: FFT vs Raw Diffraction vs SAXS\n'
                f'Scan {ptycho_scan} (φ={ptycho_phi:.3f}°) vs SAXS Scan {saxs_scan} (φ={saxs_phi:.3f}°)', 
                fontsize=16, fontweight='bold')
    
    extent = [qx_common.min(), qx_common.max(), qy_common.min(), qy_common.max()]
    
    # Top row: Individual datasets
    # Plot 1: Ptychography FFT
    im1 = axes[0, 0].imshow(ptycho_fft_norm, extent=extent, cmap='hot', origin='lower', vmin=0, vmax=1)
    axes[0, 0].set_title('Ptycho FFT (log scale)')
    axes[0, 0].set_xlabel('qx (nm⁻¹)')
    axes[0, 0].set_ylabel('qy (nm⁻¹)')
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
    
    # Plot 2: Raw Diffraction
    im2 = axes[0, 1].imshow(raw_norm, extent=extent, cmap='viridis', origin='lower', vmin=0, vmax=1)
    axes[0, 1].set_title('Raw Ptycho Diffraction (log scale)')
    axes[0, 1].set_xlabel('qx (nm⁻¹)')
    axes[0, 1].set_ylabel('qy (nm⁻¹)')
    plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
    
    # Plot 3: SAXS
    im3 = axes[0, 2].imshow(saxs_norm, extent=extent, cmap='plasma', origin='lower', vmin=0, vmax=1)
    axes[0, 2].set_title('SAXS (log scale)')
    axes[0, 2].set_xlabel('qx (nm⁻¹)')
    axes[0, 2].set_ylabel('qy (nm⁻¹)')
    plt.colorbar(im3, ax=axes[0, 2], shrink=0.8)
    
    # Bottom row: Overlays
    # Plot 4: 3-way RGB overlay (Red: SAXS, Green: Raw, Blue: FFT)
    rgb_3way = np.zeros((saxs_norm.shape[0], saxs_norm.shape[1], 3))
    rgb_3way[:, :, 0] = saxs_norm         # Red: SAXS
    rgb_3way[:, :, 1] = raw_norm          # Green: Raw Diffraction
    rgb_3way[:, :, 2] = ptycho_fft_norm   # Blue: FFT
    
    axes[1, 0].imshow(rgb_3way, extent=extent, origin='lower')
    axes[1, 0].set_title('3-Way RGB Overlay\nRed=SAXS, Green=Raw, Blue=FFT')
    axes[1, 0].set_xlabel('qx (nm⁻¹)')
    axes[1, 0].set_ylabel('qy (nm⁻¹)')
    
    # Plot 5: FFT vs Raw difference
    diff_fft_raw = ptycho_fft_norm - raw_norm
    im5 = axes[1, 1].imshow(diff_fft_raw, extent=extent, cmap='RdBu_r', origin='lower', vmin=-1, vmax=1)
    axes[1, 1].set_title('FFT - Raw Diffraction')
    axes[1, 1].set_xlabel('qx (nm⁻¹)')
    axes[1, 1].set_ylabel('qy (nm⁻¹)')
    plt.colorbar(im5, ax=axes[1, 1], shrink=0.8)
    
    # Plot 6: SAXS vs Raw difference
    diff_saxs_raw = saxs_norm - raw_norm
    im6 = axes[1, 2].imshow(diff_saxs_raw, extent=extent, cmap='RdBu_r', origin='lower', vmin=-1, vmax=1)
    axes[1, 2].set_title('SAXS - Raw Diffraction')
    axes[1, 2].set_xlabel('qx (nm⁻¹)')
    axes[1, 2].set_ylabel('qy (nm⁻¹)')
    plt.colorbar(im6, ax=axes[1, 2], shrink=0.8)
    
    plt.tight_layout()
    plt.show()
    
    print(f"3-way overlay plots completed successfully!")


def create_two_way_overlay(ptycho_fft_norm: np.ndarray, saxs_norm: np.ndarray,
                          qx_common: np.ndarray, qy_common: np.ndarray,
                          ptycho_scan: int, saxs_scan: int, ptycho_phi: float, saxs_phi: float) -> None:
    """Create a 2-way overlay plot with FFT and SAXS data."""
    
    print(f"\n{'='*60}")
    print("CREATING 2-WAY FFT vs SAXS OVERLAY")
    print(f"{'='*60}")
    
    # Create 2-way overlay figure (2x2 layout)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'2-Way Q-space Overlay: Ptycho FFT vs SAXS\n'
                f'Scan {ptycho_scan} (φ={ptycho_phi:.3f}°) vs SAXS Scan {saxs_scan} (φ={saxs_phi:.3f}°)', 
                fontsize=14, fontweight='bold')
    
    extent = [qx_common.min(), qx_common.max(), qy_common.min(), qy_common.max()]
    
    # Plot 1: Ptychography FFT in q-space (log scale, normalized)
    im1 = axes[0, 0].imshow(ptycho_fft_norm, extent=extent, cmap='hot', origin='lower', vmin=0, vmax=1)
    axes[0, 0].set_title('Ptycho FFT (log scale)')
    axes[0, 0].set_xlabel('qx (nm⁻¹)')
    axes[0, 0].set_ylabel('qy (nm⁻¹)')
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
    
    # Plot 2: SAXS in q-space (log scale, normalized)
    im2 = axes[0, 1].imshow(saxs_norm, extent=extent, cmap='plasma', origin='lower', vmin=0, vmax=1)
    axes[0, 1].set_title('SAXS (log scale)')
    axes[0, 1].set_xlabel('qx (nm⁻¹)')
    axes[0, 1].set_ylabel('qy (nm⁻¹)')
    plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
    
    # Plot 3: RGB overlay (Red: SAXS, Blue: Ptycho FFT)
    rgb_overlay = np.zeros((saxs_norm.shape[0], saxs_norm.shape[1], 3))
    rgb_overlay[:, :, 0] = saxs_norm       # Red: SAXS
    rgb_overlay[:, :, 2] = ptycho_fft_norm # Blue: FFT
    rgb_overlay[:, :, 1] = 0.3 * (saxs_norm + ptycho_fft_norm)  # Green: blend
    
    axes[1, 0].imshow(rgb_overlay, extent=extent, origin='lower')
    axes[1, 0].set_title('RGB Overlay\nRed=SAXS, Blue=Ptycho FFT')
    axes[1, 0].set_xlabel('qx (nm⁻¹)')
    axes[1, 0].set_ylabel('qy (nm⁻¹)')
    
    # Plot 4: Difference map (SAXS - Ptycho FFT)
    diff_map = saxs_norm - ptycho_fft_norm
    im4 = axes[1, 1].imshow(diff_map, extent=extent, cmap='RdBu_r', origin='lower', vmin=-1, vmax=1)
    axes[1, 1].set_title('Difference Map\n(SAXS - Ptycho FFT)')
    axes[1, 1].set_xlabel('qx (nm⁻¹)')
    axes[1, 1].set_ylabel('qy (nm⁻¹)')
    plt.colorbar(im4, ax=axes[1, 1], shrink=0.8)
    
    plt.tight_layout()
    plt.show()
    
    print(f"2-way overlay plots completed successfully!")


def plot_common_peaks(ptycho_norm: np.ndarray, saxs_norm: np.ndarray,
                     qx_common: np.ndarray, qy_common: np.ndarray,
                     ptycho_scan: int, saxs_scan: int,
                     ptycho_phi: float, saxs_phi: float) -> None:
    """Plot analysis of common peaks between SAXS and ptychography FFT."""
    
    print(f"\n{'='*60}")
    print("COMMON PEAK ANALYSIS")
    print(f"{'='*60}")
    
    #
    central_stop_radius = 60
    
    # Find peaks in SAXS data first (they're typically more defined)
    saxs_peaks = find_saxs_peaks(
        saxs_norm, qx_common, qy_common,
        min_distance=8, threshold_rel=0.05, max_peaks=200, central_stop_radius=central_stop_radius
    )
    
    if not saxs_peaks:
        print("No peaks found in SAXS data")
        return
    
    # Find corresponding ptychography peaks
    matched_peaks = find_corresponding_ptycho_peaks(
        saxs_peaks, ptycho_norm, qx_common, qy_common,
        q_search_radius=0.1, min_ptycho_intensity=0.5, central_stop_radius=central_stop_radius
    )
    
    print(f"Found {len(saxs_peaks)} SAXS peaks, {len(matched_peaks)} have corresponding ptycho peaks")
    
    if not matched_peaks:
        print("No corresponding ptycho peaks found")
        return
    
    # Create simplified peak analysis figure - just 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'Common Peak Analysis: SAXS → Ptycho FFT Peak Search\n'
                 f'Scan {ptycho_scan} (φ={ptycho_phi:.3f}°) vs Scan {saxs_scan} (φ={saxs_phi:.3f}°)', 
                 fontsize=14, fontweight='bold')
    
    extent = [qx_common.min(), qx_common.max(), qy_common.min(), qy_common.max()]
    
    # Plot 1: SAXS with all detected peaks
    axes[0].imshow(saxs_norm, extent=extent, cmap='plasma', origin='lower')
    
    # Calculate central stop region in q-space (convert from pixels)
    # Estimate q per pixel from grid
    q_per_pixel = (qx_common.max() - qx_common.min()) / qx_common.shape[1]
    central_stop_q = central_stop_radius * q_per_pixel  # 30 pixels * q/pixel
    central_stop_circle = plt.Circle((0, 0), central_stop_q, fill=False, color='white', alpha=0.8, linewidth=2, linestyle='--')
    axes[0].add_patch(central_stop_circle)
    
    # Show ALL SAXS peaks without any limit
    matched_saxs_peaks = [mp[0] for mp in matched_peaks]  # Extract SAXS peaks that have matches
    
    for i, (qx, qy, intensity, py, px) in enumerate(saxs_peaks):
        # Check if this SAXS peak has a corresponding ptycho peak
        is_matched = any(sp[0] == qx and sp[1] == qy and sp[2] == intensity for sp in matched_saxs_peaks)
        color = 'lime' if is_matched else 'yellow'
        markersize = 6 if is_matched else 3
        
        axes[0].plot(qx, qy, 'o', color=color, markersize=markersize, markeredgecolor='black', markeredgewidth=0.5)
    axes[0].set_title(f'SAXS Peaks: {len(saxs_peaks)} total, {len(matched_peaks)} matched\n(Green = matched with ptycho, Yellow = no match)')
    axes[0].set_xlabel('qx (nm⁻¹)')
    axes[0].set_ylabel('qy (nm⁻¹)')
    
    # Plot 2: Ptycho FFT with corresponding peaks
    axes[1].imshow(ptycho_norm, extent=extent, cmap='hot', origin='lower')
    
    # Draw central stop region for ptycho too
    central_stop_circle_ptycho = plt.Circle((0, 0), central_stop_q, fill=False, color='white', alpha=0.8, linewidth=2, linestyle='--')
    axes[1].add_patch(central_stop_circle_ptycho)
    
    # Show ALL matched ptycho peaks without any limit
    for i, (saxs_peak, ptycho_peak, distance) in enumerate(matched_peaks):
        qx_p, qy_p, int_p, py_p, px_p = ptycho_peak
        axes[1].plot(qx_p, qy_p, 'o', color='cyan', markersize=4, markeredgecolor='black', markeredgewidth=0.5)
    axes[1].set_title(f'Ptycho FFT: {len(matched_peaks)} corresponding peaks found')
    axes[1].set_xlabel('qx (nm⁻¹)')
    axes[1].set_ylabel('qy (nm⁻¹)')
    
    # Plot 3: Overlay with matched peaks and search circles
    rgb_overlay = np.zeros((saxs_norm.shape[0], saxs_norm.shape[1], 3))
    rgb_overlay[:, :, 0] = saxs_norm      # Red: SAXS
    rgb_overlay[:, :, 2] = ptycho_norm    # Blue: Ptycho FFT
    rgb_overlay[:, :, 1] = 0.3 * (saxs_norm + ptycho_norm)  # Green: blend
    
    axes[2].imshow(rgb_overlay, extent=extent, origin='lower')
    
    # Show ALL matched peak connections without any limit
    for i, (saxs_peak, ptycho_peak, distance) in enumerate(matched_peaks):
        qx_s, qy_s, int_s, py_s, px_s = saxs_peak
        qx_p, qy_p, int_p, py_p, px_p = ptycho_peak
        
        # Draw connection line
        axes[2].plot([qx_s, qx_p], [qy_s, qy_p], 'w-', linewidth=0.8, alpha=0.6)
        
        # Draw SAXS peak (orange/red)
        axes[2].plot(qx_s, qy_s, 'o', color='orange', markersize=3, markeredgecolor='black', markeredgewidth=0.3)
        
        # Draw corresponding ptycho peak (cyan)
        axes[2].plot(qx_p, qy_p, 'o', color='cyan', markersize=3, markeredgecolor='black', markeredgewidth=0.3)
    
    axes[2].set_title(f'Peak Matching: {len(matched_peaks)} connected pairs\n(Orange = SAXS, Cyan = Ptycho FFT)')
    axes[2].set_xlabel('qx (nm⁻¹)')
    axes[2].set_ylabel('qy (nm⁻¹)')
    
    plt.tight_layout()
    plt.show()
    
    # Print basic summary
    if matched_peaks:
        distances = [dist for _, _, dist in matched_peaks]
        print(f"\nPeak matching summary:")
        print(f"  SAXS peaks found: {len(saxs_peaks)}")
        print(f"  Ptycho matches: {len(matched_peaks)}")
        print(f"  Success rate: {100*len(matched_peaks)/len(saxs_peaks):.1f}%")
        print(f"  Average position difference: {np.mean(distances):.4f} nm⁻¹")
        print(f"  Max position difference: {np.max(distances):.4f} nm⁻¹")


def plot_ptycho_raw_diffraction(scan_number: int, crop_size: int = 256, 
                               center: tuple = (773, 626)) -> None:
    """Plot the summed ptychography raw diffraction pattern."""
    print(f"\n{'='*60}")
    print("LOADING PTYCHOGRAPHY RAW DIFFRACTION PATTERNS")
    print(f"{'='*60}")
    
    summed_diffraction = load_ptycho_raw_diffraction_sum(scan_number, crop_size, center)
    
    if summed_diffraction is None:
        print("Cannot plot - no diffraction data loaded")
        return
    
    # Create figure for diffraction pattern
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Ptychography Raw Diffraction Patterns (Scan {scan_number})', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Linear scale
    im1 = axes[0, 0].imshow(summed_diffraction, cmap='viridis', origin='lower')
    axes[0, 0].set_title('Summed Diffraction (Linear Scale)')
    axes[0, 0].set_xlabel('Detector X (pixels)')
    axes[0, 0].set_ylabel('Detector Y (pixels)')
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.8, label='Counts')
    
    # Plot 2: Log scale
    im2 = axes[0, 1].imshow(summed_diffraction, cmap='viridis', origin='lower',norm=colors.LogNorm())
    axes[0, 1].set_title('Summed Diffraction (Log Scale)')
    axes[0, 1].set_xlabel('Detector X (pixels)')
    axes[0, 1].set_ylabel('Detector Y (pixels)')
    plt.colorbar(im2, ax=axes[0, 1], shrink=0.8, label='Log10(Counts + 1)')
    
    # Plot 3: Radial average
    # For the cropped image, the center is now at the geometric center
    crop_center = (summed_diffraction.shape[0] // 2, summed_diffraction.shape[1] // 2)
    r_diff, intensity_diff = azimuthal_average(summed_diffraction, center=crop_center)
    
    axes[1, 0].plot(r_diff, intensity_diff, 'b-', linewidth=2)
    axes[1, 0].set_xlabel('Radial distance (pixels)')
    axes[1, 0].set_ylabel('Azimuthally averaged intensity')
    axes[1, 0].set_title('Radial Profile (Linear Scale)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Radial average (log scale)
    axes[1, 1].semilogy(r_diff, intensity_diff + 1, 'r-', linewidth=2)
    axes[1, 1].set_xlabel('Radial distance (pixels)')
    axes[1, 1].set_ylabel('Log(Intensity + 1)')
    axes[1, 1].set_title('Radial Profile (Log Scale)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\nDiffraction Pattern Statistics:")
    print(f"  Shape: {summed_diffraction.shape}")
    print(f"  Total counts: {summed_diffraction.sum():.2e}")
    print(f"  Max intensity: {summed_diffraction.max():.2e}")
    print(f"  Mean intensity: {summed_diffraction.mean():.2e}")
    print(f"  Center pixel intensity: {summed_diffraction[crop_center[0], crop_center[1]]:.2e}")
    print(f"  Original beam center was: {center}")
    print(f"  Cropped image center: {crop_center}")


def plot_matches_overview(matches: List[Tuple[PtychoEntry, SAXSMeasurement, float]], 
                         ptycho_scan: int, saxs_scan: int) -> None:
    """Plot an overview of all phi angle matches."""
    if not matches:
        print("No matches to plot.")
        return
    
    ptycho_phis = [match[0].phi_angle for match in matches]
    saxs_phis = [match[1].phi_angle for match in matches]
    phi_diffs = [match[2] for match in matches]
    indices = [match[1].phi_index for match in matches]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Phi angles comparison
    ax1.scatter(ptycho_phis, saxs_phis, c=phi_diffs, cmap='viridis', s=100, alpha=0.7)
    ax1.plot([min(ptycho_phis + saxs_phis), max(ptycho_phis + saxs_phis)], 
             [min(ptycho_phis + saxs_phis), max(ptycho_phis + saxs_phis)], 
             'r--', alpha=0.5, label='Perfect match')
    ax1.set_xlabel(f'Ptychography φ (degrees) - Scan {ptycho_scan}')
    ax1.set_ylabel(f'SAXS φ (degrees) - Scan {saxs_scan}')
    ax1.set_title('Phi Angle Correlation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar for phi differences
    cbar1 = plt.colorbar(ax1.collections[0], ax=ax1)
    cbar1.set_label('Phi Difference (degrees)')
    
    # Plot 2: SAXS indices vs phi differences
    ax2.scatter(indices, phi_diffs, c=ptycho_phis, cmap='plasma', s=100, alpha=0.7)
    ax2.set_xlabel(f'SAXS Phi Index - Scan {saxs_scan}')
    ax2.set_ylabel('Phi Difference (degrees)')
    ax2.set_title('SAXS Index vs Phi Difference')
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar for ptycho phi angles
    cbar2 = plt.colorbar(ax2.collections[0], ax=ax2)
    cbar2.set_label('Ptychography φ (degrees)')
    
    plt.tight_layout()
    plt.show()


def run_correlation():
    """Run the correlation analysis with hardcoded parameters for Jupyter notebook."""
    
    # ====================== CONFIGURATION ======================
    # Modify these parameters as needed:
    ptycho_scan = 676#355#318                    # Ptychography scan number
    saxs_scan = 566#346#344                      # SAXS scan number
    log_file = 'Jul22_2025.log'         # Path to log file
    tolerance = 90.0                     # Tolerance for phi angle matching (increased for testing)
    verbose = True                       # Enable verbose output
    export_file = None                   # Set to filename (e.g., 'results.csv') to export
    list_scans_only = False              # Set to True to just list available scans
    enable_plotting = True               # Set to True to enable plotting functionality
    
    # Ptychography data loading options
    use_h5_ptycho = True                 # Use HDF5 file instead of TIFF (more accurate)
    complex_mode = 'phase'               # How to process complex data: 'amplitude', 'phase', 'intensity'
    pad_to_square = True                 # Pad rectangular images to square (preserves all data, recommended for FFT)
    pad_mode = 'constant'                # Padding mode: 'constant', 'edge', 'reflect', 'symmetric'
    
    # Q-space scaling test options
    ptycho_q_scaling = 1.0#0.725#1.08              # Scaling factor for ptycho q-space (1.0 = no scaling, >1.0 = expand)
    
    # Raw diffraction pattern analysis
    plot_raw_diffraction = False#True          # Plot summed raw ptychography diffraction patterns
    
    # Optional: manually specify SAXS line range and combined file prefix
    saxs_line_start = None  # e.g., 480333
    saxs_line_end = None    # e.g., 481055
    saxs_combined_prefix = 'SZC8SAXS'  # For ZC8 data use 'SZC8SAXS'; for ZC3 keep 'SZC3SAXS'
    # ===========================================================
    
    # Check if log file exists
    try:
        with open(log_file, 'r') as f:
            pass
    except FileNotFoundError:
        print(f"Error: Log file '{log_file}' not found.")
        return
    
    # Handle list-scans option
    if list_scans_only:
        list_available_scans(log_file)
        return
    
    print(f"Analyzing ptychography scan {ptycho_scan} with SAXS scan {saxs_scan}")
    print(f"Tolerance: {tolerance}")
    
    # Parse the log file
    log_parser = LogParser(log_file)
    log_parser.parse_log()
    
    # Filter by separate scan numbers for ptycho always
    ptycho_entries, _ = log_parser.filter_by_separate_scans(ptycho_scan, saxs_scan)
    
    # Obtain SAXS entries either by explicit line range or by normal parsing
    if saxs_line_start is not None and saxs_line_end is not None:
        saxs_entries = log_parser.parse_saxs_by_line_range(saxs_line_start, saxs_line_end, saxs_scan)
    else:
        _, saxs_entries = log_parser.filter_by_separate_scans(ptycho_scan, saxs_scan)
    
    print(f"\nFiltered results:")
    print(f"  Ptychography entries (scan {ptycho_scan}): {len(ptycho_entries)}")
    print(f"  SAXS entries (scan {saxs_scan}): {len(saxs_entries)}")
    
    if not ptycho_entries:
        print(f"No ptychography entries found for scan {ptycho_scan}")
        return
    
    if not saxs_entries:
        print(f"No SAXS entries found for scan {saxs_scan}")
        return
    
    # Find matches
    matcher = CorrelationMatcher(tolerance=tolerance)
    matches = matcher.find_matches(ptycho_entries, saxs_entries)
    
    # Print results
    print_results_cross_scan(matches, ptycho_scan, saxs_scan)
    
    # Export results if requested
    if export_file and matches:
        export_results_cross_scan(matches, export_file, ptycho_scan, saxs_scan)
    
    if verbose and matches:
        print(f"\nDetailed information:")
        for i, (ptycho, saxs_measurement, diff) in enumerate(matches):
            print(f"\nMatch {i+1}:")
            print(f"  Ptycho: {ptycho.detector_filename} (phi={ptycho.phi_angle:.4f}, log_line={ptycho.log_line})")
            print(f"  SAXS:   {saxs_measurement.detector_filename} (phi={saxs_measurement.phi_angle:.4f}, index={saxs_measurement.phi_index}, log_line={saxs_measurement.log_line})")
            print(f"  Phi difference: {diff:.4f}")
    
    # Plot raw diffraction patterns if requested
    #if plot_raw_diffraction and enable_plotting:
        #plot_ptycho_raw_diffraction(ptycho_scan)
        #continue
    
    # Plotting functionality
    if enable_plotting and matches:
        print(f"\n{'='*60}")
        print("LOADING DATA FOR PLOTTING")
        print(f"{'='*60}")
        
        # Load ptychography reconstruction
        ptycho_data, actual_pixel_size = get_ptycho_data_for_analysis(
            ptycho_scan, use_h5=use_h5_ptycho, complex_mode=complex_mode, 
            pad_to_square=pad_to_square, padding_mode=pad_mode
        )
        
        if ptycho_data is not None:
            # Plot overview of all matches
            print(f"\nPlotting matches overview...")
            plot_matches_overview(matches, ptycho_scan, saxs_scan)
            
            # Plot detailed results for ONLY THE BEST MATCH (first one, since matches are sorted by phi difference)
            if matches:
                best_match = matches[0]  # Best match is first (lowest phi difference)
                ptycho, saxs_measurement, diff = best_match
                
                print(f"\nLoading data for best match (phi difference: {diff:.4f}°)...")
                print(f"  Ptycho: {ptycho.detector_filename} (phi={ptycho.phi_angle:.4f}°)")
                print(f"  SAXS: {saxs_measurement.detector_filename} (phi={saxs_measurement.phi_angle:.4f}°, index={saxs_measurement.phi_index})")
                
                # Load corresponding SAXS data
                print(f"Loading SAXS data for scan {saxs_scan} and phi index {saxs_measurement.phi_index-1} instead of {saxs_measurement.phi_index}")
                saxs_data = load_saxs_data(saxs_scan, saxs_measurement.phi_index-1, combined_prefix=saxs_combined_prefix)
                
                if saxs_data is not None:
                    print(f"Plotting correlation results for best match...")
                    
                    # Set flag to enable raw diffraction overlay
                    plot_correlation_results._plot_raw_diffraction = plot_raw_diffraction
                    
                    plot_correlation_results(
                        ptycho_data, saxs_data, 
                        ptycho_scan, saxs_scan, saxs_measurement.phi_index,
                        ptycho.phi_angle, saxs_measurement.phi_angle,
                        ptycho_pixel_size_nm=actual_pixel_size,
                        ptycho_q_scaling=ptycho_q_scaling
                    )
                else:
                    print(f"Skipping plot - SAXS data not available for best match")
        else:
            print("Skipping plots - ptychography reconstruction not available")
    elif enable_plotting and not matches:
        print("\nNo matches found - skipping plots")
    
    return matches


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Correlate SAXS and ptychography data based on phi angles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --scan 323 --log Jul22_2025.log
  %(prog)s --scan 323 --tolerance 0.1 --log Jul22_2025.log
  %(prog)s --list-scans --log Jul22_2025.log
  %(prog)s --scan 323 --export results.csv
        """
    )
    
    parser.add_argument('--scan', '-s', type=int,
                       help='Scan number to analyze')
    parser.add_argument('--log', '-l', type=str, default='Jul22_2025.log',
                       help='Path to the log file (default: Jul22_2025.log)')
    parser.add_argument('--tolerance', '-t', type=float, default=0.01,
                       help='Tolerance for phi angle matching (default: 0.01)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--export', '-e', type=str, metavar='FILE',
                       help='Export results to CSV file')
    parser.add_argument('--list-scans', action='store_true',
                       help='List all available scan numbers and exit')
    parser.add_argument('--saxs-lines', type=str, metavar='START:END',
                       help='Specify raw line range (1-based) in log for SAXS phi (e.g., 480333:481055)')
    parser.add_argument('--saxs-prefix', type=str, default='SZC3SAXS',
                       help="SAXS combined filename prefix (e.g., 'SZC3SAXS' or 'SZC8SAXS')")
    
    args = parser.parse_args()
    
    # Check if log file exists
    try:
        with open(args.log, 'r') as f:
            pass
    except FileNotFoundError:
        print(f"Error: Log file '{args.log}' not found.")
        sys.exit(1)
    
    # Handle list-scans option
    if args.list_scans:
        list_available_scans(args.log)
        return
    
    # Require scan number if not listing scans
    if args.scan is None:
        print("Error: --scan is required unless using --list-scans")
        parser.print_help()
        sys.exit(1)
    
    # Parse the log file
    log_parser = LogParser(args.log)
    log_parser.parse_log()
    
    # If user provided explicit SAXS line range, use it for SAXS; otherwise filter normally
    if args.saxs_lines:
        try:
            start_str, end_str = args.saxs_lines.split(':')
            saxs_start = int(start_str)
            saxs_end = int(end_str)
        except Exception:
            print("Error: --saxs-lines must be in START:END format with integers")
            sys.exit(1)
        # Get ptycho entries for this scan and SAXS by explicit range
        ptycho_entries, _ = log_parser.filter_by_scan(args.scan)
        saxs_entries = log_parser.parse_saxs_by_line_range(saxs_start, saxs_end, args.scan)
    else:
        # Filter by scan number
        ptycho_entries, saxs_entries = log_parser.filter_by_scan(args.scan)
    
    print(f"\nFiltered for scan {args.scan}:")
    print(f"  Ptychography entries: {len(ptycho_entries)}")
    print(f"  SAXS entries: {len(saxs_entries)}")
    
    if not ptycho_entries:
        print(f"No ptychography entries found for scan {args.scan}")
        return
    
    if not saxs_entries:
        print(f"No SAXS entries found for scan {args.scan}")
        return
    
    # Find matches
    matcher = CorrelationMatcher(tolerance=args.tolerance)
    matches = matcher.find_matches(ptycho_entries, saxs_entries)
    
    # Print results
    print_results(matches, args.scan)
    
    # Export results if requested
    if args.export and matches:
        export_results(matches, args.export, args.scan)
    
    if args.verbose and matches:
        print(f"\nDetailed information:")
        for i, (ptycho, saxs, diff) in enumerate(matches):
            print(f"\nMatch {i+1}:")
            print(f"  Ptycho: {ptycho.detector_filename} (phi={ptycho.phi_angle:.4f}, log_line={ptycho.log_line})")
            print(f"  SAXS:   {saxs.detector_filename} (phi={saxs.phi_angle:.4f}, log_line={saxs.log_line})")
            print(f"  Phi difference: {diff:.4f}")


if __name__ == '__main__':
    # Check if we're running in a Jupyter environment
    try:
        # If we're in Jupyter, this will succeed
        get_ipython()
        # In Jupyter, don't run main() - let the notebook cell handle execution
        pass
    except NameError:
        # We're not in Jupyter, so run the command-line version
        main()

# %%
# Run the correlation analysis (for Jupyter notebook usage)
# To use this cell, simply execute it - no command line arguments needed!
matches = run_correlation()

# %%

E_keV = 15.7
lambda_m = 12.4 / E_keV
print(f"lambda = {lambda_m:.3f} \AA")

z = 10.0
N = 128
detector_pixel_size_m = 172.0 * 1e-6
pixel_size_recon = z*lambda_m/(N*detector_pixel_size_m)
print(f"pixel_size_recon = {pixel_size_recon/10:.3f} nm")

# %%
