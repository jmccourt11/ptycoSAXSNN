#%%
#!/usr/bin/env python3
"""
Script to create a combined plot comparing loss values across different loss functions and epochs.

Supports two different file naming patterns and paths:

1. Original Pattern (Lattice Types):
   Path: /scratch/trained_model
   Files: best_loss_epoch_LatticeSC_Probe256x256_ZCB_9_3D__Noise_sim_ZCB_9_3D_S5065_N600_steps4_dp256_Unet_{epoch}_{loss}_symmetry_0.0.txt
   Groups by: LatticeSC, LatticeClathII

2. New Pattern (Unet Statuses):
   Path: /net/micdata/data2/12IDC/ptychosaxs/models/ZCB_9_3D
   Files: 
   - L1: best_loss_epoch_{epoch}.txt (no Unet) or best_loss_epoch_Unet_{epoch}.txt (with Unet)
   - L2: best_loss_epoch_32_{unet_status}_{epoch}_L2.txt
   - Pearson: best_loss_epoch_{unet_status}_{epoch}_pearson_loss.txt
   Groups by: Unet, no_Unet

To switch between paths, modify the data_dir variable in the main() function.
"""

import os
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def parse_filename(filename):
    """
    Parse filename to extract epoch, loss function, and unet status.
    Handles multiple patterns:
    1. Original: best_loss_epoch_LatticeSC_Probe256x256_ZCB_9_3D__Noise_sim_ZCB_9_3D_S5065_N600_steps4_dp256_Unet_{epoch}_{loss}_symmetry_0.0.txt
    2. L1: best_loss_epoch_{unet_status}_{epoch}.txt
    3. L2: best_loss_epoch_32_{unet_status}_{epoch}_L2.txt
    4. Pearson: best_loss_epoch_{unet_status}_{epoch}_pearson_loss.txt
    """
    # Try L2 pattern first: best_loss_epoch_32_{unet_status}_{epoch}_L2.txt
    l2_match = re.search(r'best_loss_epoch_32_(no_Unet|Unet)_(\d+)_L2\.txt', filename)
    if l2_match:
        unet_status = l2_match.group(1)
        epoch = int(l2_match.group(2))
        return epoch, 'L2', unet_status
    
    # Try L1 pattern with Unet: best_loss_epoch_Unet_{epoch}.txt
    l1_unet_match = re.search(r'best_loss_epoch_Unet_(\d+)\.txt$', filename)
    if l1_unet_match:
        epoch = int(l1_unet_match.group(1))
        return epoch, 'L1', 'Unet'
    
    # Try L1 pattern without Unet: best_loss_epoch_{epoch}.txt
    l1_no_unet_match = re.search(r'best_loss_epoch_(\d+)\.txt$', filename)
    if l1_no_unet_match:
        epoch = int(l1_no_unet_match.group(1))
        return epoch, 'L1', 'no_Unet'
    
    # Try Pearson pattern: best_loss_epoch_{unet_status}_{epoch}_pearson_loss.txt
    pearson_match = re.search(r'best_loss_epoch_(no_Unet|Unet)_(\d+)_pearson_loss\.txt', filename)
    if pearson_match:
        unet_status = pearson_match.group(1)
        epoch = int(pearson_match.group(2))
        return epoch, 'pearson_loss', unet_status
    
    # Try original pattern: best_loss_epoch_LatticeSC_Probe256x256_ZCB_9_3D__Noise_sim_ZCB_9_3D_S5065_N600_steps4_dp256_Unet_{epoch}_{loss}_symmetry_0.0.txt
    epoch_match = re.search(r'_Unet_(\d+)_', filename)
    epoch = int(epoch_match.group(1)) if epoch_match else None
    
    loss_match = re.search(r'_Unet_\d+_(L1|L2|pearson_loss)_', filename)
    loss_func = loss_match.group(1) if loss_match else None
    
    lattice_match = re.search(r'best_loss_epoch_(LatticeSC|LatticeClathII)_', filename)
    lattice_type = lattice_match.group(1) if lattice_match else None
    
    return epoch, loss_func, lattice_type

def print_model_loss_mapping(data_dir):
    """
    Print all matching models with their corresponding loss txt files to help clarify relationships.
    """
    print(f"\nModel-Loss File Mapping for directory: {data_dir}")
    print("=" * 80)
    
    # Get all files in the directory
    all_files = os.listdir(data_dir)
    
    # Separate model files (.pth) and loss files (.txt)
    model_files = [f for f in all_files if f.endswith('.pth')]
    loss_files = [f for f in all_files if f.endswith('.txt') and 'best_loss_epoch' in f]
    
    print(f"\nFound {len(model_files)} model files (.pth)")
    print(f"Found {len(loss_files)} loss files (.txt)")
    
    # Show all model files to understand the naming pattern
    print(f"\nAll model files found:")
    for model_file in sorted(model_files):
        print(f"  {model_file}")
    
    print(f"\nAll loss files found:")
    for loss_file in sorted(loss_files):
        print(f"  {loss_file}")
    
    # Parse loss files to extract information
    loss_info = {}
    for loss_file in loss_files:
        epoch, loss_func, config = parse_filename(loss_file)
        if epoch and loss_func and config:
            key = (epoch, loss_func, config)
            loss_info[key] = loss_file
    
    # Group by configuration and loss function
    configs = {}
    for (epoch, loss_func, config), loss_file in loss_info.items():
        if config not in configs:
            configs[config] = {}
        if loss_func not in configs[config]:
            configs[config][loss_func] = []
        configs[config][loss_func].append((epoch, loss_file))
    
    # Print organized mapping
    for config in sorted(configs.keys()):
        print(f"\n{config} Configuration:")
        print("-" * 40)
        
        for loss_func in sorted(configs[config].keys()):
            print(f"\n  {loss_func} Loss:")
            epochs_and_files = sorted(configs[config][loss_func])
            
            for epoch, loss_file in epochs_and_files:
                # Try to find corresponding model file
                model_patterns = []
                
                if loss_func == 'L1':
                    if config == 'Unet':
                        model_patterns = [f"best_model_ZCB_9_Unet_epoch_{epoch}.pth"]
                    else:  # no_Unet
                        model_patterns = [f"best_model_ZCB_9_epoch_{epoch}.pth"]
                elif loss_func == 'L2':
                    if config == 'Unet':
                        model_patterns = [
                            f"best_model_ZCB_9_32_Unet_epoch_{epoch}_L2.pth",
                            f"best_model_ZCB_9_Unet_epoch_{epoch}.pth"
                        ]
                    else:  # no_Unet
                        model_patterns = [
                            f"best_model_ZCB_9_32_epoch_{epoch}_L2.pth",
                            f"best_model_ZCB_9_epoch_{epoch}.pth"
                        ]
                elif loss_func == 'pearson_loss':
                    if config == 'Unet':
                        model_patterns = [
                            f"best_model_ZCB_9_Unet_epoch_{epoch}_pearson_loss.pth",
                            f"best_model_ZCB_9_Unet_epoch_{epoch}.pth"
                        ]
                    else:  # no_Unet
                        model_patterns = [
                            f"best_model_ZCB_9_epoch_{epoch}_pearson_loss.pth",
                            f"best_model_ZCB_9_epoch_{epoch}.pth"
                        ]
                
                # Check which model files actually exist
                existing_models = []
                for pattern in model_patterns:
                    if pattern in model_files:
                        existing_models.append(pattern)
                
                print(f"    Epoch {epoch:3d}: {loss_file}")
                if existing_models:
                    for model in existing_models:
                        print(f"             -> {model}")
                else:
                    print(f"             -> No matching model file found")
                    print(f"             -> Looking for: {', '.join(model_patterns)}")
    
    print(f"\n" + "=" * 80)

def normalize_losses(data):
    """
    Normalize losses using min-max normalization for each loss function within each configuration.
    Formula: (x - min) / (max - min)
    """
    normalized_data = defaultdict(lambda: defaultdict(dict))
    
    for config in data.keys():
        for loss_func in data[config].keys():
            losses = list(data[config][loss_func].values())
            if losses:
                min_loss = min(losses)
                max_loss = max(losses)
                
                # Avoid division by zero
                if max_loss == min_loss:
                    normalized_data[config][loss_func] = data[config][loss_func].copy()
                else:
                    for epoch, loss_value in data[config][loss_func].items():
                        normalized_value = (loss_value - min_loss) / (max_loss - min_loss)
                        normalized_data[config][loss_func][epoch] = normalized_value
    
    return normalized_data

def main():
    # Directory containing the loss files - can be either path
    #data_dir = "/scratch/trained_model"  # Original path with lattice types
    data_dir = "/net/micdata/data2/12IDC/ptychosaxs/models/ZCB_9_3D"  # New path with unet statuses
    
    # Expected epochs and loss functions
    expected_epochs = [2, 5, 10, 25, 50, 100, 150, 200, 250, 300, 400, 500]
    expected_losses = ['L1', 'L2', 'pearson_loss']
    
    # Data structure: {lattice_type/unet_status: {loss_func: {epoch: loss_value}}}
    data = defaultdict(lambda: defaultdict(dict))
    
    # Scan directory for relevant files
    print("Scanning directory for loss files...")
    files_found = 0
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt') and 'best_loss_epoch' in filename:
            epoch, loss_func, config = parse_filename(filename)
            
            if epoch and loss_func and config:
                # Process files matching the new patterns or original pattern with symmetry_0.0
                should_process = False
                if ('_symmetry_0.0.txt' in filename or  # Original pattern
                    re.match(r'best_loss_epoch_Unet_\d+\.txt$', filename) or  # L1 with Unet pattern
                    re.match(r'best_loss_epoch_\d+\.txt$', filename) or  # L1 without Unet pattern
                    re.match(r'best_loss_epoch_32_(no_Unet|Unet)_\d+_L2\.txt$', filename) or  # L2 pattern
                    re.match(r'best_loss_epoch_(no_Unet|Unet)_\d+_pearson_loss\.txt$', filename)):  # Pearson pattern
                    should_process = True
                
                if should_process:
                    filepath = os.path.join(data_dir, filename)
                    loss_value = read_loss_value(filepath)
                    
                    if loss_value is not None:
                        data[config][loss_func][epoch] = loss_value
                        files_found += 1
                        print(f"Found: {config}, {loss_func}, epoch {epoch}, loss = {loss_value:.6f}")
    
    print(f"\nTotal files processed: {files_found}")
    
    # Print model-loss file mapping to clarify relationships
    print_model_loss_mapping(data_dir)
    
    # Normalize the data
    print("\nNormalizing losses using min-max normalization...")
    normalized_data = normalize_losses(data)
    
    # Create plots for each configuration (original data)
    for config in data.keys():
        print(f"\nCreating plot for {config}...")
        
        plt.figure(figsize=(16, 12))
        
        # Define colors and markers based on configuration
        config_colors = {'Unet': 'blue', 'no_Unet': 'red', 'LatticeSC': 'blue', 'LatticeClathII': 'orange'}
        loss_markers = {'L1': 'o', 'L2': 's', 'pearson_loss': '^'}
        
        for loss_func in expected_losses:
            if loss_func in data[config]:
                epochs = []
                losses = []
                
                for epoch in expected_epochs:
                    if epoch in data[config][loss_func]:
                        epochs.append(epoch)
                        losses.append(data[config][loss_func][epoch])
                
                if epochs:  # Only plot if we have data
                    plt.plot(epochs, losses, 
                           color=config_colors.get(config, 'black'), 
                           marker=loss_markers[loss_func],
                           linewidth=2, 
                           markersize=12,
                           label=f'{loss_func} Loss',
                           alpha=0.8)
        
        plt.xlabel('Epoch', fontsize=20)
        plt.ylabel('Loss Value', fontsize=20)
        plt.title(f'Loss Comparison Across Different Loss Functions - {config}', fontsize=22)
        plt.legend(fontsize=18)
        plt.grid(True, alpha=0.3)
        plt.xscale('log')  # Log scale for epochs for better visualization
        
        # Set x-axis ticks to show actual epoch values
        plt.xticks(expected_epochs, expected_epochs, fontsize=18)
        plt.yticks(fontsize=18)
        
        plt.tight_layout()
        plt.show()
    
    # Create normalized plots for each configuration
    for config in normalized_data.keys():
        print(f"\nCreating normalized plot for {config}...")
        
        plt.figure(figsize=(16, 12))
        
        # Define colors and markers based on configuration
        config_colors = {'Unet': 'blue', 'no_Unet': 'red', 'LatticeSC': 'blue', 'LatticeClathII': 'orange'}
        loss_markers = {'L1': 'o', 'L2': 's', 'pearson_loss': '^'}
        
        for loss_func in expected_losses:
            if loss_func in normalized_data[config]:
                epochs = []
                losses = []
                
                for epoch in expected_epochs:
                    if epoch in normalized_data[config][loss_func]:
                        epochs.append(epoch)
                        losses.append(normalized_data[config][loss_func][epoch])
                
                if epochs:  # Only plot if we have data
                    if loss_func == 'pearson_loss':
                        plt.plot(epochs, losses, 
                                color=config_colors.get(config, 'black'), 
                                marker=loss_markers[loss_func],
                                linewidth=2, 
                                markersize=12,
                                label=f'NPCC Loss (Normalized)',
                                alpha=0.8)
                    else:
                        plt.plot(epochs, losses, 
                            color=config_colors.get(config, 'black'), 
                            marker=loss_markers[loss_func],
                            linewidth=2, 
                            markersize=12,
                            label=f'{loss_func} Loss (Normalized)',
                            alpha=0.8)
        
        plt.xlabel('Epoch', fontsize=20)
        plt.ylabel('Normalized Loss Value (0-1)', fontsize=20)
        plt.title(f'Normalized Loss Comparison - {config}', fontsize=22)
        plt.legend(fontsize=18)
        plt.grid(True, alpha=0.3)
        plt.xscale('log')  # Log scale for epochs for better visualization
        plt.ylim(-0.05, 1.05)  # Set y-axis with buffer room for normalized data
        
        # Set x-axis ticks to show actual epoch values
        plt.xticks(expected_epochs, expected_epochs, fontsize=18)
        plt.yticks(fontsize=18)
        
        plt.tight_layout()
        plt.show()
    
    # Create a combined plot showing all configurations
    if len(data) > 1:
        print(f"\nCreating combined plot for all configurations...")
        plt.figure(figsize=(20, 14))
        
        # Define colors and markers for different configurations and loss functions
        config_colors = {
            'Unet': 'blue', 'no_Unet': 'red',  # For unet statuses
            'LatticeSC': 'blue', 'LatticeClathII': 'orange'  # For lattice types
        }
        loss_markers = {'L1': 'o', 'L2': 's', 'pearson_loss': '^'}
        
        for config in data.keys():
            for loss_func in expected_losses:
                if loss_func in data[config]:
                    epochs = []
                    losses = []
                    
                    for epoch in expected_epochs:
                        if epoch in data[config][loss_func]:
                            epochs.append(epoch)
                            losses.append(data[config][loss_func][epoch])
                    
                    if epochs:  # Only plot if we have data
                        plt.plot(epochs, losses, 
                               color=config_colors.get(config, 'black'), 
                               marker=loss_markers[loss_func],
                               linewidth=2, 
                               markersize=12,
                               label=f'{config} - {loss_func}',
                               alpha=0.8)
        
        plt.xlabel('Epoch', fontsize=20)
        plt.ylabel('Loss Value', fontsize=20)
        plt.title('Combined Loss Comparison: All Configurations and Loss Functions', fontsize=22)
        plt.legend(fontsize=16, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.xscale('log')  # Log scale for epochs for better visualization
        
        # Set x-axis ticks to show actual epoch values
        plt.xticks(expected_epochs, expected_epochs, fontsize=18)
        plt.yticks(fontsize=18)
        
        plt.tight_layout()
        plt.show()
    
    # Create a combined normalized plot showing all configurations
    if len(normalized_data) > 1:
        print(f"\nCreating combined normalized plot for all configurations...")
        plt.figure(figsize=(20, 14))
        
        # Define colors and markers for different configurations and loss functions
        config_colors = {
            'Unet': 'blue', 'no_Unet': 'red',  # For unet statuses
            'LatticeSC': 'blue', 'LatticeClathII': 'orange'  # For lattice types
        }
        loss_markers = {'L1': 'o', 'L2': 's', 'pearson_loss': '^'}
        
        for config in normalized_data.keys():
            for loss_func in expected_losses:
                if loss_func in normalized_data[config]:
                    epochs = []
                    losses = []
                    
                    for epoch in expected_epochs:
                        if epoch in normalized_data[config][loss_func]:
                            epochs.append(epoch)
                            losses.append(normalized_data[config][loss_func][epoch])
                    
                    if epochs:  # Only plot if we have data
                        plt.plot(epochs, losses, 
                               color=config_colors.get(config, 'black'), 
                               marker=loss_markers[loss_func],
                               linewidth=2, 
                               markersize=12,
                               label=f'{config} - {loss_func} (Normalized)',
                               alpha=0.8)
        
        plt.xlabel('Epoch', fontsize=20)
        plt.ylabel('Normalized Loss Value (0-1)', fontsize=20)
        plt.title('Combined Normalized Loss Comparison: All Configurations and Loss Functions', fontsize=22)
        plt.legend(fontsize=16, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.xscale('log')  # Log scale for epochs for better visualization
        plt.ylim(-0.05, 1.05)  # Set y-axis with buffer room for normalized data
        
        # Set x-axis ticks to show actual epoch values
        plt.xticks(expected_epochs, expected_epochs, fontsize=18)
        plt.yticks(fontsize=18)
        
        plt.tight_layout()
        plt.show()
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print("=" * 50)
    for config in data.keys():
        print(f"\n{config}:")
        for loss_func in expected_losses:
            if loss_func in data[config]:
                losses = list(data[config][loss_func].values())
                if losses:
                    print(f"  {loss_func} (Original): min={min(losses):.6f}, max={max(losses):.6f}, mean={np.mean(losses):.6f}")
                    
            if loss_func in normalized_data[config]:
                norm_losses = list(normalized_data[config][loss_func].values())
                if norm_losses:
                    print(f"  {loss_func} (Normalized): min={min(norm_losses):.6f}, max={max(norm_losses):.6f}, mean={np.mean(norm_losses):.6f}")

if __name__ == "__main__":
    main()

# %%
