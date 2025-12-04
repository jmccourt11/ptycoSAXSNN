# Ptychography-SAXS Deconvolution Neural Network (pty-co-SAXSNN)

Deep learning framework for deconvolving SAXS diffraction patterns using ptychography probe information. Removes probe-induced convolution effects to recover ideal diffraction patterns for structural analysis.

## Project Structure

Sequential workflow pipeline:

- **`00_Intro/`** - Introduction and overview materials
- **`01_Data_Generation/`** - Simulation scripts for training data (probe/lattice-based simulations)
- **`02_Preprocess/`** - Data preprocessing and normalization
- **`03_Train_Validation_Test/`** - Model training and validation, inference on simulated data
- **`04_Inference_Experimental/`** - Application to experimental data, visualization
- **`05_Postprocessing/`** - Organizing deconvolved results into H5/mat files, contains **`matlab/`**  which has MATLAB tools (SAXS4assembly, SAXSimagviewer)
- **`06_Tomographic_Analysis/`** - 3D analysis, reciprocal space mapping (RSM)
- **`07_Performance_Metrics/`** - Model evaluation and comparison
- **`08_Batch_Mode/`** - Batch processing scripts
- **`09_Clathrate_Lattice/`** - Colloidal clathrate assembly tools
- **`src/models/`** - Neural network model definitions (U-Net, encoder-decoder)
- **`src/utils/`** - Utility functions (data processing, visualization, tomographic)
- **`data/`** - Reference data, lattice files, masks
- **`example/`** - Example notebook results

## Key Features

- Encoder-decoder networks for probe deconvolution
- Physics-based simulation pipeline for training data
- Experimental data processing for APS 12-ID-E beamline

## Technical Stack

- **Framework**: PyTorch
- **Input/Output**: 256×256 diffraction patterns
- **Architecture**: U-Net (with skip connections)
- **Loss Functions**: Pearson correlation, L1/L2 with symmetry penalties
