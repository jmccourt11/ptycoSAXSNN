% Runner script for batch processing reciprocal space data
% 
% This script provides examples of how to run the batch processing functions

%% Option 1: Process ALL possible combinations (48 total combinations)
% Uncomment the line below to process all files:
% batch_process_reciprocal_space();

%% Option 2: Process specific combinations only
% Examples of processing specific subsets:

% Process only ClathII lattice with pearson_loss
% process_specific_files({'ClathII'}, {'Noise', 'noNoise'}, {'50', '250'}, {'pearson_loss'});

% Process only epoch 250 files
% process_specific_files({'SC', 'ClathII'}, {'Noise', 'noNoise'}, {'250'}, {'L1', 'L2', 'pearson_loss'});

% Process only noNoise conditions
% process_specific_files({'SC', 'ClathII'}, {'noNoise'}, {'50', '250'}, {'L1', 'L2', 'pearson_loss'});

%% Option 3: Process a single specific file
% Example: Process SC lattice, Noise condition, epoch 250, pearson_loss
% process_specific_files({'SC'}, {'Noise'}, {'50'}, {'pearson_loss'});


process_specific_files({'ClathII'}, {'Noise'}, {'250'}, {'pearson_loss'});
process_specific_files({'ClathII'}, {'Noise'}, {'50'}, {'pearson_loss','L1','L2'});


fprintf('Batch processing script completed.\n');
fprintf('Check the output for processing results.\n');
