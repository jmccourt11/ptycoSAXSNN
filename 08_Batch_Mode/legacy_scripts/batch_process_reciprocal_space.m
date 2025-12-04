function batch_process_reciprocal_space()
% Batch processing script for reciprocal space construction
% Processes multiple input files and saves output to corresponding directories
%
% This script automatically finds all matching input files and processes them
% using the construct_RecpSpace_fromImgMtrx function

%% Configuration - SAXS parameters (same as original script)
saxs.SDD = 10.200;
saxs.waveln = 0.123984; % 1.23984 Å
saxs.tthi = 0;
saxs.ai = 0;
saxs.edensity = 0;
saxs.beta = 0;
saxs.center = [127.5, 129.5];
saxs.psize = 172*10^(-6);

% Processing parameters
ROIX = [1:256];
ROIY = [1:256];
qN = 100;
qmax = 0.2;

%% Define all possible parameter combinations
base_path = 'Z:\12IDC\ptychosaxs\batch_mode_250\RSM\';
lattice_types = {'SC', 'ClathII'};
noise_conditions = {'Noise', 'noNoise'};
epochs = {'50', '250'};
loss_types = {'L1', 'L2', 'pearson_loss'};

% Initialize counters
total_files = 0;
processed_files = 0;
failed_files = 0;
failed_list = {};

fprintf('Starting batch processing of reciprocal space data...\n');
fprintf('Base path: %s\n', base_path);

%% Process all combinations
for i = 1:length(lattice_types)
    for j = 1:length(noise_conditions)
        for k = 1:length(epochs)
            for l = 1:length(loss_types)
                
                % Construct directory path
                dir_name = sprintf('best_model_Lattice%s_Probe256x256_ZCB_9_3D__%s_sim_ZCB_9_3D_S5065_N600_steps4_dp256_Unet_epoch_%s_%s_symmetry_0.0', ...
                    lattice_types{i}, noise_conditions{j}, epochs{k}, loss_types{l});
                
                full_dir_path = fullfile(base_path, dir_name);
                input_file = fullfile(full_dir_path, 'combined_diffraction_patterns_processed.mat');
                output_file = fullfile(full_dir_path, 'DECONV_RSM_PROCESSED.mat');
                
                total_files = total_files + 1;
                
                fprintf('\n--- Processing file %d ---\n', total_files);
                fprintf('Lattice: %s, Noise: %s, Epoch: %s, Loss: %s\n', ...
                    lattice_types{i}, noise_conditions{j}, epochs{k}, loss_types{l});
                fprintf('Input file: %s\n', input_file);
                
                % Check if input file exists
                if ~exist(input_file, 'file')
                    fprintf('WARNING: Input file does not exist, skipping...\n');
                    failed_files = failed_files + 1;
                    failed_list{end+1} = sprintf('%s - File not found', input_file);
                    continue;
                end
                
                % Check if output file already exists
                if exist(output_file, 'file')
                    fprintf('WARNING: Output file already exists, skipping...\n');
                    fprintf('Output file: %s\n', output_file);
                    continue;
                end
                
                try
                    % Load the input data
                    fprintf('Loading input data...\n');
                    loaded_data = load(input_file);
                    
                    % Check if required variables exist in the loaded data
                    if ~isfield(loaded_data, 'img') || ~isfield(loaded_data, 'phi')
                        fprintf('ERROR: Required variables (img, phi) not found in input file\n');
                        failed_files = failed_files + 1;
                        failed_list{end+1} = sprintf('%s - Missing required variables', input_file);
                        continue;
                    end
                    
                    img = loaded_data.img;
                    phi = loaded_data.phi;
                    
                    fprintf('Data loaded successfully. Image size: %dx%dx%d\n', size(img));
                    
                    % Prepare input data structure
                    inp_data.mask = ones(size(img(:,:,1)));
                    inp_data.img_mtrx = img;
                    inp_data.phi = phi;
                    inp_data.norm_factor = ones(size(phi));
                    inp_data.isfliped = false;
                    inp_data.background = false;
                    inp_data.gen_back = false;
                    inp_data.switch_axes = false;
                    
                    % Switch axes if requested
                    if inp_data.switch_axes
                        inp_data.img_mtrx = permute(inp_data.img_mtrx, [2 1 3]);
                        inp_data.mask = permute(inp_data.mask, [2 1]);
                    end
                    
                    % Process the data
                    fprintf('Processing reciprocal space construction...\n');
                    [Qv_d, DATA_d] = construct_RecpSpace_fromImgMtrx(inp_data, saxs, ROIX, ROIY, qN, qmax);
                    
                    % Save the results
                    fprintf('Saving results to: %s\n', output_file);
                    save(output_file, 'DATA_d', 'Qv_d');
                    
                    processed_files = processed_files + 1;
                    fprintf('SUCCESS: File processed and saved successfully!\n');
                    
                catch ME
                    fprintf('ERROR: Failed to process file\n');
                    fprintf('Error message: %s\n', ME.message);
                    failed_files = failed_files + 1;
                    failed_list{end+1} = sprintf('%s - %s', input_file, ME.message);
                end
            end
        end
    end
end

%% Summary report
fprintf('\n=== BATCH PROCESSING COMPLETE ===\n');
fprintf('Total files checked: %d\n', total_files);
fprintf('Successfully processed: %d\n', processed_files);
fprintf('Failed/Skipped: %d\n', failed_files);

if ~isempty(failed_list)
    fprintf('\nFailed files:\n');
    for i = 1:length(failed_list)
        fprintf('  %d. %s\n', i, failed_list{i});
    end
end

fprintf('\nBatch processing completed at: %s\n', datestr(now));

end
