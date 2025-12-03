function process_specific_files(lattice_types, noise_conditions, epochs, loss_types)
% Process specific combinations of reciprocal space files
% 
% Usage examples:
%   process_specific_files({'SC'}, {'Noise'}, {'250'}, {'pearson_loss'})
%   process_specific_files({'SC', 'ClathII'}, {'noNoise'}, {'50', '250'}, {'L1'})
%
% Parameters:
%   lattice_types   - cell array of lattice types: {'SC', 'ClathII'}
%   noise_conditions - cell array of noise conditions: {'Noise', 'noNoise'}  
%   epochs          - cell array of epochs: {'50', '250'}
%   loss_types      - cell array of loss types: {'L1', 'L2', 'pearson_loss'}

if nargin < 4
    error('All four parameter arrays must be provided');
end

%% Configuration - SAXS parameters
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

%% Process specified combinations
base_path = 'Z:\12IDC\ptychosaxs\batch_mode_250\RSM\';

processed_count = 0;
total_combinations = length(lattice_types) * length(noise_conditions) * length(epochs) * length(loss_types);

fprintf('Processing %d specific file combinations...\n', total_combinations);

for i = 1:length(lattice_types)
    for j = 1:length(noise_conditions)
        for k = 1:length(epochs)
            for l = 1:length(loss_types)
                
                % Construct paths
                dir_name = sprintf('best_model_Lattice%s_Probe256x256_ZCB_9_3D__%s_sim_ZCB_9_3D_S5065_N600_steps4_dp256_Unet_epoch_%s_%s_symmetry_0.0', ...
                    lattice_types{i}, noise_conditions{j}, epochs{k}, loss_types{l});
                
                full_dir_path = fullfile(base_path, dir_name);
                input_file = fullfile(full_dir_path, 'combined_diffraction_patterns_processed.mat');
                output_file = fullfile(full_dir_path, 'DECONV_RSM_PROCESSED.mat');
                
                fprintf('\nProcessing: %s_%s_%s_%s\n', lattice_types{i}, noise_conditions{j}, epochs{k}, loss_types{l});
                
                if ~exist(input_file, 'file')
                    fprintf('  Input file not found: %s\n', input_file);
                    continue;
                end
                
                if exist(output_file, 'file')
                    fprintf('  Output already exists, skipping: %s\n', output_file);
                    continue;
                end
                
                try
                    % Load and process
                    loaded_data = load(input_file);
                    img = loaded_data.img;
                    phi = loaded_data.phi;
                    
                    % Prepare input data
                    inp_data.mask = ones(size(img(:,:,1)));
                    inp_data.img_mtrx = img;
                    inp_data.phi = phi;
                    inp_data.norm_factor = ones(size(phi));
                    inp_data.isfliped = false;
                    inp_data.background = false;
                    inp_data.gen_back = false;
                    inp_data.switch_axes = false;
                    
                    % Process
                    [Qv_d, DATA_d] = construct_RecpSpace_fromImgMtrx(inp_data, saxs, ROIX, ROIY, qN, qmax);
                    
                    % Save
                    save(output_file, 'DATA_d', 'Qv_d');
                    processed_count = processed_count + 1;
                    fprintf('  SUCCESS: Saved to %s\n', output_file);
                    
                catch ME
                    fprintf('  ERROR: %s\n', ME.message);
                end
            end
        end
    end
end

fprintf('\nCompleted processing %d out of %d combinations.\n', processed_count, total_combinations);

end
