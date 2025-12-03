% load the data
img, phi = load('../example/05_Postprocessing/combined_diffraction_patterns_ZCB_9_3D_best_model_Lattice400_Typeclathrate_II_DP256_Sim256_Grid1024_nsteps4_nscans60_centerConc0_sim_ZCB_9_3D_S5065_N60_steps4_dp256_Unet_epoch_10_pearson_loss_processed.mat');

% set the parameters
saxs.SDD = 11.;%5570=5.57 m
saxs.waveln=0.0789; %1.23984 \AA
%saxs.SDD = 10.200;
%saxs.waveln = 0.123984; % 1.23984 Å
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


%%
%inp_data.mask = mask;
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

[Qv_d, DATA_d] = construct_RecpSpace_fromImgMtrx(inp_data, saxs, ROIX, ROIY, qN, qmax);

% Save DATA to MAT file
%output_file = 'reciprocal_space_data.mat';
%save(output_file, 'DATA');
%disp(sprintf('Data saved to %s', output_file));
 
draw_3dmap(DATA_d-mean(D,[Qv_d(:,1),Qv_d(:,2),Qv_d(:,3)]))