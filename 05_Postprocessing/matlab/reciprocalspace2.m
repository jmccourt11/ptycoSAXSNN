img, phi = load('../example/05_Postprocessing/combined_diffraction_patterns_ZCB_9_3D_best_model_Lattice400_Typeclathrate_II_DP256_Sim256_Grid1024_nsteps4_nscans60_centerConc0_sim_ZCB_9_3D_S5065_N60_steps4_dp256_Unet_epoch_10_pearson_loss_processed.mat');

saxs.SDD = 11.; 
saxs.waveln=0.0789; %0.123984; %1.23984 \AA
saxs.tthi = 0;
saxs.ai = 0;
saxs.edensity = 0;
saxs.beta = 0;
ROIX = [1:300];
ROIY = [1:300];
saxs.center=[150,150];
qN = 150;
qmax = 0.2;

recon_pixel_size=39;
pixel_size=2*pi/recon_pixel_size/numel(ROIX);
saxs.pxQ = pixel_size; %q-space
saxs.psize=saxs.waveln*saxs.pxQ*saxs.SDD/(2*pi);%real-space

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

[Qv, DATA] = construct_RecpSpace_fromImgMtrx(inp_data, saxs, ROIX, ROIY, qN);

% swap y and x
draw_3dmap(DATA, [Qv(:,2),Qv(:,1),Qv(:,3)]);
