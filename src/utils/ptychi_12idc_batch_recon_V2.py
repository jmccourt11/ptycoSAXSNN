import os
import ptychi.pear as pear
import time
import logging
import traceback
import glob


scan_start=3021
scanNo_end = 3384  # set to '' to get up-to-date scan number

number_of_iterations=1000
diff_pattern_size_pix = 128   #the size used in this reconstruction, should be equal and smaller than preprocess_size
preprocess_size =256 #in the filename of  data_roi0_Ndp{:03d}


# Get current directory and go up one level for data_main_dir
#current_dir = os.path.dirname(os.path.abspath(__file__))
#data_main_dir = os.path.dirname(current_dir)
data_main_dir ='/mnt/micdata2/12IDC/2025_Feb'
print(f'{data_main_dir=}')
h5_dir=os.path.join(data_main_dir,'ptycho1')

reconstructed_probe_file = '/mnt/micdata2/12IDC/2025_Feb/ptychi_recons/S{:04d}/Ndp128_LSQML_c1000_m0.5_p10_cp_mm_opr3_ic_pc_ul2/recon_Niter1000.h5' 
init_probe_path='/mnt/micdata2/12IDC/2025_Feb/ptychi_recons/S3176/Ndp128_LSQML_c1000_m0.5_p15_cp_mm_opr3_ic_pc_ul2/recon_Niter1000.h5'  

while True:
	if not scanNo_end:
		subdirs = [int(d) for d in os.listdir(h5_dir) 
					if os.path.isdir(os.path.join(h5_dir, d)) and d.isdigit() 
					and int(d) >= 0 and d.zfill(3) =='{:03d}'.format(int(d))]
		max_dir = os.path.join(h5_dir, str(max(subdirs)))

		print(f'{max_dir=}')
		
		# Get initial state of directory
		initial_file_count = len(glob.glob(os.path.join(max_dir, '*')))
		initial_mod_time = os.path.getmtime(max_dir)
		print(f'Initial mod time: {initial_mod_time}')	# Wait and check for changes
		time.sleep(5)
		# Get current state of directory
		current_file_count = len(glob.glob(os.path.join(max_dir, '*')))
		current_mod_time = os.path.getmtime(max_dir)
		print(f'Current mod time: {current_mod_time}')
		
		# Determine if directory is still being written to
		if current_mod_time != initial_mod_time or current_file_count != initial_file_count:
			print(f"*************************************************************************")
			print(f"!!!Directory is still changing - scan {max(subdirs):03d} is in progress!!!!")
			print(f"*************************************************************************")
			scanNo_max = max(subdirs)
		else:
			print(f"*********Directory is stable - scan {max(subdirs):03d} is complete*******")
			scanNo_max = max(subdirs) + 1

	else:
		scanNo_max = scanNo_end + 1

	print(f'{scanNo_max}')

	for scan_num in range (scan_start,scanNo_max,1):
		#Getting sample name
		files= glob.glob(os.path.join(data_main_dir, 'ptycho1', '{:03d}'.format(scan_num), '*{:03d}_00001*.h5'.format(scan_num)))
		if not files:
			print(f"****No data found in scan{scan_num}! Please check if the data exists!****")
			continue		
		sample_name = files[0].split('/')[-1].split('{:03d}_'.format(scan_num))[0]
		print(f'{sample_name=}')

		# Check if reconstruction already exists
		if os.path.isfile(reconstructed_probe_file.format(scan_num)):
			print(f'fly{scan_num} has been reconstructed, delete the result if you want to reprocess!')
			continue
		# Check if preprocessed data exists
		data_file_path = os.path.join(data_main_dir, 'results', sample_name, f'fly{scan_num:03d}', f'data_roi0_Ndp{preprocess_size}_dp.hdf5')
		if not os.path.isfile(data_file_path):
			print(f'fly{scan_num} data doesn\'t exist! Please preprocess the data first!')
			continue

		
		params = {
			'data_directory': data_main_dir,
			#path_to_init_probe': reconstructed_probe_file.format(scan_num-1) if os.path.isfile(reconstructed_probe_file.format(scan_num-1)) else init_probe_path,  # init_probe_path,
			'path_to_init_probe':  init_probe_path,
			'path_to_init_object': '',                  #init_object_path,
    		'path_to_init_positions': '',   #path to init position
			'scan_num': scan_num,
			'instrument': '12idc',
			'beam_energy_kev': 10.,
			'det_sample_dist_m': 10.2,   #saxs 10, waxs9.77
			'diff_pattern_size_pix': diff_pattern_size_pix,
			'diff_pattern_center_x': 743,
			'diff_pattern_center_y': 720,
			'load_processed_hdf5': True,
			'path_to_processed_hdf5_dp': os.path.join(data_main_dir, 'results', sample_name, f'fly{scan_num:03d}', f'data_roi0_Ndp{preprocess_size}_dp.hdf5'),
			'path_to_processed_hdf5_pos': os.path.join(data_main_dir, 'results', sample_name, f'fly{scan_num:03d}', f'data_roi0_Ndp{preprocess_size}_para.hdf5'),
			'position_correction': True,
			'position_correction_update_limit': 2,
			'position_correction_affine_constraint': False,
			'intensity_correction': True,
			'center_probe': True,
			'number_probe_modes':10,
			'update_object_w_higher_probe_modes': True,
			'number_opr_modes':3,
			'update_batch_size': 1000,
			'batch_selection_scheme': 'compact',
			'momentum_acceleration': True,
			'number_of_slices': 1,
			'object_thickness_m': 8e-6,
			#'slice_distance': 0,
			'layer_regularization': 0.1, 
    		'position_correction_layer': None,
			'number_of_iterations': number_of_iterations,
			'recon_parent_dir': '',
			'recon_dir_suffix': '',
			'save_freq_iterations': 100,
			'recon_dir_suffix':'',
			'gpu_id': None,
			'save_diffraction_patterns': False
		}

		try:
			# Check if required files exist
			file_paths = [
				params['path_to_init_probe'],
				params['path_to_processed_hdf5_dp'],
				params['path_to_processed_hdf5_pos']
			]
			
			for path in file_paths:
				if not os.path.exists(path):
					raise ValueError(f"Required file not found: {path}")

			# Attempt reconstruction
			pear.ptycho_recon(**params)

			print(f"Successfully completed reconstruction for scan {scan_num}")

		except Exception as e:
			print(f"\nError in reconstruction for scan {scan_num}:")
			print(f"Error type: {type(e).__name__}")
			print(f"Error message: {str(e)}")
			print("\nFull traceback:")
			traceback.print_exc()
			
			# Log parameter state for debugging
			# print("\nParameter values at time of error:")
			# for key, value in sorted(params.items()):
			# 	print(f"  {key}: {value}")
			
			print(f"\nSkipping scan {scan_num} and continuing with reconstruction of next scan...")
			continue
	time.sleep(10)
    
    
