# Load Packages
import numpy as np
import pyroomacoustics as pra
import librosa

# import os
# os.getcwd()

import sys
sys.path.insert(1, '/mnt/home/ayamaguchi/scripts/function_files')

import function_files.functions as fun
import function_files.util as muse
import function_files.jackknife_main as jk

# MUSE function
def run_MUSE(room_dims:np.ndarray,      # room dimensions (small or large)
             mic_pos:np.ndarray,        # microphone positions (3 x n_mic)
             speaker_pos:np.ndarray,    # speaker position (3 x 1)
             eval_mode:str,             # evaluation mode (snippets or jackknife)
             wall_type:str,             # wall type (reverberant or non-reverberant)
             signal:np.ndarray,         # audio signal
             sampling_rate:int=125000,  # microphone sample rate
             dir_val:int=3              # 1:center, 2:horizontal out, 3:down, 4: horizontally in
             ): 

    # Convert the indices (bins) to (x,y) coordinates
    n_mic = mic_pos.shape[1]
    dir_idx = np.ones(n_mic)*dir_val 

    # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
    # Create a room (Next: make this into a function?)
    # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
    if wall_type=='rev':
        ceiling_floor, wall_material = "fibre_absorber_2", "hard_surface"
    elif wall_type=='nrev':
        ceiling_floor, wall_material = "curtains_fabric_folded", "curtains_fabric_folded"
    else:
        raise ValueError('wall type is not specified.')
    materials = pra.make_materials(
        ceiling = ceiling_floor, 
        floor = ceiling_floor,
        east = wall_material, 
        west = wall_material, 
        north = wall_material, 
        south = wall_material)
    
    room = pra.ShoeBox(
        [room_dims[0], room_dims[1], room_dims[2]],
        fs=sampling_rate,
        materials=materials,
        max_order=9  # Maximum number of times a sound wave can reflect
    ) 

    # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
    # Add microphones and a source
    # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
    _, mic_dir,directivities = fun.place_mics(room, mic_pos, dir_idx)
    # Add microphones
    room.add_microphone_array(mic_array=mic_pos, directivity=directivities)
    # Add sound source
    if room.fs != sampling_rate:
        print('resampling audio')
        signal_resampled = librosa.resample(signal.astype('float'), 
                                              orig_sr=sampling_rate,
                                              target_sr=room.fs)
        room.add_source(speaker_pos, signal=signal_resampled, delay=0)
    else:
        room.add_source(speaker_pos, signal=signal, delay=0)
    # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
    # Simulate
    # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
    room.simulate()

    # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
    # Evaluate
    # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-    
    # Set MUSE (Mouse Ultrasonic Source Estimator) parameters
    x_grid, y_grid = muse.make_xy_grid(room_dims[0], room_dims[1], resolution=0.0025)
    in_cage = None                      # Unused argument
    audio = room.mic_array.signals.T    # Transpose shape from (n_mics, n_samp) to (n_samp, n_mics)
    f_lo, f_hi = 0, 62500               # frequency range of interest
    temp = 20                           # Temperature oF environment (C)
    
    # Run MUSE
    if eval_mode=='sn': # Snippets
        r_est, _, rsrp_grid, _, _, _, _, _, _ \
            = muse.r_est_from_clip_simplified(audio, sampling_rate,
                                            f_lo,f_hi,
                                            temp,
                                            x_grid,y_grid,
                                            in_cage, mic_pos, 1)
    elif eval_mode=='jk': # Use the r_avg (average estimate) of the jackknife
        r_est, _, _ =\
            jk.r_est_jackknife(audio.T, sampling_rate,\
                f_lo, f_hi, temp, room_dims[0],room_dims[1], 0.0025, mic_pos.T)
    else:
        raise ValueError('Evaluation mode is not specified')

    err = np.sqrt(sum((np.array(speaker_pos[:2]) - r_est.flatten())**2))
    # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
    # Return the result
    # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
    return err, r_est