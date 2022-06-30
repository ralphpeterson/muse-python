#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 09:51:38 2022

@author: akihiro

    Jackknife implementation
    
    Ref: https://doi.org/10.1016/j.jneumeth.2017.12.013
"""
import numpy as np
from tqdm import tqdm
from scripts import functions as fun
from muse import util as muse

def jackknife(mic_pos,dir_idx, 
              speaker_pos,
              stimulus,sr_stimulus, sr,
              x_dim, y_dim, z_dim, materials):

    n_mic = mic_pos.shape[1]
    
    loc_est = []
    for remove_mic_idx in tqdm(range(n_mic)):
        # Generate a room
        room = fun.generate_room(x_dim, y_dim, z_dim, materials)

        # remove one microphone
        mic_jk = np.delete(mic_pos.T,remove_mic_idx,0).T
        dir_jk = np.delete(dir_idx, remove_mic_idx)

        # Orient the microphones (This is already defined outside the function.)
        # This shouldn't need to be redefined here. (Issue <-- needs to be updated, check Aman's code also)
        # if mic_orient == 'center':
        #     fun.place_mics_center(room, mic_jk)
        # elif mic_orient == 'h_out':
        #     fun.place_mics_h_out(room, mic_jk)
        # elif mic_orient == 'down':
        #     fun.place_mics_down(room, mic_jk)
        # else:
        #     raise ValueError('Specify the microphone orientation correctly.')
        fun.place_mics(room, mic_jk, dir_jk)
        
        # Play audio and simulate
        sr_stimulus, stimulus = \
            fun.add_source(speaker_pos, stimulus, sr_stimulus, room)
        room.simulate()
        
        # set MUSE parameters
        audio,fs,f_lo,f_hi,temp,x_grid,y_grid,in_cage,R = \
            fun.set_MUSE_params(room, x_dim, y_dim, mic_jk, sr, 0.0025)

        # Run MUSE
        r_est, _, _, _, _, _, _, _, _ \
        = muse.r_est_from_clip_simplified(audio,
            fs,
            f_lo, f_hi,
            temp,
            x_grid, y_grid,
            in_cage,
            R,  # Expected to have shape (3, n_mics)
            1)
        
        del room
        loc_est.append(r_est)
    
        r_mean = np.mean(loc_est,axis=0)
        r_std = np.std(loc_est, axis=0)
        
    return r_mean, r_std


