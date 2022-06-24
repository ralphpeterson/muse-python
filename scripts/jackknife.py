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

def jackknife(mic_pos,speaker_pos,
              stimulus,sr_stimulus, sr,
              x_dim, y_dim, z_dim, materials):

    n_mic = mic_pos.shape[1]
    
    loc_est = []
    for remove_mic_idx in tqdm(range(n_mic)):
        room = fun.generate_room(x_dim, y_dim, z_dim, materials)

        # remove one microphone
        mic_jk = np.delete(mic_pos.T,remove_mic_idx,0).T
    
        # Play audio and simulate
        # fun.place_mics(room, mic_jk)
        _ = fun.orient_mic(room, mic_jk)
        sr_stimulus, stimulus = \
            fun.add_source(speaker_pos, stimulus, sr_stimulus, room)
        room.simulate()
        
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


