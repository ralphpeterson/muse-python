#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 12:13:59 2022
@author: akihiro

    multiprocessing test: SIFT approach

Reference: 
    https://github.com/amanchoudhri/calibrationtools/blob/a560117f12965235f3684cb94938a6061c484bb6/calibrationtools/calculate.py#L168
    
"""
import argparse
from mailbox import NotEmptyError
import multiprocessing as mp
from multiprocessing.sharedctypes import Value
import numpy as np
import itertools
from scipy.io import wavfile
import os
import time

# import logging
# logging.basicConfig(level=logging.DEBUG)
# logger = mp.log_to_stderr()

import sys
sys.path.insert(1, '/mnt/home/ayamaguchi/scripts/function_files')
from function_files.run_MUSE import run_MUSE
import function_files.mic_pos_f as mpos
import function_files.mic_arrays as ma

def f(room_dims: np.ndarray, 
      mic_pos: np.ndarray, 
      speaker_pos: np.ndarray,
      signal: np.ndarray,
      src_idx: int,
      wall_type: str,
      eval_mode:str='jk'
      ):

    # fs = 192000 # sampling rate of the microphone? (-- from )
    fs = 125000 # sampling rate of the microphone (-- from Ralph: CM16/CMPA48AAF-5V)
    # mode='sn' for 2 mic. 'jk' for 3 or more mics.
    err, _ = run_MUSE(room_dims, mic_pos, speaker_pos, eval_mode, wall_type, signal, sampling_rate=fs, dir_val=4)
    # print(err)
    # logger.warning(f'current n_mics: {mic_pos.shape[1]}')

    return err, src_idx

def n_xy(room_type:str):
    ''' Get the room type and output the number of grid points (nx, ny) and the room dimensions
    '''
    if room_type=='small':
        rd = np.array([0.40, 0.60, 0.355]) # room_dimensions (small room)
        d = 1.25
    elif room_type=='large':
        rd = np.array([1.83, 2.44, 0.76]) # room_dimensions (large room)
        d = 5
    else:
        raise ValueError('room type is not specified')
    nx, ny = int(-(rd[0]*100//-d)), int(-(rd[1]*100//-d))
    return nx, ny, rd

def get_mic_pos(rd:np.ndarray, opt_type:str, opt_var:int, mic_config:str, offset:float):
    if mic_config=='wall':
        if opt_type=='hght':
            z_val = opt_var/1000 # in m. opt_var:[5..345..5] for the small room
            mic_pos = ma.mic_edges4n(4,rd[0],rd[1], z_val, offset)
            var_temp = z_val
        else:
            raise ValueError('Wrong optimization type or undefined')
    elif mic_config=='ceiling':
        if opt_type=='diag' or opt_type=='vert':
            if opt_var<=0 or opt_var>32:
                raise ValueError('Invalid opt_var')
            else:
                alpha = 0.03*opt_var
            mpos_2D = mpos.f(alpha, rd, opt_type)
            mic_pos = np.vstack((mpos_2D.T, np.ones(4)*(rd[2]-offset)))
            var_temp = alpha
        elif opt_type=='rot':
            if opt_var<=0 or opt_var>180:
                raise ValueError('Invalid opt_var')
            else:
                theta = opt_var # theta = 1~180 (int)
            mpos_2D = mpos.g(theta, rd, beta=0.4, alpha=0.9)
            mic_pos = np.vstack((mpos_2D.T, np.ones(4)*(rd[2]-offset)))
            var_temp = theta
        else:
            raise ValueError('optimization type is not defined properly')
    else:
        raise ValueError('mic_config is not defined properly')
    
    return mic_pos, var_temp

def compute(
    opt_type: str,      # Optimization type: 'diag', 'vert', 'rot', or 'hght'
    opt_var: float,     # Optimization variable
    signal: np.ndarray, # Audio signal
    room_type: str,     # Room type: 'small' or 'large'
    mic_config: str,    # Microphone configuration type: 'wall' or 'ceiling'
    wall_type: str,     # Wall type: 'rv' or 'nrev'
    eval_type: str      # Evaluation type: 'sn' or 'jk'
    ):
    """
    Given an array of mic and source locations, find the minimum max error
    between the true source location and the predicted location for mic config, 
    using multiprocessing across all the source locations in the arena.
    """
    # Environment/Grid
    nx, ny, rd = n_xy(room_type)
    xs, ys = np.arange(nx), np.arange(ny)

    # Nall = np.array(np.meshgrid(np.arange(nx), np.arange(ny))).T.reshape(-1,2)
    offset = 0.01
    x_ary = np.linspace(offset,rd[0]-offset, nx)
    y_ary = np.linspace(offset,rd[1]-offset, ny)

    mic_pos, var_temp = get_mic_pos(rd, opt_type, opt_var, mic_config, offset)

    # Iteration arguments
    def arg_iter():
        # print('mic config: '+mic_config)
        # print('opt_type: '+opt_type)

        # print(mic_pos)
        for src_idx in itertools.product(xs, ys):
            speaker_pos = np.array([x_ary[src_idx[0]],y_ary[src_idx[1]], 0.01])
            yield (rd, mic_pos, speaker_pos, signal, src_idx, wall_type, eval_type)

    # with mp.Pool(processes=mp.cpu_count()) as pool:
    with mp.Pool(processes=96) as pool:
        print('Processes used   :', 96,'/', mp.cpu_count())
        arg_iterator = arg_iter()
        results = pool.starmap(f, arg_iterator)

    return np.array(results, dtype=object), mic_pos, var_temp

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()  # Add an argument
    parser.add_argument('--opt_var', type=int, required=True) # Parse the argument
    args = parser.parse_args()
    opt_var = args.opt_var

    # Select the stimulus
    # stim_dir = '/mnt/home/ayamaguchi/stimuli/sine_2khz.wav'
    stim_dir = '/mnt/home/ayamaguchi/stimuli/gerbil-warble.wav'
    stim_name = stim_dir.split('/')[-1]
    fs, signal = wavfile.read(os.path.abspath(stim_dir))

    # -*-*-*-*-*-*-*-*-*-*- Define the room type/search type -*-*-*-*-*-*-*-*-*-*-
    room_type   = 'small'         # 'small' or 'large'
    wall_type   = 'rev'           # 'rev' or 'nrev'
    eval_type   = 'jk'            # evaluation type: 'sn' or 'jk'
    mic_config  = 'ceiling'       # 'wall' or 'ceiling'
    opt_type    = 'rot'          # optimization type: 'diag', 'vert', 'rot', or 'hght'
    # opt_type    = 'vert'          # optimization type: 'diag', 'vert', 'rot', or 'hght'
    dir_name = room_type+'_'+wall_type+'_'+eval_type+'_'+mic_config+'_'+opt_type

    # print('Optimization variable: %1.0f'%opt_var)
    print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')
    print('Source frequency : %1.0f Hz'%fs)
    print('Sampling freq.   : %1.0f Hz'%125000)
    print('Evaluation type  : '+dir_name)

    # -*-*-*-*-*-*-*-*-*-*- Optimal Mic Config Search -*-*-*-*-*-*-*-*-*-*-
    print('Evaluating...    : %1.0f'%opt_var)
    ti = time.time()
    results, mic_pos, var_temp = compute(opt_type, opt_var, signal, room_type, mic_config, wall_type, eval_type)

    # print('indices')
    tf = time.time()
    print('Evaluation done  : %1.0f (%1.2f sec)'%(opt_var, tf-ti))
 
    # -*-*-*-*-*-*-*-*-*-*- Print Results -*-*-*-*-*-*-*-*-*-*-
    print('max_error        : %1.3f m'%max(results[:,0]))

    # -*-*-*-*-*-*-*-*-*-*- Save Data in .npy -*-*-*-*-*-*-*-*-*-*-
    # save_data = np.array([results[:,0], opt_var]) # save error (1831,) and mic_idx (np.array)
    save_data = np.array([[results], [mic_pos]], dtype=object)
    
    if not os.path.exists('/mnt/home/ayamaguchi/output/'+dir_name):
        print('New directory is created: '+dir_name)
        os.makedirs('/mnt/home/ayamaguchi/output/'+dir_name)
    else:
        # print('saving directroy exists')
        pass
    npy_path = '/mnt/home/ayamaguchi/output/'+dir_name+'/'+opt_type+'_%1.3f.npy'%(var_temp)
    np.save(npy_path, save_data)


