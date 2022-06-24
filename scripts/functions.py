#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 17:00:46 2022

@author: akihiro
"""
import librosa
import pyroomacoustics as pra
import numpy as np
from pyroomacoustics.directivities import CardioidFamily, DirectionVector, DirectivityPattern
from muse import util as muse

def generate_room(x_dim, y_dim, z_dim, wall_materials, sampling_rate=125000):
    # generate the showbox using pyroom acoustics
    # Materials of the room's walls
    # See different materials at https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.materials.database.html
    materials = pra.make_materials(
        ceiling = wall_materials[0],
        floor   = wall_materials[1],
        east    = wall_materials[2],
        west    = wall_materials[3],
        north   = wall_materials[4],
        south   = wall_materials[5]
    )
    
    shoebox_object = pra.ShoeBox(
        [x_dim, y_dim, z_dim],
        fs=sampling_rate,
        materials=materials,
        max_order=9  # Maximum number of times a sound wave can reflect (?)
    ) 
    return shoebox_object

def orient_mic(shoebox, mic_pos):
    ''' 
    Orient the microphones horizontally outward
    '''
    mic_pos = mic_pos.T  # (3, n_mics) -> (n_mics, 3)
    shoebox_dims = shoebox.shoebox_dim
    if any([coord >= box_size for coord, box_size in zip(mic_pos.max(axis=0), shoebox_dims)]) \
        or any([coord <= 0 for coord in mic_pos.min(axis=0)]):
        raise ValueError("The microphones must be within the box. They cannot be located along an egde or face.")
    
    room_center = np.array(shoebox_dims) / 2
    room_center[2] = 0
    
    # List of vectors pointing from each mic to or away from the room_center
    mic_directions = room_center[np.newaxis, ...] - mic_pos
    mic_directions[:,2] = 0 # horizontal
    mic_directions /= -1*np.sqrt((mic_directions**2).sum(axis=1, keepdims=True))

    directivities = []
    mic_dir = np.zeros(mic_directions.shape)
    for i,direction in enumerate(mic_directions):
        theta = np.arctan2(direction[1], direction[0])
        xy_vec_size = np.sqrt((direction[:2] ** 2).sum())
        phi = np.pi/2 - np.arctan(direction[2] / xy_vec_size)
        pra_direction = DirectionVector(
            azimuth=theta,
            colatitude=phi,
            degrees=False
        )
        directivities.append(CardioidFamily(
            orientation=pra_direction,
            pattern_enum=DirectivityPattern.HYPERCARDIOID
        ))
        
        xd,yd,zd = np.sin(phi)*np.cos(theta),np.sin(phi)*np.sin(theta),np.cos(phi)
        mic_dir[i] = np.array([xd,yd,zd])

    # Transpose mic_pos back to (3, n_mics)
    shoebox.add_microphone_array(mic_array=mic_pos.T, directivity=directivities)
    return mic_dir

def place_mics(shoebox, mic_pos:float):
    # Assuming mic_pos has shape (3, n_mics), which, while strange, is what MUSE expects
    # I think this is just a side-effect of MATLAB matrix convention
    mic_pos = mic_pos.T  # (3, n_mics) -> (n_mics, 3)
    shoebox_dims = shoebox.shoebox_dim
    if any([coord >= box_size for coord, box_size in zip(mic_pos.max(axis=0), shoebox_dims)]) \
        or any([coord <= 0 for coord in mic_pos.min(axis=0)]):
        raise ValueError("The microphones must be within the box. They cannot be located along an egde or face.")
    
    # Compute directivity of microphones
    # Assumes the microphones are all pointed toward the center of the room's floor
    # Can be simplified by just pointing them downward
    room_center = np.array(shoebox_dims) / 2
    room_center[2] = 0          # set the z-center to be zero
    
    # List of vectors pointing from each mic to or away from the room_center
    mic_directions = room_center[np.newaxis, ...] - mic_pos
    # Normalize the vectors
    mic_directions /= np.sqrt((mic_directions**2).sum(axis=1, keepdims=True))

    # Convert the direction vectors into polar coordinates for the directivities used by PRA
    directivities = []
    mic_dir = np.zeros(mic_directions.shape)
    for i,direction in enumerate(mic_directions):
        # Angle within the x-y plane. 0 radians is toward the positive x direction
        # In spherical coordinates, this is \theta
        theta = np.arctan2(direction[1], direction[0])
        # Same thing here, but between the z axis and the x-y plane
        xy_vec_size = np.sqrt((direction[:2] ** 2).sum())
        # Using arctan instead of arctan2 because I need the result to be within Q1 and Q4
        # This would be called \phi in spherical coordinates
        phi = np.pi/2 - np.arctan(direction[2] / xy_vec_size)
        pra_direction = DirectionVector(
            azimuth=theta,
            colatitude=phi,
            degrees=False
        )
        # For visualizations, see https://en.wikipedia.org/wiki/Microphone#Polar_patterns
        # For supported options, see https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.directivities.html?highlight=DirectivityPattern#pyroomacoustics.directivities.DirectivityPattern
        directivities.append(CardioidFamily(
            orientation=pra_direction,
            pattern_enum=DirectivityPattern.HYPERCARDIOID
        ))
        xd = np.sin(phi)*np.cos(theta)
        yd = np.sin(phi)*np.sin(theta)
        zd = np.cos(phi)
        mic_dir[i] = np.array([xd,yd,zd])

    # Transpose mic_pos back to (3, n_mics)
    shoebox.add_microphone_array(mic_array=mic_pos.T, directivity=directivities)
    
    return mic_dir

def add_source(speaker_pos, stimulus, sr_stimulus, room):
    if not isinstance(speaker_pos, list):
        speaker_pos = list(speaker_pos)
    
    if room.fs != sr_stimulus:
        print('resampling audio')
        stimulus_resampled = librosa.resample(stimulus.astype('float'), orig_sr=sr_stimulus, 
                                                                      target_sr=room.fs)
        room.add_source(speaker_pos, signal=stimulus_resampled, delay=0)
        return room.fs, stimulus_resampled
        
    else:
        room.add_source(speaker_pos, signal=stimulus, delay=0)
        return sr_stimulus, stimulus

def set_MUSE_params(room, x_dim, y_dim, mic_pos, sr, res=0.0025):
    # Set MUSE (Mouse Ultrasonic Source Estimator) parameters
    x_grid, y_grid = muse.make_xy_grid(x_dim, y_dim, resolution=res)
    in_cage = None  # unused argument
    #x, y, zmic positions
    R = mic_pos
    #audio from mic array from simulation
    audio = room.mic_array.signals.T  # Transpose shape from (n_mics, n_samp) to (n_samp, n_mics)
    #frequency range of interest
    f_lo = 0
    f_hi = 62500
    fs = sr #audio sampling rate
    # dt = 1 / fs #time step
    temp = 20 #temperature oF environment (C)
    N, n_mics = audio.shape #num mics
    
    return audio,fs,f_lo,f_hi,temp,x_grid,y_grid,in_cage,R




