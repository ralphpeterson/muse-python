#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 14:44:09 2022

@author: akihiro

Material reference:
    https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.materials.database.html

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import seaborn as sns
from scipy.spatial import distance

from muse import util as muse
from scripts import plot_functions as pf
from scripts import functions as fun
from scripts import mic_arrays as ma

# %matplotlib qt        # interactive widow
# %matplotlib inline    # html

import os
print('Current directory:', os.getcwd())
# %% Setup the conditions and Generate a Room
sr = 125000 #audio sampling rate

#room dimensions (in m)
# x_dim, y_dim, z_dim = 0.4, 0.6, 0.355 # room 1
x_dim, y_dim, z_dim = 1.83, 2.44, 0.76 # room 2

# room materials
# ceiling, floor = "fibre_absorber_2", "fibre_absorber_2"
# east, west, north, south = "hard_surface","hard_surface","hard_surface","hard_surface"
ceiling, floor = "curtains_fabric_folded", "curtains_fabric_folded"
east, west, north, south = "curtains_fabric_folded","curtains_fabric_folded","curtains_fabric_folded","curtains_fabric_folded"
materials = [ceiling, floor, east, west, north, south]

# microphone positions to place in top corners, otherwise, put in x,y,z coords 
offset = .1 # in meters
# mic_pos = mic_edges4(x_dim, y_dim, z_dim, offset, 'center')
# mic_pos = mic_corners4(x_dim, y_dim, z_dim, offset)
# mic_pos = mic_edges8(x_dim, y_dim, z_dim, offset)
mic_pos = ma.mic_edge_corner(x_dim, y_dim, z_dim, offset)
# mic_pos = ma.mic_array(8, x_dim/2,y_dim/2, zc=0.1, r=0.5) #radius in m
n_mic = mic_pos.shape[1]

# load the stimulus to play in the room
stim_dir = '../muse-python/stimuli/tones/sine_2khz.wav'
stim_name = stim_dir.split('/')[-1]
sr_stimulus, stimulus = wavfile.read(os.path.abspath(stim_dir))

#position of a speaker in the room (in m)
x_pos, y_pos = x_dim/2, y_dim/2 # center of the room
z_pos = 0.01

#%% Generate a room
room = fun.generate_room(x_dim, y_dim, z_dim, materials)

# visualize the room
axes_offset = 0.5
# pf.plot_room(room, axes_offset, x_dim, y_dim, z_dim)

# Add mics in the room & get the direction of the microphones
mic_dir = fun.place_mics(room, mic_pos) # mics pointing to the room center
# mic_dir = fun.orient_mic(room, mic_pos)
# pf.plot_room(room, axes_offset, x_dim, y_dim, z_dim)

# Add a speaker
speaker_pos = (x_pos, y_pos, z_pos)

# Play audio
sr_stimulus, stimulus = fun.add_source(speaker_pos, stimulus, sr_stimulus, room)
ax = pf.plot_room(room, axes_offset, x_dim, y_dim, z_dim, 
                  mic_pos, mic_dir, arrows=True)

# %% Run the simulation
room.simulate()

# %% Plot the audio (original and mics)
pf.plot_mic_audio(room, stimulus, n_mic, sr, stim_name)

# %% Set MUSE (Mouse Ultrasonic Source Estimator) parameters
# #create grid to predict sound source over
# x_grid, y_grid = muse.make_xy_grid(x_dim, y_dim, resolution=0.0025)  # I increased this ten-fold because it took forever to run
# in_cage = None  # unused argument
# #x, y, zmic positions
# R = mic_pos
# #audio from mic array from simulation
# audio = room.mic_array.signals.T  # Transpose shape from (n_mics, n_samp) to (n_samp, n_mics)
# #frequency range of interest
# f_lo = 0
# f_hi = 62500
# fs = sr #audio sampling rate
# # dt = 1 / fs #time step
# temp = 20 #temperature oF environment (C)
# N, n_mics = audio.shape #num mics

audio,fs,f_lo,f_hi,temp,x_grid,y_grid,in_cage,R = fun.set_MUSE_params(room, x_dim, y_dim, mic_pos, sr, 0.0025)

# %% Run MUSE
print('Running MUSE')
r_est, _, rsrp_grid, _, _, _, _, _, _ \
= muse.r_est_from_clip_simplified(audio,
    fs,
    f_lo,
    f_hi,
    temp,
    x_grid,
    y_grid,
    in_cage,
    R,  # Expected to have shape (3, n_mics)
    1)

# Plot the RSRP grid 
fig, ax = plt.subplots(1,1,figsize=(7,7))
cs = ax.contourf(x_grid,y_grid,rsrp_grid)
ax.plot(r_est[0],r_est[1], 'o', color='r')
cbar = fig.colorbar(cs)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_aspect('equal', 'box')
plt.tight_layout()

# %% Run jackknife
from scripts_ay import jackknife as jk

r_mean, r_std = jk.jackknife(mic_pos,speaker_pos,
              stimulus,sr_stimulus, sr,
              x_dim, y_dim, z_dim, materials)


# %% Plot the source and the estimated location
axes_offset = 0.5

sns.set_style('white')
sns.set_style('ticks')
sns.set_context('paper')

fig, ax = room.plot(plot_directivity=False, mic_marker_size=50,img_order=-1)

ax.plot3D(x_pos, y_pos, z_pos, marker='o', color='k', markersize=5)
ax.plot3D(r_est[0], r_est[1], 0, marker='o', color='r', markersize=5)

ax.set_xlim([-axes_offset, x_dim+axes_offset])
ax.set_ylim([-axes_offset, y_dim+axes_offset])
ax.set_zlim([-axes_offset, z_dim+axes_offset])

ax.set_xlabel('Width (m)', labelpad=10)
ax.set_ylabel('Length (m)', labelpad=10)
ax.set_zlabel('Height (m)', labelpad=10);

print('MUSE error:', np.around(distance.euclidean(r_est, speaker_pos[:2])*1e2, decimals=3), 'cm')





