#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 17:00:19 2022

@author: akihiro

    Functions for plotting a room & show microphone directions
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch

class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)
def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)

setattr(Axes3D, 'arrow3D', _arrow3D)
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

def plot_room(room, plot_offset, 
              x_dim:float, y_dim:float, z_dim:float,
              mic_pos, mic_dir,
              arrows=False):
    sns.set_style('white')
    sns.set_style('ticks')
    sns.set_context('paper')

    fig, ax = room.plot(plot_directivity=False, mic_marker_size=50)
    ax.set_xlim([-plot_offset, x_dim+plot_offset])
    ax.set_ylim([-plot_offset, y_dim+plot_offset])
    ax.set_zlim([-plot_offset, z_dim+plot_offset])

    ax.set_xlabel('Width (m)', labelpad=10, fontsize=14)
    ax.set_ylabel('Length (m)', labelpad=10, fontsize=14)
    ax.set_zlabel('Height (m)', labelpad=10, fontsize=14)
    
    if arrows:
        # Add the direction of the microphones
        for i in range(mic_pos.T.shape[0]):
            ax.arrow3D(mic_pos.T[i,0],mic_pos.T[i,1], mic_pos.T[i,2],
                        mic_dir[i,0],mic_dir[i,1],mic_dir[i,2],
                        mutation_scale=20,
                        arrowstyle="-|>",
                        linestyle='-',
                        ec='black',
                        lw='2')    
    return ax

def plot_mic_audio(room, stimulus, n_mic, sr, stim_name):

    # tick_labels = ["Stimulus"]
    
    t = np.arange(room.mic_array.signals.shape[1])/sr
    t_stim = np.arange(stimulus.shape[0])/sr
    
    n_mic = room.mic_array.signals.shape[0]
    
    fig, ax = plt.subplots(n_mic+1,1)
    # plt.subplot(n_mic+1,1,1)
    ax[0].plot(t_stim, stimulus, 'k')
    ax[0].grid()
    ax[0].set_xticklabels([])
    # plt.xticks([])
    ax[0].set_yticklabels([])
    ax[0].set_ylabel('Original', fontsize=14)
    for i in range(0,n_mic):
        # ax[i].subplot(n_mic+1,1,i+2)
        ax[i+1].plot(t, room.mic_array.signals[i], 'k')
        ax[i+1].grid()
        ax[i+1].set_ylabel('Mic {}'.format(i+1), fontsize=14)
        # plt.yticks([])
        ax[i+1].set_yticklabels([])
        if i!=n_mic-1:
            ax[i+1].set_xticklabels([])
    
    xmin, xmax = ax[4].get_xlim()
    ax[0].set_xlim(xmin, xmax)
    
    plt.xlabel('Time (s)', fontsize=14)
    fig.suptitle('Stimulus (Audio): '+ stim_name)
    sns.despine()
    

