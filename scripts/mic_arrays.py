#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 20:06:36 2022

@author: akihiro

    Setup different microphone configurations
"""
import numpy as np

def mic_array(n_mic: int, xc:float,yc:float, zc:float, r:float):
    '''
    n_mic: int
        number of microphones (must be greater than 3)
    xc, yc : float
        center of the circular array in x and y
    zc: float
        position in z (offset < z < z_dim-offset)
    r : float
        radius of the array.
    offset : float
        offset in z.
    '''
    try: 
        if not n_mic>3:
            raise ValueError()
    except ValueError:
        raise ValueError('n_mic has to be greater than 3.')
    
    theta = 2*np.pi/n_mic * np.arange(n_mic) + np.pi/2
    x = np.around(r*np.cos(theta),decimals=2) + xc
    y = np.around(r*np.sin(theta),decimals=2) + yc
    z = np.ones(n_mic) * zc
    
    mic_pos = np.array([x,y,z])

    return mic_pos
    
    

def mic_corners4(x_dim, y_dim, z_dim, offset):
    '''
        4 microphones on each corner of the box.
    '''
    mic_pos = np.array([
        [x_dim-offset, y_dim-offset, z_dim-offset],
        [offset, offset, z_dim-offset],
        [offset, y_dim-offset, z_dim-offset],
        [x_dim-offset, offset, z_dim-offset]
    ]).T
    
    return mic_pos

def mic_edges4(x_dim, y_dim, z_dim, offset, loc):
    if loc=='center':
        mic_pos = np.array([
            [x_dim-offset,  y_dim/2,      z_dim-offset],
            [offset,        y_dim/2,      z_dim-offset],
            [x_dim/2,     y_dim-offset,   z_dim-offset],
            [x_dim/2,     offset,         z_dim-offset]
        ]).T
    elif loc=='right':
        mic_pos = np.array([
            # [x_dim-offset,  y_dim/3*2,      z_dim/2],
            [x_dim-offset,  y_dim/3,        z_dim-offset],
            [offset,        y_dim/3*2,      z_dim-offset],
            # [offset,        y_dim/3,        z_dim/2],
            # [x_dim/3*2,     offset,         z_dim/2],
            [x_dim/3*2,     y_dim-offset,   z_dim-offset],
            # [x_dim/3,       y_dim-offset,   z_dim/2],
            [x_dim/3,       offset,         z_dim-offset]
        ]).T
    elif loc=='left':
        mic_pos = np.array([
            [x_dim-offset,  y_dim/3*2,      z_dim-offset],
            # [x_dim-offset,  y_dim/3,        z_dim/2],
            # [offset,        y_dim/3*2,      z_dim/2],
            [offset,        y_dim/3,        z_dim-offset],
            [x_dim/3*2,     offset,         z_dim-offset],
            # [x_dim/3*2,     y_dim-offset,   z_dim/2],
            [x_dim/3,       y_dim-offset,   z_dim-offset],
            # [x_dim/3,       offset,         z_dim/2]
        ]).T
        
    return mic_pos

def mic_edges8(x_dim, y_dim, z_dim, offset):
    # offset = 3
    mic_pos = np.array([
        # [x_dim-offset, y_dim-offset, z_dim-offset],
        # [offset, offset, z_dim-offset],
        # [offset, y_dim-offset, z_dim-offset],
        # [x_dim-offset, offset, z_dim-offset],
        [x_dim-offset,  y_dim/3*2,      z_dim-offset],
        [offset,        y_dim/3*2,      z_dim-offset],
        [x_dim/3*2,     y_dim-offset,   z_dim-offset],
        [x_dim/3*2,     offset,         z_dim-offset],
        [x_dim-offset,  y_dim/3,        z_dim-offset],
        [offset,        y_dim/3,        z_dim-offset],
        [x_dim/3,       y_dim-offset,   z_dim-offset],
        [x_dim/3,       offset,         z_dim-offset]
    ]).T
    return mic_pos

def mic_edge_corner(x_dim, y_dim, z_dim, offset):
    mic_pos = np.array([
            # 4 edges
            [x_dim-offset,  y_dim/2,      z_dim-offset],
            [offset,        y_dim/2,      z_dim-offset],
            [x_dim/2,     y_dim-offset,   z_dim-offset],
            [x_dim/2,     offset,         z_dim-offset],
            # 4 corners
            [x_dim-offset, y_dim-offset, z_dim-offset],
            [offset, offset, z_dim-offset],
            [offset, y_dim-offset, z_dim-offset],
            [x_dim-offset, offset, z_dim-offset]
    ]).T
    return mic_pos
