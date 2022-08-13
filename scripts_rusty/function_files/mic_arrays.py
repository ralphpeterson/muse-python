#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 20:06:36 2022

@author: akihiro

    Setup different microphone configurations
"""
#%%
import numpy as np
import matplotlib.pyplot as plt

def mic_circular_array(n_mic: int, xc:float,yc:float, zc:float, r:float):
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
        if not n_mic>2:
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
            [x_dim-offset,  y_dim/3,        z_dim-offset],
            [offset,        y_dim/3*2,      z_dim-offset],
            [x_dim/3*2,     y_dim-offset,   z_dim-offset],
            [x_dim/3,       offset,         z_dim-offset]
        ]).T
    elif loc=='left':
        mic_pos = np.array([
            [x_dim-offset,  y_dim/3*2,      z_dim-offset],
            [offset,        y_dim/3,        z_dim-offset],
            [x_dim/3*2,     offset,         z_dim-offset],
            [x_dim/3,       y_dim-offset,   z_dim-offset],
        ]).T
        
    return mic_pos

def mic_edges4n(n_mic:int, x_dim, y_dim, z_dim, offset:float):
    '''
    n: number of microphones
    '''
    if n_mic%4!=0:
        raise ValueError("The number of microphones sould be a multiple of 4.")

    x_pos = np.concatenate([
        np.ones(int(n_mic/4))*offset, # left
        np.linspace(0,x_dim, 2+int(n_mic/4))[1:-1], #top
        np.ones(int(n_mic/4))*(x_dim-offset), # right
        np.linspace(0,x_dim, 2+int(n_mic/4))[1:-1][::-1] #bottom
        ])
    
    y_pos = np.concatenate([
        np.linspace(0,y_dim, 2+int(n_mic/4))[1:-1], #left
        np.ones(int(n_mic/4))*(y_dim-offset), # top
        np.linspace(0,y_dim, 2+int(n_mic/4))[1:-1][::-1], # right
        np.ones(int(n_mic/4))*offset # bottom
        ])
    z_pos = np.ones(n_mic) * z_dim

    mic_pos = np.array([x_pos, y_pos, z_pos])

    return mic_pos

def mic_edge_corner(n_mic:int, x_dim, y_dim, z_dim, offset:float):
    '''
    n: number of microphones
    '''
    if n_mic%4!=0:
        raise ValueError("The number of microphones sould be a multiple of 4.")
    # set corners
    x_corner = np.concatenate([
        np.ones(2)*offset,
        np.ones(2)*(x_dim-offset)
        ])
    y_corner = np.concatenate([
        [offset],
        np.ones(2)*(y_dim-offset),
        [offset]
        ])
    z_corner = np.ones(4)*(z_dim-offset)
    mic_corner = np.array([x_corner, y_corner, z_corner])
    # set edges/walls
    mic_edge = mic_edges4n(n_mic-4, x_dim, y_dim, z_dim, offset)
    mic_pos = np.concatenate([mic_corner,mic_edge],axis=1)

    return mic_pos

def mic_ceiling(n_mic:int, x_dim:float, y_dim:float, z_dim:float, offset:float):
    '''
        Uniformly distribute the microphones 
    '''
    if n_mic<3:
        raise ValueError("n_mic should be greater than 3.")
    elif n_mic%2!=0:
        if n_mic!=np.sqrt(n_mic)**2:
            raise ValueError("n_mic should be an even number or an n**2 value.")

    if n_mic == np.sqrt(n_mic)**2: # n**2 case
        n_row = int(np.sqrt(n_mic))
        x_pos = np.reshape(
            np.tile(
                (2*np.arange(1,n_row+1)-1)/(n_row*2)*x_dim, n_row
                ),(n_row,-1)
            ).flatten()
        y_pos = np.reshape(
            np.tile(
                (2*np.arange(1,n_row+1)-1)/(n_row*2)*y_dim, n_row
                ),(n_row,-1)
            ).T.flatten()
    else: # 2*n case
        n_col = 2
        n_row = int(n_mic/2)

        x_pos = np.reshape(np.tile(
            (2*np.arange(1,n_col+1)-1)/(n_col*2)*x_dim, n_row
            ),(n_row,-1)
        ).flatten()
        
        y_pos = np.reshape(np.tile(
            (2*np.arange(1,n_row+1)-1)/(n_row*2)*y_dim, n_col
            ),(n_col,-1)
        ).T.flatten()

    z_pos = np.ones(n_mic)*(z_dim-offset)

    mic_pos = np.array([x_pos, y_pos, z_pos])

    return mic_pos

#%%
if __name__ == "__main__":
    offset = 0.1
    x_dim, y_dim, z_dim = 2,2,2
    # mic_pos_a = mic_edge_corner(8, x_dim, y_dim, z_dim, offset)
    # mic_pos_b = mic_circular_array(8, x_dim/2, y_dim/2, z_dim, 0.1)
    mic_pos = mic_ceiling(9, x_dim, y_dim, z_dim, offset)
    
    # mic_pos = np.concatenate((mic_pos_a, mic_pos_b),axis=1)
    fig, ax = plt.subplots(1,1)
    ax.plot(mic_pos[0], mic_pos[1], 'o')
    ax.grid()
    ax.set_aspect('equal', 'box')
    ax.set(xlim=(-offset, x_dim+offset), ylim=(-offset, y_dim+offset))
