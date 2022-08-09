#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 17:36:30 2022

@author: akihiro
"""

import numpy as np
import matplotlib.pyplot as plt

x_dim, y_dim = 1.83, 2.44 # m
offset = 0.1 # m

# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
def f(alpha: float, mode:str='diag'):
    '''
    Parameters
    ----------
    alpha : float
    mode : TYPE, optional
        DESCRIPTION. The default is 'diag':int.
    Returns
    -------
    mic_pos : np.ndarray

    '''
    
    if alpha<=0 or alpha>=1:
        raise ValueError('alpha must be between (0,1)')
    
    if mode=='diag':
        # diagonal config matrix
        Mx, Mxb = np.eye(4), np.zeros((4,4))
        Mx[range(2,4),range(2,4)]=-1        
        Mxb[range(2,4),range(2,4)]=1
    
        My, Myb = np.eye(4), np.zeros((4,4))
        My[range(1,3),range(1,3)]=-1
        Myb[range(1,3),range(1,3)]=1
    elif mode=='vert':
        # vertical/horizontal config matrix
        Mx, Mxb = np.zeros((4,4)), np.zeros((4,4))
        Mx[0,0], Mx[2,2] = 1, -1
        Mxb[1,1], Mxb[2,2], Mxb[3,3] = 1/2,1, 1/2
    
        My, Myb = np.zeros((4,4)), np.zeros((4,4))
        My[1,1], My[3,3] = -1, 1
        Myb[0,0], Myb[1,1], Myb[2,2] = 1/2,1, 1/2     
    else:
        ValueError('mode is undefined.')        
    
    X = (x_dim)/2 * Mx * alpha + (x_dim) * Mxb
    Y = (y_dim)/2 * My * alpha + (y_dim) * Myb

    mic_pos = []
    for i in range(4):
        mic_pos.append([X[i,i], Y[i,i]])
    
    return np.asarray(mic_pos)
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
def g(theta_deg:int, xd:float, yd:float, beta:float, alpha:float=1):
    '''
    Parameters
    ----------
    theta_deg : int
    xd : float
    yd : float
    alpha : float, optional
    beta : float

    Returns
    -------
    mic_pos : np.ndarray 
    '''
    if beta<=0 or beta>=1:
        raise ValueError('beta must be between (0,1)')

    theta= theta_deg * np.pi/180
    m1 = np.array([xd/2*np.cos(theta)*alpha +x_dim/2, yd/2*np.sin(theta)*alpha +y_dim/2])
    m2 = np.array([xd/2*np.cos(theta)*beta +x_dim/2, yd/2*np.sin(theta)*beta +y_dim/2])
    m3 = np.array([xd/2*np.cos(theta+np.pi)*alpha +x_dim/2, yd/2*np.sin(theta+np.pi)*alpha +y_dim/2])
    m4 = np.array([xd/2*np.cos(theta+np.pi)*beta +x_dim/2, yd/2*np.sin(theta+np.pi)*beta +y_dim/2])    
    return np.array([m1,m2,m3,m4])

# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

if __name__ =="__main__":

    # mic_pos = f(0.78, mode='diag')
    # mic_pos = f(0.96, mode='vert')
    # mpos_d = f(0.78, mode='diag')
    # mpos_v = f(0.96, mode='vert')
    mpos_d = g(93, x_dim, y_dim, 0.2, 0.9)
    mpos_v = g(0, x_dim, y_dim, 0.2, 0.9)
    
    mic_pos = np.concatenate((mpos_v, mpos_d))

    # mic_pos = g(93, x_dim, y_dim, 0.2, 0.9)
    
    fig, ax = plt.subplots()
    # for theta in range(0,180):
    #     # alpha = 0.03*i
    #     mic_pos = g(theta, x_dim, y_dim, 0.4, 0.9)
    #     ax.plot(mic_pos[:,0], mic_pos[:,1], 'o')
    ax.plot(mic_pos[:,0], mic_pos[:,1], 'o')
    ax.set_aspect('equal')
    ax.set_xlim(0,x_dim)
    ax.set_ylim(0,y_dim)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.grid()






