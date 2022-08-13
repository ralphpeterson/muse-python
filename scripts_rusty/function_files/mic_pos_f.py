#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 17:36:30 2022

@author: akihiro
"""
import numpy as np
import matplotlib.pyplot as plt

# x_dim, y_dim = 1.83, 2.44 # m
offset = 0.1 # m

# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
def f(alpha: float, room_dims:np.ndarray, mode:str='diag'):
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
    
    X = (room_dims[0])/2 * Mx * alpha + (room_dims[0]) * Mxb
    Y = (room_dims[1])/2 * My * alpha + (room_dims[1]) * Myb

    mic_pos = []
    for i in range(4):
        mic_pos.append([X[i,i], Y[i,i]])
    
    return np.asarray(mic_pos)
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
def g(theta_deg:int, room_dims:np.ndarray, beta:float, alpha:float=1):
    '''
    Parameters
    ----------
    theta_deg : int
    room_dims : np.ndarray 
    alpha : float, optional
    beta : float

    Returns
    -------
    mic_pos : np.ndarray 
    '''
    if beta<=0 or beta>=1:
        raise ValueError('beta must be between (0,1)')
    if type(theta_deg)!=int:
        raise ValueError('Wrong data type:', type(theta_deg))

    theta= theta_deg * np.pi/180
    m1 = np.array([room_dims[0]/2*np.cos(theta)*alpha +room_dims[0]/2, room_dims[1]/2*np.sin(theta)*alpha +room_dims[1]/2])
    m2 = np.array([room_dims[0]/2*np.cos(theta)*beta +room_dims[0]/2, room_dims[1]/2*np.sin(theta)*beta +room_dims[1]/2])
    m3 = np.array([room_dims[0]/2*np.cos(theta+np.pi)*alpha +room_dims[0]/2, room_dims[1]/2*np.sin(theta+np.pi)*alpha +room_dims[1]/2])
    m4 = np.array([room_dims[0]/2*np.cos(theta+np.pi)*beta +room_dims[0]/2, room_dims[1]/2*np.sin(theta+np.pi)*beta +room_dims[1]/2])    
    return np.array([m1,m2,m3,m4])

# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

if __name__ =="__main__":
    # rd = np.array([1.83, 2.44, 0.76]) # room_dimensions (large room)
    rd = np.array([0.40, 0.60, 0.355]) # room_dimensions (small room)

    # mic_pos = f(0.5, rd, mode='vert')
    mic_pos = f(0.1, rd, mode='diag')
    # mic_pos = g(90, rd, 0.2, 0.9)
    
    fig, ax = plt.subplots()
    ax.plot(mic_pos[:,0], mic_pos[:,1], 'o')
    ax.set_aspect('equal')
    ax.set_xlim(0,rd[0])
    ax.set_ylim(0,rd[1])
    ax.grid()

    fig, ax = plt.subplots()
    for i in range(1,180):
        # print('alpha:', 0.03*i)
        # mic_pos = f(0.03*i, rd, mode='vert')

        # print('theta:',i)
        mic_pos = g(i, rd, 0.3, 0.9)

        ax.plot(mic_pos[:,0], mic_pos[:,1], 'o')
    ax.set_aspect('equal')
    ax.set_xlim(0,rd[0])
    ax.set_ylim(0,rd[1])
    ax.grid()



