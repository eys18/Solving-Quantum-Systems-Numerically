#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 11:50:41 2020

@author: ewansaw
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import time


#add timer for this method 

#after substitution, we get simplified integral
def SHO(u):
    return np.sqrt(1 / np.pi) * np.exp(-u ** 2)

#def SHO_3d(u):
#    return (1 / np.pi)**(3/2) * np.exp(-u[0]**2 - u[1]**2 - u[2]**2)

def SHO_3d(x, y, z):
    return (1 / np.pi)**(3/2) * np.exp(-x ** 2 - y ** 2 - z ** 2)

#equation for a 3d sphere centred at origin, radius 1
def sphere(u):
    r = (u[0]**2 + u[1]**2 + u[2]**2)    
    a = np.where(r < 1, 1, 0)
    return a

def trap3d(f, a, b, ε = 0.0001, simps = False):
    if (b <= a).any():
        raise ValueError("b has to be greater than a to evaluate integral")
    if isinstance(a, np.ndarray) == False or isinstance(b, np.ndarray) == False:
        raise TypeError("both a and b need to be numpy arrays")
    if len(a) != len(b):
        raise ValueError("a and b need to have the same shape")
    h = b - a #first step size h1
    
    lims = np.transpose(np.append([a], [b], axis = 0)) #array of all limits
    
    #making array of endpoints(8 corners of a cube)
    endpoint_total = 0
    for k in range(2):
        for j in range(2):
            for i in range(2):
                s = f(lims[0][i], lims[1][j], lims[2][k])
                endpoint_total += s

    T = np.zeros(2)
    T[0] = np.prod(h) * endpoint_total / 8 #calculate T1 
    d = 1 #iteration count
    
    while True: #calculate next iteration
        #edges
        edge_total = 0 # edges of cube along varying x: 4 edges
        for j in range(2):
            for k in range(2):
                s = sum(f(a[0] + (2*i-1)/2 * h[0], lims[1][j], lims[2][k]) for i in range(1, 2**(d-1)+1))
                edge_total += s     
        
        #edges along varying y
        for i in range(2):
            for k in range(2):
                s = sum(f(lims[0][i], a[1] + (2*j-1)/2 * h[1], lims[2][k]) for j in range(1, 2**(d-1)+1))
                edge_total += s
                
        #edges along varying z
        for i in range(2):
            for j in range(2):
                s = sum(f(lims[0][i], lims[1][j], a[2] + (2*k-1)/2 * h[2]) for k in range(1, 2**(d-1)+1))
                edge_total += s
        #faces
        #varying x and y
        face_total = 0
        for k in range(2):
            for j in range(1, 2**d):
                if (j % 2) == 0: #check if j is even
                    for i in range(1, 2**(d-1)+1):
                        s = f(a[0] + (2*i-1)/2 * h[0], a[1] + j/2 * h[1], lims[2][k])
                        face_total += s
                else: #if j is odd
                    for i in range(1, 2**d):
                        s = f(a[0] + i/2 * h[0], a[1] + j/2 * h[1], lims[2][k])
                        face_total += s
        #varying y and z
        for i in range(2):
            for j in range(1, 2**d):
                if (j % 2) == 0: #check if j is even
                    for k in range(1, 2**(d-1)+1):
                        s = f(lims[0][i], a[1] + j/2 * h[1], a[2] + (2*k-1)/2 * h[2])
                        face_total += s
                else: #if j is odd
                    for k in range(1, 2**d):
                        s = f(lims[0][i], a[1] + j/2 * h[1], a[2] + k/2 * h[2])
                        face_total += s
        #varying x and z
        for j in range(2):
            for k in range(1, 2**d):
                if (k % 2) == 0: #check if k is even
                    for i in range(1, 2**(d-1)+1):
                        s = f(a[0] + (2*i-1)/2 * h[0], lims[1][j], a[2] + k/2 * h[2])
                        face_total += s
                else: #if k is odd
                    for i in range(1, 2**d):
                        s = f(a[0] + i/2 * h[0], lims[1][j], a[2] + k/2 * h[2])
                        face_total += s
        #cube centres
        centre_total = 0
        for k in range(1, 2**d):
            if (k % 2) != 0: #if k is odd
                for j in range(1, 2**d):
                    for i in range(1, 2**d):
                        s = f(a[0] + i/2 * h[0],  a[1] + j/2 * h[1], a[2] + k/2 * h[2])
                        centre_total += s   
            else: #if k is even
                for j in range(1, 2**d):
                    if (j % 2) == 0: #check if j is even
                       for i in range(1, 2**(d-1)+1):
                          s = f(a[0] + (2*i-1)/2 * h[0], a[1] + j/2 * h[1], a[2] + k/2 * h[2])
                          centre_total += s    
                    else: #if j is odd
                        for i in range(1, 2**d):
                            s = f(a[0] + i/2 * h[0], a[1] + j/2 * h[1], a[2] + k/2 * h[2])
                            centre_total += s
            
        T[1] = T[0]/8 + np.prod(h)/64 * (2*edge_total + 4*face_total + 8*centre_total)     
        print(T[1])
        print("ε = %s"%abs((T[1] - T[0])/T[0]))
        if abs((T[1] - T[0])/T[0]) < ε:
            return T[0]
        else:
            d +=  1
            T[0] = T[1]
            h = h/2 #halve step size
            continue
             
        
true = 0.49766113
true_3d = 0.123254

#%%
I = trap3d(SHO_3d, np.array([0,0,0]), np.array([2,2,2]), 0.00001)


