#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 22:56:43 2020

@author: ewansaw
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import time

def SHO(x):
    '''
    Ground state wavefunction integrand in 1D. Simplified from substitution.

    Parameters
    ----------
    x : float
        x-coordinate/input.

    Returns
    -------
    float
        resulting value of integrand at position.

    '''
    return np.sqrt(1 / np.pi) * np.exp(-x ** 2)

def SHO_3d(u):
    '''
    Ground state wavefunction integrand in 3D. Simplified from substitution.

    Parameters
    ----------
    u : numpy.ndarray
        numpy array of shape (1,3). Each element corresponds to x, y & z inputs.

    Returns
    -------
    float
        resulting value of integrand at position.

    '''
    return (1 / np.pi)**(3/2) * np.exp(-u[0]**2 - u[1]**2 - u[2]**2)
    

def PSI1(u):
    '''
    Integrand of Psi 1. Simplified from substitution.

    Parameters
    ----------
    u : numpy.ndarray
        numpy array of shape (1,3). Each element corresponds to x, y & z inputs.
        
    Returns
    -------
    float
        resulting value of integrand at position.

    '''
    return (1 / np.pi)**(3/2) * np.exp(-u[0]**2 - u[1]**2 - u[2]**2) * (u[0]**2 + u[1]**2)

   
#Define importance sampling pdf
def P(x, A = -0.4151, B = 0.9151):
    '''
    Importance sampling PDF

    Parameters
    ----------
    x : flat
        x-input
    A : float, optional
        gradient of line. The default is -0.4151.
    B : float, optional
        y-intercept of line. The default is 0.9151.

    Returns
    -------
    float
        Probability at given position.

    '''
    
    return A * x + B



def monte_carlo(f, a, b, ε = 0.00001, imp_samp = False, P = P, A = -0.4151, B = 0.9151):
    '''
    Numerical Integrator that calculates estimates of an integral using 
    Monte Carlo methods. Works up to d-dimensions.

    Parameters
    ----------
    f : function
        function to be integrated.
        
    a : numpy.ndarray
        1D array which specifies lower limits of integral; each element
        corresponds to a respective dimension.
        
    b : numpy.ndarray
        1D array which specifies upper limits of integral; each element
        corresponds to a respective dimension. Each element of b has to be greater
        than the respective element of a.
        
    ε : float, optional
        user-specified value for relative accuracy desired. The default is 0.00001.
        
    imp_samp : boolean, optional
        determines which Monte Carlo method is used. If True, samples are picked
        using sampling PDF P. Otherwise, flat sampling is used.The default is False.
        
    P : function, optional
        Sampling PDF to be used for importance sampling case. The default is P.
        
    A : float, optional
        gradient of sampling PDF. The default is -0.4151.
        
    B : float, optional
        y-intercept of sampling PDF. The default is 0.9151.

    Raises
    ------
    TypeError
        Raised if both a and b are not numpy arrays.
        
    ValueError
        Raised if a and b do not have the same shape OR if any element of b is
        smaller than its respective element of a.

    Returns
    -------
    I : float
        Integral estimate
    Nsamp : float
        Array of all number of samples generated for every iteration.
    errors : float
        relative errors associated with each estimate at every iteration.

    '''
    if isinstance(a, np.ndarray) == False or isinstance(b, np.ndarray) == False:
        raise TypeError("both a and b need to be numpy arrays")
    if len(a) != len(b):
        raise ValueError("a and b need to have the same shape")
    if (b <= a).any():
        raise ValueError("b has to be greater than a to evaluate integral")
        
    #to record execution time 
    start_time = time.time()
    
    #number of dimensions
    d = len(a) 
    
    #calculating volume of integrand limits
    V = np.prod(b - a)
    
    N = s = s_squares = 0
    errors = [] #to store values of relative accuracies at each iteration
    Nsamp = [] # to store number of samples for each iteration

    while True:
        #next iteration has N + 100000 samples
        N_new = N + 100000
        
        #for importance sampling case
        if imp_samp == True:
            
            #N x d random numbers in range[0,1]
            rand = random.random((d, N_new - N))
            
            #using transformation method
            samples = (-np.sqrt(2/A * (rand + B ** 2/(2 * A))) - (B / A))
            
            #calculate integral estimate
            Q = f(samples) / np.prod(P(samples), axis = 0) 
            s += np.sum(Q)
            I = s / N_new 
            
            #accumulate sum of squares for calculating variance
            s_squares += np.sum(Q ** 2)
            
            
        else: #flat sampling case
            rand = np.transpose(random.uniform(a, b, (N_new - N, d)))
            F = f(rand)
            
            #calculate integral estimate
            s += np.sum(F)
            I = V / N_new * s  
            
            #accumulate sum of squares for calculating variance
            s_squares += np.sum(F ** 2)
             
        #calculate standard deviation
        std_dev = np.sqrt(1/(N_new * (N_new-1)) * (s_squares - 2 * I * s 
                                               + N_new * I ** 2))
        
        #std deviation for non-importance sampling has extra V factor
        if imp_samp == False:
            std_dev *= V

        #appending new values
        Nsamp.append(N_new * d)
        errors.append(std_dev / I)
            
        if abs((std_dev / I)) < ε:  #if convergense is satisfied
            print("std dev = %s"%std_dev)
            print("--- %s seconds ---" % (time.time() - start_time)) #total execution time
            return I, Nsamp, errors
        else: #rewrite old results as new results
            N = N_new
            continue 

#%% Fig. 1 in report
'''
Plotting relative accuracies against Number of samples used Fig. 1 in report
'''

P_sho, Nsamp, errors = monte_carlo(SHO, np.array([0]), np.array([2]), ε = 1e-4, imp_samp= False)
P_sho, Nsamp1, errors1 = monte_carlo(SHO, np.array([0]), np.array([2]), ε = 1e-4, imp_samp= True)

plt.plot(Nsamp, errors, label="Without Importance Sampling")
plt.plot(Nsamp1, errors1, label="Wtih Importance Sampling")
plt.xlim(0,1e7)
plt.xlabel("Number of Samples Generated")
plt.ylabel("Relative Accuracy ε (log scale)")
plt.yscale("log")
plt.legend()
plt.show()


#%% Fig. 2 in report
'''
Checking how samples vary with importance sampling Fig. 2 in report
'''

# generate points of wavefunction
x = np.linspace(0,4,10000)
y = np.zeros(len(x))
for i in range(len(x)):
    y[i] = SHO(x[i])  
    

#generate samples using transformation method
N = 100000
samples = np.zeros(N)
y0 = np.zeros(N)
A = -0.4
B = 0.9
for i in range(len(samples)):
    samples[i] = (-np.sqrt(2/A * (random.uniform(0, 1) + B ** 2/(2 * A)))
                      - (B / A))
    
plt.xlim((0,2))
plt.ylim((0,1))
plt.hist(samples, bins = 30, density=True, label ="Transformation Method Samples")
plt.plot(x,y, "--" ,label = "Ground State Wavefunction")
plt.plot(x,P(x,-0.4, 0.9), label ="Sampling PDF")
plt.legend()
plt.ylabel("Probability")
plt.xlabel("Sample Points")
plt.show()


#%% Fig. 3 in report
'''
Gaussian Validation Fig. 3 in report
'''

#NOTE: THIS CELL TAKES LONG TO RUN

f=SHO
a = np.array([0])
b = np.array([2])
integrals = []

# True value given by Wolfram Alpha
true = 0.49766113

#generate 10000 estimates
for i in range(10000):
    I, N, errors = monte_carlo(f, a, b, ε = 1e-3, imp_samp= True)
    print(i)
    integrals.append(I)
    
#plot histogram
plt.plot(true, 2500, 'x', label = 'True Value')
plt.hist(integrals, bins = 30, edgecolor='black', density=True, label ="Integral Estimates")
plt.legend()
plt.ylabel("Frequency")
plt.xlabel("Integral Estimates")
plt.show()


#%%1D Ground state at ε = 1e-4
'''
Monte Carlo Integration for 1D Ground state at ε = 1e-4 (Table 2 in report)
'''
f = SHO
a = np.array([0])
b = np.array([2])
ε = 1e-4  # set error

#Uniform Sampling
print('Uniform Sampling')
print('=====================================')
U = monte_carlo(f, a, b, ε, imp_samp = False)
print('I = %s'%U[0])
print('# Samples = %s'%U[1][-1])
print('relative error = %s'%U[2][-1])
print('-------------------------------------')

#Linear Sampling
print('Linear Sampling')
print('=====================================')
L = monte_carlo(f, a, b, ε, imp_samp = True)
print('I = %s'%L[0])
print('# Samples = %s'%L[1][-1])
print('relative error = %s'%L[2][-1])

#%%1D Ground state at ε = 1e-5
'''
Monte Carlo Integration for 1D Ground state at ε = 1e-5 (Table 2 in report)
'''
# NOTE: This cell takes longer to run

f = SHO
a = np.array([0])
b = np.array([2])
ε = 1e-5  # set error

#Uniform Sampling
print('Uniform Sampling')
print('=====================================')
U = monte_carlo(f, a, b, ε, imp_samp = False)
print('I = %s'%U[0])
print('# Samples = %s'%U[1][-1])
print('relative error = %s'%U[2][-1])
print('-------------------------------------')

#Linear Sampling
print('Linear Sampling')
print('=====================================')
L = monte_carlo(f, a, b, ε, imp_samp = True)
print('I = %s'%L[0])
print('# Samples = %s'%L[1][-1])
print('relative error = %s'%L[2][-1])


#%%3D Ground state at ε = 1e-4
'''
Monte Carlo Integration for 3D Ground state at ε = 1e-4 (Table 4 in report)
'''
# NOTE: This cell takes longer to run ~400 seconds

f = SHO_3d
a = np.array([0,0,0])
b = np.array([2,2,2])
ε = 1e-4  # set error

#Uniform Sampling (MC Uniform)
print('Uniform Sampling')
print('=====================================')
U = monte_carlo(f, a, b, ε, imp_samp = False)
print('I = %s'%U[0])
print('# Samples = %s'%U[1][-1])
print('relative error = %s'%U[2][-1])
print('-------------------------------------')


#Linear Sampling (MC Linear)
print('Linear Sampling')
print('=====================================')
L = monte_carlo(f, a, b, ε, imp_samp = True)
print('I = %s'%L[0])
print('# Samples = %s'%L[1][-1])
print('relative error = %s'%L[2][-1])

