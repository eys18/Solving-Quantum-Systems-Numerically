#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 16:18:15 2020

@author: ewansaw
"""

import numpy as np
import matplotlib.pyplot as plt
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

def SHO_3d(x, y, z):
    '''
    Ground state wavefunction integrand in 3D. Simplified from substitution.
    
    Parameters
    ----------
    x : float
        x input
    y : float
        y input
    z : float
        z input

    Returns
    -------
    float
        resulting value of integrand at given position.

    '''
    return (1 / np.pi)**(3/2) * np.exp(-x ** 2 - y ** 2 - z ** 2)

def PSI1(x,y,z):
    '''
    Integrand of Psi 1. Simplified from substitution.

    Parameters
    ----------
    x : float
        x input
    y : float
        y input
    z : float
        z input

    Returns
    -------
    float
        resulting value of integrand at given position.

    '''
    return (1 / np.pi)**(3/2) * np.exp(-x**2 - y**2 - z**2) * (x**2 + y**2)


def trap_int(f, a, b, ε = 0.0001, simps = False):
    '''
    Numerical Integrator that calculates estimates of an integral using 
    Extended Trapezoidal rule for 1D.

    Parameters
    ----------
    f : function
        function to be integrated.
        
    a : float
        lower limit of integral
        
    b : float
        upper limit of integral
        
    ε : float, optional
        user-specified value for relative accuracy desired. The default is 0.0001.
        
    simps : boolean, optional
         is True when called by 'simp' function below. If True, calculates values
         of convergence for Simpson's rule instead. The default is False.

    Raises
    ------
    ValueError
        b has to be greater than a to evaluate integral as a is the lower limit.

    Returns
    -------
    (if simps == False):
    T[1] : float
        Estimate of integral up to desired relative accuracy.
        
    (if simps == True):
    T : array
        array of 3 most recent integrals that satisfy Simpson's rule convergence.

    '''
    if b <= a:
        raise ValueError("b has to be greater than a to evaluate integral")
        
    else:
        #to record execution time 
        start_time = time.time()
        
        #calculating initial stepsize h1
        h = b - a
        
        #empty array to save results
        T = np.zeros(3)
        
        #calculate T1
        T[0] = h/2 * (f(a) + f(b))
                      
        #calculate T2             
        T[1] = T[0]/2 + h/2 * f(a + h/2) 

        #check if T1 reaches convergence criterion for trapezoidal rule
        if simps == False:
            if abs((T[1] - T[0]) / T[0]) < ε: 
                return T[0]
        
        #iteration count after calculating first two results
        j = 2
        
        #create loop that breaks when convergence is reached
        while True: 
            #halve step size
            h = h / 2
            
            #calculate result of new iteration
            s = sum(f(a + (2 * i - 1) * h/2) for i in range(1, 2 ** (j - 1) + 1))       
            T[2] = 0.5 * T[1] + h/2 * s
            
            #break loop if convergence of simpsons rule is met (only for simp function)
            if simps == True:
                
                #calculate relative accuracy for simpsons rule
                conv = ((4 * T[2] -  T[1]) / 
                        (4 * T[1] -  T[0]) - 1)
                
                #if convergence satisfied
                if abs(conv) < ε:
                    print('Results \n----------------------------')
                    print('Number of samples = %s'%((b - a)/h))
                    print("Final relative error = %s"%conv)
                    
                    #returns array of results to simp function   
                    return T

            #convergence criterion for extended trapezoidal rule
            elif abs((T[2] - T[1]) / T[1]) < ε:
                
                #print final execution time and error
                print('Results \n----------------------------')
                print('Number of samples = %s'%((b - a)/h))
                print("Final relative error = %s"%((T[2] - T[1]) / T[1]))   
                print("--- %s seconds ---" % (time.time() - start_time))
                
                #return second last element as that is the element that reaches convergence 
                return T[1]
                
            #increase count at end of iteration
            j += 1
            
            #updating new terms and by overwriting old results
            T[0] = T[1]
            T[1] = T[2]
            
            #repeat loop
            continue
            
            
def simp(f, a, b, ε = 0.0001):
    '''
    Numerical Integrator that calculates estimates of an integral using 
    Extended Simpson's rule for 1D.

    Parameters
    ----------
    f : function
        function to be integrated.
        
    a : float
        lower limit of integral
        
    b : float
        upper limit of integral
        
    ε : float, optional
        user-specified value for relative accuracy desired. The default is 0.0001.

    Raises
    ------
    ValueError
        b has to be greater than a to evaluate integral as a is the lower limit.

    Returns
    -------
    S : float
        Estimate of integral up to desired relative accuracy.        

    '''
    if b <= a:
        raise ValueError("b has to be greater than a to evaluate integral")
        
    else:
        #to record execution time 
        start_time = time.time()
        
        #call trapezoidal function to retrieve integral values
        T = trap_int(f, a, b, ε, simps = True)
        
        #calculate resulting integral for simpson's rule
        S = 4/3 * T[1] - 1/3 * T[0]
        
    #print final execution time
    print("--- %s seconds ---" % (time.time() - start_time))
    return S


def trap_3d(f, a, b, ε = 0.0001):
    '''
    Numerical Integrator that calculates estimates of an integral using 
    Extended Trapezoidal rule for 3D functions.

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

    Raises
    ------
    TypeError
        Raised if both a and b are not numpy arrays.
        
    ValueError
        Raised if a and b do not have the same shape OR if any element of b is
        smaller than its respective element of a.

    Returns
    -------
    float
        Estimate of integral up to desired relative accuracy.

    '''
    
    if isinstance(a, np.ndarray) == False or isinstance(b, np.ndarray) == False:
        raise TypeError("both a and b need to be numpy arrays")
    if len(a) != len(b):
        raise ValueError("a and b need to have the same shape")
    if (b <= a).any():
        raise ValueError("b has to be greater than a to evaluate integral")
        
    #integrate f(x, y, z) over dx
    def g(y,z):
        return trap_int(lambda x: f(x, y, z), a[0], b[0], ε)
    
    #integrate g(y, z) over dy 
    def h(z):
        return trap_int(lambda y: g(y,z), a[1], b[1], ε)

    #integrate h(z) over dz
    return trap_int(h, a[2], b[2], ε) #integrate over dz


def simp_3d(f, a, b, ε = 0.0001):
    '''
    Numerical Integrator that calculates estimates of an integral using 
    Extended Simson's rule for 3D functions.

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

    Raises
    ------
    TypeError
        Raised if both a and b are not numpy arrays.
        
    ValueError
        Raised if a and b do not have the same shape OR if any element of b is
        smaller than its respective element of a.

    Returns
    -------
    float
        Estimate of integral up to desired relative accuracy.

    '''
    if isinstance(a, np.ndarray) == False or isinstance(b, np.ndarray) == False:
        raise TypeError("both a and b need to be numpy arrays")
    if len(a) != len(b):        
        raise ValueError("a and b need to have the same shape")
    if (b <= a).any():
        raise ValueError("b has to be greater than a to evaluate integral")
        
    #integrate f(x, y, z) over dx
    def g(y, z):
        return simp(lambda x: f(x, y, z), a[0], b[0], ε)
    
    #integrate g(y, z) over dy 
    def h(z):
        return simp(lambda y: g(y, z), a[1], b[1], ε)
    
    #integrate h(z) over dz
    return (simp(h, a[2], b[2], ε))    


#%%1D Ground state at ε = 1e-6
'''
Newton-Cotes Integration for 1D Ground state at ε = 1e-6 (Table 1 in report)
'''
f = SHO
a = 0
b = 2
ε = 1e-6  # set error

#Trapezoidal Rule
print('Trapezoidal Rule')
T = trap_int(f, a, b, ε)
print('I = %s'%T)


#Simpson's Rule
print('Simpsons Rule')
S = simp(f, a, b, ε)
print('I = %s'%S)

#%%3D Ground state at ε = 1e-4
'''
Newton-Cotes Integration for 3D Ground state at ε = 1e-4 (Table 4 in report)
'''
f = SHO_3d
a = np.array([0,0,0])
b = np.array([2,2,2])
ε = 1e-4  # set error

T = trap_3d(f, a, b, ε)
S = simp_3d(f, a, b, ε)

print('=====================================')
#Trapezoidal Rule
print('Trapezoidal Rule I = %s' %T)

#Simpson's Rule
print('Simpsons Rule I = %s' %S)

#%%Final Results using Simpsons
'''
Simpson's for 1D ground state at ε = 1e-15 (Table 5 in report, row 1)
'''
f = SHO
a = 0
b = 2
ε = 1e-15  # set error

S = simp(f, a, b, ε)
print('=====================================')
print('Simpsons Rule I = %s' %S)

#%%Final Results using Simpsons
'''
Simpson's for 3D ground state at ε = 1e-8 (Table 5 in report, row 2)
'''
f = SHO_3d
a = np.array([0,0,0])
b = np.array([2,2,2])
ε = 1e-8  # set error

S = simp_3d(f, a, b, ε)
print('=====================================')
print('Simpsons Rule I = %s' %S)


#%%Final Results using Simpsons
'''
Simpson's for PSI1 at ε = 1e-8 (Table 5 in report, row 3)
'''
f = PSI1
a = np.array([0,0,0])
b = np.array([2,2,2])
ε = 1e-8  # set error


S = simp_3d(f, a, b, ε)
print('=====================================')
print('Simpsons Rule I = %s' %S)

