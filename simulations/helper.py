# This module contains all the helper functions for my dissertation research on 
# pedestrian avoidance
import numpy as np
from numpy.linalg import norm
import math


def rate_of_expansion(p0, p1, v0, v1, w, relative=False):
    '''
    Calculate the (relative) rate of optical expansion of p1 in the perspective
    of p0. This function assumes symmetrical expansion of p1. 
    
    Args:
        p0, p1 (2-d np array of float): Time series of positions in meter,
            with the shape (n_step, 2).
        v0, v1 (2-d np array of float): Time series of velocities in meter / second,
            with the shape (n_step, 2).
        w (float): The width of the leader in meters.
        relative (boolean): Whether compute relative rate of expansion.
        
    Return:
        RE (1-d np array of float): An array of (relative) rate of optical expansion
            in radians / second.
    '''
    r01 = p1 - p0 # Vector pointing to agent 1 from agent 0.
    r = np.linalg.norm(r01, axis=1) # Distance between agent 0 and agent 1.
    drdt = np.zeros_like(r) # The time derivative of distance
    RE = np.zeros_like(r) # Rate of optical expansion
    for i in range(len(drdt)):
        # The time derivative of distance is the sum of speed along the signed distance
        drdt[i] = (np.inner(v0[i], r01[i]) + np.inner(v1[i], -r01[i])) / r[i]
        RE[i] = w * drdt[i] / (r[i] ** 2 + w ** 2 / 4)
    if relative:
        RE /= 2 * np.arctan(w / (2 * r))
    return RE
    
def rate_of_bearing(p0, p1, v0):
    '''
    Calculate the rate of change of bearing angle of p1 in the perspective
    of p0, using Euler method (not analytical). Bearing angle is defined as 
    the angle between the line of sight and heading direction.
    
    Args:
        p0, p1 (2-d np array of float): Time series of positions in meter,
            with the shape (n_step, 2).
        v0 (2-d np array of float): Time series of velocity of agent 0 
            in meter / second, with the shape (n_step, 2).
        
    Return:
        (1-d np array of float): An array of rate of change of bearing angle
            in radians / step. Note that np.diff will make the length
            n_Step - 1. Therefore, last value is duplicated to pad to n_step. 
    '''
    b = bearing(p0, p1, v0)
    return np.append(np.diff(b), np.diff(b)[-1])

def bearing(p0, p1, v0):
    '''
    Calculate bearing angle of p1 in the perspective of p0 in radians.
    Bearing angle is defined as the angle between the line of sight and
    heading direction.
    
    Args:
        p0, p1 (2-d np array of float): Time series of positions in meter,
            with the shape (n_step, 2).
        v0 (2-d np array of float): Time series of velocity of agent 0 
            in meter / second, with the shape (n_step, 2).
        
    Return:
        (1-d np array of float): Bearing angle in radians.
    '''
    r01 = p1 - p0 # Vector pointing to agent 1 from agent 0.
    r = np.linalg.norm(r01, axis=1) # Distance between agent 0 and agent 1.
    s0 = np.linalg.norm(v0, axis=1)# The speed of agent 0.
    bearing = np.zeros_like(r)
    for i in range(len(bearing)):
        bearing[i] = np.arccos(np.inner(v0[i], r01[i]) / (s0[i] * r[i]))
    return bearing

def simple_gaussian(x, a=2, b=0.1):
    '''
    Probability Density Function (pdf) of Gussian distribution.
    
    Args:
        x (1-d np array of float): x in pdf.
        a, b (float): Parameters of the pdf.
        
    Return:
        (1-d np array of float): y in pdf
    '''
    return np.exp(-x ** 2 / a) / b
    
    
def broken_sigmoid(x, a=1, b=20):
    '''
    A distorted sigmoid function. Equivalent to sigmoid function
    when x < 0. It is symmetrical about the y-axis.
    
    Args:
        x (1-d np array of float): Input of the function.
        a, b (float): Parameters of the function.
        
    Return:
        (1-d np array of float): Output of the function
    '''
    return b / (1 + np.exp(np.absolute(x / a)))

def minimum_separation(p0, p1, v0, v1):
    '''
    This function computes the minimum separation or distance at closest approach (dca) 
    and time to minimum separation or time to closest approach (ttca) of two trajectiles.
    
    Args:
        p0, p1 (tuple): Initial positions of agent 0 and agent 1 in meters.
        v0, v1 (tuple): Velocities of agent 0 and agent 1 in meters/second.
        
    Return:
        dca (float): Minimum separation or distance at closest approach in meters. Positive
            value means agent 0 will pass in front on agent 1, negative means the opposite.
        ttca (float): Time to minimum separation or time to closest approach in seconds.
            negative value means the minimum separation already happend.
    '''
    if p0[0] == p1[0] and p0[1] == p1[1]:
        return 0, 0
    p01 = np.array(p1) - np.array(p0)
    if v0[0] == v1[0] and v0[1] == v1[1]:
        return norm(p01), 0
    v10 = np.array(v0) - np.array(v1)
    dca = np.sqrt(1 - (np.inner(p01, v10) / (norm(p01) * norm(v10))) ** 2) * norm(p01)
    ttca = np.inner(p01, v10) / norm(v10) ** 2 # np.inner takes care of the sign
    if np.inner(np.array(v1), np.array(p0) - np.array(p1) + v10 * ttca) < 0: # Check passing order
        dca *= -1
    return dca, ttca

    
    
    
    
    
    
    
    
    