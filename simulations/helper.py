# This module contains all the helper functions for the dissertation research of Jiuyang Bai
# on the control law of pedestrian collision avoidance please contact baijiuyang@hotmail.com
# for support. Github: https://github.com/baijiuyang/collision-avoidance.git

import numpy as np
from numpy.linalg import norm
from numpy import sqrt
import math
from matplotlib import animation, pyplot as plt


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
    
def rate_of_bearing(p0, p1, v0, Hz):
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
            in radians / second. Note that np.diff will make the length
            n_Step - 1. Therefore, last value is duplicated to pad to n_step. 
    '''
    b = bearing(p0, p1, v0)
    return np.append(np.diff(b), np.diff(b)[-1]) * Hz

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

    
def collision_trajectory(bearing, side, spd1=2.0, w=1.5, r=10, Hz=100, animate=False, interval=None):
    '''
    This function produces near-collision trajectories of circular agents 0 and 1 with 
    a defined bearing angle. The speed of agent 1 is constant. The speed of agent 0 will 
    vary to guarantee near-collision. 
    
    Args:
        bearing ((-90,90), float): The initial bearing angle between A and B in degree. Bearing 
            angle is defined as the angle between the line of sight and heading direction.        
        side (char): Agent 0 pass in front ('f'), or from behind ('b') agent 1. 
        spd1 (float): The speed of agent 1 in meter/second.
        w (float): The diameter of both agents in meters.
        r (float): The distance from agent 1 to origin of the coordinate system. 
        Hz (int): Number of Hz for the simulation.
        animate (bool): Whether animate the trajectories.
        interval (int): Delay between frames in milliseconds.
        
    Return:
        traj0, traj1 (2-d np array of float): Near-collision trajectories of agent 1 and agent 0 in 
            meters.
        np.tile(v0, (n, 1)), np.tile(v1, (n, 1)) (2-d np array of float): The velocity of agent 0 to 
            achieve near-collision with agent 1.
    '''
    alpha = (180 - 2 * bearing) * 2 * math.pi / 360 # Central angle of line of sight
    p1 = np.array([-r * np.sin(alpha), -r * np.cos(alpha)])
    p0 = np.array([0, -r])
    p01 = p1 - p0
    v1 = np.array([spd1 * np.sin(alpha), spd1 * np.cos(alpha)])
    a = r ** 2 * np.sin(alpha) ** 2 - w ** 2
    b = -2 * spd1 * (np.cos(alpha) * (norm(p01) ** 2 - w ** 2) + r ** 2 * (1 - np.cos(alpha)) ** 2)
    c = spd1 ** 2 * (norm(p01) ** 2 - w ** 2 - r ** 2 * (1 - np.cos(alpha)) ** 2)
    d = (b ** 2) - (4 * a * c)

    # find two solutions
    sol1 = (-b - sqrt(d)) / (2 * a)
    sol2 = (-b + sqrt(d)) / (2 * a)
    
    if side == 'f':
        v0 = np.array([0, max(sol1, sol2)])
    elif side == 'b': 
        v0 = np.array([0, min(sol1, sol2)])
    else:
        raise Exception('Argument side should be either \'f\' or \'b\'')
    t = 2 * r / spd1
    n = int(t * Hz)
    traj0 = np.cumsum(np.tile(v0 / Hz, (n, 1)), axis=0) + np.expand_dims(p0, axis=0)
    traj1 = np.cumsum(np.tile(v1 / Hz, (n, 1)), axis=0) + np.expand_dims(p1, axis=0)
    
    if animate:
        if not interval: interval = 1000 / Hz
        fig = plt.figure(figsize=(7, 7))
        ax = plt.axes(xlim=(-12, 12), ylim=(-12, 12))
        angles = np.linspace(0, 2 * math.pi, num=12)
        circle = np.stack((w / 2 * np.cos(angles), w / 2 * np.sin(angles)), axis=-1)
        agent0, = ax.plot(traj0[0, 0] + circle[:, 0], traj0[0, 1] + circle[:, 1], 'b')
        agent1, = ax.plot(traj1[0, 0] + circle[:, 0], traj1[0, 1] + circle[:, 1], 'r')
        def animate_fast(i):
            '''
                Fast animation function update without clear. Good for
                watching in real time, but will leave trace if saved.
            '''
            # ms is the short for markersize
            agent0.set_data(traj0[i, 0] + circle[:, 0], traj0[i, 1] + circle[:, 1])
            agent1.set_data(traj1[i, 0] + circle[:, 0], traj1[i, 1] + circle[:, 1])

            return agent0, agent1
        # call the animator.  blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(fig, animate_fast, frames=len(traj0), interval=interval, blit=True)
    
    return traj0, traj1, np.tile(v0, (n, 1)), np.tile(v1, (n, 1))
    
    
    
    
    
    