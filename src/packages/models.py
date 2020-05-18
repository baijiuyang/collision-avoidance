# This module contains the pedestrian models for the dissertation research of Jiuyang Bai
# on the control law of pedestrian collision avoidance, please contact baijiuyang@hotmail.com
# for support. Git: https://github.com/baijiuyang/collision-avoidance.git


import numpy as np
from numpy import sin, cos, tan, arcsin, arccos, arctan, inner
from numpy.linalg import norm
from packages.helper import dist, d_dist, sp2a, rotate
from packages import helper
from math import pi as PI

class Model:
    '''
    This class represents the model of pedestrian locomotion.
    
    Fields:
        name (char array): Model names.
        args (dict): Model arguments in form of {'arg_name': arg_value}.   
        ref (2-d vector): Allocentric reference axis.
    '''

    def __init__(self, model, ref=[0, 1]):
        self.name = model['name']
        self.args = model
        self.ref = [i / norm(ref) for i in ref]

    def __call__(self, inputs):
        '''
        Predicts the influence of one entity on the
            kinematics of the pedestrian.

        Args:
            inputs (dict): {'p0', 'p1', 'v0', 'w', ...}.
        
        Return:
            (dict): The model prediction in the form of
                {'ax': ax, 'ay': ay, 'd_s': d_s, 'dd_phi': dd_phi}. 
        '''
        p0, v0, p1, v1 = inputs.get('p0', None), inputs.get('v0', None), inputs.get('p1', None), inputs.get('v1', None)
        s0, phi, d_phi = inputs.get('s0', None), inputs.get('phi', None), inputs.get('d_phi', None)
        w = inputs.get('w', None)
        args = self.args
        ax, ay, d_s, dd_phi = None, None, None, None
        
        # Make prediction
        if self.name == 'mass_spring_approach':
            psi = helper.psi(p0, p1, ref=self.ref)
            r = dist(p0, p1)
            d_s, dd_phi = mass_spring_approach(s0, phi, d_phi, psi, r, args['ps'], args['t_relax'], args['b'], args['k_g'], args['c_1'], args['c_2'])
        
        elif self.name == 'parallel_perpendicular_approach':
            beta = helper.beta(p0, p1, v0)
            a_para_mag, a_perp_mag = parallel_perpendicular_approach(beta, s0, args['ps'], args['t_relax0'], args['t_relax1'])
            a_para = rotate([i * a_para_mag for i in self.ref], phi)
            a_perp = rotate([i * a_perp_mag for i in self.ref], phi + np.sign(beta) * PI / 2)
            a = a_para + a_perp
            if len(np.shape(a)) == 1:
                ax = a[0]
                ay = a[1]
            else:
                ax = a[:,0]
                ay = a[:,1]
            
        elif self.name == 'perpendicular_acceleration_avoid':
            beta = helper.beta(p0, p1, v0)
            psi = helper.psi(p0, p1, ref=self.ref)
            theta = helper.theta(p0, p1, w)
            d_theta = helper.d_theta(p0, p1, v0, v1, w)
            d_psi = helper.d_psi(p0, p1, v0, v1)
            alpha, a_mag = perpendicular_acceleration_avoid(beta, psi, theta, d_theta, d_psi, args['k'], args['c'])
            a = rotate([i * a_mag for i in self.ref], alpha)
            if len(np.shape(a)) == 1:
                ax = a[0]
                ay = a[1]
            else:
                ax = a[:,0]
                ay = a[:,1]
            # print(beta * 180 / PI, alpha * 180 / PI, a_mag)
        
        elif self.name == 'cohen_avoid':
            beta = helper.beta(p0, p1, v0)
            d_psi = helper.d_psi(p0, p1, v0, v1)
            r = dist(p0, p1)
            d_s, dd_phi = cohen_avoid(beta, d_psi, d_phi, r, s0, args['ps'], args['b'], args['k_mo'], args['c_5'], args['c_6'])
            
        elif self.name == 'cohen_avoid1':
            beta = helper.beta(p0, p1, v0)
            d_theta = helper.d_theta(p0, p1, v0, v1, w)
            d_psi = helper.d_psi(p0, p1, v0, v1)
            d_s, dd_phi = cohen_avoid1(beta, d_theta, d_psi, d_phi, s0, args['ps'], args['b'], args['k_mo'], args['c_5'], args['c_6'])
        
        elif self.name == 'vector_approach':
            a = vector_approach(p0, p1, v0, args['ps'], args['t_relax'])
            if len(np.shape(a)) == 1:
                ax = a[0]
                ay = a[1]
            else:
                ax = a[:,0]
                ay = a[:,1]
            
        return {'ax': ax, 'ay': ay, 'd_s': d_s, 'dd_phi': dd_phi}
        
        
def mass_spring_approach(s0, phi, d_phi, psi_g, r, ps, t_relax, b, k_g, c_1, c_2):
    d_s = (ps - s0) / t_relax
    dd_phi = -b * d_phi - k_g * (phi - psi_g) * (np.exp(-c_1 * r) + c_2)
    return d_s, dd_phi

def vector_approach(p0, p1, v0, ps, t_relax):
    vi = np.array(p1) - np.array(p0)
    vp = vi * ps / norm(vi, axis=-1)
    return (vp - np.array(v0)) / t_relax

def parallel_perpendicular_approach(beta, s0, ps, t_relax0, t_relax1):
    a_para_mag = (ps - s0) / t_relax0
    a_perp_mag = np.absolute(beta) * s0 / t_relax1
    return a_para_mag, a_perp_mag
    
def perpendicular_acceleration_avoid(beta, psi, theta, d_theta, d_psi, k, c):
    '''
    Computes the direction and magnituide of acceleration for moving obstacle avoidance.
    
    Args:
        beta (float or np array of floats): The angle between line of sight and heading.
            positive mean line of sight is on the right of heading.
        psi (float or np array of floats): The direction of line of sight given an
            allocentric reference axis.
        theta (float or np array of floats): The visual angle of the obstacle.
        d_theta, d_psi (float or np array of floats): The rate of change of theta and psi.
        k, c (float or np array of floats): 
    '''
    alpha = psi - np.sign(d_psi) * (PI / 2)
    ratio = k * ((np.maximum(0, d_theta / theta) + c) / (np.absolute(d_psi) + c) - 1)
    sigmoid = 1 / (1 + np.exp(20 * (np.absolute(beta) - 1.3)))
    a_mag = ratio * sigmoid # multiply a sigmoid function of beta
    return alpha, a_mag
    
def cohen_avoid(beta, d_psi, d_phi, r, s0, ps, b, k, c_5, c_6):
    step = (np.sign(PI / 2 - np.absolute(beta)) + 1) / 2
    d_s = -b * (s0 - ps) + k * d_psi * np.exp(-c_5 * np.absolute(d_psi) - c_6 * r) * step
    dd_phi = -b * d_phi - k * d_psi * np.exp(-c_5 * np.absolute(d_psi) - c_6 * r) * step
    return d_s, dd_phi
    
def cohen_avoid1(beta, d_theta, d_psi, d_phi, s0, ps, b, k, c_5, c_6):
    sigmoid = 1 / (1 + np.exp(20 * (np.absolute(beta) - 1.3)))
    d_s = -b * (s0 - ps) + d_psi * k * np.exp(-c_5 * np.absolute(d_psi)) * (1 - np.exp(-c_6 * np.maximum(0, d_theta))) * sigmoid
    dd_phi = -b * d_phi - d_psi * k * np.exp(-c_5 * np.absolute(d_psi)) * (1 - np.exp(-c_6 * np.maximum(0, d_theta))) * sigmoid
    return d_s, dd_phi
    