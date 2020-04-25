# This module contains the pedestrian models for the dissertation research of Jiuyang Bai
# on the control law of pedestrian collision avoidance, please contact baijiuyang@hotmail.com
# for support. Git: https://github.com/baijiuyang/collision-avoidance.git


import numpy as np
from numpy import sin, cos, tan, arcsin, arccos, arctan, inner
from numpy.linalg import norm
from helper import dist, d_dist, sp2a, rotate
import helper
from math import pi as PI

class Model:
    '''
    This class represents the model of pedestrian locomotion.
    
    Fields:
        model_name (char array): Model names.
        model_args (dict): Model arguments in form of {'arg_name': arg_value}.   
        ref (2-d vector): Allocentric reference axis.
    '''

    def __init__(self, model, ref=[0, 1]):
        self.model_name = model['name']
        self.model_args = model
        self.ref = [i / norm(ref) for i in ref]

    def __call__(self, agent, source, source_type):
        '''
        Predicts the influence of one entity on the kinematics of the pedestrian.

        Args:
            agent, source (Agent object): The agent to be predicted and sources of influence.
        
        Return:
            (dict): The model prediction in the form of
                {'ax': ax, 'ay': ay, 'd_s': d_s, 'dd_phi': dd_phi}. 
        '''
        p0, v0, p1, v1 = agent.p, agent.v, source.p, source.v
        args = self.model_args
        ax, ay, d_s, dd_phi = 0, 0, 0, 0
        
        # Make prediction
        if self.model_name == 'mass_spring_approach' and ('g' in source_type):
            s, phi, d_phi = agent.s, agent.phi, agent.d_phi
            psi_g = helper.psi(p0, p1, ref=self.ref)
            r_g = dist(p0, p1)
            d_s, dd_phi = mass_spring_approach(s, phi, d_phi, psi_g, r_g, args['p_spd'], args['t_relax'], args['b'], args['k_g'], args['c_1'], args['c_2'])
        
        elif self.model_name == 'optical_ratio_avoid' and ('o' in source_type):
            w = source.w
            d_theta = helper.d_theta(p0, p1, v0, v1, w)
            beta = helper.beta(p0, p1, v0)
            d_beta = helper.d_beta(p0, p1, v0, v1, agent.d_phi)
            d_s, dd_phi = optical_ratio_avoid(d_theta, beta, d_beta, args['k_s'], args['k_h'], args['c'])
        
        elif self.model_name == 'perpendicular_acceleration_avoid' and ('o' in source_type):
            w = source.w
            d_phi = agent.d_phi
            psi = helper.psi(p0, p1, ref=self.ref)
            theta = helper.theta(p0, p1, w)
            d_theta = helper.d_theta(p0, p1, v0, v1, w)
            d_psi = helper.d_psi(p0, p1, v0, v1)
            beta = helper.beta(p0, p1, v0)
            alpha, a_mag = perpendicular_acceleration_avoid(beta, psi, theta, d_theta, d_psi, args['k'], args['c'])
            a = rotate([i * a_mag for i in self.ref], alpha)
            ax = a[0]
            ay = a[1]
            # print(beta * 180 / PI, alpha * 180 / PI, a_mag)
        
        elif self.model_name == 'cohen_avoid' and ('o' in source_type):
            d_s, dd_phi = cohen_avoid(beta, d_psi, d_phi, r, s, args['s0'], args['b'], args['k'], args['c5'], args['c6'])
            
        elif self.model_name == 'cohen_avoid1' and ('o' in source_type):
            d_s, dd_phi = cohen_avoid1(beta, d_theta, d_psi, d_phi, s, args['s0'], args['b'], args['k'], args['c5'], args['c6'])
            
        return {'ax': ax, 'ay': ay, 'd_s': d_s, 'dd_phi': dd_phi}
        
        
def mass_spring_approach(s, phi, d_phi, psi_g, r_g, p_spd, t_relax, b, k_g, c_1, c_2):
    d_s = (p_spd - s) / t_relax
    dd_phi = -b * d_phi - k_g * (phi - psi_g) * (np.exp(-c_1 * r_g) + c_2)
    return d_s, dd_phi

def optical_ratio_avoid(d_theta, beta, d_beta, k_s, k_h, c):
    if d_theta <= 0 or abs(beta) >= PI / 2:
        return 0, 0
    d_s = k_s * sin(beta) * (np.sign(d_beta) * c - d_beta / d_theta)
    dd_phi = k_h * cos(beta) * (-np.sign(d_beta) * c + d_beta / d_theta)
    # print('beta = ', beta)
    # print('d_beta = ', d_beta)
    # print('d_theta = ', d_theta)
    return d_s, dd_phi
    
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
    
def cohen_avoid(beta, d_psi, d_phi, r, s, s0, b, k, c5, c6):
    step = (np.sign(PI / 2 - np.absolute(beta)) + 1) / 2
    d_s = -b * (s - s0) + k * d_psi * np.exp(-c5 * np.absolute(d_psi) - c6 * r) * step
    dd_phi = -b * d_phi - k * d_psi * np.exp(-c5 * np.absolute(d_psi) - c6 * r) * step
    return d_s, dd_phi
    
def cohen_avoid1(beta, d_theta, d_psi, d_phi, s, s0, b, k, c5, c6):
    sigmoid = 1 / (1 + np.exp(20 * (np.absolute(beta) - 1.3)))
    d_s = -b * (s - s0) + np.sign(d_psi) * k * np.exp(-c5 * np.absolute(d_psi)) * (1 - np.exp(-c6 * np.maximum(0, d_theta))) * sigmoid
    dd_phi = -b * d_phi - np.sign(d_psi) * k * np.exp(-c5 * np.absolute(d_psi)) * (1 - np.exp(-c6 * np.maximum(0, d_theta))) * sigmoid
    
    