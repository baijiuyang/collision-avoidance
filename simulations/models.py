# This module contains the pedestrian models for the dissertation research of Jiuyang Bai
# on the control law of pedestrian collision avoidance, please contact baijiuyang@hotmail.com
# for support. Git: https://github.com/baijiuyang/collision-avoidance.git


import numpy as np
from numpy import sin, cos, tan, arcsin, arccos, arctan, inner
from numpy.linalg import norm
from helper import dist, d_dist
import helper
from math import pi as PI

class Model:
    '''
    This class represents the model of pedestrian locomotion.
    
    Fields:
        p (2-element list of float): The current position of the agent.
        v (2-element list of float): The current velocity of the agent.
        a (2-element list of float): The current acceleration of the agent.
        p_spd (float): The preferred speed of the agent.
        models (dict): Keys are model names. Values are model functions.
        models (1-d list of char arrays): Model names.
        model_args (2-d list of floats): Model arguments.   
        ref (2-d vector): Allocentric reference axis.
    '''

    def __init__(self, model_name, model_args, ref=[0, 1]):
        self.model_name = model_name
        self.model_args = model_args
        self.ref = ref

    def __call__(self, agent, source, source_type, p0_old, p1_old, v0_old, v1_old, Hz):
        '''
        Predicts the influence of one entity on the kinematics of the pedestrian.

        Args:
            p1, v1 (2-d np array of float): The time series of positions and velocities of the 
            influencing entity with the shape of (n_steps, 2).
        
        Return:
            d_s, dd_phi (1-d np array of float): The time series of tangential and rotational
                acceleration of the pedestrian.   
        '''
        p0, v0, p1, v1 = agent.p, agent.v, source.p, source.v
        
        args = self.model_args
        
        # Make prediction
        if self.model_name == 'approach_stationary_goal' and ('g' in source_type):
            s0, phi, d_phi= agent.s, agent.phi, agent.d_phi
            psi_g = helper.psi(p0, p1, self.ref)
            r_g = dist(p0, p1)
            return approach_stationary_goal(s0, phi, d_phi, psi_g, r_g, args['p_spd'], args['t_relax'], args['b'], args['k_g'], args['c_1'], args['c_2'])
            
        elif self.model_name == 'optical_ratio_model' and ('o' in source_type):
            r01 = [i - j for i, j in zip(p1, p0)]
            v01 = [i - j for i, j in zip(v1, v0)]
            r = norm(r01)
            d_r = d_dist(p0, p1, v0, v1)
            w = source.w
            beta = helper.beta(p0, p1, v0)
            d_beta = helper.d_beta_numeric([p0_old, p0], [p1_old, p1], [v0_old, v0], Hz)[0]
            return optical_ratio_model(r, d_r, w, beta, d_beta, args['k_s'], args['k_h'], args['c'])
        
        else:
            return null(1)
        
def null(n):
    if n == 1:
        return 0, 0
    else:
        return np.zeros(n), np.zeros(n)
        
def approach_stationary_goal(s0, phi, d_phi, psi_g, r_g, p_spd, t_relax, b, k_g, c_1, c_2):
    d_s = (s0 - p_spd) / t_relax
    dd_phi = -b * d_phi - k_g * (phi - psi_g) * (np.exp(-c_1 * r_g) + c_2)
    return d_s, dd_phi
    
def optical_ratio_model(r, d_r, w, beta, d_beta, k_s, k_h, c):
    theta = 2 * arctan(w / (2 * r))
    d_theta = -w * d_r / (r**2 + w**2 / 4)
    if d_theta <= 0 or abs(beta) >= PI / 2:
        return 0, 0
    d_s = k_s * sin(beta) * (np.sign(d_beta) * c - d_beta / d_theta)
    dd_phi = k_h * cos(beta) * (-np.sign(d_beta) * c + d_beta / d_theta)
    # print('beta = ', beta)
    # print('d_beta = ', d_beta)
    # print('d_theta = ', d_theta)
    return d_s, dd_phi
    

    