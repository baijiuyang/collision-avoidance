# This module contains the pedestrian models for the dissertation research of Jiuyang Bai
# on the control law of pedestrian collision avoidance, please contact baijiuyang@hotmail.com
# for support. Git: https://github.com/baijiuyang/collision-avoidance.git


import numpy as np

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
    '''

    def __init__(self, model_name, model_args):
        self.model_name = model_name
        self.model_args = model_args

    def __call__(self, p1, v1, source_type):
        '''
        Predicts the influence of one entity on the kinematics of the pedestrian.

        Args:
            p1, v1 (2-d np array of float): The time series of positions and velocities of the 
            influencing entity with the shape of (n_steps, 2).
        
        Return:
            d_s, dd_phi (1-d np array of float): The time series of tangential and rotational
                acceleration of the pedestrian.   
        '''
        # Convert 1-d input to 2-d input
        if len(np.shape(p1)) == 1:
            p1 = np.expand_dims(p1, axis=0)
        if len(np.shape(v1)) == 1:
            v1 = np.expand_dims(v1, axis=0)
        
        # Make prediction
        if self.model_name == 'approach_stationary_goal' and ('g' in source_type):
            print("Agent approached goal = ", self.approach_stationary_goal(p1, v1))
            return self.approach_stationary_goal(p1, v1)
            
        elif self.model_name == 'optical_ratio_model' and ('o' in source_type):
            print("Agent avoided obstacle = ", self.optical_ratio_model(p1, v1))
            return self.optical_ratio_model(p1, v1)
        
        else:
            return self.null(len(p1))
        
    def null(self, n):
        if n == 1:
            return 0, 0
        else:
            return np.zeros(n), np.zeros(n)
        
    def approach_stationary_goal(self, p1, v1):
        return 0, 0
    
    def optical_ratio_model(self, p1, v1):
        return 0, 0
    

    