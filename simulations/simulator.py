# This module contains the simulation and agent classes for the dissertation research of Jiuyang Bai
# on the control law of pedestrian collision avoidance, please contact baijiuyang@hotmail.com
# for support. Git: https://github.com/baijiuyang/collision-avoidance.git

import numpy as np
from numpy.linalg import norm
from helper import sp2v, v2sp, sp2a
from numpy import sqrt
from models import Model



class Simulation:
    '''
    This class provides animiation, trajectory plots, variable plots functions.
    
    Fields:
        t (list of floats): Time stamps in seconds.
        p, v, a ({agent.id: 2-d list of float}): Time series of positions, velocities, accelerations.
        agents (Object of Agent class): See definition in Agent class.
        Hz (int): Temporal frequency of the simulation.
    '''
    def __init__(self, agents, Hz):
        self.t = [0]
        self.p = {}
        self.v = {}
        self.a = {}
        self.agents = {}
        self.add_agents(agents)
        self.Hz = Hz
        
    def add_agent(self, agent):
        self.agents[agent.id] = agent
        self.p[agent.id] = [agent.p[:]]
        self.v[agent.id] = [agent.v[:]]
        self.a[agent.id] = [agent.a[:]]
        
    def add_agents(self, agents):
        for agent in agents:
            self.add_agent(agent)
    
    def update(self, Hz=None):
        if not Hz: Hz = self.Hz
        # Parallel update: 1. Computing without updating, 2. Updating
        # Compute
        for agent in self.agents.values():
            agent.interact(self.agents.values())

        # Update
        for agent in self.agents.values():
            agent.move(Hz)
            # Record new states
            self.t.append(self.t[-1] + 1.0 / Hz)
            self.p[agent.id].append(agent.p[:])
            self.v[agent.id].append(agent.v[:])
            self.a[agent.id].append(agent.a[:])
        
    def simulate(self, t_total, Hz=None):
        if not Hz: Hz = self.Hz
        # Run update t_total * Hz times
        for i in range(int(t_total * Hz)):
            self.update(Hz=Hz)
    
    def play(self):
        pass
    
    def plot_positions(self):
        pass
        
    def plot_speeds(self):
        pass
    
class Agent:
    '''
    This class represents the agent to be simulated, who has kinematic status and
    obey the control laws reflected by the models.
    
    Fields:
        id (int): Agent id.
        p, v, a (2-d vector of floats): The current position, velocity
            and acceleration of the agent.
        s, phi (1-d array of floats): The current speed and heading.
        d_s, d_phi (1-d array of floats): The current rate of change of
            speed and heading.
        dd_phi (1-d array of floats): The current rate of change of d_phi.
        p_spd (float): The preferred speed of the agent.
        models (dict): Keys are model names. Values are model functions.
        ref (2-d vector): The allocentric reference axis.
        d_s_next, dd_phi_next (floats): Backstage d_s, and dd_phi.
    '''
    def __init__(self, id, goal_id, w, p0, v0, a0, s, phi, d_phi, dd_phi, d_s, p_spd, model_names, model_args, constant, ref=[0,1]):
        '''
        This constructor initialize an agent (1) through vectoral velocity and acceleration or
        (2) through speed, phi (orientation given ref as the allocentric reference axis), d_phi
        '''
        self.id = id
        self.goal_id = goal_id
        self.w = w
        self.p = p0
        self.v = v0
        self.a = a0
        self.s = s
        self.d_s = d_s
        self.phi = phi
        self.d_phi = d_phi
        self.dd_phi = dd_phi
        self.p_spd = p_spd
        self.models = {}
        self.ref = ref
        self.constant = constant
        self.d_s_next = 0
        self.dd_phi_next = 0
        for name, args in zip(model_names, model_args):
            self.models[name] = Model(name, args)
        if v0 == None:
            self.sp2av()
        else:
            self.av2sp()

    def sp2av(self):
        self.v = sp2v(self.s, self.phi, ref=self.ref)
        self.a = sp2a(self.s, self.d_s, self.phi, self.d_phi)
    
    def av2sp(self):
        s, phi = v2sp(self.v, ref=self.ref)
        if s == 0:
            phi = 0
        self.s, self.phi = s, phi
    
    def interact_pairwise(self, source):
        '''
        Computes the tangential acceleration (d_s) and rotational acceleration (dd_phi)
        caused by one source based on the models.
        
        Args:
            source (Object from the Agent class): See definition in the Agent class.
    
        Return:
            d_s, dd_phi (1-d np array of float): The time series of tangential and rotational
                acceleration of the pedestrian.
        '''
        # print("%d is interacting with %d\n" % (self.id, source.id))
        # Pre-allocation of return variables
        d_s, dd_phi = 0, 0
        
        # Do not interact with self or if self is constant
        if source.id == self.id or self.constant: 
            return d_s, dd_phi
        
        p1, v1 = source.p, source.v
        source_type = ''
        if source.id == self.goal_id:
            source_type += 'g' # goal
        if source.w:
            source_type += 'o' # obstacle
            
        # Combine the influences of all models
        for model in self.models.values():
            _d_s, _dd_phi = model(p1, v1, source_type)
            d_s += _d_s
            dd_phi += _dd_phi
        return d_s, dd_phi
        
    def interact(self, sources):
        '''
        Combine the influences from all sources.
        '''
        if self.constant: return
        # Pre-allocation of return variables
        d_s, dd_phi = 0, 0
        
        # Combine the influence of all sources
        for source in sources:
            _d_s, _dd_phi = self.interact_pairwise(source)
            d_s += _d_s
            dd_phi += _dd_phi
        self.d_s_next, self.dd_phi_next = d_s, dd_phi
            
    def move(self, Hz):
        '''
        Updates position, velocity, and acceleration using current velocity, acceleration, 
        d_s and dd_phi.
        
        Args:
            Hz (float): The temporal frequency of simulation.
        '''
        # Update variables based on the current values of their derivatives
        self.p[0] += self.v[0] * (1.0 / Hz) + 0.5 * self.a[0] * (1.0 / Hz) ** 2
        self.p[1] += self.v[1] * (1.0 / Hz) + 0.5 * self.a[1] * (1.0 / Hz) ** 2        
        self.phi += self.d_phi * (1.0 / Hz) + 0.5 * self.dd_phi * (1.0 / Hz) ** 2
        self.d_phi += self.dd_phi * (1.0 / Hz)
        self.s += self.d_s * (1.0 / Hz)
        
        # Update the derivatives
        self.d_s, self.dd_phi = self.d_s_next, self.dd_phi_next
        self.sp2av()
     
