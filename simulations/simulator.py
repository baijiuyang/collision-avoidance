# This module contains the simulation and agent classes for the dissertation research of Jiuyang Bai
# on the control law of pedestrian collision avoidance, please contact baijiuyang@hotmail.com
# for support. Git: https://github.com/baijiuyang/collision-avoidance.git

import numpy as np
from numpy.linalg import norm
from helper import sp2v, v2sp, sp2a
from numpy import sqrt, sin, cos
from models import Model
import time
from matplotlib import animation, pyplot as plt
import math


class Simulation:
    '''
    This class provides animiation, trajectory plots, variable plots functions.
    
    Fields:
        t (list of floats): Time stamps in seconds.
        p, v, a, phi, d_phi, dd_phi, d_s ({agent.id: 2-d list of float}): Time series of positions, velocities, accelerations,
            heading, 1st order d of heading, 2nd order d of heading, 1st order d of speed.
        agents (Object of Agent class): See definition in Agent class.
        Hz (int): Temporal frequency of the simulation.
    '''
    def __init__(self, agents, Hz):
        self.t = [0]
        self.p = {}
        self.v = {}
        self.a = {}
        self.phi = {}
        self.d_phi = {}
        self.dd_phi = {}
        self.d_s = {}
        self.agents = {}
        self.add_agents(agents)
        self.Hz = Hz
        
    def add_agent(self, agent):
        self.agents[agent.id] = agent
        self.p[agent.id] = [agent.p[:]]
        self.v[agent.id] = [agent.v[:]]
        self.a[agent.id] = [agent.a[:]]
        self.phi[agent.id] = [agent.phi]
        self.d_phi[agent.id] = [agent.d_phi]
        self.dd_phi[agent.id] = [agent.dd_phi]
        self.d_s[agent.id] = [agent.d_s]
        
        
    def add_agents(self, agents):
        for agent in agents:
            self.add_agent(agent)
    
    def update(self, Hz=None):
        if not Hz: Hz = self.Hz
        # Parallel update: 1. Computing without updating, 2. Updating
        # Compute  
        p_old, v_old = {}, {}
        for i, p in self.p.items():
            if len(self.t) == 1:
                p_old[i] = p[-1]
            else:
                p_old[i] = p[-2]
        for i, v in self.v.items():
            if len(self.t) == 1:
                v_old[i] = v[-1]
            else:
                v_old[i] = v[-2]
        # print('p_old = ', p_old)
        # print('v_old = ', v_old)
        for agent in self.agents.values():
            agent.interact(self.agents.values(), p_old, v_old, Hz)

        # Update
        self.t.append(self.t[-1] + 1.0 / Hz)
        for agent in self.agents.values():
            agent.move(Hz)
            # Record new states
            self.p[agent.id].append(agent.p[:])
            self.v[agent.id].append(agent.v[:])
            self.a[agent.id].append(agent.a[:])
            self.phi[agent.id].append(agent.phi)
            self.d_phi[agent.id].append(agent.d_phi)
            self.dd_phi[agent.id].append(agent.dd_phi)
            self.d_s[agent.id].append(agent.d_s)
        
    def simulate(self, t_total, Hz=None):
        if not Hz: Hz = self.Hz
        # Run update t_total * Hz times
        for i in range(int(t_total * Hz) - 1):
            self.update(Hz=Hz)
    
    def play(self, interval=None, save=False):
        '''
        Args:
            interval (float): The pause between two frames in millisecond.
            save (bool): Flag for saving the animation in the current working directory.
        '''
        if not interval: interval = 1000 / self.Hz # real time
        # Create a figure
        fig = plt.figure(figsize=(7,7))
        ax = plt.axes(xlim=(-12, 12), ylim=(-12, 12))
        # Set the aspect ratio of x and y axis equal to the true value
        ax.set_aspect('equal')
        
        # Initialize data
        agents = []
        ids = []
        angles = np.linspace(0, 2 * math.pi, num=12) # Use 12 points to approximate a circle
        for i, traj in self.p.items():
            w = self.agents[i].w if self.agents[i].w else 0.1
            circle = np.stack((w / 2 * cos(angles), w / 2 * sin(angles)), axis=-1) # 12 points on the perimeter
            agent, = ax.plot(traj[0][0] + circle[:, 0], traj[0][1] + circle[:, 1])
            ax.plot([p[0] for p in traj], [p[1] for p in traj], agent.get_color())
            agents.append(agent)
            ids.append(i)
        
        def animate_fast(i):
            '''
            Fast animation function update without redraw everything. 
            Good for watching in real time, but will leave trace if saved.
            '''
            for agent, id in zip(agents, ids):
                w = self.agents[id].w if self.agents[id].w else 0.1
                circle = np.stack((w / 2 * cos(angles), w / 2 * sin(angles)), axis=-1) # 12 points on the perimeter
                agent.set_data(self.p[id][i][0] + circle[:, 0], self.p[id][i][1] + circle[:, 1])
            return agents
        
        def animate_slow(i):
            '''
            slow animation function redraw everything at each frame. 
            Good for saving video but too slow to watch in real time.
            '''
            ax.clear()
            # Redefine the ax
            ax.set_xlim(-12, 12)
            ax.set_ylim(-12, 12)
            ax.set_aspect('equal')
            for j in range(len(agents)):
                id = ids[j]
                w = self.agents[id].w if self.agents[id].w else 0.1
                circle = np.stack((w / 2 * cos(angles), w / 2 * sin(angles)), axis=-1) # 12 points on the perimeter
                agents[j], = ax.plot(self.p[id][i][0] + circle[:, 0], self.p[id][i][1] + circle[:, 1])
                ax.plot([p[0] for p in self.p[id]], [p[1] for p in self.p[id]], agents[j].get_color())
            return agents
        
        # call the animator.  blit=True means only re-draw the parts that have changed.
        animate = animate_slow if save else animate_fast
        anim = animation.FuncAnimation(fig, animate, frames=len(self.t), interval=interval, blit=True)

        # save the animation as an mp4.  This requires ffmpeg or mencoder to be
        # installed.  The extra_args ensure that the x264 codec is used, so that
        # the video can be embedded in html5.  You may need to adjust this for
        # your system: for more information, see
        # http://matplotlib.sourceforge.net/api/animation_api.html
        if save:
            t = time.localtime()
            t = [str(i) for i in t[:6]]
            t = "-".join(t)
            filename = 'simulation' + t + '.mp4'
            anim.save(filename)
        return anim
        # For command line usage
        # plt.show()
        
        
    
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
        
        Args:
            model_args (dict): {'arg_name': arg value}.
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
            self.models[name] = Model(name, args, self.ref)
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
    
    def interact_pairwise(self, source, p0_old, p1_old, v0_old, v1_old, Hz):
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
        
        source_type = ''
        if source.id == self.goal_id:
            source_type += 'g' # goal
        if source.w:
            source_type += 'o' # obstacle
            
        # Combine the influences of all models
        for model in self.models.values():
            _d_s, _dd_phi = model(self, source, source_type, p0_old, p1_old, v0_old, v1_old, Hz)
            d_s += _d_s
            dd_phi += _dd_phi
        return d_s, dd_phi
        
    def interact(self, sources, p_old, v_old, Hz):
        '''
        Combine the influences from all sources.
        
        Args:
            sources (Agent object): Source of influence.
            p_old, v_old (dict): {agent_id: position of last frame}, {agent_id: velocity of last frame}.
        '''
        if self.constant: return
        # Pre-allocation of return variables
        d_s, dd_phi = 0, 0
        p0_old = p_old[self.id]
        v0_old = v_old[self.id]
        # Combine the influence of all sources
        for source in sources:
            p1_old = p_old[source.id]
            v1_old = v_old[source.id]
            _d_s, _dd_phi = self.interact_pairwise(source, p0_old, p1_old, v0_old, v1_old, Hz)
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
     
