# This module contains the simulation and agent classes for the dissertation research of Jiuyang Bai
# on the control law of pedestrian collision avoidance, please contact baijiuyang@hotmail.com
# for support. Git: https://github.com/baijiuyang/collision-avoidance.git

import numpy as np
from numpy.linalg import norm
from helper import sp2v, v2sp, sp2a, av2dsdp
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
        self.agents = {}
        self.add_agents(agents)
        self.Hz = Hz

    def add_agent(self, agent):
        self.agents[agent.id] = agent
        self.p[agent.id] = [agent.p[:]]

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
        self.t.append(self.t[-1] + 1.0 / Hz)
        for agent in self.agents.values():
            agent.move(Hz)
            # Record new states
            self.p[agent.id].append(agent.p[:])

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
        agents, circles, ws, ids = [], [], [], []
        angles = np.linspace(0, 2 * math.pi, num=12) # Use 12 points to approximate a circle
        for i, traj in self.p.items():
            w = self.agents[i].w if self.agents[i].w else 0.1
            circle = np.stack((w / 2 * cos(angles), w / 2 * sin(angles)), axis=-1) # 12 points on the perimeter
            agent, = ax.plot(traj[0][0] + circle[:, 0], traj[0][1] + circle[:, 1])
            ax.plot([p[0] for p in traj], [p[1] for p in traj], color=agent.get_color()) # Plot trajectory
            agents.append(agent)
            ws.append(w)
            circles.append(circle)
            ids.append(i)
        
        def animate(i):
            '''
            Fast animation function update without redraw everything. 
            Good for watching in real time, but will leave trace of movement if saved.
            '''
            for agent, circle, id in zip(agents, circles, ids):
                agent.set_data(self.p[id][i][0] + circle[:, 0], self.p[id][i][1] + circle[:, 1])
            return agents
        
        # call the animator.  blit=True means only re-draw the parts that have changed.
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
        goal_id (int): The id of the goal of the agent.
        w (float or None): The width of the agent.
        p, v (2-d vector): The current position and velocity of the agent.
        s, phi (float): The current speed and heading.
        d_s, d_phi (float): The current rate of change of speed and heading.
        a, dd_phi (float): Acceleration and derivative of d_phi from the output of different models.
        p_spd (float): The preferred speed of the agent.
        models (dict): Keys are model names. Values are model functions.
        ref (2-d vector): The allocentric reference axis.
        constant (bool): Whether this agent has constant state.
    '''
    def __init__(self, id, init_state, goal_id=None, w=None, p_spd=None, model_names=None, model_args=None, constant=False, ref=[0, 1]):
        '''
        This constructor initialize an agent (1) through vectoral velocity and acceleration or
        (2) through speed, phi (orientation given ref as the allocentric reference axis), d_phi
        
        Args:
            model_args (dict): {'arg_name': arg value}.
            init_state (dict): {'p': p0, 'v': v0, 's': s0, 'phi': phi0}
        '''
        self.id = id
        self.goal_id = goal_id
        self.w = w
        self.ref = [i / norm(ref) for i in ref]
        self.p_spd = p_spd
        self.constant = constant
        self.p = init_state['p']
        if 'v' in init_state:
            self.v = init_state['v']
            self.s, self.phi = v2sp(self.v, ref=self.ref)
        else:
            self.s = init_state['s']
            self.phi = init_state['phi']
            self.v = sp2v(self.s, self.phi, ref=self.ref)
        self.a = [0, 0]
        self.d_s = 0
        self.d_phi = 0
        self.dd_phi = 0
        self.models = {}
        self.a_next = [0, 0]
        self.d_s_next = 0
        self.dd_phi_next = 0
        if not self.constant:
            for name, args in zip(model_names, model_args):
                self.models[name] = Model(name, args, self.ref)

    def interact_pairwise(self, source):
        '''
        Computes vector acceleration caused by one source based on the models.
        
        Args:
            source (Object from the Agent class): See definition in the Agent class.
    
        Return:
            pred_sum (dict): The sum of model predictions in the form of
                {'ax': ax, 'ay': ay, 'd_s': d_s, 'dd_phi': dd_phi}.
        '''
        pred_sum = {}
        # Do not interact with self or if self is constant
        if source.id == self.id or self.constant: 
            return pred_sum
        
        source_type = ''
        if source.id == self.goal_id:
            source_type += 'g' # goal
        if source.w:
            source_type += 'o' # obstacle
            
        # Combine the influences of all models
        for model in self.models.values():
            pred = model(self, source, source_type) # Model objects return a dictionary 
            for key in pred:
                pred_sum[key] = pred_sum.get(key, 0) + pred[key]
        return pred_sum
        
    def interact(self, sources):
        '''
        Combine the influences from all sources.
        
        Args:
            sources (Agent object): Source of influence.
        '''
        if self.constant: return

        # Gather the influence of all sources in backstage
        for source in sources:
            src = self.interact_pairwise(source)
            self.a_next[0] += src.get('ax', 0)
            self.a_next[1] += src.get('ay', 0)
            self.d_s_next += src.get('d_s', 0)
            self.dd_phi_next += src.get('dd_phi', 0)
        
    def move(self, Hz):
        '''
        Updates position, velocity, and acceleration using current velocity, acceleration, 
        d_s and dd_phi.
        
        Args:
            Hz (float): The temporal frequency of simulation.
        '''
        # Update variables based on the current values of their derivatives
        dt = 1.0 / Hz        
        self.p[0] += self.v[0] * dt 
        self.p[1] += self.v[1] * dt
        if self.constant:
            return
        self.s += self.d_s * dt
        self.phi += self.d_phi * dt
        self.v = sp2v(self.s, self.phi, ref=self.ref)
        
        # Apply backstage influence then clear them
        self.a[0], self.a[1], self.d_s, self.dd_phi = self.a_next[0], self.a_next[1], self.d_s_next, self.dd_phi_next
        self.a_next[0], self.a_next[1], self.d_s_next, self.dd_phi_next = 0, 0, 0, 0
        
        # Apply a to d_s and d_phi
        v_next = [i + j * dt for i, j in zip(self.v, self.a)]
        s_next, phi_next = v2sp(v_next, ref=self.ref)
        self.d_s += s_next - self.s
        self.d_phi += phi_next - self.phi
        
        # Apply dd_phi to d_phi
        self.d_phi += self.dd_phi * dt
        
        # Debug log
        # if self.id == 1:
            # print(self.p, self.v, self.a, self.s, self.phi, self.d_s, self.d_phi, self.dd_phi)
        
        
        
     
