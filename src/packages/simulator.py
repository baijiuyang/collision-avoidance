# This module contains the simulation and agent classes for the dissertation research of Jiuyang Bai
# on the control law of pedestrian collision avoidance, please contact baijiuyang@hotmail.com
# for support. Git: https://github.com/baijiuyang/collision-avoidance.git

import numpy as np
from numpy.linalg import norm
from packages.helper import sp2v, v2sp, sp2a, av2dsdp
from numpy import sqrt, sin, cos
from packages.models import Model
import time
from matplotlib import gridspec, animation, pyplot as plt
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
        fig0 = plt.figure(num=0, figsize=(15,7))
        spec = gridspec.GridSpec(ncols=3, nrows=1,
                         width_ratios=[1, 1, 2])
        # Create speed plot
        ax0 = fig0.add_subplot(spec[0])
        ax0.set_ylim(0, 1.75)
        ax0.set_xlabel('time (s)')
        ax0.set_ylabel('speed (m/s)')
        ax0.set_title('Speed')
        for id, s in self.get_s().items():
            ax0.plot(self.t, s, label=str(id))
        ax0.legend()
        time_bar0, = ax0.plot([self.t[0], self.t[0]], ax0.get_ylim(), color=(0.7, 0.7, 0.7))       
        # Create heading plot
        ax1 = fig0.add_subplot(spec[1])
        ax1.set_ylim(-3.2, 3.2)
        ax1.set_xlabel('time (s)')
        ax1.set_ylabel('heading (radian)')
        ax1.set_title('Heading')
        for id, phi in self.get_phi().items():
            ax1.plot(self.t, phi, label=str(id))
        ax1.legend()
        time_bar1, = ax1.plot([self.t[0], self.t[0]], ax1.get_ylim(), color=(0.7, 0.7, 0.7))       
        # Create path plot
        ax2 = fig0.add_subplot(spec[2])
        ax2.set_xlim(-12, 12)
        ax2.set_ylim(-12, 12)
        ax2.set_xlabel('postion x (m)')
        ax2.set_ylabel('postion y (m)')
        ax2.set_title('Path')
        ax2.set_aspect('equal') # Use the true ratio of x and y axis
        agents, circles, ws, ids = [], [], [], []
        angles = np.linspace(0, 2 * math.pi, num=12) # Use 12 points to approximate a circle
        for i, traj in self.p.items():
            w = self.agents[i].w if self.agents[i].w else 0.1
            circle = np.stack((w / 2 * cos(angles), w / 2 * sin(angles)), axis=-1) # 12 points on the perimeter
            agent, = ax2.plot(traj[0][0] + circle[:, 0], traj[0][1] + circle[:, 1])
            ax2.plot([p[0] for p in traj], [p[1] for p in traj], color=agent.get_color()) # Plot trajectory
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
            time_bar0.set_data([self.t[i], self.t[i]], ax0.get_ylim())
            time_bar1.set_data([self.t[i], self.t[i]], ax1.get_ylim())
            lines = agents[:]
            lines.append(time_bar0)
            lines.append(time_bar1)
            return lines
        
        # call the animator.  blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(fig0, animate, frames=len(self.t), interval=interval, blit=True)

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
        return fig0, anim
        # For command line usage
        # plt.show()
    
    def get_v(self):
        vs = {}
        for id, p in self.p.items():
            vs[id] = np.gradient(p, axis=0) * self.Hz
        return vs
        
    def get_s(self):
        ss = {}
        vs = self.get_v()
        for id, v in vs.items():
            ss[id] = norm(v, axis=-1)
        return ss
    
    def get_phi(self):
        phis = {}
        vs = self.get_v()
        for id, v in vs.items():
            phis[id] = v2sp(v, ref=self.agents[id].ref)[1]
        return phis
        
    def plot_speeds(self, hide=False):
        speeds = self.get_s()
        fig = plt.figure(num=1, figsize=(7,7))
        ax = fig.add_subplot(111)
        ax.set_ylim(0.25, 1.75)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('speed (m/s)')
        for id, s in speeds.items():
            ax.plot(self.t, s, label=str(id))
        ax.legend()
        return fig
        
    
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
        approach_model, avoid_model (Model objects): See Model class in models.py.
        ref (2-d vector): The allocentric reference axis.
        constant (bool): Whether this agent has constant state.
    '''
    def __init__(self, id, init_state=None, goal_id=None, w=None, p_spd=None, models=None, constant=False, ref=[0, 1]):
        '''
        This constructor initialize an agent (1) through vectoral velocity and acceleration or
        (2) through speed, phi (orientation given ref as the allocentric reference axis), d_phi
        
        Args:
            models (dict): {'approach': [model dict], 'avoid': [model dict]}. Model dict has the format of 
                {'name': '[model name]', 'type': ['approach' or 'avoid'], '[parameter1]': [para1], ...}.
            init_state (dict): {'p': p0, 'v': v0, 's': s0, 'phi': phi0}
        '''
        self.id = id
        self.goal_id = goal_id
        self.w = w
        self.ref = [i / norm(ref) for i in ref]
        self.p_spd = p_spd
        self.constant = constant
        if init_state:
            self.set_state(init_state)
        self.a = [0, 0]
        self.d_s = 0
        self.d_phi = 0
        self.dd_phi = 0
        self.a_next = [0, 0]
        self.d_s_next = 0
        self.dd_phi_next = 0
        if not self.constant:
            self.approach_model = Model(models['approach'], self.ref) if models['approach'] else None
            self.avoid_model = Model(models['avoid'], self.ref) if models['avoid'] else None
        
    def set_state(self, init_state):
        self.p = init_state['p']
        if 'v' in init_state:
            self.v = init_state['v']
            self.s, self.phi = v2sp(self.v, ref=self.ref)
        else:
            self.s = init_state['s']
            self.phi = init_state['phi']
            self.v = sp2v(self.s, self.phi, ref=self.ref)
            
    def interact_pairwise(self, source):
        '''
        Computes vector acceleration caused by one source based on the models.
        
        Args:
            source (Object from the Agent class): See definition in the Agent class.
    
        Return:
            pred_sum (dict): The sum of model predictions in the form of
                {'ax': ax, 'ay': ay, 'd_s': d_s, 'dd_phi': dd_phi}.
        '''
        inputs = {}
        inputs['p0'] = self.p
        inputs['v0'] = self.v        
        inputs['p1'] = source.p
        inputs['v1'] = source.v
        inputs['s0'] = self.s
        inputs['phi'] = self.phi
        inputs['d_phi'] = self.d_phi
        inputs['w'] = source.w
        pred_sum = {}
        preds = []
        # Do not interact with self or if self is constant
        if source.id == self.id or self.constant: 
            return None
        
        source_type = ''
        if source.id == self.goal_id:
            source_type += 'g' # goal
            preds.append(self.approach_model(inputs))
        if source.w:
            source_type += 'o' # obstacle
            preds.append(self.avoid_model(inputs))
            
        # Combine the influences of all models
        for pred in preds:
            for key, val in pred.items():
                pred_sum[key] = pred_sum.get(key, 0) + val if val != None else None
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
            if not src: continue
            if src['d_s'] == None:
                self.a_next[0] += src['ax']
                self.a_next[1] += src['ay']
                self.d_s_next = None
                self.dd_phi_next = None
            else:
                self.d_s_next += src['d_s']
                self.dd_phi_next += src['dd_phi']
                self.a_next = None
                
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
            
        # Update by d_s, dd_phi
        if self.d_s_next != None:
            self.s += self.d_s * dt
            self.phi += self.d_phi * dt
            self.v = sp2v(self.s, self.phi, ref=self.ref)        
            # Apply backstage influence then clear them
            self.d_s, self.dd_phi = self.d_s_next, self.dd_phi_next
            self.d_s_next, self.dd_phi_next = 0, 0
            # Apply dd_phi to d_phi
            self.d_phi += self.dd_phi * dt
        # Update by a
        else:
            self.v[0] += self.a[0] * dt
            self.v[1] += self.a[1] * dt
            # Apply backstage influence then clear them
            self.a = self.a_next
            self.a_next[0], self.a_next[1] = 0, 0
            # Covert to s, d_s, phi, d_phi
            self.s, self.phi = v2sp(self.v, ref=self.ref)
            self.d_s, self.d_phi = av2dsdp(self.v, self.a)
        
        
        
        # Debug log
        # if self.id == 1:
            # states = [self.phi, self.d_phi, self.dd_phi]
            # print(np.round(states, decimals=2))
        
        
        
     
