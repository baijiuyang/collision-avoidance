# This module contains all the helper functions for the dissertation research of Jiuyang Bai
# on the control law of pedestrian collision avoidance, please contact baijiuyang@hotmail.com
# for support. Github: https://github.com/baijiuyang/collision-avoidance.git

import numpy as np
from numpy.linalg import norm
from numpy import sqrt, sin, cos, arccos, arcsin, arctan
import math
from matplotlib import animation, pyplot as plt
import time

def inner(x, y):
    '''
    A wraper for numpy.inner() to make it be able to compute element wise
    inner product between x and y, which can be single vectors or lists of
    vectors.
    '''
    if len(np.shape(x)) == 1:
        return np.inner(x, y)
    else:
        retval = np.zeros(len(x))
        for i in range(len(retval)):
            retval[i] = inner(x[i], y[i])
        return retval

def dist(p0, p1):
    return norm(np.array(p1) - np.array(p0), axis=-1)
    
def d_dist(p0, p1, v0, v1):
    '''
    Computes the derivative of distance as a scaler.
    '''
    if len(np.shape(p0)) == 1:
        return ((p1[0] - p0[0]) * (v1[0] - v0[0]) + (p1[1] - p0[1]) * (v1[1] - v0[1])) / dist(p0, p1)
    else:
        return ((p1[:, 0] - p0[:, 0]) * (v1[:, 0] - v0[:, 0]) + (p1[:, 1] - p0[:, 1]) * (v1[:, 1] - v0[:, 1])) / dist(p0, p1)
    
def d_speed(v, a):
    '''
    Computes the derivative of speed as a scaler.
    '''
    if len(np.shape(v)) == 1:
        return (v[0] * a[0] + v[1] * a[1]) / norm(v)
    else:
        return (v[:, 0] * a[:, 0] + v[:, 1] * a[:, 1]) / norm(v, axis=1)

def rotate(vec, angle):
    '''
    Rotate vector 'vec' clockwise 'angle' radians.
    
    Args:
        vec (2-d vector or np array of 2-d vectors): The vector to be rotated.
        angle (float or np array of floats): The angle of rotation in radian.
    
    Return:
        (2-d vector or np array of 2-d vectors): The rotated vector.
    '''
    def _rotate(vec, angle):
        # Rotation matrix
        M = [[cos(angle), -sin(angle)], [sin(angle), cos(angle)]]
        return np.matmul(vec, M)
    
    # When only one angle
    if len(np.shape(angle)) == 0:
        return _rotate(vec, angle)
    # When there are multiple angles
    else:
        vecs = np.zeros((len(angle), 2))
        # When only one vector
        if len(np.shape(vec)) == 1:
            for i in range(len(vecs)):
                vecs[i] = _rotate(vec, angle[i])
        # When there are multiple vectors
        elif len(vec) == len(angle):
            for i in range(len(vecs)):
                vecs[i] = _rotate(vec[i], angle[i])
        else:
            raise Exception('Arguments length do not match')
        return vecs
        
def sp2v(s, phi, ref=[0, 1]):
    '''
    Convert speeds to velocities given phi and reference axis of phi. 
    
    Args:
        s, phi(floats or np array of floats): Speeds, and headings.
        ref (2-d vector): The allocentric reference axis that defines phi.
            Clockwise rotation up to pi is positive, counter-clockwise rotation
            up to pi is negative.
        
    Return:
        (2-d vector or np array of 2-d vectors): Velocities.
    '''
    ref = [i / norm(ref) for i in ref]
    return rotate(ref, phi) * np.expand_dims(s, axis=-1)
    
def v2sp(v, ref=[0, 1]):
    '''
    Convert velocities to speeds and phi. 
    
    Args:
        v (2-d vector or np array of 2-d vectors): Velocities.
        ref (2-d vector): The allocentric reference axis that defines phi.
            Clockwise rotation up to pi is positive, counter-clockwise rotation
            up to pi is negative. [0, 1] or [1, 0].
        
    Return:
        s, phi(floats or np array of floats): Speeds, and headings.
    '''
    ref = [i / norm(ref) for i in ref]
    def _v2sp(v, ref):
        s = norm(v)
        if s == 0:
            phi = None
        else:
            phi = arccos(inner(v, ref) / s)
            # Phi is negative when it's on the left side of ref
            if ref[1] * v[0] - ref[0] * v[1] < 0:
                phi = -phi
        return s, phi
        
    if len(np.shape(v)) == 1:
        return _v2sp(v, ref)
    else:
        ss = np.zeros(len(v))
        phis = np.zeros(len(v))
        for i in range(len(v)):
            ss[i], phis[i] = _v2sp(v[i], ref)
        return np.stack((ss, phis), axis=-1)

def sp2a(s, d_s, phi, d_phi, ref=[0, 1]):
    '''
    Compute vectoral acceleration from d_s, phi, and d_phi.
    
    Args:
        s, phi (floats or np array of floats): Speed and heading.
        d_s, d_phi (floats or np array of floats): Rate of change of speed and phi.
        ref (2-d vector): The allocentric reference axis that defines phi.
            Clockwise rotation up to pi is positive, counter-clockwise rotation
            up to pi is negative.
        
    Return:
        (2-d vector or np array of 2-d vectors): Vectoral accelerations.
    '''
    ref = [i / norm(ref) for i in ref]
    def _sp2a(s, d_s, phi, d_phi, ref):
        v_unit = rotate(ref, phi)
        _, phi = v2sp(v_unit, ref=[0, 1]) # Change ref to [0 , 1] so that the following equations work
        dd_x = d_s * sin(phi) + s * cos(phi) * d_phi
        dd_y = d_s * cos(phi) - s * sin(phi) * d_phi
        return [dd_x, dd_y]
    if len(np.shape(s)) == 0:
        return _sp2a(s, d_s, phi, d_phi, ref)
    else:
        accs = np.zeros((len(s), 2))
        for i in range(len(accs)):
            accs[i] = _sp2a(s[i], d_s[i], phi[i], d_phi[i], ref)
        return accs

def av2dsdp(v, a):
    '''
    Computes rate of change of speed and heading based on vector acceleration.
    
    Args:
        a (2-d vector or np array of 2-d vectors): Vector acceleration.
        v (2-d vector or np array of 2-d vectors): Vector velocity.
    Returns:
        d_s (float or np array of floats): Rate of change of speed in m/s^2.
        d_phi (float or np array of floats): Rate of change of heading in rad/s).
            Positive mean clockwise heading change.
    '''
    def _av2dsdp(v, a):
        if v[0] == 0 and v[1] == 0:
            d_s = norm(a)
            d_phi = 0
        else:
            d_s = inner(a, v) / norm(v) # correctly signed
            d_phi = inner(a, rotate(v, math.pi / 2)) / norm(v) ** 2 # correctly signed
        return d_s, d_phi
    if len(np.shape(a)) == 1:
        return _av2dsdp(v, a)
    else:
        d_ss, d_phis = np.zeros(len(a)), np.zeros(len(a))
        for i in range(len(d_ss)):
            d_ss[i], d_phis[i] = _av2dsdp(v[i], a[i])
        return np.stack((d_ss, d_phis), axis=-1)
    
def theta(p0, p1, w):
    '''
    Computes the visual angle of p1 in the perspective of p0
    
    Args:
        w (float): The width of p1 in meters.
        
    Return:
        (float or np array of floats): Theta in radians.
    '''
    r = dist(p0, p1)
    return 2 * arctan(w / (2 * r))
    
def d_theta(p0, p1, v0, v1, w):
    '''
    Computes the analytical solution of optical expansion of p1 in the perspective
    of p0. This function assumes symmetrical expansion of p1. 
    
    Args:
        p0, p1 (2-d vectors or np array of float): Time series of positions in meter,
            with the shape (n_step, 2).
        v0, v1 (2-d vectors or np array of float): Time series of velocities in meter / second,
            with the shape (n_step, 2).
        w (float): The width of the leader in meters.
        
    Return:
        RE (float or np array of floats): An array of rate of optical expansion
            in radians / second.
    '''
    def _d_theta(p0, p1, v0, v1, w):
        r01 = [i - j for i, j in zip(p1, p0)] # Vector pointing to agent 1 from agent 0.
        r10 = [-i for i in r01]
        r = norm(r01) # Distance between agent 0 and agent 1.
        # The time derivative of distance is the sum of speed along the signed distance
        dr = (inner(v0, r01) + inner(v1, r10)) / r
        RE = 1 * w * dr / (r ** 2 + w ** 2 / 4)
        return RE
        
    if len(np.shape(p0)) == 1:
        return _d_theta(p0, p1, v0, v1, w)
    else:
        REs = np.zeros(len(p0))
        for i in range(len(REs)):
            REs[i] = _d_theta(p0[i], p1[i], v0[i], v1[i], w)
        return REs

def d_theta_numeric(p0, p1, w, Hz):
    thetas = theta(p0, p1, w)
    d_thatas = np.diff(thetas)
    return np.append(d_thatas, d_thatas[-1]) * Hz # Pad to the length of p0.

def beta(p0, p1, v0):
    '''
    Compute beta angle of p1 in the perspective of p0 in radians.
    Beta angle is defined as the angle between the line of sight and
    heading direction. beta = psi (bearing) - phi (heading).
    
    Args:
        p0, p1 (2-d vector or np array of 2-d vectors): Time series of positions in meter,
            with the shape (n_step, 2).
        v0 (2-d vector ot np array of floats): Time series of velocity of agent 0 
            in meter / second, with the shape (n_step, 2).
        
    Return:
        (float or np array of floats): Beta angle in radians. [-pi, pi].
        Positive value means p1 is on the left hand side of p0.
    '''
    def _beta(p0, p1, v0):
        r01 = [i - j for i, j in zip(p1, p0)]
        r = norm(r01)
        s0 = norm(v0)
        angle = arccos(inner(v0, r01) / (s0 * r))
        # Decide the side of agent 1 on agent 0. Rotate v0 90 degree clockwise 
        # v0' = (y, -x) if the sign of its dot product with r01 is negative, 
        # agent 1 is on the left side of agent 0, which means a negative beta. Vice versa.
        if v0[1] * r01[0] - v0[0] * r01[1] < 0: 
            angle = -angle
        return angle
        
    if len(np.shape(p0)) == 1:
        return _beta(p0, p1, v0)
    else:
        betas = np.zeros(len(p0))
        for i in range(len(betas)):
            betas[i] = _beta(p0[i], p1[i], v0[i])
        return betas

def d_beta(p0, p1, v0, v1, d_phi):
    '''
    Compute the analytical solution the rate of change of beta angle of p1 
    in the perspective of p0. Beta angle is defined as the angle between 
    the line of sight and heading direction. beta = psi - phi.
    
    Args:
        p0, p1 (2-d vectors or np array of floats): Time series of positions in meter,
            with the shape (n_step, 2).
        v0, v1 (2-d vectors or np array of floats): Time series of velocities 
            in meter / second, with the shape (n_step, 2).
        d_phi (float or list of floats): Rate of change of heading direction.
        
    Return:
        d_beta (float or np array of floats): An array of rate of change of beta 
            angle in radians / second.
    '''
    d_phi = np.array(d_phi)
    return d_psi(p0, p1, v0, v1) - d_phi
  
def d_beta_numeric(p0, p1, v0, Hz):
    '''
    Computes the numeric solution of rate of change of beta angle.
    Beta angle is defined as the angle between the line of sight and 
    heading direction.
    
    Args:
        p0, p1 (2-d np array of float): Time series of positions in meter,
            with the shape (n_step, 2).
        v0 (2-d np array of float): Time series of velocity of agent 0 
            in meter / second, with the shape (n_step, 2).
        
    Return:
        d_betas (np array of float): An array of rate of change of beta 
            angle in radians / second. Note that np.diff will make the length
            n_Step - 1. Therefore, last value is duplicated to pad to n_step. 
    '''
    betas = beta(p0, p1, v0)
    d_betas = np.append(np.diff(betas), np.diff(betas)[-1])
    indices = np.where(np.absolute(d_betas) > math.pi) # When beta switch sign around pi
    # Correct for the change of sign of beta angle
    for i in indices[0]:
        if d_betas[i] < 0:
            d_betas[i] += 2 * math.pi
        elif d_betas[i] > 0:
            d_betas[i] -= 2 * math.pi
    return d_betas * Hz


def psi(p0, p1, ref=[0,1]):
    '''
    Computes the angle between line of sight between p0 and p1 and ref. 
    
    Args:
        p0, p1 (2-d vector or np array of 2-d vectors): Positions of agent 0 and agent 1.
        ref (2-d vector): The allocentric reference axis. [0, 1] or [1, 0].
        
    Return:
        (float or np array of floats): Psi in (-pi, pi). Positive means clockwise
            rotation from ref up to pi.
    '''
    ref = [i / norm(ref) for i in ref]
    def _psi(p0, p1, ref):
        
        r01 = [i - j for i, j in zip(p1, p0)]
        angle = arccos(inner(r01, ref) / norm(r01))
        # Psi is negative when it's on the left side of ref
        if ref[1] * r01[0] - ref[0] * r01[1] < 0:
            angle = -angle
        return angle
    if len(np.shape(p0)) == 1:
        return _psi(p0, p1, ref)
    else:
        psis = np.zeros(len(p0))
        for i in range(len(psis)):
            psis[i] = _psi(p0[i], p1[i], ref)
        return psis
    

def d_psi(p0, p1, v0, v1):
    '''
    Computes the analytical solutions of rate of change of psi angle, which is
    the angle between line of sight and ref (an allocentric reference axis). 
    
    Args:
        p0, p1 (2-d vector or np array of 2-d vectors): Positions of agent 0 and 1.
        v0, v1 (2-d vector or np array of 2-d vectors): Velocities of agent 0 and 1.
    
    Return:
        (float or np array of floats): Rate of change of psi.
    '''
    def _d_psi(p0, p1, v0, v1):
        r01 = [i - j for i, j in zip(p1, p0)]
        r = norm(r01)
        r01_i = rotate(r01, math.pi / 2) / r
        return (inner(v1, r01_i) - inner(v0, r01_i)) / r
    if len(np.shape(p0)) == 1:
        return _d_psi(p0, p1, v0, v1)
    else:
        d_psis = np.zeros(len(p0))
        for i in range(len(d_psis)):
            d_psis[i] = _d_psi(p0[i], p1[i], v0[i], v1[i])
        return d_psis
        
def d_psi_numeric(p0, p1, Hz, ref=[0, 1]):
    '''
    Computes the numeric solution of rate of change of psi angle, the accuracy 
    of which depends on Hz.
    
    Args:
        p0, p1 (2-d vector or np array of 2-d vectors): Positions of agent 0 and 1.
        Hz (int): Frequency of simulation.
        ref (2-d vector): The allocentric reference axis.
    
    Return:
        (np array of floats): Rate of change of psi.
    '''
    psis = psi(p0, p1, ref)
    d_psis = np.diff(psis)
    d_psis = np.append(d_psis, d_psis[-1]) # pad to make d_psis the length as p0
    i = np.where(np.absolute(d_psis) > math.pi) # When psi switch sign around pi
    if d_psis[i] > 0:
        d_psis[i] -= 2 * math.pi
    else:
        d_psis[i] += 2 * math.pi
    return d_psis * Hz
    
def phi(v, ref=[0, 1]):
    '''
    Computes the heading direction in terms of a reference axis.
    
    Args:
        v (2-d vector or np array of 2-d vectors): Velocity.
        ref (2-d vector): The allocentric reference axis.
    Return:
        (float or np array of floats): Heading direction in range (-pi, pi).
            Positive heading is clockwise rotation from ref.
    '''
    if len(np.shape(v)) == 1:
        return v2sp(v, ref)[1]
    else:
        return v2sp(v, ref)[:, 1]
    
def simple_gaussian(x, a=2, b=0.1):
    '''
    Probability Density Function (pdf) of Gussian distribution.
    
    Args:
        x (np array of float): x in pdf.
        a, b (float): Parameters of the pdf.
        
    Return:
        (np array of float): y in pdf
    '''
    return np.exp(-x ** 2 / a) / b
    
    
def broken_sigmoid(x, a=1, b=20):
    '''
    A distorted sigmoid function. Equivalent to sigmoid function
    when x < 0. It is symmetrical about the y-axis.
    
    Args:
        x (np array of float): Input of the function.
        a, b (float): Parameters of the function.
        
    Return:
        (np array of float): Output of the function
    '''
    return b / (1 + np.exp(np.absolute(x / a)))

def min_sep(p0, p1, v0, v1):
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
    dca = np.sqrt(1 - (inner(p01, v10) / (norm(p01) * norm(v10))) ** 2) * norm(p01)
    ttca = inner(p01, v10) / norm(v10) ** 2 # np.inner takes care of the sign
    if inner(np.array(v1), np.array(p0) - np.array(p1) + v10 * ttca) < 0: # Check passing order
        dca *= -1
    return dca, ttca
    
    
    
def collision_trajectory(beta, side, spd1=1.3, w=1.5, r=10, r_min=0, Hz=100, animate=False, interval=None, save=False):
    '''
    This function produces near-collision trajectories of circular agents 0 and 1 with 
    a defined beta angle. The speed of agent 1 is constant. The speed of agent 0 will 
    vary to guarantee near-collision. 
    
    Args:
        beta ((-90,90), float): The initial beta angle between A and B in degree. beta 
            angle is defined as the angle between the line of sight and heading direction. obstacles
            on the left have positive betas, obstacles on the right have negative betas.
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
    alpha = (180 + 2 * beta) * 2 * math.pi / 360 # Central angle of line of sight
    p1 = np.array([-r * sin(alpha), -r * cos(alpha)])
    p0 = np.array([0, -r])
    p01 = p1 - p0
    v1 = np.array([spd1 * sin(alpha), spd1 * cos(alpha)])
    a = r ** 2 * sin(alpha) ** 2 - r_min ** 2
    b = -2 * spd1 * (cos(alpha) * (norm(p01) ** 2 - r_min ** 2) + r ** 2 * (1 - cos(alpha)) ** 2)
    c = spd1 ** 2 * (norm(p01) ** 2 - r_min ** 2 - r ** 2 * (1 - cos(alpha)) ** 2)
    d = (b ** 2) - (4 * a * c)

    if r_min == 0:
        v0 = np.array([0, spd1])
    else:
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
        trajs = np.stack((traj0, traj1))
        play_trajs(trajs, [w, w], Hz, colors='br', interval=interval, save=save)
    
    return traj0, traj1, np.tile(v0, (n, 1)), np.tile(v1, (n, 1))
    
def play_trajs(trajs, ws, Hz, colors=None, interval=None, save=False):
    '''
    trajs (2-d np array)
    '''
    if not interval: interval = 1000 / Hz
    if not colors: colors = [None] * len(trajs)
    # Create a figure
    fig = plt.figure(figsize=(7, 7))
    ax = plt.axes(xlim=(-12, 12), ylim=(-12, 12))
    ax.set_aspect('equal')
    
    # Initialize plots
    angles = np.linspace(0, 2 * math.pi, num=12)
    circles = []
    agents = []

    for traj, w, color in zip(trajs, ws, colors):
        circle = np.stack((w / 2 * cos(angles), w / 2 * sin(angles)), axis=-1)
        agent, = ax.plot(traj[0, 0] + circle[:, 0], traj[0, 1] + circle[:, 1], color=color)
        ax.plot([p[0] for p in traj], [p[1] for p in traj], color=agent.get_color()) # Plot trajectory
        agents.append(agent)
        circles.append(circle)

    def animate(i):
        # ms is the short for markersize
        for agent, traj, circle in zip(agents, trajs, circles):
            agent.set_data(traj[i][0] + circle[:, 0], traj[i][1] + circle[:, 1])
        return agents

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, frames=len(trajs[0]), interval=interval, blit=True)
    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
        # installed.  The extra_args ensure that the x264 codec is used, so that
        # the video can be embedded in html5.  You may need to adjust this for
        # your system: for more information, see
        # http://matplotlib.sourceforge.net/api/animation_api.html
    if save:
        t = time.localtime()
        t = [str(i) for i in t[:6]]
        t = "-".join(t)
        filename = 'play_trial' + t + '.mp4'
        anim.save(filename)
    return anim
    
    
    