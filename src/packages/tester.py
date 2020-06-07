import numpy as np
from numpy import gradient
from numpy.linalg import norm
from packages.simulator import Agent 
from packages.helper import dist, v2sp, beta, min_dist

class Tester:
    '''
    The class of objects that can test a given model on a given dataset.
    The test results can be shown in different variables using different
    metrics, averaged or grouped by subject or condition.  
    
    Fields:
        agent (Agent object): Contains states and models.
            See definition in Agent class.
        goals (list of lists): Elements are trials [goal0,goal1,...],
            each goal is a trajectory [(x0, y0),(x1, y1),...].
        obsts (list of lists): Elements are trials [obst0,obst1,...],
            each obstacle is a trajectory [(x0, y0),(x1, y1),...].
        subjs (list of lists): Elements are subject trajectories, [(x0, y0),...].
        info (dict): {'w': {'goals','obsts'}, 'ps': [trial0,trial1,...], ...}.
        Hz (int): Temporal frequency of the simulation.
        t (list of lists): Time stamps in seconds.
        p_pred (list of lists): Elements are predicted trajectories, [(x0, y0),...].
    '''
    def __init__(self, agent, goals, obsts, subjs, info, Hz):
        '''Constructor'''
        self.agent = agent
        self.goals = goals
        self.obsts = obsts
        self.subjs = subjs
        self.info = info
        self.Hz = Hz
        self.reset()
        
    def reset(self):
        self.p_pred = []
        self.t = []
        
    def set_model(self, model):
        if 'avoid' in model['name']:
            self.agent.set_avoid_model(model)
        elif 'approach' in model['name']:
            self.agent.set_approach_model(model)        
    
    def simulate_trial(self, init_state, goals, obsts):
        length = len(goals[0])
        # Initialize agent
        self.agent.set_state(init_state)
        self.agent.goal_id = 0
        t = [0]
        p_pred = [tuple(self.agent.p[:])]
        
        # Initialize sources
        goal_scrs = []
        obst_scrs = []
        for i, goal in enumerate(goals):
            source = Agent(0, constant=True)            
            source.w = self.info['w']['goals'][i]
            source.p = goal[0][:]
            source.v = ((goal[1][0] - goal[0][0]) * self.Hz, (goal[1][1] - goal[0][1]) * self.Hz)
            goal_scrs.append(source)
        for i, obst in enumerate(obsts):
            source = Agent(-1, constant=True)
            source.w = self.info['w']['obsts'][i]
            source.p = obst[0][:]
            source.v = ((obst[1][0] - obst[0][0]) * self.Hz, (obst[1][1] - obst[0][1]) * self.Hz)
            obst_scrs.append(source)
        # Simulate frame by frame
        for i in range(length - 1):
            for j, goal in enumerate(goals):
                goal_scrs[j].p = goal[i][:]
                goal_scrs[j].v = ((goal[i+1][0] - goal[i][0]) * self.Hz, (goal[i+1][1] - goal[i][1]) * self.Hz)
            for j, obst in enumerate(obsts):
                obst_scrs[j].p = obst[i][:]
                obst_scrs[j].v = ((obst[i+1][0] - obst[i][0]) * self.Hz, (obst[i+1][1] - obst[i][1]) * self.Hz)
            self.agent.interact(goal_scrs)
            self.agent.interact(obst_scrs)
            self.agent.move(self.Hz)           
            t.append(t[-1] + 1.0 / self.Hz)
            p_pred.append((self.agent.p[0], self.agent.p[1]))
        
        self.t.append(t)
        self.p_pred.append(p_pred)
        
    def simulate(self):
        self.reset()
        for i in range(len(self.subjs)):
            # Get preferred speed from data
            if 'ps' in self.info:
                self.agent.approach_model.args['ps'] = self.info['ps'][i]
            # Get initial state from data
            v0 = [(a - b) * self.Hz for a, b in zip(self.subjs[i][1], self.subjs[i][0])]
            v1 = [(a - b) * self.Hz for a, b in zip(self.subjs[i][2], self.subjs[i][1])]
            s0, phi0 = v2sp(v0, ref=self.agent.ref)
            s1, phi1 = v2sp(v1, ref=self.agent.ref)
            init_state = {'p': self.subjs[i][0][:], 
                          'v': v0,
                          'd_s': (s1 - s0) * self.Hz,
                          'd_phi': (phi1 - phi0) * self.Hz}
            self.simulate_trial(init_state,
                                self.goals[i] if self.goals else [],
                                self.obsts[i] if self.obsts else [])
    
    def dca(self):
        '''
        Compute signed distance of the closest approach.
        Postive means passing from the left.        
        '''
        subjs, preds = [], []
        # Iterate through trials
        for i in range(len(self.subjs)):
            pred, dat = [], []
            # Iterate through obstacles
            for obst in self.obsts[i]:                
                pred.append(min_dist(self.p_pred[i], obst)[1])         
                dat.append(min_dist(self.subjs[i], obst)[1])
            preds.append(pred)
            subjs.append(dat)
        return subjs, preds
    
    def test(self, metric, i_trial=None):
        if not self.p_pred: self.simulate()
        Hz = self.Hz
        var = metric.split('_')[0]
        alg = metric.split('_')[1]
        preds = []
        subjs = []
        # Compute variable
        for i in range(len(self.subjs)):
            if i_trial != None and i != i_trial:
                continue
            if var == 'p':
                preds.append(self.p_pred[i])
                subjs.append(self.subjs[i])
            elif var == 'v':
                preds.append(gradient(self.p_pred[i], axis=0) * Hz)
                subjs.append(gradient(self.subjs[i], axis=0) * Hz)
            elif var == 'a':
                preds.append(gradient(gradient(self.p_pred[i], axis=0) * Hz, axis=0) * Hz)
                subjs.append(gradient(gradient(self.subjs[i], axis=0) * Hz, axis=0) * Hz)
            elif var == 's':
                preds.append(norm(gradient(self.p_pred[i], axis=0) * Hz, axis=-1))
                subjs.append(norm(gradient(self.subjs[i], axis=0) * Hz, axis=-1))
            elif var == 'phi':
                ref = self.agent.ref
                preds.append(v2sp(gradient(self.p_pred[i], axis=0) * Hz, ref=ref)[1])
                subjs.append(v2sp(gradient(self.subjs[i], axis=0) * Hz, ref=ref)[1])
            elif var == 'dca':
                subjs, preds = self.dca()
                break
                
                           
        # Compute metric
        vals = []
        for i in range(len(preds)):
            if alg == 'dist':
                vals.append(np.mean(dist(preds[i], subjs[i])))
            elif alg == 'MAE':
                vals.append(np.mean(np.absolute(np.array(preds[i]) - np.array(subjs[i]))))
            elif alg == 'MSE':
                vals.append(np.mean((np.array(preds[i]) - np.array(subjs[i]))**2))
            elif alg == 'RMSE':
                vals.append(np.sqrt(np.mean((np.array(preds[i]) - np.array(subjs[i]))**2)))
        return vals
    
    