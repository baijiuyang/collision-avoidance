import os
import sys
import math
import numpy as np
from scipy.optimize import minimize
import pickle
from packages.tester import Tester
from packages.simulator import Agent
from packages.helper import beta
from packages import data_container
sys.modules['data_container'] = data_container


mass_spring_approach = {'name': 'mass_spring_approach', 'ps': 1.3,
                        't_relax': 0.7,
                        'b': 3.25, 'k_g': 7.5, 'c1': 0.4, 'c2': 0.4}
mass_spring_approach1 = {'name': 'mass_spring_approach1', 'ps': 1.3,
                         'b1': 3.25, 'k1':7.5, 'c1': 0.4, 'c2': 0.4,
                         'b2': 4.8, 'k2': 6}
cohen_avoid = {'name': 'cohen_avoid', 'ps': 1.3,
               'b1': 4, 'k1': 400, 'c5': 4, 'c6': 4,
               'b2': 1, 'k2': 300, 'c7': 6, 'c8': 2}
cohen_avoid2 = {'name': 'cohen_avoid2', 'ps': 1.3,
                'b1': 4, 'k1': 400, 'c5': 4, 'c6': 4,
                'b2': 1, 'k2': 300, 'c7': 6, 'c8': 2}
cohen_avoid3 = {'name': 'cohen_avoid3',
                'k1': 530, 'c5': 6, 'c6': 1.3,
                'k2': 50, 'c7': 6, 'c8': 1.3}
cohen_avoid4 = {'name': 'cohen_avoid4',
                'k1': 530, 'c5': 6, 'c6': 1.3,
                'k2': 50, 'c7': 6, 'c8': 1.3}

i_iter = 0
                
def Cohen_movObst_exp1():    
    info = {'w': {'goals': [0.1], 'obsts': [0.1]}, 'ps': []}
    goals = []
    obsts = []
    subjs = []
    file = os.path.abspath(os.path.join(os.getcwd(),
                                        os.pardir,
                                        'data',
                                        'Cohen_movObst_exp1_data.pickle'))
    
    with open(file, 'rb') as f:
        data = pickle.load(f)
        Hz = data.Hz
        for i in range(len(data.trajs)):
            if i in data.dump:
                continue
            
            goal = data.info['p_goal'][i]
            obst = data.info['p_obst'][i]
            subj = np.array(data.get_traj(i))[:,0:2].tolist()
            # Define the start and end of a trial for fitting
            t0 = data.info['stimuli_onset'][i]
            for j in range(t0, len(subj)):
                p0 = subj[j]
                p1 = obst[j]
                v0 = [(a - b) * data.Hz for a, b in zip(subj[j+1], subj[j])]
                if abs(beta(p0, p1, v0)) > math.pi / 2:
                    t1 = j
                    break                
            goals.append([goal[t0:t1]])
            obsts.append([obst[t0:t1]])
            subjs.append(subj[t0:t1])
            info['ps'].append(data.info['ps'][i])
    agent = Agent(1, goal_id=0, w=0.4)
    return Tester(agent, goals, obsts, subjs, info, Hz)
    
def error(x, tester):
    global i_iter
    i_iter += 1
    print(f'iteration {i_iter:03d}, x = ' + [f'{a:0.3f}' for a in x])
    approach = mass_spring_approach
    avoid = {'name': 'cohen_avoid',
             'b1': x[0], 'k1': x[1], 'c5': x[2], 'c6': x[3],
             'b2': x[4], 'k2': x[5], 'c7': x[6], 'c8': x[7]}
    tester.set_model(approach)
    tester.set_model(avoid)
    tester.simulate()
    err = np.mean(tester.test('p_dist'))
    print(f'error is {err:.6f}')
    return err
    
def main():
    tester = Cohen_movObst_exp1()
    x0 = np.array([4, 400, 4, 4, 1, 300, 6, 2])
    res = minimize(error, x0, args=(tester), method='nelder-mead',
                    options={'xatol': 1e-8, 'disp': True})
    print(res.x)

if __name__ == "__main__":
    main()
    