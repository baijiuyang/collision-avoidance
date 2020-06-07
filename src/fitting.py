import os
import sys
import time
import numpy as np
from numpy.linalg import norm
from itertools import product
import multiprocessing
import pickle
from packages.tester import Tester
from packages.simulator import Agent
from packages.helper import hms, ymdhms, beta
from packages import data_container
sys.modules['data_container'] = data_container
import math

def process_func(job):
    global tester, _total_jobs, _outfile, X
    tic = time.perf_counter()
    print(os.getpid(), ' starting ', job)
    i_val = 0
    # Apply parameters
    for i, name in enumerate(job[0]):
        model = {'name': name}
        for arg in job[1][i]:
            model[arg] = job[2][i_val]
            i_val += 1
        tester.set_model(model)
    tester.simulate()
    # print(os.getpid(), ' done with simulation ', job)
    result = []
    for metric in job[3]:
        # print(os.getpid(), ' start ', metric)
        result.append(np.mean(tester.test(metric)))
        # print(os.getpid(), ' done with ', metric)
    
    with open(_outfile, 'a') as file:
        model = '/'.join(job[0])
        args = '/'.join([arg for mdl in job[1] for arg in mdl])
        vals = '/'.join([str(val) for val in job[2]])
        metrics = '/'.join(job[3])
        result = '/'.join(["%.4f" % res for res in result])
        output = (model, args, vals, metrics, result)
        line = ','.join(output)
        file.write(line + '\n')
        
    toc = time.perf_counter()
    print(f'{os.getpid():d} spent {hms(toc - tic):s} on ', job)
    progress = sum(1 for i in open(_outfile, 'rb')) * 100.0 / _total_jobs
    if progress > 20000:
        _outfile = _outfile[:-4] + X + '.csv'
        X += 'X'
    print(f'======%{progress:0.1f} done======')
        
def process_init(goals, obsts, subjs, info, Hz, total_jobs, outfile):
    global tester, _total_jobs, _outfile, X
    X = 'X'
    _total_jobs = total_jobs
    _outfile = outfile
    agent = Agent(1, goal_id=0, w=0.4)
    tester = Tester(agent, goals, obsts, subjs, info, Hz)

def Cohen_movObst_exp1():
    print('Loading input')  
    # Create job list
    model_names = ['mass_spring_approach1', 'cohen_avoid3']
    arg_names = [('b_1', 'b_2', 'k_1', 'k_2', 'c_1', 'c_2'),
                 ('k_1', 'k_2', 'c_5', 'c_6', 'c_7', 'c_8')]
    arg_vals = list(product([3.25],
                            [4.8],
                            [7.5],
                            [6],
                            [0.4],
                            [0.4],
                            np.linspace(0, 400, 5).round(2),
                            np.linspace(0, 400, 5).round(2),
                            np.linspace(0, 10, 5).round(2), 
                            np.linspace(0, 10, 5).round(2),
                            np.linspace(0, 10, 5).round(2),
                            np.linspace(0, 10, 5).round(2)))
    metrics = ['phi_RMSE', 'phi_MAE', 'phi_MSE']
    job_list = []
    for vals in arg_vals:
        job_list.append((model_names, arg_names, vals, metrics))
    
    # initargs for processes    
    data_name = 'Cohen_movObst_exp1'  
    info = {'w': {'goals': [0.1], 'obsts': [0.1]}, 'ps': []}
    goals = []
    obsts = []
    subjs = []
    file = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'data\\Cohen_movObst_exp1_data.pickle'))
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
    return job_list, goals, obsts, subjs, info, Hz, data_name
    
def Fajen_steer_exp1a():
    # Create job list
    model_names = ['mass_spring_approach1']
    arg_names = [('b_1', 'b_2', 'k_1', 'k_2', 'c_1', 'c_2')]
    arg_vals = list(product([3.25],
                            np.linspace(6, 7, 5).round(2),
                            [7.5],
                            np.linspace(7, 9, 10).round(2),
                            [0.4],
                            [0.4]))
    metrics = ['s_RMSE', 's_MAE', 's_MSE']
    job_list = []
    for vals in arg_vals:
        job_list.append((model_names, arg_names, vals, metrics))

    # initargs for processes    
    data_name = 'Fajen_steer_exp1a'  
    info = {'w': {'goals': [0.1]}, 'ps':[]}
    goals = []
    obsts = []
    subjs = []
    file = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'data\\Fajen_steer_exp1a_data.pickle'))
    with open(file, 'rb') as f:
        data = pickle.load(f)
        Hz = data.Hz
        goals = []
        subjs = []
        obsts = []
        Hz = data.Hz
        for i in range(len(data.trajs)):
            if i in data.dump:
                continue

            goal = data.info['p_goal'][i]
            subj = np.array(data.get_traj(i))[:,0:2].tolist()
            # Define the start and end of a trial for fitting
            for j in range(1, len(subj)):
                spd1 = norm([(a - b) * Hz for a, b in zip(subj[j], subj[j-1])])
                spd2 = norm([(a - b) * Hz for a, b in zip(subj[j+1], subj[j])])
                if spd1 > 0.22 and spd2 > spd1:
                    t0 = j
                    break
            t1 = t0 + 3 * Hz
            goals.append([goal[t0:t1]])
            subjs.append(subj[t0:t1])
            info['ps'].append(data.info['ps'][i])

    return job_list, goals, obsts, subjs, info, Hz, data_name

if __name__ == "__main__":
    #========control panel========#
    fitting_func = Cohen_movObst_exp1
    #=============================#

    tic = time.perf_counter()    
    # Initialize inputs
    n_worker = os.cpu_count()
    job_list, goals, obsts, subjs, info, Hz, data_name = fitting_func()
    t = ymdhms()
    outfile = 'fitting_results_' + data_name + '_' + t + '.csv'
    total_jobs = len(job_list)
    initargs = (goals, obsts, subjs, info, Hz, total_jobs, outfile)    
    # Run jobs
    print('Start testing')
    with multiprocessing.Pool(processes=n_worker, initializer=process_init, initargs=initargs) as pool:
        pool.map(process_func, job_list)
    toc = time.perf_counter()
    print(f'A total time of {hms(toc - tic):s} was spent on {total_jobs:d} iterations ({hms((toc - tic)/total_jobs):s}/iter)')