# -*- coding: utf-8 -*-

###############################################################################

"""
Description: 
    - An implementation of the Hyperband hyperparameter tuning algorithm (described in https://arxiv.org/pdf/1603.06560),
    - Implemented such that each trial can be run in parallel to all other trials as a separate batch job
    - Useful for avoiding memory leaks from one trial round to the next

Inputs:
    - experiment_id: The name of the set of hyperparameter trials being tested
    - out_path: The path to the parent directory to which all trial outputs are saved
    - in_path: The path from which the tuning_schedule is loaded
    - max_epochs: [Hyperband parameter] corresponds to R in the Hyperband paper
    - factor: [Hyperband parameter] corresponds to eta in the Hyperband paper
    - bracket: [Hyperband parameter] corresponds to s in the Hyperband paper
        
Outputs:
    - This script runs the next available round for the next available trail as indicated 
    in the tuning_schedule file specified by the script inputs (experiment_id,max_epochs,factor,bracket)
    and updates the tuning_schedule with the final metric
    - This script also manages the survival logic of the trials as they progress through the rounds 
    (i.e. with the top 1/factor trials progressing to the next round)

Notes:
    - The evaluation metric must be nonnegative and larger is better
    - The run_trial function defined as a shell below needs to be updated to the specifications of the given experiment
    - This script can be run using e.g. Script_Parallel_Hyperband_Tuning
"""

###############################################################################

import os
import time
import argparse
import functools

import numpy as np
import pandas as pd

from lock import lock_wrapper

###############################################################################

parser = argparse.ArgumentParser()

parser.add_argument('--experiment_id', default='Exp1', type=str, metavar='N',help='The name of the set of hyperparameter trials being tested')
parser.add_argument('--out_path', default='/', type=str, metavar='N',help='The path to the parent directory to which all trial outputs are saved')
parser.add_argument('--max_epochs', default=800, type=int, metavar='N',help='[Hyperband parameter] corresponds to R in the Hyperband paper')
parser.add_argument('--factor', default=3, type=int, metavar='N',help='[Hyperband parameter] corresponds to eta in the Hyperband paper')
parser.add_argument('--in_path', default='/', type=str, metavar='N',help='The path from which the tuning_schedule is loaded')
parser.add_argument('--bracket', default=0, type=int, metavar='N',help='[Hyperband parameter] corresponds to s in the Hyperband paper')

args = parser.parse_args()
print(f'args: {args}')
experiment_id = args.experiment_id
out_path = args.out_path
in_path = args.in_path
eta = args.factor
R = args.max_epochs
s = args.bracket

#ASSUMPTION: Metric is higher is better and is non negative
###############################################################################
def get_and_set_first_available_trial(in_path,eta,R,s):    
    tuning_schedule = pd.read_csv(in_path)
    
    trial_index = -1
    
    s_max = np.floor(np.emath.logn(n=eta,x=R)).astype(int)
    B = (s_max + 1)*R
    n = np.ceil((B/R)*((eta**s)/(s+1)))
    r = R*eta**-s
    
    rounds = s+1
    
    for rd in range(rounds):
        status = tuning_schedule[f'round_{rd}_status'].copy() #io: awaiting survival decision, i: in, o: out, p: in progress, f: finished
        trials_undecided = status=='io'
        if sum(trials_undecided)>0:
            print(f'Initializing round {rd}')
            if rd == 0:
                status = pd.Series(['i']*(len(status)))
            else:
                prev_result = tuning_schedule[f'round_{rd-1}_result']
                prev_status = tuning_schedule[f'round_{rd-1}_status']
                n_promote = np.ceil(sum(prev_status=='f')/eta).astype(int)
                top_k_positions = np.argsort(prev_result,kind='stable')[-n_promote:][::-1].tolist()
                
                status = np.full(len(status),'o',dtype=object)
                status[top_k_positions] = 'i'
                status = pd.Series(status)
                print(f'Promoted {n_promote} trials from round {rd-1} to round {rd}')
                        
        trails_available = status=='i'
        if sum(trails_available):
            trial_index = (status=='i').idxmax()
            status[trial_index] = 'p'
            print(f'Beginning trial: {trial_index} (round: {rd})')
            break
        else:
            trails_in_progress = status=='p'
            if sum(trails_in_progress) and rd!=rounds-1:
                print(f'Waiting for {sum(trails_in_progress)} round {rd} trials to complete')
                return {},tuning_schedule,2
            
    if trial_index == -1:
        print(f'Tuning search completed for bracket, all {tuning_schedule.shape[0]} trials completed for all {rd+1} rounds') 
        return {},tuning_schedule,1
    
    tuning_schedule[f'round_{rd}_status'] = status
    
    params = [c for c in tuning_schedule.columns if not 'round' in c]
    trial = tuning_schedule.loc[:,params].iloc[trial_index]
    params_dict = trial.to_dict()
    
    if rd == 0:
        marginal_epochs = np.round(r*eta**rd).astype(int)
    else:
        marginal_epochs = (np.round(r*eta**rd) - np.round(r*eta**(rd-1))).astype(int)
    
    d = {'trial_index':trial_index,'round':rd,'marginal_epochs':marginal_epochs}
    d.update(params_dict)
    print('trial params:',d)
    
    return d,tuning_schedule,0
    
def get_and_set_trial(in_path,eta,R,s):
    trial,tuning_schedule,status = get_and_set_first_available_trial(in_path,eta,R,s)
    tuning_schedule.to_csv(in_path,index=False)
    return trial,status

def set_trial_finished(in_path,trial,metric):
    tuning_schedule = pd.read_csv(in_path)
    rd = trial.get('round')
    trial_val = trial.get('trial')
    tuning_schedule.loc[tuning_schedule['trial']==trial_val,f'round_{rd}_status'] = 'f'
    tuning_schedule.loc[tuning_schedule['trial']==trial_val,f'round_{rd}_result'] = metric
    tuning_schedule.to_csv(in_path,index=False)
    
def run_trial(trial):
    if trial.get('round') == 0:
        pass #TODO: initialize new model
    else:
        pass #TODO: load model from previous round
    
    #TODO: train model
    #TODO: save model
    
    metric = -1 #TODO: evaluate metric
    
    return metric
    
    
def do_trial(out_path,max_epochs,trial):
    print(f'Beginning trial {trial}')
    
    out_path = os.path.join(out_path,f"{trial.get('bracket')}_{trial.get('trial')}_{trial.get('round')}")
    
    os.makedirs(out_path,exist_ok=True)
        
    metric = run_trial(trial=trial)
    
    print(f'Completed trial {trial}')
    
    return metric

###############################################################################

tuning_schedule_path = os.path.join(in_path,f'tuning_tracker_{experiment_id}_epochs_{R}_factor_{eta}_bracket_{s}.csv')

get_and_set_trial_p = functools.partial(get_and_set_trial,
                                        in_path=tuning_schedule_path,
                                        eta=eta,
                                        R=R,
                                        s=s)

max_repeats = 4 #Arbitrary number of attempts at starting trial run

repeat_attempt = True
repeat_counter = 0
while repeat_attempt:
    trial,status = lock_wrapper(tuning_schedule_path,get_and_set_trial_p)
    if status==2:
        if repeat_counter > max_repeats:
            repeat_attempt = False
        else:
            time.sleep(60)
            repeat_counter+=1
    else:
        repeat_attempt = False

if trial:
    metric = do_trial(out_path=out_path,max_epochs=R,trial=trial)
    
    set_trial_finished_p = functools.partial(set_trial_finished,
                                             in_path=tuning_schedule_path,
                                             trial=trial,
                                             metric=metric)
    
    lock_wrapper(tuning_schedule_path,set_trial_finished_p)
else:
    if repeat_counter < max_repeats:
        print(f'Skipped trial, all trials finished')
    else:
        print(f'Skipped trial, timed out waiting for previous round to be completed')
###############################################################################







