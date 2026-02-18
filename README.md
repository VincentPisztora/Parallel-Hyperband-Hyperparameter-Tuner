# Parallel-Hyperband-Hyperparameter-Tuner
This repository contains code implementing the Hyperband hyperparameter tuning algorithm (described in [Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization](https://arxiv.org/pdf/1603.06560)). It is implemented such that each trial can be run in parallel to all other trials as a separate batch job. Useful for avoiding memory leaks from one trial round to the next.

Notes:

    - The evaluation metric must be nonnegative and "larger is better"
    
    - The parallel_hyperband_tuning_schedule.py script needs to be updated to the specifications of the given experiment
    
    - The run_trial function defined as a shell in parallel_hyperband_tuning_worker.py needs to be updated to the specifications of the given experiment
    
    - parallel_hyperband_tuning_worker.py should be simultaneously launched many times as separate batch jobs using e.g. Script_Parallel_Hyperband_Tuning
    
