from matplotlib import pyplot as plt
import numpy as np
import os, sys
import pandas as pd
from scipy.linalg import expm, block_diag
from scipy.integrate import odeint, solve_ivp
from scipy.io import loadmat
import scipy.stats as stats
from scipy.stats import multivariate_normal as mvn
from scipy.stats.distributions import chi2

def NEES(x_truth, x_est_post, P_post):
    
    # n: number of states to estimate 
    #
    # Inputs
    # x_truth: the statevector from truth model (nx1)
    # x_est_post: the statevector post KF prediction (nx1)
    # P_post: the filter covariance post KF update (nxn)
    #
    # Outputs
    # NEES: the normalized estimation error squared (1x1)
    # 
    # Comments
    # Run this at every k time and store NEES into an array, one column array per monte carlo run
    
    est_error = x_truth - x_est_post
    NEES = est_error.T @ P_post @ est_error
    
    return NEES

def NEES_Chi2_Test(NEES_array, num_states, num_runs, alpha):
    
    # k: number of timesteps
    # r: number of monte carlo runs
    # n: number of states to estimate 
    #
    # Inputs
    # NEES_array: the normalized estimation error squared per k timestep per monte carlo run (kxr)
    # num_states: n (number of states to estimate) 
    # num_runs: r (number of monte carlo runs)
    # alpha: confidence interval
    #
    # Outputs
    # NEES_avg: the normalized estimation error squared averaged over runs for each timestep (kx1)
    # 
    # Comments
    # Run this after the monte carlo sim
    
    k = np.shape(NEES_array)[0]
    
    NEES_avg = np.average(NEES_array, axis=1)
    NEES_exp = num_states * np.ones(np.shape(NEES_avg))
    r1 = chi2.ppf(alpha/2, df=num_runs*num_states)/num_runs
    r2 = chi2.ppf(1 - (alpha/2), df=num_runs*num_states)/num_runs
    
    res = stats.chisquare(NEES_avg, NEES_exp)
    
    timesteps = np.arange(0,k)
    fig, ax = plt.subplot()
    fig.suptitle('NEES Testing')
    ax.plot(timesteps, NEES_avg, color='b')
    ax.plot(timesteps, r1, color="g")
    ax.plot(timesteps, r2, color="g")
    ax.set_title('NEES Estimation Results')
    ax.set_xlabel('Timesteps, k')
    ax.set_ylabel('NEES Statistic')

    plt.tight_layout()
    plt.show()
    plt.savefig("NEES Testing.png")
    
    return NEES_avg
    
    