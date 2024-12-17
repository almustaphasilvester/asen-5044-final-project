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
    
    est_error = x_truth.flatten() - x_est_post.flatten()
    NEES = est_error.T @ np.linalg.inv(P_post) @ est_error
    
    return NEES

def NEES_Chi2_Test(NEES_array, num_states, num_runs, alpha, title="NEES Testing"):
    
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
    # res: get chi statistic (res[0]) and p-value (res[1])
    # 
    # Comments
    # Run this after the monte carlo sim
    
    k = np.shape(NEES_array)[0]
    
    NEES_avg = np.nanmean(NEES_array, axis=1)
    NEES_exp = num_states * np.ones(np.shape(NEES_avg[np.isfinite(NEES_avg)]))
    r1 = chi2.ppf(alpha/2, df=num_runs*num_states)/num_runs
    r2 = chi2.ppf(1 - (alpha/2), df=num_runs*num_states)/num_runs
    try:
        res = stats.chisquare(NEES_avg[np.isfinite(NEES_avg)], NEES_exp)
    except:
        res = None
    
    timesteps = np.arange(0,k)
    ax = plt.subplot()
    ax.plot(timesteps, NEES_avg, color='b', linestyle='', marker='o', fillstyle = 'none')
    ax.plot(timesteps, r1 * np.ones(np.shape(timesteps)), color="g")
    ax.plot(timesteps, r2 * np.ones(np.shape(timesteps)), color="g")
    ax.set_title(title)
    ax.set_xlabel('Timesteps, k')
    ax.set_ylabel('NEES Statistic')

    plt.tight_layout()
    plt.savefig(title + ".png")
    plt.show()
    
    return NEES_avg, res
    
def NIS(y_real, y_sim, S_k):
    
    # p: number of measurements recorded
    #
    # Inputs
    # y_real: the measurement vector from real data (px1)
    # y_sim: the innovation vector post KF update, also know as H matrix times x estimated post prediction (px1)
    # S_k: the innovation error covariance, also known as HPH.T + R (easily get it from the KF) (pxp)
    #
    # Outputs
    # NIS: the normalized innovation squared (1x1)
    # 
    # Comments
    # Run this at every k time and store NIS into an array, one column array per monte carlo run
    
    innov_error = y_real.flatten() - y_sim.flatten()
    NIS = innov_error.T @ np.linalg.inv(S_k) @ innov_error
    
    return NIS

def NIS_Chi2_Test(NIS_array, num_meas, num_runs, alpha, title="NIS Testing"):
    
    # k: number of timesteps
    # r: number of monte carlo runs
    # p: number of measurements recorded
    #
    # Inputs
    # NIS_array: the normalized innovation squared per k timestep per monte carlo run (kxr)
    # num_meas: p (number of measurements recorded)
    # num_runs: r (number of monte carlo runs)
    # alpha: confidence interval
    #
    # Outputs
    # NIS_avg: the normalized innovation squared averaged over runs for each timestep (kx1)
    # res: get chi statistic (res[0]) and p-value (res[1]) 
    # 
    # Comments
    # Run this after the monte carlo sim
    
    k = np.shape(NIS_array)[0]
    
    NIS_avg = np.nanmean(NIS_array, axis=1)
    NIS_exp = num_meas * np.ones(np.shape(NIS_avg[np.isfinite(NIS_avg)]))
    r1 = chi2.ppf(alpha/2, df=num_runs*num_meas)/num_runs
    r2 = chi2.ppf(1 - (alpha/2), df=num_runs*num_meas)/num_runs
    #import pdb; pdb.set_trace()
    try:
        res = stats.chisquare(NIS_avg[np.isfinite(NIS_avg)], NIS_exp)
    except:
        res = None
    
    timesteps = np.arange(0,k)
    ax = plt.subplot()
    ax.plot(timesteps, NIS_avg, color='b', linestyle='', marker='o', fillstyle = 'none')
    ax.plot(timesteps, r1 * np.ones(np.shape(timesteps)), color="g")
    ax.plot(timesteps, r2 * np.ones(np.shape(timesteps)), color="g")
    ax.set_title(title)
    ax.set_xlabel('Timesteps, k')
    ax.set_ylabel('NIS Statistic')

    plt.tight_layout()
    plt.savefig(title + ".png")
    plt.show()
    
    return NIS_avg, res