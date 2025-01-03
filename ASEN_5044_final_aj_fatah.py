from matplotlib import pyplot as plt
import numpy as np
import sympy as sp
import os, sys
import pandas as pd
from scipy.linalg import expm, block_diag
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.io import loadmat
from scipy.stats import multivariate_normal as mvn
from nees_nis import *
from ekf import *
from time import time

# NOMINAL/NONLINEAR MODELS ----------------------------------------------------------------

class tracking_stations:
    """
    Equations for tracking station position and velocity measurements
    """

    def __init__(self, RE, omegaE):
        self.RE = RE
        self.omegaE = omegaE

    def theta0(self, i):
        return i*(np.pi/6)
    
    def Xi(self, t, i):
        return RE*np.cos(omegaE * t + self.theta0(i))

    def Yi(self, t, i):
        return RE*np.sin(omegaE * t + self.theta0(i))

    def Xidot(self, t, i):
        return -RE*omegaE*np.sin(omegaE * t + self.theta0(i))

    def Yidot(self, t, i):
        return RE*omegaE*np.cos(omegaE * t + self.theta0(i))
        
    def theta(self, t, i):
        theta_ang = self.omegaE * t + self.theta0(i)
        if theta_ang > 2*np.pi:
            theta_ang -= 2*np.pi
        return theta_ang


def nominal_orbit(t):
    max_vel = r0*np.sqrt(mu/(r0**3))

    theta = (t/T_tot)*2*np.pi
    x = r0*np.cos(theta)
    xdot = -max_vel*np.sin(theta)
    y = r0*np.sin(theta)
    ydot = max_vel*np.cos(theta)

    return x, xdot, y, ydot

def nominal_measurements(t, i):
    # nominal orbit
    X, Xdot, Y, Ydot = nominal_orbit(t)

    # tracking station data
    Xi = tracking_station_data.Xi(t, i)
    Yi = tracking_station_data.Yi(t, i)
    Xidot = tracking_station_data.Xidot(t, i)
    Yidot = tracking_station_data.Yidot(t, i)

    # nominal measurements
    rho    = np.sqrt((X-Xi)**2 + (Y-Yi)**2)
    rhodot = ((X-Xi)*(Xdot-Xidot) + (Y-Yi)*(Ydot-Yidot))/rho
    phi    = np.arctan2((Y-Yi),(X-Xi))

    return rho, rhodot, phi

def dyn_sys(t, state, Qtrue=np.array([])):
    """
    define dynamical system with given equations
    """
    if Qtrue.size != 0:
        wtilde = np.random.multivariate_normal([0,0], Qtrue)
        w1 = float(wtilde[0])
        w2 = float(wtilde[1])
    else:
        w1 = 0
        w2 = 0

    x, v_x, y, v_y = state
    # Compute the derivatives
    dx_dt = v_x
    ddx_dt = ((-mu * x) / (x**2 + y**2)**(3/2)) + w1
    dy_dt = v_y
    ddy_dt = ((-mu * y) / (x**2 + y**2)**(3/2)) + w2
    
    return [dx_dt, ddx_dt, dy_dt, ddy_dt]

def dyn_measurements(state, station_state, Rtrue=np.array([])):
    """
    define dynamical system with given equations
    """
    if Rtrue.size != 0:
        vtilde = np.random.multivariate_normal([0,0,0], Rtrue)
        v1 = float(vtilde[0])
        v2 = float(vtilde[1])
        v3 = float(vtilde[2])
    else:
        v1 = 0
        v2 = 0
        v3 = 0

    x, v_x, y, v_y = state
    xi, v_xi, yi, v_yi = station_state

    # Compute the range (rho)
    rho = (np.sqrt((x - xi)**2 + (y - yi)**2)) + v1
    
    # Compute the radial velocity (dot(rho))
    rho_dot = ((((x - xi) * (v_x - v_xi)) + ((y - yi) * (v_y - v_yi))) / rho) + v2
    
    # Compute the elevation angle (phi)
    phi = (np.arctan2(y - yi, x - xi)) + v3
    
    return rho, rho_dot, phi

# PART 1 ----------------------------------------------------------------------------------

def dt_linearization_states(x_nom, dT):
    """
    Linearize CT system about specified equilibrium/nominal operating point and find correspond DT linearized model matrices
    """

    # nominal point
    X_nom = x_nom[0][0]
    Y_nom = x_nom[2][0]

    # CT nonlinear model Jacobians - evaluated at nominal point
    A_nom = np.array([[0,1,0,0]\
            ,[(-mu*(Y_nom**2 - 2*(X_nom**2)))/((X_nom**2 + Y_nom**2)**2.5), 0, (3*mu*X_nom*Y_nom)/((X_nom**2 + Y_nom**2)**2.5), 0]\
            ,[0,0,0,1]\
            ,[3*mu*X_nom*Y_nom/((X_nom**2 + Y_nom**2)**2.5), 0, (-mu*(X_nom**2 - 2*(Y_nom**2)))/((X_nom**2 + Y_nom**2)**2.5), 0]])
    B_nom = np.array([[0, 0]\
            ,[1,0]\
            ,[0,0]\
            ,[0,1]])
    
    # DT linearized model matrices
    A_hat = np.concatenate((A_nom, B_nom), axis=1)
    A_hat = np.concatenate((A_hat, np.zeros((np.shape(A_hat)[1]-np.shape(A_hat)[0], np.shape(A_hat)[1]))), axis=0)
    A_hat_expm = expm(dT*A_hat)
    F = A_hat_expm[0:np.shape(A_nom)[0], 0:np.shape(A_nom)[1]]
    G = A_hat_expm[0:np.shape(A_nom)[0]:, np.shape(A_nom)[1]:]
    
    return F, G

def dt_linearization_measurements(x_nom, t):
    """
    Linearize CT system about specified equilibrium/nominal operating point and find correspond DT linearized model matrices
    """

    # nominal point
    X_nom = x_nom[0][0]
    Xdot_nom = x_nom[1][0]
    Y_nom = x_nom[2][0]
    Ydot_nom = x_nom[3][0]

    # CT nonlinear model Jacobians - evaluated at nominal point
    C_nom = {}
    for i in range(12):
        Xi_nom = tracking_station_data.Xi(t, i)
        Yi_nom = tracking_station_data.Yi(t, i)
        Xidot_nom = tracking_station_data.Xidot(t, i)
        Yidot_nom = tracking_station_data.Yidot(t, i)
        C_nom[i] = np.array([[(X_nom-Xi_nom)/np.sqrt((X_nom-Xi_nom)**2 + (Y_nom-Yi_nom)**2), 0, (Y_nom-Yi_nom)/np.sqrt((X_nom-Xi_nom)**2 + (Y_nom-Yi_nom)**2), 0],\
                    [((Y_nom-Yi_nom)*((Xdot_nom-Xidot_nom)*(Y_nom-Yi_nom) - (Ydot_nom - Yidot_nom)*(X_nom-Xi_nom)))/(((X_nom-Xi_nom)**2 + (Y_nom-Yi_nom)**2)**1.5),\
                        (X_nom-Xi_nom)/np.sqrt((X_nom-Xi_nom)**2 + (Y_nom-Yi_nom)**2),\
                        ((X_nom-Xi_nom)*((Ydot_nom-Yidot_nom)*(X_nom-Xi_nom) - (Xdot_nom - Xidot_nom)*(Y_nom-Yi_nom)))/(((X_nom-Xi_nom)**2 + (Y_nom-Yi_nom)**2)**1.5),\
                        (Y_nom-Yi_nom)/np.sqrt((X_nom-Xi_nom)**2 + (Y_nom-Yi_nom)**2),],\
                    [-(Y_nom-Yi_nom)/((X_nom-Xi_nom)**2 + (Y_nom-Yi_nom)**2), 0, (X_nom-Xi_nom)/((X_nom-Xi_nom)**2 + (Y_nom-Yi_nom)**2), 0]])
    D_nom = np.zeros((3,2))

    H = C_nom
    M = D_nom

    return H, M

def dt_linearized_state_sim(x0, dT, T):
    """
    Simulate linearized DT dynamics model near nominal point
    Validate against numerical integration routine (i.e. solve_ivp)
    """

    # initialize F,G,H,M matrices at t=0
    F, G = dt_linearization_states(x0, dT)

    # intial conditions - perturbation
    # (no process noise, measurement noise, or control input perturbations)
    dx = np.array([[0],[0.075],[0],[-0.021]])
    du = np.zeros((2,1))

    # # initial conditions - state, i.e. nominal point
    x_nom = np.array(x0) + dx

    # # simulate linearized DT dynamics
    x_tot_list = x_nom
    dx_tot_list = dx
    for t in range(dT,int(T),dT):
        # calculate dx at time k, using dx at k-1
        dx = F@dx + G@du

        # calculate new nominal orbit, ie at time k
        x, xdot, y, ydot = nominal_orbit(t)
        x_nom = [[x],[xdot],[y],[ydot]]

        x_tot = x_nom + dx
        x_tot_list = np.concatenate((x_tot_list, np.array(x_tot)), axis=1)
        dx_tot_list = np.concatenate((dx_tot_list, np.array(dx)), axis=1)

        # FOR NEXT TIMESTEP, calculate F and G 
        F, G = dt_linearization_states(x_nom, dT)

    timesteps = np.arange(0,int(T),dT)
    fig, ax = plt.subplots(4,1,sharex=True)
    fig.suptitle('Simulated Linearized DT Dynamics')
    ax[0].plot(timesteps, np.squeeze(np.asarray(x_tot_list[0])), color='b')
    ax[0].set_title('X')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Position (km)')
    ax[1].plot(timesteps, np.squeeze(np.asarray(x_tot_list[1])), color='b')
    ax[1].set_title('X_dot')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Velocity (km/s)')
    ax[2].plot(timesteps, np.squeeze(np.asarray(x_tot_list[2])), color='b')
    ax[2].set_title('Y')
    ax[2].set_xlabel('Time (s)')
    ax[2].set_ylabel('Position (km)')
    ax[3].plot(timesteps, np.squeeze(np.asarray(x_tot_list[3])), color='b')
    ax[3].set_title('Y_dot')
    ax[3].set_xlabel('Time (s)')
    ax[3].set_ylabel('Velocity (km/s)')
    plt.tight_layout()

    # validate using scipy.integrate.solve_ivp
    t_eval = np.linspace(0, T, int(T/10))
    soln = solve_ivp(dyn_sys, [0, T], np.asarray(x0).flatten(), t_eval=t_eval, method='RK45', rtol=1e-5)

    # Extract the results from the solution
    x_vals   = soln.y[0]
    v_x_vals = soln.y[1]
    y_vals   = soln.y[2]
    v_y_vals = soln.y[3]

    # Plot the results
    fig, ax = plt.subplots(4,1,sharex=True)
    fig.suptitle('Full Nonlinear Dynamics Simulation (using scipy.solve_ivp)')
    ax[0].plot(t_eval, x_vals, color='r')
    ax[0].set_title('X')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Position (km)')
    ax[1].plot(t_eval, v_x_vals, color='r')
    ax[1].set_title('X_dot')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Velocity (km/s)')
    ax[2].plot(t_eval, y_vals, color='r')
    ax[2].set_title('Y')
    ax[2].set_xlabel('Time (s)')
    ax[2].set_ylabel('Position (km)')
    ax[3].plot(t_eval, v_y_vals, color='r')
    ax[3].set_title('Y_dot')
    ax[3].set_xlabel('Time (s)')
    ax[3].set_ylabel('Velocity (km/s)')
    plt.tight_layout()

    # Plot the state perturbations
    fig, ax = plt.subplots(4,1,sharex=True)
    fig.suptitle('Linearized Approx. Perturbations vs. Time')
    ax[0].plot(t_eval, np.squeeze(np.asarray(dx_tot_list[0])))
    ax[0].set_title('X')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Position (km)')
    ax[1].plot(t_eval, np.squeeze(np.asarray(dx_tot_list[1])))
    ax[1].set_title('X_dot')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Velocity (km/s)')
    ax[2].plot(t_eval, np.squeeze(np.asarray(dx_tot_list[2])))
    ax[2].set_title('Y')
    ax[2].set_xlabel('Time (s)')
    ax[2].set_ylabel('Position (km)')
    ax[3].plot(t_eval, np.squeeze(np.asarray(dx_tot_list[3])))
    ax[3].set_title('Y_dot')
    ax[3].set_xlabel('Time (s)')
    ax[3].set_ylabel('Velocity (km/s)')
    plt.tight_layout()
    plt.show()
    plt.close()

def dt_linearized_measurements_sim(x0, dT, T):
    """
    Simulate linearized DT measurement model near nominal point
    Validate against numerical integration routine (i.e. solve_ivp)
    """
    colors = [
        '#E57373',  # Red
        '#64B5F6',  # Blue
        '#81C784',  # Green
        '#FFB74D',  # Orange
        '#BA68C8',  # Purple
        '#F06292',  # Pink
        '#FFF176',  # Yellow
        '#4DD0E1',  # Cyan
        '#8D6E63',  # Brown
        '#4DB6AC',  # Teal
        '#5C6BC0',  # Indigo
        '#D4E157'   # Lime
        ]

    fig, ax = plt.subplots(4,1,sharex=True)
    fig.suptitle('Simulated Linearized DT Measurements')
    ax[0].set_title('rho_i')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Range (km)')
    ax[1].set_title('rho_dot_i')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Range Rate (km/s)')
    ax[2].set_title('phi_i')
    ax[2].set_xlabel('Time (s)')
    ax[2].set_ylabel('Elevation (rad)')
    ax[3].set_title('Visible Stations')
    ax[3].set_xlabel('Time (s)')
    ax[3].set_ylabel('Station ID')

    # simulate linearized DT measurements
    for i in range(12):
        # initialize F,G,H,M matrices at t=0
        F, G = dt_linearization_states(x0, dT)
        x_nom = x0

        # intial conditions - perturbation
        # (no process noise, measurement noise, or control input perturbations)
        dx = np.array([[0],[0.075],[0],[-0.021]])
        du = np.zeros((2,1))
        
        vis_list = []
        # simulate linearized DT dynamics
        for t in range(dT,int(T),dT):
            # calculate dx at time t, using dx at t-1
            dx = F@dx + G@du

            # visibility, at time t
            theta    = np.arctan2(tracking_station_data.Yi(t,i),tracking_station_data.Xi(t,i))
            theta    = tracking_station_data.theta(t,i)
            el_L_lim = (-0.5*np.pi)+theta
            el_H_lim = (0.5*np.pi)+theta
            # calculate new nominal orbit, ie at time t
            x, xdot, y, ydot = nominal_orbit(t)
            x_nom = [[x],[xdot],[y],[ydot]]
            # calculate new nominal measurements
            rho, rhodot, phi = nominal_measurements(t, i)
            y_nom_new = np.array([[rho],[rhodot],[phi]])

            # calculate H and M at timestep t
            H, M = dt_linearization_measurements(x_nom, t)
            # dy at timestep t, calculated using dx at timestep t
            dy = (H[i])@dx + M@du

            y_tot = y_nom_new + dy

            site_pos = np.array([tracking_station_data.Xi(t,i),tracking_station_data.Yi(t,i)])
            sat_pos  = np.array([x, y])
            rel_pos = sat_pos - site_pos
            el_mask = np.arccos(np.dot(rel_pos, site_pos)/(np.linalg.norm(rel_pos)*np.linalg.norm(site_pos)))

            if el_mask < 0.5*np.pi:
            # if (y_tot[2][0] >= el_L_lim) and (y_tot[2][0] <= el_H_lim):
                vis_list.append(i+1)
            else:
                y_tot = np.array([[np.nan],[np.nan],[np.nan]])
                vis_list.append(np.nan)

            if t == dT:
                y_tot_list = y_tot
            else:
                y_tot_list = np.concatenate((y_tot_list, np.array(y_tot)), axis=1)

            # FOR NEXT TIMESTEP, calculate F and G 
            F, G = dt_linearization_states(x_nom, dT)

        timesteps = np.arange(dT,int(T),dT)
        ax[0].plot(timesteps, np.squeeze(np.asarray(y_tot_list[0])), color=colors[i], marker='x')
        ax[1].plot(timesteps, np.squeeze(np.asarray(y_tot_list[1])), color=colors[i], marker='o', fillstyle='none')
        ax[2].plot(timesteps, np.squeeze(np.asarray(y_tot_list[2])), color=colors[i], marker='o', fillstyle='none')
        ax[3].plot(timesteps, vis_list, color=colors[i], marker='^')

    plt.tight_layout()

    # validate using scipy.integrate.solve_ivp
    t_eval = np.linspace(0, T, int(T/10))
    soln = solve_ivp(dyn_sys, [0, T], np.asarray(x0).flatten(), t_eval=t_eval, method='RK45', rtol=1e-5)
    fig, ax = plt.subplots(4,1,sharex=True)
    fig.suptitle('Full Nonlinear Measurements Simulation (using scipy.integrate.solve_ivp)')
    ax[0].set_title('rho_i')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Range (km)')
    ax[1].set_title('rho_dot_i')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Range Rate (km/s)')
    ax[2].set_title('phi_i')
    ax[2].set_xlabel('Time (s)')
    ax[2].set_ylabel('Elevation (rad)')
    ax[3].set_title('Visible Stations')
    ax[3].set_xlabel('Time (s)')
    ax[3].set_ylabel('Station ID')
    
    for i in range(12):
        vis_list = []
        t_idx = 1
        for t in range(dT,int(T),dT):
            # tracking station data
            Xi = tracking_station_data.Xi(t, i)
            Yi = tracking_station_data.Yi(t, i)
            Xidot = tracking_station_data.Xidot(t, i)
            Yidot = tracking_station_data.Yidot(t, i)
            station_state = [Xi, Xidot, Yi, Yidot]

            # visibility, at time t
            theta    = np.arctan2(Yi,Xi)
            theta    = tracking_station_data.theta(t, i)
            el_L_lim = (-0.5*np.pi)+theta
            el_H_lim = (0.5*np.pi)+theta

            #state = soln[t_idx, :]
            state  = soln.y[:,t_idx]
            rho, rho_dot, phi = dyn_measurements(state, station_state)

            site_pos = np.array([Xi,Yi])
            sat_pos  = np.array([state[0], state[2]])
            rel_pos = sat_pos - site_pos
            el_mask = np.arccos(np.dot(rel_pos, site_pos)/(np.linalg.norm(rel_pos)*np.linalg.norm(site_pos)))

            if el_mask < 0.5*np.pi:
                y = np.array([[rho],[rho_dot],[phi]])
                vis_list.append(i+1)
            else:
                y = np.array([[np.nan],[np.nan],[np.nan]])
                vis_list.append(np.nan)

            if t == dT:
                y_soln = y
            else:
                y_soln = np.concatenate((y_soln, y), axis=1)

            t_idx += 1

        # Plot results
        timesteps = np.arange(dT,int(T),dT)
        ax[0].plot(timesteps, np.squeeze(np.asarray(y_soln[0])), color=colors[i], marker='x')
        ax[1].plot(timesteps, np.squeeze(np.asarray(y_soln[1])), color=colors[i], marker='o', fillstyle='none')
        ax[2].plot(timesteps, np.squeeze(np.asarray(y_soln[2])), color=colors[i], marker='o', fillstyle='none')
        ax[3].plot(timesteps, vis_list, color=colors[i], marker='^')
        ax[0].set_xlim([0,T])
        ax[1].set_xlim([1,T])
        ax[2].set_xlim([2,T])
        ax[3].set_xlim([3,T])

    plt.tight_layout()
    plt.show()
    plt.close()

# PART 2 ----------------------------------------------------------------------------------
def monte_carlo_states_tmt(x0,Qtrue,T,plot=False):
    t_eval = np.linspace(0, T, int(T/10))
    noisy_soln = solve_ivp(dyn_sys, [0, T], np.asarray(x0).flatten(), t_eval=t_eval, method='RK45',args=(Qtrue,), rtol=1e-12)
    soln = solve_ivp(dyn_sys, [0, T], np.asarray(x0).flatten(), t_eval=t_eval, method='RK45', rtol=1e-12)

    # sanity check plots
    # Plot the results
    if plot:
        fig, ax = plt.subplots(4,1,sharex=True)
        fig.suptitle('Monte Carlo Sim (using scipy.integrate.solve_ivp)')
        ax[0].plot(t_eval, noisy_soln.y[0], color='b')
        ax[0].plot(t_eval, soln.y[0], color='r')
        ax[0].set_title('X')
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel('Position (km)')
        ax[1].plot(t_eval, noisy_soln.y[1], color='b')
        ax[1].plot(t_eval, soln.y[1], color='r')
        ax[1].set_title('X_dot')
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Velocity (km/s)')
        ax[2].plot(t_eval, noisy_soln.y[2], color='b')
        ax[2].plot(t_eval, soln.y[2], color='r')
        ax[2].set_title('Y')
        ax[2].set_xlabel('Time (s)')
        ax[2].set_ylabel('Position (km)')
        ax[3].plot(t_eval, noisy_soln.y[3], color='b')
        ax[3].plot(t_eval, soln.y[3], color='r')
        ax[3].set_title('Y_dot')
        ax[3].set_xlabel('Time (s)')
        ax[3].set_ylabel('Velocity (km/s)')

        ax[0].set_ylim([-7500,7500])
        ax[1].set_ylim([-9,9])
        ax[2].set_ylim([-7500,7500])
        ax[3].set_ylim([-9,9])

        fig.legend(['With Noise', 'Without Noise'], loc='lower center', bbox_to_anchor=(0.5,0))

        plt.tight_layout()
        plt.show()
        plt.close()

    return noisy_soln, soln

def monte_carlo_measurements_tmt(x0,Qtrue,Rtrue,T,plot=False):
    # validate using scipy.integrate.solve_ivp
    t_eval = np.linspace(0, T, int(T/10))
    noisy_soln = solve_ivp(dyn_sys, [0, T], np.asarray(x0).flatten(), t_eval=t_eval, method='RK45',args=(Qtrue,), rtol=1e-12)

    if plot:
        colors = [
        '#E57373',  # Red
        '#64B5F6',  # Blue
        '#81C784',  # Green
        '#FFB74D',  # Orange
        '#BA68C8',  # Purple
        '#F06292',  # Pink
        '#FFF176',  # Yellow
        '#4DD0E1',  # Cyan
        '#8D6E63',  # Brown
        '#4DB6AC',  # Teal
        '#5C6BC0',  # Indigo
        '#D4E157'   # Lime
        ]
        fig, ax = plt.subplots(4,1,sharex=True)
        fig.suptitle('Noisy Simulated Measurement Data')
        ax[0].set_title('rho_i')
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel('Range (km)')
        ax[1].set_title('rho_dot_i')
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Range Rate (km/s)')
        ax[2].set_title('phi_i')
        ax[2].set_xlabel('Time (s)')
        ax[2].set_ylabel('Elevation (rad)')
        ax[3].set_title('Visible Stations')
        ax[3].set_xlabel('Time (s)')
        ax[3].set_ylabel('Station ID')
    
    y_soln = [np.array([]) for i in range(int(T/dT))]
    for i in range(12):
        vis_list_noisy = []
        t_idx = 1
        for t in range(dT,int(T),dT):
            # tracking station data
            Xi = tracking_station_data.Xi(t, i)
            Yi = tracking_station_data.Yi(t, i)
            Xidot = tracking_station_data.Xidot(t, i)
            Yidot = tracking_station_data.Yidot(t, i)
            station_state = [Xi, Xidot, Yi, Yidot]

            # visibility, at time t
            theta    = np.arctan2(Yi,Xi)
            # theta    = tracking_station_data.theta(t, i)
            el_L_lim = (-0.5*np.pi) + theta
            el_H_lim = (0.5*np.pi) + theta

            #state = soln.y[:,t_idx]
            state_noisy = noisy_soln.y[:,t_idx]
            rho_noisy, rho_dot_noisy, phi_noisy = dyn_measurements(state_noisy, station_state,Rtrue)

            site_pos = np.array([Xi,Yi])
            sat_pos  = np.array([state_noisy[0], state_noisy[2]])
            rel_pos = sat_pos - site_pos
            el_mask = np.arccos(np.dot(rel_pos, site_pos)/(np.linalg.norm(rel_pos)*np.linalg.norm(site_pos)))

            if el_mask < 0.5*np.pi:
                y_noisy = np.array([[rho_noisy],[rho_dot_noisy],[phi_noisy]])
                vis_list_noisy.append(i+1)
                if y_soln[t_idx-1].size == 0:
                    y_soln[t_idx-1] = np.array([[rho_noisy],[rho_dot_noisy],[phi_noisy],[i+1]])
                else:
                    y_soln[t_idx-1] = np.concatenate((y_soln[t_idx-1], np.array([[rho_noisy],[rho_dot_noisy],[phi_noisy],[i+1]])), axis=1)
            else:
                y_noisy = np.array([[np.nan],[np.nan],[np.nan]])
                vis_list_noisy.append(np.nan)

            if t == dT:
                y_noisy_soln = y_noisy
            else:
                y_noisy_soln = np.concatenate((y_noisy_soln, y_noisy), axis=1)

            t_idx += 1

        # Plot results
        if plot:
            timesteps = np.arange(dT,int(T),dT)
            ax[0].plot(timesteps, np.squeeze(np.asarray(y_noisy_soln[0])), color=colors[i], marker='x')
            ax[1].plot(timesteps, np.squeeze(np.asarray(y_noisy_soln[1])), color=colors[i], marker='o', fillstyle='none')
            ax[2].plot(timesteps, np.squeeze(np.asarray(y_noisy_soln[2])), color=colors[i], marker='o', fillstyle='none')
            ax[3].plot(timesteps, vis_list_noisy, color=colors[i], marker='^')

    if plot:
        ax[0].set_ylim([0,2250])
        ax[1].set_ylim([-10,10])
        ax[2].set_ylim([-5,5])
        ax[3].set_ylim([0,12])
        plt.tight_layout()
        plt.show()
        plt.close()

    return y_soln

def LKF(x0, dx0, P0, dT, T, Qtrue, Rtrue, ydata, Q_LKF = np.eye(2)*1e-10, plot=False):
    """
    Implement and tune a linearized KF using the specified nominal state trajectory
    """
    
    def eulerized_dt_jacobians(x_nom, dT, t):
        # nominal point
        X_nom    = x_nom[0][0]
        Xdot_nom = x_nom[1][0]
        Y_nom    = x_nom[2][0]
        Ydot_nom = x_nom[3][0]

        # CT nonlinear model Jacobians - evaluated at nominal point
        A_nom = np.array([[0,1,0,0]\
                ,[(-mu*(Y_nom**2 - 2*(X_nom**2)))/((X_nom**2 + Y_nom**2)**2.5), 0, (3*mu*X_nom*Y_nom)/((X_nom**2 + Y_nom**2)**2.5), 0]\
                ,[0,0,0,1]\
                ,[3*mu*X_nom*Y_nom/((X_nom**2 + Y_nom**2)**2.5), 0, (-mu*(X_nom**2 - 2*(Y_nom**2)))/((X_nom**2 + Y_nom**2)**2.5), 0]])
        B_nom = np.array([[0, 0]\
                ,[1,0]\
                ,[0,0]\
                ,[0,1]])
        C_nom = {}
        for i in range(12):
            Xi_nom = tracking_station_data.Xi(t, i)
            Yi_nom = tracking_station_data.Yi(t, i)
            Xidot_nom = tracking_station_data.Xidot(t, i)
            Yidot_nom = tracking_station_data.Yidot(t, i)
            C_nom[i] = np.array([[(X_nom-Xi_nom)/np.sqrt((X_nom-Xi_nom)**2 + (Y_nom-Yi_nom)**2), 0, (Y_nom-Yi_nom)/np.sqrt((X_nom-Xi_nom)**2 + (Y_nom-Yi_nom)**2), 0],\
                        [((Y_nom-Yi_nom)*((Xdot_nom-Xidot_nom)*(Y_nom-Yi_nom) - (Ydot_nom - Yidot_nom)*(X_nom-Xi_nom)))/(((X_nom-Xi_nom)**2 + (Y_nom-Yi_nom)**2)**1.5),\
                            (X_nom-Xi_nom)/np.sqrt((X_nom-Xi_nom)**2 + (Y_nom-Yi_nom)**2),\
                            ((X_nom-Xi_nom)*((Ydot_nom-Yidot_nom)*(X_nom-Xi_nom) - (Xdot_nom - Xidot_nom)*(Y_nom-Yi_nom)))/(((X_nom-Xi_nom)**2 + (Y_nom-Yi_nom)**2)**1.5),\
                            (Y_nom-Yi_nom)/np.sqrt((X_nom-Xi_nom)**2 + (Y_nom-Yi_nom)**2),],\
                        [-(Y_nom-Yi_nom)/((X_nom-Xi_nom)**2 + (Y_nom-Yi_nom)**2), 0, (X_nom-Xi_nom)/((X_nom-Xi_nom)**2 + (Y_nom-Yi_nom)**2), 0]])
        Gamma_nom = np.array([[0 ,0]\
                            ,[1, 0]\
                            ,[0, 0]\
                            ,[0, 1]])
        
        F_tilde = np.identity(np.shape(A_nom)[0]) + dT*A_nom
        G_tilde = dT*B_nom
        Omega_tilde = dT*Gamma_nom
        H_tilde = C_nom

        return F_tilde, G_tilde, Omega_tilde, H_tilde

    # initialize at t=0
    x_hat_plus = np.array(x0)
    P_plus = P0
    dx_hat_vals = np.random.multivariate_normal(dx0, P_plus)
    dx_hat_plus = np.array([[dx_hat_vals[0]],[dx_hat_vals[1]],[dx_hat_vals[2]],[dx_hat_vals[3]]])
    du = np.zeros((2,1))

    F_tilde, G_tilde, Omega_tilde, H_tilde_all = eulerized_dt_jacobians(x_hat_plus + dx_hat_plus, dT, 0)
    
    x_hat_plus_tot   = x_hat_plus + dx_hat_plus       # ie nominal + perturb
    vis_station_list = []
    P_list           = [P_plus]
    t_list           = [0]
    NEES_list        = []
    NIS_list         = []

    # ground truth values
    x_tmt, xstar = monte_carlo_states_tmt(x0,Qtrue,T,False)

    # t_idx = 1
    for t in range(dT,T,dT):
        t_idx = int(t/dT)
        # calculate nominal orbit at time k+1
        x_nom_t = xstar.y[:,t_idx]
        x_nom = np.array([[x_nom_t[0]],[x_nom_t[1]],[x_nom_t[2]],[x_nom_t[3]]])

        # TIME UPDATE/PREDICTION STEP FOR TIME k+1
        dx_hat_minus = F_tilde@dx_hat_plus + G_tilde@du
        P_minus      = (F_tilde@P_plus@(F_tilde.T)) + (Omega_tilde@Q_LKF@(Omega_tilde.T))

        # actual received sensor measurement
        y_full_vect = ydata[t_idx]
        if y_full_vect.size != 0:
            visible_stations = y_full_vect[3,:]
            H_tilde = np.array([])
            yk1     = np.array([])
            ystar   = np.array([])
            R_all   = np.array([])
            idx     = 0
            vis_station_list.append([visible_stations])
            for id in visible_stations:
                id -= 1
                # stack measurement vectors
                y_id = y_full_vect[0:3,idx]
                y_id = np.array([[y_id[0]],[y_id[1]],[y_id[2]]])
                if yk1.size == 0:
                    yk1 = y_id
                else:
                    yk1 = np.concatenate((yk1, np.array(y_id)), axis=0)

                # stack H matrices
                H_tilde_id = H_tilde_all[id]
                if H_tilde.size == 0:
                    H_tilde = np.array(H_tilde_id)
                else:
                    H_tilde = np.concatenate((H_tilde, np.array(H_tilde_id)), axis=0)

                # stack nominal measurements
                Xi = tracking_station_data.Xi(t, id)
                Yi = tracking_station_data.Yi(t, id)
                Xidot = tracking_station_data.Xidot(t, id)
                Yidot = tracking_station_data.Yidot(t, id)
                station_state = [Xi, Xidot, Yi, Yidot]
                
                # nominal sensor measurement at time k+1
                state = x_nom.flatten()
                rho, rho_dot, phi = dyn_measurements(state, station_state)
                ystar_id = np.array([[rho],[rho_dot],[phi]])
                if ystar.size == 0:
                    ystar = np.array(ystar_id)
                else:
                    ystar = np.concatenate((ystar, np.array(ystar_id)), axis=0)
                    
                if R_all.size == 0:
                    R_all = Rtrue
                else:
                    R_all = block_diag(R_all, Rtrue)

                idx += 1

            # nominal sensor measurement at time k+1
            dy = yk1 - ystar

            # Kalman Gain
            S_k = H_tilde@P_minus@(H_tilde.T) + R_all
            K = P_minus@(H_tilde.T)@(np.linalg.inv(S_k))

            # Covariance Matrix
            P_plus = (np.identity(np.shape(P_minus)[0]) - (K@H_tilde))@P_minus

            # state perturbation estimate
            dx_hat_plus = dx_hat_minus + K@(dy - H_tilde@dx_hat_minus)

            # ADD TO NOMINAL STATE ESTIMATE
            x_hat_plus = x_nom + dx_hat_plus
            
            # NIS Calculation
            NIS_list.append(NIS(dy, H_tilde@dx_hat_minus, S_k))
        
        else: 
            # Covariance Matrix
            P_plus = P_minus

            # state estimate
            dx_hat_plus = dx_hat_minus
            x_hat_plus = x_nom
            
            # NIS Calculation
            NIS_list.append(np.nan)
        
        x_hat_plus_tot  = np.concatenate((x_hat_plus_tot, np.array(x_hat_plus)), axis=1)
        P_list.append(P_plus)
        #print(np.diag(P_plus))    
        t_list.append(t)
        
        # MEASUREMENT UPDATE/CORRECTION STEP FOR TIME k+1
        # FOR NEXT TIMESTEP, calculate new F_tilde, G_tilde, Omega_tilde, H_tilde_all
        F_tilde, G_tilde, Omega_tilde, H_tilde_all = eulerized_dt_jacobians(x_nom, dT, t)
        
        # NEES Calculation
        NEES_list.append(NEES(x_tmt.y[:,t_idx], x_hat_plus, P_plus))
        
        # NEES/NIS Debug
        # print("NEES: ", NEES_list[t_idx-1])
        # print("NIS: ", NIS_list[t_idx-1])
        
        t_idx += 1
        
    # PLOTS
    # LKF state
    if plot:
        fig, ax = plt.subplots(4,1,sharex=True)
        fig.suptitle('Linearized Kalman Filter')
        ax[0].plot(t_list, np.squeeze(np.asarray(x_hat_plus_tot[0])), color='b')
        ax[0].set_title('X')
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel('Position (km)')
        ax[1].plot(t_list, np.squeeze(np.asarray(x_hat_plus_tot[1])), color='b')
        ax[1].set_title('X_dot')
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Velocity (km/s)')
        ax[2].plot(t_list, np.squeeze(np.asarray(x_hat_plus_tot[2])), color='b')
        ax[2].set_title('Y')
        ax[2].set_xlabel('Time (s)')
        ax[2].set_ylabel('Position (km)')
        ax[3].plot(t_list, np.squeeze(np.asarray(x_hat_plus_tot[3])), color='b')
        ax[3].set_title('Y_dot')
        ax[3].set_xlabel('Time (s)')
        ax[3].set_ylabel('Velocity (km/s)')

        ax[0].set_ylim([-7500,7500])
        ax[1].set_ylim([-9,9])
        ax[2].set_ylim([-7500,7500])
        ax[3].set_ylim([-9,9])

        # plot confidence bounds
        for i in range(4):
            upper_bounds    = [2*np.sqrt(array[i, i]) for array in P_list]  # Extract the diagonal element (i, i) at each timestep
            lower_bounds    = [-2*np.sqrt(array[i, i]) for array in P_list]  # Extract the diagonal element (i, i) at each timestep
            upper_bounds = upper_bounds + x_hat_plus_tot[i]
            lower_bounds = lower_bounds + x_hat_plus_tot[i]
            ax[i].plot(t_list, upper_bounds, 'r-')
            ax[i].plot(t_list, lower_bounds, 'r-')

        fig.legend(['Filter State', 'Filter 2sigma bounds'], loc='lower center', bbox_to_anchor=(0.5,0))

        plt.tight_layout()

    # Linearized KF State Estimation Errors
    x_err    = []
    xdot_err = []
    y_err    = []
    ydot_err = []
    t_all = np.arange(0,int(T),dT)
    for t in t_all:
        if t in t_list:
            t_all_idx  = int(t/10)
            t_list_idx = t_list.index(t)
            x_err.append(x_tmt.y[0,t_all_idx] - np.squeeze(np.asarray(x_hat_plus_tot[0]))[t_list_idx])
            xdot_err.append(x_tmt.y[1,t_all_idx] - np.squeeze(np.asarray(x_hat_plus_tot[1]))[t_list_idx])
            y_err.append(x_tmt.y[2,t_all_idx] - np.squeeze(np.asarray(x_hat_plus_tot[2]))[t_list_idx])
            ydot_err.append(x_tmt.y[3,t_all_idx] - np.squeeze(np.asarray(x_hat_plus_tot[3]))[t_list_idx])

    if plot:
        fig, ax = plt.subplots(4,1,sharex=True)
        fig.suptitle('Linearized KF State Estimation Errors')
        ax[0].plot(t_list, x_err, color='b')
        ax[0].set_title('X')
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel('Position (km)')
        ax[1].plot(t_list, xdot_err, color='b')
        ax[1].set_title('X_dot')
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Velocity (km/s)')
        ax[2].plot(t_list, y_err, color='b')
        ax[2].set_title('Y')
        ax[2].set_xlabel('Time (s)')
        ax[2].set_ylabel('Position (km)')
        ax[3].plot(t_list, ydot_err, color='b')
        ax[3].set_title('Y_dot')
        ax[3].set_xlabel('Time (s)')
        ax[3].set_ylabel('Velocity (km/s)')

        # plot confidence bounds
        for i in range(4):
            upper_bounds    = [2*np.sqrt(array[i, i]) for array in P_list]  # Extract the diagonal element (i, i) at each timestep
            lower_bounds    = [-2*np.sqrt(array[i, i]) for array in P_list]  # Extract the diagonal element (i, i) at each timestep
            upper_bounds = upper_bounds
            lower_bounds = lower_bounds
            ax[i].plot(t_list, upper_bounds, 'r-')
            ax[i].plot(t_list, lower_bounds, 'r-')

        ax[0].set_ylim([-250,250])
        ax[1].set_ylim([-2.5,2.5])
        ax[2].set_ylim([-250,250])
        ax[3].set_ylim([-2.5,2.5])

        fig.legend(['Filter Error', 'Filter 2sigma bounds'], loc='lower center', bbox_to_anchor=(0.5,0))

        plt.tight_layout()
        plt.show()
        plt.close()
    
    est_errors = np.array([x_err, xdot_err, y_err, ydot_err])
    
    return x_hat_plus_tot, est_errors, P_list, NEES_list, NIS_list

def EKF(x0, P0, dT, T, Qtrue, Rtrue, ydata, Q_EKF = np.eye(2)*1e-10, plot=False):
    """
    Implement and tune an extended KF using the specified nominal state trajectory
    """
    
    def eulerized_dt_jacobians(x_minus, x_plus, dT, t):
        # predicted point
        X_minus    = x_minus[0][0]
        Xdot_minus = x_minus[1][0]
        Y_minus    = x_minus[2][0]
        Ydot_minus = x_minus[3][0]
        
        # updated point
        X_plus    = x_plus[0][0]
        Xdot_plus = x_plus[1][0]
        Y_plus    = x_plus[2][0]
        Ydot_plus = x_plus[3][0]

        # CT nonlinear model Jacobians - evaluated at nominal point
        A_nom = np.array([[0,1,0,0]\
                ,[(-mu*(Y_plus**2 - 2*(X_plus**2)))/((X_plus**2 + Y_plus**2)**2.5), 0, (3*mu*X_plus*Y_plus)/((X_plus**2 + Y_plus**2)**2.5), 0]\
                ,[0,0,0,1]\
                ,[3*mu*X_plus*Y_plus/((X_plus**2 + Y_plus**2)**2.5), 0, (-mu*(X_plus**2 - 2*(Y_plus**2)))/((X_plus**2 + Y_plus**2)**2.5), 0]])
        B_nom = np.array([[0, 0]\
                ,[1,0]\
                ,[0,0]\
                ,[0,1]])
        C_nom = {}
        for i in range(12):
            Xi_nom = tracking_station_data.Xi(t, i)
            Yi_nom = tracking_station_data.Yi(t, i)
            Xidot_nom = tracking_station_data.Xidot(t, i)
            Yidot_nom = tracking_station_data.Yidot(t, i)
            C_nom[i] = np.array([[(X_minus-Xi_nom)/np.sqrt((X_minus-Xi_nom)**2 + (Y_minus-Yi_nom)**2), 0, (Y_minus-Yi_nom)/np.sqrt((X_minus-Xi_nom)**2 + (Y_minus-Yi_nom)**2), 0],\
                        [((Y_minus-Yi_nom)*((Xdot_minus-Xidot_nom)*(Y_minus-Yi_nom) - (Ydot_minus - Yidot_nom)*(X_minus-Xi_nom)))/(((X_minus-Xi_nom)**2 + (Y_minus-Yi_nom)**2)**1.5),\
                            (X_minus-Xi_nom)/np.sqrt((X_minus-Xi_nom)**2 + (Y_minus-Yi_nom)**2),\
                            ((X_minus-Xi_nom)*((Ydot_minus-Yidot_nom)*(X_minus-Xi_nom) - (Xdot_minus - Xidot_nom)*(Y_minus-Yi_nom)))/(((X_minus-Xi_nom)**2 + (Y_minus-Yi_nom)**2)**1.5),\
                            (Y_minus-Yi_nom)/np.sqrt((X_minus-Xi_nom)**2 + (Y_minus-Yi_nom)**2),],\
                        [-(Y_minus-Yi_nom)/((X_minus-Xi_nom)**2 + (Y_minus-Yi_nom)**2), 0, (X_minus-Xi_nom)/((X_minus-Xi_nom)**2 + (Y_minus-Yi_nom)**2), 0]])
        Gamma_nom = np.array([[0 ,0]\
                            ,[1, 0]\
                            ,[0, 0]\
                            ,[0, 1]])
        
        F_tilde = np.identity(np.shape(A_nom)[0]) + dT*A_nom
        G_tilde = dT*B_nom
        Omega_tilde = dT*Gamma_nom
        H_tilde = C_nom

        return F_tilde, G_tilde, Omega_tilde, H_tilde

    # initialize at t=0
    P_plus = P0
    #print(x0, type(x0), x0[0])
    #x0 = [x0[0][0], x0[1][0], x0[2][0], x0[3][0]]
    x_hat_plus = np.random.multivariate_normal(mean=[x0[0][0], x0[1][0], x0[2][0], x0[3][0]], cov=P_plus).reshape((4,1))
    du = np.zeros((2,1))

    F_tilde, G_tilde, Omega_tilde, H_tilde_all = eulerized_dt_jacobians(x0, x_hat_plus, dT, 0)
  
    x_hat_plus_tot   = x_hat_plus       # ie nominal + perturb
    vis_station_list = []
    P_list           = [P_plus]
    t_list           = [0]
    NEES_list        = []
    NIS_list         = []

    # ground truth values
    xstar, dum = monte_carlo_states_tmt(x0,Qtrue,T,False)
    
    t_eval = np.linspace(0, T, int(T/10))

    t_idx = 1
    for t in range(dT,T,dT):

        # TIME UPDATE/PREDICTION STEP FOR TIME k+1
        x_hat_ivp   = solve_ivp(dyn_sys, [0, dT], np.asarray(x_hat_plus).flatten(), t_eval=[dT], method='RK45', rtol=1e-12)
        x_hat_minus = x_hat_ivp.y[:,0].reshape((4,1))
        
        # FOR NEXT TIMESTEP, calculate new F_tilde, G_tilde, Omega_tilde, H_tilde_all
        F_tilde, G_tilde, Omega_tilde, H_tilde_all = eulerized_dt_jacobians(x_hat_minus, x_hat_plus, dT, t)
        
        #import pdb; pdb.set_trace()
        P_minus     = (F_tilde@P_plus@(F_tilde.T)) + (Omega_tilde@Q_EKF@(Omega_tilde.T))


        # MEASUREMENT UPDATE/CORRECTION STEP FOR TIME k+1
        # actual received sensor measurement
        y_full_vect = ydata[t_idx]
        if y_full_vect.size != 0:
            visible_stations = y_full_vect[3,:]
            H_tilde = np.array([])
            yk1     = np.array([])
            ystar   = np.array([])
            R_all   = np.array([])
            idx     = 0
            vis_station_list.append([visible_stations])
            for id in visible_stations:
                id -= 1
                # stack measurement vectors
                y_id = y_full_vect[0:3,idx]
                y_id = np.array([[y_id[0]],[y_id[1]],[y_id[2]]])
                if yk1.size == 0:
                    yk1 = y_id
                else:
                    yk1 = np.concatenate((yk1, np.array(y_id)), axis=0)

                # stack H matrices
                H_tilde_id = H_tilde_all[id]
                if H_tilde.size == 0:
                    H_tilde = np.array(H_tilde_id)
                else:
                    H_tilde = np.concatenate((H_tilde, np.array(H_tilde_id)), axis=0)

                # stack nominal measurements
                Xi = tracking_station_data.Xi(t, id)
                Yi = tracking_station_data.Yi(t, id)
                Xidot = tracking_station_data.Xidot(t, id)
                Yidot = tracking_station_data.Yidot(t, id)
                station_state = [Xi, Xidot, Yi, Yidot]
                
                # nominal sensor measurement at time k+1
                state = x_hat_minus.flatten()
                rho, rho_dot, phi = dyn_measurements(state, station_state)
                ystar_id = np.array([[rho],[rho_dot],[phi]])
                if ystar.size == 0:
                    ystar = np.array(ystar_id)
                else:
                    ystar = np.concatenate((ystar, np.array(ystar_id)), axis=0)
                    
                if R_all.size == 0:
                    R_all = Rtrue
                else:
                    R_all = block_diag(R_all, Rtrue)

                idx += 1

            # nominal sensor measurement at time k+1
            innov = yk1  - ystar
            # Kalman Gain
            S_k = H_tilde@P_minus@(H_tilde.T) + R_all
            K = P_minus@(H_tilde.T)@(np.linalg.inv(S_k))

            # Covariance Matrix
            P_plus = (np.identity(np.shape(P_minus)[0]) - (K@H_tilde))@P_minus

            # state estimate
            x_hat_plus = x_hat_minus + K@innov
            
            # NIS Calculation
            NIS_list.append(NIS(innov, H_tilde@x_hat_minus, S_k))
        
        else: 
            
            # Covariance Matrix
            P_plus = P_minus

            # state estimate
            x_hat_plus = x_hat_minus
            
            # NIS Calculation
            NIS_list.append(np.nan)
        
        x_hat_plus_tot  = np.concatenate((x_hat_plus_tot, np.array(x_hat_plus)), axis=1)
        P_list.append(P_plus)

        t_list.append(t)
        
        # NEES Calculation
        NEES_list.append(NEES(xstar.y[:,t_idx], x_hat_plus, P_plus))
        
        # NEES/NIS Debug
        # print("NEES: ", NEES_list[t_idx-1])
        # print("NIS: ", NIS_list[t_idx-1])
        
        t_idx += 1
        
    # PLOTS
    # EKF state
    if plot:
        fig, ax = plt.subplots(4,1,sharex=True)
        fig.suptitle('Extended Kalman Filter')
        ax[0].plot(t_list, np.squeeze(np.asarray(x_hat_plus_tot[0])), color='b')
        ax[0].set_title('X')
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel('Position (km)')
        ax[1].plot(t_list, np.squeeze(np.asarray(x_hat_plus_tot[1])), color='b')
        ax[1].set_title('X_dot')
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Velocity (km/s)')
        ax[2].plot(t_list, np.squeeze(np.asarray(x_hat_plus_tot[2])), color='b')
        ax[2].set_title('Y')
        ax[2].set_xlabel('Time (s)')
        ax[2].set_ylabel('Position (km)')
        ax[3].plot(t_list, np.squeeze(np.asarray(x_hat_plus_tot[3])), color='b')
        ax[3].set_title('Y_dot')
        ax[3].set_xlabel('Time (s)')
        ax[3].set_ylabel('Velocity (km/s)')

        ax[0].set_ylim([-7500,7500])
        ax[1].set_ylim([-9,9])
        ax[2].set_ylim([-7500,7500])
        ax[3].set_ylim([-9,9])

        # plot confidence bounds
        for i in range(4):
            upper_bounds    = [2*np.sqrt(array[i, i]) for array in P_list]  # Extract the diagonal element (i, i) at each timestep
            lower_bounds    = [-2*np.sqrt(array[i, i]) for array in P_list]  # Extract the diagonal element (i, i) at each timestep
            upper_bounds = upper_bounds + x_hat_plus_tot[i]
            lower_bounds = lower_bounds + x_hat_plus_tot[i]
            ax[i].plot(t_list, upper_bounds, 'r-')
            ax[i].plot(t_list, lower_bounds, 'r-')

        fig.legend(['Filter State', 'Filter 2sigma bounds'], loc='lower center', bbox_to_anchor=(0.5,0))

        plt.tight_layout()

    # Extended KF State Estimation Errors
    x_err    = []
    xdot_err = []
    y_err    = []
    ydot_err = []
    t_all = np.arange(0,int(T),dT)
    for t in t_all:
        if t in t_list:
            t_all_idx  = int(t/10)
            t_list_idx = t_list.index(t)
            x_err.append(xstar.y[0,t_all_idx] - np.squeeze(np.asarray(x_hat_plus_tot[0]))[t_list_idx])
            xdot_err.append(xstar.y[1,t_all_idx] - np.squeeze(np.asarray(x_hat_plus_tot[1]))[t_list_idx])
            y_err.append(xstar.y[2,t_all_idx] - np.squeeze(np.asarray(x_hat_plus_tot[2]))[t_list_idx])
            ydot_err.append(xstar.y[3,t_all_idx] - np.squeeze(np.asarray(x_hat_plus_tot[3]))[t_list_idx])

    if plot:
        fig, ax = plt.subplots(4,1,sharex=True)
        fig.suptitle('Extended KF State Estimation Errors')
        ax[0].plot(t_list, x_err, color='b')
        ax[0].set_title('X')
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel('Position (km)')
        ax[1].plot(t_list, xdot_err, color='b')
        ax[1].set_title('X_dot')
        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Velocity (km/s)')
        ax[2].plot(t_list, y_err, color='b')
        ax[2].set_title('Y')
        ax[2].set_xlabel('Time (s)')
        ax[2].set_ylabel('Position (km)')
        ax[3].plot(t_list, ydot_err, color='b')
        ax[3].set_title('Y_dot')
        ax[3].set_xlabel('Time (s)')
        ax[3].set_ylabel('Velocity (km/s)')

        # plot confidence bounds
        for i in range(4):
            upper_bounds    = [2*np.sqrt(array[i, i]) for array in P_list]  # Extract the diagonal element (i, i) at each timestep
            lower_bounds    = [-2*np.sqrt(array[i, i]) for array in P_list]  # Extract the diagonal element (i, i) at each timestep
            upper_bounds = upper_bounds
            lower_bounds = lower_bounds
            ax[i].plot(t_list, upper_bounds, 'r-')
            ax[i].plot(t_list, lower_bounds, 'r-')

        fig.legend(['Filter Error', 'Filter 2sigma bounds'], loc='lower center', bbox_to_anchor=(0.5,0))

        plt.tight_layout()
        plt.show()
        plt.close()
    
    est_errors = np.array([x_err, xdot_err, y_err, ydot_err])
    
    return x_hat_plus_tot, est_errors, P_list, NEES_list, NIS_list


# MAIN =====================================================================================

if __name__ == "__main__":
    # given parameters
    mu = 398600 # gravitational constant, km3/s2
    r0 = 6678   # orbital radius, km
    T = 14000   # HARD-CODED
    
    # Monte-Carlo parameters
    #np.random.seed(100)  # Random Set Seed
    num_mc_runs = 50     # Number of Monte-Carlo Runs
    alpha = 0.05         # Confidence
    
    T_tot = round(np.sqrt((4*(np.pi**2)*(r0**3))/mu))   # orbital period, s

    dT = 10 # 10 sec
    omegaE = 2*np.pi/86400  # rad/s
    RE = 6378   # earth radius, km

    # state definitions + initial conditions
    X = [r0]
    Xdot = [0]
    Y = [0]
    Ydot = [r0*np.sqrt(mu/(r0**3))]
    x0 = [X,Xdot,Y,Ydot]
    num_states = len(x0) # Number of States (x, xdot, y, ydot)
    num_meas = 3         # Number of Measurements (rho, rhodot, phi)

    # tracking stations
    tracking_station_data = tracking_stations(RE, omegaE)

    # simulate the linearize DT dynamics and measurement models
    # dt_linearized_state_sim(x0,dT,T)
    # dt_linearized_measurements_sim(x0,dT,T)

    # linearized kalman filter (LKF)
    # input files
    input_files_dir = os.path.join(os.getcwd(), 'orb_determ_data_csv')
    Qtrue = np.genfromtxt(os.path.join(input_files_dir, 'Qtrue.csv'), delimiter=',')     # process noise covar
    Rtrue = np.genfromtxt(os.path.join(input_files_dir, 'Rtrue.csv'), delimiter=',')     # measurement noise covar
    tvec  =(pd.read_csv(os.path.join(input_files_dir, 'tvec.csv'), header=None)).values.tolist()      # time vector
    measLabels = (pd.read_csv(os.path.join(input_files_dir, 'measLabels.csv'), header=None)).values.tolist()    # labels for measurements dataframe
    ydata = loadmat(os.path.join(input_files_dir,'orbitdeterm_finalproj_KFdata.mat'))['ydata'][0]

    ydata_sim = monte_carlo_measurements_tmt(x0,Qtrue,Rtrue,T,plot=False)
    
    # KF Tunings
    dx0 = [0,0.075,0,-0.021]
    pos_sigma = 10
    vel_sigma = 3
    P0_LKF = np.diag([pos_sigma**2,vel_sigma**2,pos_sigma**2,vel_sigma**2])
    Q_LKF = np.array([[5e-6, 0], [0, 5e-6]])
    
    start = time()
    NEES_array = []
    NIS_array = []

    for i in range(num_mc_runs):
        if i == 0:
            plot_arg = True
        else:
            plot_arg = False
        res_lkf = LKF(x0, dx0, P0_LKF, dT, T, Qtrue, Rtrue, ydata, Q_LKF=Q_LKF, plot=plot_arg)#, plot=True)
        NEES_array.append(res_lkf[3])
        NIS_array.append(res_lkf[4])
        if i % 10 == 9:
            print("LKF elapsed:", time() - start)
    #print(NIS_array[0])
        
    nis_lkf, stat_nis_lkf = NIS_Chi2_Test(np.asarray(NIS_array).T, num_meas, num_mc_runs, alpha, title="LKF NIS Testing Real", plot=True)    
    NEES_Chi2_Test(np.asarray(NEES_array).T, num_states, num_mc_runs, alpha, title="LKF NEES Testing Real", plot=True)
    
    print(stat_nis_lkf)
    
    # def run_ekf(Q_EKF):
        # NEES_array = []
        # NIS_array = []

        # try:
            # Q_EKF = Q_EKF.reshape((2,2))
        # except:
            # pass
    
        
        # for i in range(num_mc_runs):
            # if i == 0:
                # plot_arg = True
            # else:
                # plot_arg = False
            # res_ekf = EKF(x0, P0_EKF, dT, T, Qtrue, Rtrue, ydata_sim, Q_EKF = Q_EKF, plot=plot_arg)
            # NEES_array.append(res_ekf[3])
            # NIS_array.append(res_ekf[4])
        # print("EKF elapsed:", time() - start)
            
        # NIS_avg, res = NIS_Chi2_Test(np.asarray(NIS_array).T, num_meas, num_mc_runs, alpha, title="EKF NIS Testing")    
        # NEES_avg, res = NEES_Chi2_Test(np.asarray(NEES_array).T, num_states, num_mc_runs, alpha, title="EKF NEES Testing")

        # return -1 *np.average(NIS_avg)

    # Q_EKF = np.eye(2) * 1e-3
    # #Q_EKF = np.array([[1e-6, 1e-8], [1e-8, 1e-6]])
    # P0_EKF = np.diag([10,0.1,10,0.1])
    # run_ekf(Q_EKF)
    # bounds = [(1e-8, 1e-5)]*Q_EKF.size
    # #result = minimize(run_ekf, Q_EKF.flatten(), method='L-BFGS-B', bounds=bounds)


    #print(result)