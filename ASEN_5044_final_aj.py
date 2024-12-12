from matplotlib import pyplot as plt
import numpy as np
import os, sys
import pandas as pd
from scipy.linalg import expm
from scipy.integrate import odeint, solve_ivp

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

def dyn_sys(state, t):
        """
        define dynamical system with given equations
        note: NO CTRL ACCEL OR DISTURBANCES
        """
        x, v_x, y, v_y = state
    
        # Compute the derivatives
        dx_dt = v_x
        ddx_dt = (-mu * x) / (x**2 + y**2)**(3/2)
        dy_dt = v_y
        ddy_dt = (-mu * y) / (x**2 + y**2)**(3/2)
        
        return [dx_dt, ddx_dt, dy_dt, ddy_dt]

def dyn_measurements(state, station_state):
        """
        define dynamical system with given equations
        note: NO CTRL ACCEL OR DISTURBANCES
        """
        x, v_x, y, v_y = state
        xi, v_xi, yi, v_yi = station_state

        # Compute the range (rho)
        rho = np.sqrt((x - xi)**2 + (y - yi)**2)
        
        # Compute the radial velocity (dot(rho))
        rho_dot = (((x - xi) * (v_x - v_xi)) + ((y - yi) * (v_y - v_yi))) / rho
        
        # Compute the elevation angle (phi)
        phi = np.arctan2(y - yi, x - xi)
        
        return rho, rho_dot, phi

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

def dt_linearized_state_sim(x0,dT,T):
    """
    Simulate linearized DT dynamics model near nominal point
    Validate against numerical integration routine (i.e. odeint)
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
    for t in range(dT,int(T),dT):
        # calculate dx at time k, using dx at k-1
        dx = F@dx + G@du

        # calculate new nominal orbit, ie at time k
        x, xdot, y, ydot = nominal_orbit(t)
        x_nom = [[x],[xdot],[y],[ydot]]

        x_tot = x_nom + dx
        x_tot_list = np.concatenate((x_tot_list, np.array(x_tot)), axis=1)

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

    # validate using scipy.integrate.odeint
    t_eval = np.linspace(0, T, int(T/10))
    soln = odeint(dyn_sys, np.asarray(x0).flatten(), t_eval)

    # Extract the results from the solution
    x_vals   = soln[:, 0]
    v_x_vals = soln[:, 1]
    y_vals   = soln[:, 2]
    v_y_vals = soln[:, 3]

    # Plot the results
    timesteps = np.arange(0,int(T),1)
    fig, ax = plt.subplots(4,1,sharex=True)
    fig.suptitle('Full Nonlinear Dynamics Simulation (using scipy.odeint)')
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

    # residuals
    x_resid = np.squeeze(np.asarray(x_tot_list[0] - soln[:, 0]));
    xdot_resid = np.squeeze(np.asarray(x_tot_list[0] - soln[:, 1]));
    y_resid = np.squeeze(np.asarray(x_tot_list[0] - soln[:, 2]));
    ydot_resid = np.squeeze(np.asarray(x_tot_list[0] - soln[:, 3]));

    # Plot the residuals
    timesteps = np.arange(0,int(T),1)
    fig, ax = plt.subplots(4,1,sharex=True)
    fig.suptitle('Residuals (Linear DT Simulation vs Nonlinear Dynamics Simulation)')
    ax[0].plot(t_eval, x_resid)
    ax[0].set_title('X')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Position (km)')
    ax[1].plot(t_eval, xdot_resid)
    ax[1].set_title('X_dot')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Velocity (km/s)')
    ax[2].plot(t_eval, y_resid)
    ax[2].set_title('Y')
    ax[2].set_xlabel('Time (s)')
    ax[2].set_ylabel('Position (km)')
    ax[3].plot(t_eval, ydot_resid)
    ax[3].set_title('Y_dot')
    ax[3].set_xlabel('Time (s)')
    ax[3].set_ylabel('Velocity (km/s)')
    plt.tight_layout()
    plt.show()
    plt.close()

def dt_linearized_measurements_sim(x0,dT,T):
    """
    Simulate linearized DT measurement model near nominal point
    Validate against numerical integration routine (i.e. odeint)
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

            if (y_tot[2][0] >= el_L_lim) and (y_tot[2][0] <= el_H_lim):
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

    # validate using scipy.integrate.odeint
    t_eval = np.linspace(0, T, int(T/10))
    soln = odeint(dyn_sys, np.asarray(x0).flatten(), t_eval)

    fig, ax = plt.subplots(4,1,sharex=True)
    fig.suptitle('Full Nonlinear Measurements Simulation (using scipy.odeint)')
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
            el_L_lim = (-0.5*np.pi)+theta
            el_H_lim = (0.5*np.pi)+theta

            state = soln[t_idx, :]
            rho, rho_dot, phi = dyn_measurements(state, station_state)

            if (phi >= el_L_lim) and (phi <= el_H_lim):
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

    plt.tight_layout()
    plt.show()
    plt.close()

def LKF():
    pass

if __name__ == "__main__":
    # given parameters
    mu = 398600 # gravitational constant, km3/s2
    r0 = 6678   # orbital radius, km
    T = 14000   # HARD-CODED
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

    # tracking stations
    tracking_station_data = tracking_stations(RE, omegaE)

    # simulate the linearize DT dynamics and measurement models
    dt_linearized_state_sim(x0,dT,T)
    dt_linearized_measurements_sim(x0,dT,T)

    # linearized kalman filter (LKF)
    # input files
    input_files_dir = os.path.join(os.getcwd(), 'orb_determ_data_csv')
    Qtrue = np.genfromtxt(os.path.join(input_files_dir, 'Qtrue.csv'), delimiter=',')     # process noise covar
    Rtrue = np.genfromtxt(os.path.join(input_files_dir, 'Rtrue.csv'), delimiter=',')     # measurement noise covar
    tvec  =(pd.read_csv(os.path.join(input_files_dir, 'tvec.csv'), header=None)).values.tolist()      # time vector
    measLabels = (pd.read_csv(os.path.join(input_files_dir, 'measLabels.csv'), header=None)).values.tolist()    # labels for measurements dataframe
    ydata = pd.read_csv(os.path.join(input_files_dir, 'ydata.csv'), header=None, usecols=list(range(int(T/dT) + 1)), names=tvec[0])         # measurements
    ydata['index'] = measLabels[0]
    ydata = ydata.set_index('index', drop=True)