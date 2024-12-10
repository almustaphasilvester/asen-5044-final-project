from matplotlib import pyplot as plt
import numpy as np
from scipy.linalg import expm
from scipy.integrate import odeint, solve_ivp
import sys

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

def nominal_orbit(t, T):
    max_vel = r0*np.sqrt(mu/(r0**3))

    theta = (t/T)*2*np.pi
    x = r0*np.cos(theta)
    xdot = -max_vel*np.sin(theta)
    y = r0*np.sin(theta)
    ydot = max_vel*np.cos(theta)

    return x, xdot, y, ydot

def nominal_measurements(t, T, i):
    # nominal orbit
    X, Xdot, Y, Ydot = nominal_orbit(t, T)

    # tracking station data
    Xi = tracking_station_data.Xi(t, i)
    Yi = tracking_station_data.Yi(t, i)
    Xidot = tracking_station_data.Xidot(t, i)
    Yidot = tracking_station_data.Yidot(t, i)

    # nominal measurements
    rho    = np.sqrt((X-Xi)**2 + (Y-Yi)**2)
    rhodot = ((X-Xi)*(Xdot-Xidot) + (Y-Yi)*(Ydot-Yidot))/rho
    phi    = np.arctan((Y-Yi)/(X-Xi))

    return rho, rhodot, phi

def dt_linearization_states(x_nom, dT):
    """
    Linearize CT system about specified equilibrium/nominal operating point and find correspond DT linearized model matrices
    """

    # nominal point
    X_nom = x_nom[0][0]
    Y_nom = x_nom[1][0]

    # CT nonlinear model Jacobians - evaluated at nominal point
    A_nom = np.array([[0,1,0,0]\
            ,[(-mu*(Y_nom**2 - 2*(X_nom**2)))/((X_nom**2 + Y_nom**2)**2.5), 0, 3*mu*X_nom*Y_nom/((X_nom**2 + Y_nom**2)**2.5), 0]\
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
    Y_nom = x_nom[1][0]
    Xdot_nom = x_nom[2][0]
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

def dt_linearized_state_sim(F,G,x_nom,dT,T):
    """
    Simulate linearized DT dynamics model near nominal point
    Validate against numerical integration routine (i.e. odeint)
    """

    # intial conditions - perturbation
    # (no process noise, measurement noise, or control input perturbations)
    dx = np.array([[0],[0],[0],[0]])
    du = np.zeros((2,1))

    # # initial conditions - state, i.e. nominal point
    x_nom_new = np.array(x_nom) + dx

    # # simulate linearized DT dynamics
    x_tot_list = x_nom_new
    for t in range(dT,int(T),dT):
        dx = F@dx + G@du

        # calculate new nominal orbit
        x, xdot, y, ydot = nominal_orbit(t, T_tot)
        x_nom_new = [[x],[xdot],[y],[ydot]]

        x_tot = x_nom_new + dx
        x_tot_list = np.concatenate((x_tot_list, np.array(x_tot)), axis=1)

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
    
    t_eval = np.linspace(0, T, int(T/10))
    soln = odeint(dyn_sys, np.asarray(x_nom).flatten(), t_eval)

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
    x_resid = np.squeeze(soln[:, 0] - np.asarray(x_tot_list[0]));
    xdot_resid = np.squeeze(soln[:, 1] - np.asarray(x_tot_list[1]));
    y_resid = np.squeeze(soln[:, 2] - np.asarray(x_tot_list[2]));
    ydot_resid = np.squeeze(soln[:, 3] - np.asarray(x_tot_list[3]));

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

def dt_linearized_measurements_sim(H,M,x_nom,dT,T):
    """
    Simulate linearized DT measurement model near nominal point
    Validate against numerical integration routine (i.e. odeint)
    """
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#f4a261', '#17becf', '#ff9896', '#c5b0d5']

    fig, ax = plt.subplots(4,1,sharex=True)
    fig.suptitle('Simulated Linearized DT Dynamics')
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

    # intial conditions - perturbation
    # (no process noise, measurement noise, or control input perturbations)
    dx = np.array([[0],[-0.057],[0],[-0.057]])
    du = np.zeros((2,1))

    # dictionary for visible stations plot
    visible_stations = {}
    for i in range(12):
        # # initial conditions - measurement, i.e. nominal point
        rho, rhodot, phi = nominal_measurements(0, T, i)
        y_nom_new = np.array([[rho],[rhodot],[phi]])

        # simulate linearized DT dynamics
        y_tot = y_nom_new
        print(H[i])
        for t in range(dT,int(T),dT):
            dy = (H[i])@dx + M@du

            # calculate new nominal measurements
            rho, rhodot, phi = nominal_measurements(0, T, i)
            y_nom_new = np.array([[rho],[rhodot],[phi]])

            # FOR NEXT TIMESTEP - calculate H,M matrices
            H,M = dt_linearization_measurements(x_nom, t)

            rho, rhodot, phi = nominal_measurements(0, T, i)
            y_nom_new = np.array([[rho],[rhodot],[phi]])
            y_tot = np.concatenate((y_tot, np.array(y_nom_new)), axis=1)

            timesteps = np.arange(0,int(T),dT)
            ax[0].plot(timesteps, np.squeeze(np.asarray(y_tot[0])), color=colors[i])
            ax[1].plot(timesteps, np.squeeze(np.asarray(y_tot[1])), color=colors[i])
            ax[2].plot(timesteps, np.squeeze(np.asarray(y_tot[2])), color=colors[i])
    
    plt.tight_layout()
    plt.show()
    plt.close()

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
    Y = [0]
    Xdot = [0]
    Ydot = [r0*np.sqrt(mu/(r0**3))]
    x_nom = [X,Y,Xdot,Ydot]

    # tracking stations
    tracking_station_data = tracking_stations(RE, omegaE)
    # initialize F,G,H,M matrices at t=0
    F, G = dt_linearization_states(x_nom, dT)
    H, M = dt_linearization_measurements(x_nom, 0)

    # simulate the linearize DT dynamics and measurement models
    dt_linearized_state_sim(F,G,x_nom,dT,T)
    #dt_linearized_measurements_sim(H,M,x_nom,dT,T)