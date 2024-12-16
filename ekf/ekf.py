import pandas as pd
import time
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Calculate Nominal Points
def calculate_nominal_points(t, T):
    theta_nom = (t/T)*(2*np.pi)
    x_nom = np.cos(theta_nom) * x0
    y_nom = np.sin(theta_nom) * x0
    xd_nom = -1*np.sin(theta_nom) * x0 * (mu_static/x0**3)**(1/2)
    yd_nom = np.cos(theta_nom) * x0 * (mu_static/x0**3)**(1/2)
    return x_nom, y_nom, xd_nom, yd_nom

# Calculate Tracking Stations
def caclulate_tracking_stations(station_number, t, T):
    theta = (station_number)*(np.pi/6)
    x_s = Re*np.cos(w * t + theta)
    y_s = Re*np.sin(w * t + theta)
    xd_s = -Re*w*np.sin(w * t + theta)
    yd_s = Re*w*np.cos(w * t + theta)
    return x_s, y_s, xd_s, yd_s

def ode_simulation():
    noise = np.random.multivariate_normal([0,0], np.eye(2) * 1e-10)

    # Parameters
    u = 398600     # Constant parameter u
    u1 = 0    # Constant parameter u1
    u2 = 0     # Constant parameter u2
    w1 = noise[0]    # Constant parameter w1
    w2 = noise[1]     # Constant parameter w2

    # Define the system of first-order ODEs
    def system(t, state):
        x, v_x, y, v_y = state
        
        # Compute the derivatives
        dx_dt = v_x
        dvx_dt = (-u * x) / (x**2 + y**2)**(3/2) + u1 + w1
        dy_dt = v_y
        dvy_dt = (-u * y) / (x**2 + y**2)**(3/2) + u2 + w2
        
        return [dx_dt, dvx_dt, dy_dt, dvy_dt]

    # Initial conditions: x(0), dx(0), y(0), dy(0)
    initial_conditions = [6678, 0.0, 0.0, 7.725835197559566]  # For example, starting at x=1, y=0, with zero velocity

    # Time span for the solution
    t_span = (0, 14001)  # From t=0 to t=10
    t_eval = np.linspace(0, 14000, 14000)  # Times at which we want the solution

    # Solve the system using solve_ivp
    solution = solve_ivp(system, t_span, initial_conditions, t_eval=t_eval, reltol=1e-5 )

    # Extract the results from the solution
    x_vals = solution.y[0]
    v_x_vals = solution.y[1]
    y_vals = solution.y[2]
    v_y_vals = solution.y[3]

    # Plot the results
    plt.figure(figsize=(12, 6))

    # Plot x and y over time
    plt.subplot(2, 1, 1)
    plt.plot(solution.t, x_vals, label='x(t)', color='b')
    plt.plot(solution.t, y_vals, label='y(t)', color='r')

    plt.xlabel('Time (t)')
    plt.ylabel('Position')
    plt.legend()
    plt.title('Positions x(t) and y(t)')

    # Plot the velocities v_x and v_y over time
    plt.subplot(2, 1, 2)
    plt.plot(solution.t, v_x_vals, label='dx/dt (v_x)', color='b')
    plt.plot(solution.t, v_y_vals, label='dy/dt (v_y)', color='r')
    plt.xlabel('Time (t)')
    plt.ylabel('Velocity')
    plt.legend()
    plt.title('Velocities dx/dt (v_x) and dy/dt (v_y)')


    plt.tight_layout()
    plt.show()

    return solution

def ekf_simulation(t_span, dt):
    # Initialize state and covariance
    x_est = np.array([[x0], [xd0], [y0], [yd0]])  # Initial state estimate (4x1)
    #x_est = np.array([[0],[0.075],[0],[-0.021]])
    P = np.eye(4) * 1/100  # Initial covariance estimate (4x4)
    
    # Storage for results
    x_estimates = [x_est]
    P_history = [P]
    
    # Process and measurement noise covariances
    Q = np.eye(2) * 1e-10  # Process noise (4x4)
    R = np.array([[0.01,0,0],[0,1,0],[0,0,0.01]])  # Measurement noise for each station (3x3)


    # Get Jacobian A

    x_nom, y_nom, dx_nom, dy_nom = calculate_nominal_points(0, T_orbit)

    A_k = np.array(A.subs({
    x: x_nom, dx: dx_nom,
    y: y_nom, dy: dy_nom,
    mu: mu_static})).astype(np.float64)
        
    F_k = np.identity(np.shape(A_k)[0]) + dt*A_k
    
    # Simulation loop
    for k in range(dt, t_span, dt):

        if k%1000 == 0:
            print(f"Processing time step {k} of {t_span}")
        
        # 1. Prediction Step
        x_nom, y_nom, dx_nom, dy_nom = calculate_nominal_points(k, T_orbit)
        
        x_pred = np.array([[x_nom], [dx_nom], [y_nom], [dy_nom]]) #F_k @ x_est
        P_pred = F_k @ P @ F_k.T + Omega @ Q @ Omega.T
        
        # 2. Update Step
        # Process all tracking stations at once
        # Get all station positions
        stations_pos = np.array([caclulate_tracking_stations(i, k, T_orbit) for i in range(12)])
        xs, ys, dxs, dys = stations_pos[:,0], stations_pos[:,1], stations_pos[:,2], stations_pos[:,3]

        z_stack = []
        out_of_range = []
        for i in range(12):
            phi_t = f7.xreplace({x: x_pred[0,0]+x_nom, y: x_pred[2,0]+y_nom, x_s: xs[i], y_s: ys[i]})
            theta_t = np.arctan2(ys[i], xs[i])
            if phi_t >= (-np.pi*0.5)+theta_t and phi_t <= (np.pi*0.5)+theta_t:
                pass
            else:
                out_of_range.append(i)
        
        if len(out_of_range) == 12:
            print(f"All stations out of range at time step {k}")
            x_estimates.append(np.array([[x_nom], [dx_nom], [y_nom], [dy_nom]]))
            P_history.append(P_pred)
            continue

        # Remove out of range stations from position arrays
        xs = np.delete(xs, out_of_range)
        ys = np.delete(ys, out_of_range) 
        dxs = np.delete(dxs, out_of_range)
        dys = np.delete(dys, out_of_range)
        # Calculate measurement Jacobians for all stations (12x3x4)
        H = np.array([np.array(C.xreplace({
            x: x_nom, dx: dx_nom,
            y: y_nom, dy: dy_nom,
            x_s: xs[i], dx_s: dxs[i],
            y_s: ys[i], dy_s: dys[i]
        }).tolist()) for i in range(len(xs))])

        # Generate simulated measurements for all stations (12x3x1)
        rho = ((x_nom-xs)**2 + (y_nom-ys)**2)**(1/2)
        rhodot = ((x_nom-xs)*(dx_nom-dxs) + 
                 (y_nom-ys)*(dy_nom-dys))/rho
        phi = np.arctan2((y_nom-ys), (x_nom-xs))
        rho = np.array(rho)
        rhodot = np.array(rhodot)
        z = np.stack([rho, rhodot, phi], axis=1)[:,:,np.newaxis]

        # Stack measurements and Jacobians
        H_stack = np.vstack(H)  # (36x4)
        h_stack = np.vstack(z)  # (36x1)
        R_stack = np.kron(np.eye(len(xs)), R)  # (36x36)

        # Kalman update with stacked measurements
        S = H_stack @ P_pred @ H_stack.T + R_stack  # (36x36)
        S = S.astype(np.float64)
        K = P_pred @ H_stack.T @ np.linalg.inv(S) #S.inv()  # (4x36)

        


        # Predicted measurements
        z_stack = np.vstack([[f5.xreplace({x: solution.y[0][k%10], y: solution.y[2][k%10], x_s: xs[_], y_s: ys[_]}),
                             f6.xreplace({x: solution.y[0][k%10], y: solution.y[2][k%10], dx: solution.y[1][k%10], dy: solution.y[3][k%10], x_s: xs[_], y_s: ys[_], dx_s: dxs[_], dy_s: dys[_]}),
                             f7.xreplace({x: solution.y[0][k%10], y: solution.y[2][k%10], x_s: xs[_], y_s: ys[_]})
                             ] for _ in range(len(xs))])  # Stack measurements 12 times for 36 total rows

        z_stack = np.array(z_stack).reshape((len(xs)*3,1))
        
        # Update state and covariance
        x_pred = x_pred + K @ (z_stack - h_stack)  # (4x1)
        x_pred = np.array(x_pred).astype(np.float64)
        P_pred = (np.eye(4) - K @ H_stack) @ P_pred  # (4x4)

        # Evaluate Jacobian A at current state estimate
        A_k = np.array(A.subs({
            x: x_nom, dx: dx_nom,
            y: y_nom, dy: dy_nom,
            mu: mu_static
        })).astype(np.float64)
        
        F_k = np.identity(np.shape(A_k)[0]) + dt*A_k #expm(A_k * dt)
        
        # Store results after processing all stations
        x_est = x_pred
        P = P_pred
        #x_state = np.array([[x_nom], [dx_nom], [y_nom], [dy_nom]])
        x_estimates.append(x_est)
        P_history.append(P)
        
        # NEES Calculation
        NEES_list.append(NEES(xstar.y[:,t_idx-1], x_est, P))
    
    return np.array(x_estimates), np.array(P_history)

if __name__ == "__main__":

    # Define Symbols
    x, y, dx, dy, x_s, dx_s, y_s, dy_s, mu = sp.symbols("x y dx dy x_s dx_s y_s dy_s mu")
    u1, u2, w1, w2 = sp.symbols("u1 u2 w1 w2")

    # Define System Equations
    f1 = -mu*x*(1/((x**2 + y**2)**(1/2)))**3 + u1 + w1
    f2 = -mu*y*(1/((x**2 + y**2)**(1/2)))**3 + u2 + w2
    f3 = dx
    f4 = dy
    f5 = ((x-x_s)**2 + (y-y_s)**2)**(1/2)
    f6 = ((x - x_s)*(dx - dx_s) + (y - y_s)*(dy - dy_s))/((x - x_s)**2 + (y - y_s)**2)**(1/2)
    f7 = sp.atan2((y-y_s), (x-x_s))

    # Define State Vector
    f_x = sp.Matrix([f3, f1, f4, f2])

    # Define Jacobians
    A = f_x.jacobian([x, dx, y, dy])
    B = f_x.jacobian([u1, u2])

    # Define Output Vector
    f_y = sp.Matrix([f5, f6, f7])

    # Define Jacobians
    C = f_y.jacobian([x, dx, y, dy])
    D = f_y.jacobian([u1, u2])

    Omega = f_x.jacobian([w1, w2])
    Omega = np.array(Omega.tolist())

    # Define Initial Conditions
    mu_static = 398600
    w = 2*np.pi/86400
    Re = 6378
    x0 = 6678
    xd0 = 0
    y0 = 0
    yd0 = x0*((mu_static/x0**3)**(1/2))

    T_orbit = round(np.sqrt((4*(np.pi**2)*(x0**3))/mu_static))

    # Get ODE Solution
    solution = ode_simulation()

    # Run Simulation
    est_x, est_P = ekf_simulation(14000, 10)

    # Plot estimated states over time
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15,10))
    time = np.arange(0, 14000, 10)  # Create time array from 0 to 14000s with 10s steps

    # Plot x position
    ax1.plot(time, est_x[:,0,0])
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Estimated X Position (km)')
    ax1.set_title('EKF Estimated X Position vs Time')
    ax1.grid(True)

    # Plot x velocity 
    ax2.plot(time, np.array(est_x)[:,1,0])
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Estimated X Velocity (km/s)')
    ax2.set_title('EKF Estimated X Velocity vs Time')
    ax2.grid(True)

    # Plot y position
    ax3.plot(time, np.array(est_x)[:,2,0])
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Estimated Y Position (km)')
    ax3.set_title('EKF Estimated Y Position vs Time')
    ax3.grid(True)

    # Plot y velocity
    ax4.plot(time, np.array(est_x)[:,3,0])
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Estimated Y Velocity (km/s)')
    ax4.set_title('EKF Estimated Y Velocity vs Time')
    ax4.grid(True)

    plt.tight_layout()
    plt.show()