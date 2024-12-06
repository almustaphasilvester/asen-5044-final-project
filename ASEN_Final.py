import numpy as np
from scipy.linalg import expm
from scipy.stats import multivariate_normal
from scipy.io import loadmat
from matplotlib import pyplot as plt

dt = 10 #sec
R_E = 6378 #km
r_o = R_E + 300
mu = 398600 #km^3/s^2
omega = 2 * np.pi / 86400 #rad/s
omega = 2 * np.pi / 86400 #rad/s

#IC

x0 = np.array([r_o, 0, 0, r_o * np.sqrt(mu/(r_o ** 3))])
#print(x0)
delta_x0 = np.array([0,0.075,0,-0.021])
gs_0 = np.zeros((2,12))
y0 = np.zeros((3,12))

for i in range(0,12):
    theta_0 = np.pi * i / 6
    gs_0[:,i] = np.array([R_E * np.cos(theta_0), R_E * np.sin(theta_0)])
    x_si_0 = gs_0[0,i]
    y_si_0 = gs_0[1,i]
    dx_si_0 = -omega * y_si_0
    dy_si_0 = omega * x_si_0
    rho_0 = np.sqrt((x0[0] - x_si_0) ** 2 + (x0[2] - y_si_0) ** 2)
    drho_0 = ((x0[0] - x_si_0) * (x0[1] - dx_si_0) + (x0[2] - y_si_0) * (x0[3] - dy_si_0)) / rho_0
    phi_0 = np.arctan((x0[2] - y_si_0) / (x0[0] - x_si_0))
    y0[:,i] = np.array([rho_0, drho_0, phi_0])
    

def NL_DT_Dynamics(x_nom, dt):
    
    x = x_nom[0]
    y = x_nom[2]
    r = np.sqrt(x ** 2 + y ** 2)
    
    p = -(mu/(r**3)) * ((r ** 3 - (3 * x ** 2 * r)) / (r ** 3))
    q = -(mu/(r**3)) * ((r ** 3 - (3 * y ** 2 * r)) / (r ** 3))
    s = (3 * x * y * mu)/(r ** 5)
    
    A_nom = np.array([[0, 1, 0, 0], [p, 0, s, 0], [0, 0, 0, 1], [s, 0, q, 0]])
    B_nom = np.array([[0, 0], [1, 0], [0, 0], [0, 1]])
    
    A_shape = np.shape(A_nom)
    A_rank = A_shape[0]
    B_shape = np.shape(B_nom.T)
    B_rank = B_shape[0]
    
    A_hat_nom = np.concatenate((np.concatenate((A_nom, B_nom), axis = 1), np.concatenate((np.zeros(B_shape), np.zeros((B_rank,B_rank))), axis = 1)))
    
    #print(A_hat_nom)
    e_A_hat_nom = expm(A_hat_nom * dt)
    F_nom = e_A_hat_nom[0:A_rank,0:A_rank]
    G_nom = e_A_hat_nom[0:A_rank,A_rank:]
    
    #print("F_nom: \n", F_nom)
    #print("G_nom: \n", G_nom)
    return F_nom, G_nom

def NL_DT_Measurements(x_nom, gs_nom, omega):
    
    x = x_nom[0]
    dx = x_nom[1]
    y = x_nom[2]
    dy = x_nom[3]
    x_si = gs_nom[0]
    y_si = gs_nom[1]
    dx_si = -omega * y_si
    dy_si = omega * x_si
    rho = np.sqrt((x - x_si) ** 2 + (y - y_si) ** 2)
    drho = ((x - x_si) * (dx - dx_si) + (y - y_si) * (dy - dy_si)) / rho
    
    p = (rho * (dx - dx_si) - drho * (x - x_si)) / (rho ** 2)
    q = (rho * (dy - dy_si) - drho * (y - y_si)) / (rho ** 2)
    u = (x - x_si) / rho
    v = (y - y_si) / rho
    
    C_nom = np.array([[u, 0, v, 0], [p, u, q, v], [-(v / rho), 0, (u / rho), 0]])
    D_nom = np.zeros((3,2))
    
    H_nom = C_nom
    M_nom = D_nom
    
    #print("H_nom: \n", H_nom)
    #print("M_nom: \n", M_nom)
    return H_nom, M_nom

NL_DT_Dynamics(x0, dt)
for i in range(0,12):
    NL_DT_Measurements(x0, gs_0[:,i], omega) 

k = 1400
N = k*dt

x_nom = x0 + delta_x0
x = np.zeros((4,dt*k+1))
delta_x = np.zeros((4,dt*k+1))
gs = np.zeros((2,12,dt*k+1))
y = np.zeros((3,12,dt*k+1))
x[:,0] = x_nom
delta_x[:,0] = delta_x0
gs[:,:,0] = gs_0
y[:,:,0] = y0


for i in range(N):
    if i % dt == 0:
        print(x_nom)
        F_nom, G_nom = NL_DT_Dynamics(x_nom, dt)
        print(F_nom)
        r = np.sqrt(x_nom[0] ** 2 + x_nom[2] ** 2)
        delta_x_n = np.array([x_nom[1] * dt, -((mu*x_nom[0])/(r**3)) * dt, x_nom[3] * dt, -((mu*x_nom[2])/(r**3)) * dt])
        x_nom += delta_x_n
        delta_x[:,i] = delta_x_n #x[:,i] - x_nom
    t = i
    print(delta_x[:,i])
    delta_x[:,i+1] = F_nom @ delta_x[:,i]
    #import pdb; pdb.set_trace()
    x[:,i+1] = x_nom + delta_x[:,i+1]
    for j in range(0,12):
        theta_0 = np.pi * j / 6
        gs[:,j,i+1] = np.array([R_E * np.cos(omega * t + theta_0), R_E * np.sin(omega * t + theta_0)])
        H_nom, M_nom = NL_DT_Measurements(x[:,i+1], gs[:,j,i+1], omega)
        y[:,j,i+1] = y[:,j,i] + H_nom @ delta_x[:,i+1]
statevector = ["x (km)", "$\dot{x}$ (km/s)", "y (km)", "$\dot{y}$ (km/s)"]

fig, axs = plt.subplots(4, sharex=True, figsize=(10,8))
fig.suptitle("DT State")
n = N
dn = n/20

for i in range(len(axs)):
    axs[i].plot(range(n), x[i,:n])#, marker=".")
    axs[i].set_ylabel(statevector[i])
    axs[i].set_xticks(np.arange(0, n+1, dn))
    #axs[i].set_yticks(np.arange(-10000, 10001, 2500))
    axs[i].grid(which = "major")
plt.xlabel("k")
plt.savefig("Finals DT State.png")

fig, axs = plt.subplots(4, sharex=True, figsize=(10,8))
fig.suptitle("DT State Residuals")
n = N
dn = n/20

for i in range(len(axs)):  
    axs[i].plot(range(n), delta_x[i,:n])#, marker=".")
    axs[i].set_ylabel(statevector[i])
    axs[i].set_xticks(np.arange(0, n+1, dn))
    #axs[i].set_yticks(np.arange(-10000, 10001, 2500))
    axs[i].grid(which = "major")
plt.xlabel("k")
plt.savefig("Finals DT State Residuals.png")