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
t = 0

#IC

x0 = np.array([r_o, 0, 0, r_o * np.sqrt(mu/(r_o ** 3))]).T
gs_0 = np.zeros((12,2))
for i in range(0,12):
    theta_0 = np.pi * i / 6
    gs_0[i,:] = np.array([R_E * np.cos(omega * t + theta_0), R_E * np.sin(omega * t + theta_0)]).T

def NL_DT_Dynamics(x_nom, dt):
    
    x = x_nom[0]
    dx = x_nom[1]
    y = x_nom[2]
    dy = x_nom[3]
    r = np.sqrt(x ** 2 + y ** 2)
    
    p = -(mu/(r**3)) * (r ** 3 - (3 * x ** 2 * r)) / (r ** 3)
    q = -(mu/(r**3)) * (r ** 3 - (3 * y ** 2 * r)) / (r ** 3)
    s = 3 * x * y * (mu/(r**5))
    
    A_nom = np.array([[0, 1, 0, 0], [p, 0, s, 0], [0, 0, 0, 1], [s, 0, q, 0]])
    B_nom = np.array([[0, 0], [1, 0], [0, 0], [0, 1]])
    
    A_shape = np.shape(A_nom)
    A_rank = A_shape[0]
    B_shape = np.shape(B_nom.T)
    B_rank = B_shape[0]
    
    A_hat_nom = np.concatenate((np.concatenate((A_nom, B_nom), axis = 1), np.concatenate((np.zeros(B_shape), np.zeros((B_rank,B_rank))), axis = 1)))
    
    print(A_hat_nom)
    e_A_hat_nom = expm(A_hat_nom * dt)
    F_nom = e_A_hat_nom[0:A_rank,0:A_rank]
    print("F_nom: \n", F_nom)
    G_nom = e_A_hat_nom[0:A_rank,A_rank:]
    print("G_nom: \n", G_nom)
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
    
    print("H_nom: \n", H_nom)
    print("M_nom: \n", M_nom)
    return H_nom, M_nom




x = np.zeros((4,5))
gs = np.zeros((12,2,5))

for k in range(0,1):
    NL_DT_Dynamics(x0, dt)
    for i in range(0,12):
        print(str(i) + "\n") 
        NL_DT_Measurements(x0, gs_0[i], omega)

