import numpy as np
import matplotlib.pyplot as plt
import math


L = 2.0
c = 1.0
dx = 0.002
dt = 0.0001

# Function to calculate the analytical solution
def analytical_sol(x, t):
    ans = np.zeros(len(x))
    ans[int(c*t/dx) :int((0.5 + c*t)/dx)] = np.sin(4 * math.pi * x[:int(0.5/dx)])
    return ans

def upwind_scheme(u):
    u_new = u.copy()
    for i in range(1, len(u) - 1):
        u_new[i] = u[i] - c * (dt / dx) * (u[i] - u[i-1])
    return u_new
    
def QUICK_scheme(u):
    u_new = u.copy()
    # Boundary condition using upwind scheme
    u_new[1] = u[1] - c * (dt / dx) * (u[1] - u[0])
    for i in range(2, len(u) - 1):
        u_new[i] = u[i] - c * (dt / dx) * ((3/8) * u[i] - (7/8) * u[i-1] + (1/8) * u[i-2] + (3/8) * u[i+1])
    return u_new

T = [0.5,1]

if __name__ == "__main__":
    x = np.arange(0, L + dx, dx)
    u = np.zeros(len(x))
    u[:int(0.5/dx)] = np.sin(4 * math.pi * x[:int(0.5/dx)])
    u[int(0.5/dx):] = 0

    u_QUICK = u.copy()
    u_upwind = u.copy()

    t1 = 0
    for t in T:
        while t1<t:
            u_upwind = upwind_scheme(u_upwind)
            u_QUICK = QUICK_scheme(u_QUICK)
            t1 += dt
        
        u_analytical = analytical_sol(x, t)
        fig, axs = plt.subplots(figsize=(8, 12))

        axs.plot(x, u_analytical, label='Analytical Solution')
        axs.plot(x, u_upwind, label='Upwind Scheme', color= 'black')
        axs.plot(x, u_QUICK, label='QUICK Scheme')
        axs.set_xlabel('x')
        axs.set_ylabel('u at time t = ' + str(t))
        axs.legend()

        plt.tight_layout()
        plt.show()