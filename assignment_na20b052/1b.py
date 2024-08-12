import numpy as np
import matplotlib.pyplot as plt
import math
from mpi4py import MPI


L = 2.0
c = 1.0
dx = 0.002
dt = 0.0001

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def analytical_sol(x, t):
    ans = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] - c*t < 0.5 and x[i] - c*t > 0:
            ans[i] = np.sin(4 * math.pi * (x[i] - c*t))
    return ans

def upwind_scheme(u,left):
    u_new = u.copy()
    if rank == 0:
        u_new[0] = 0
    if rank > 0:
        u_new[0] = u[0] - c * (dt / dx) * (u[0] - left[0])
    for i in range(1, len(u)):
        u_new[i] = u[i] - c * (dt / dx) * (u[i] - u[i-1])
    return u_new

    
def QUICK_scheme(u,left, right):
    u_new = u.copy()
    if rank == 0:
        u_new[0] = 0
        u_new[1] = u[1] - c * (dt / dx) * (u[1] - u[0])
    else:
        u_new[0] = u[0] - c * (dt / dx) * ((3/8) * u[0] - (7/8) * left[-1] + (1/8) * left[-2] + (3/8) * u[1])
        u_new[1] = u[1] - c * (dt / dx) * ((3/8) * u[1] - (7/8) * u[0] + (1/8) * left[-1] + (3/8) * u[2])
    if rank != size-1:
        u_new[-1] = u[-1] - c * (dt / dx) * ((3/8) * u[-1] - (7/8) * u[-2] + (1/8) * u[-3] + (3/8) * right[0])

    for i in range(2, len(u) - 1):
        u_new[i] = u[i] - c * (dt / dx) * ((3/8) * u[i] - (7/8) * u[i-1] + (1/8) * u[i-2] + (3/8) * u[i+1])
    return u_new


T = [ 0, 0.5, 1]

n = int(L/dx) + 1 # number of points in the domain
llb = (n*rank)//size 
lrb = (n*(rank+1))//size - 1

x = np.linspace(llb*dx, lrb*dx, lrb-llb+1) 
if rank == 0:
    tx = np.linspace(0, L, n)

u0 = analytical_sol(x, 0)
u_upwind = u0.copy()
u_QUICK = u0.copy()
t1 = 0

for t in T:
    alsol = analytical_sol(x, t)
    u_analytical = comm.gather(alsol,root=0)

    while t1<t:
        # For the upwind scheme
        recv_left_upwind = []
        send_right_upwind = [u_upwind[-1]]
        if rank < size-1:
            comm.send(send_right_upwind, dest=rank+1)
        if rank > 0:
            recv_left_upwind = comm.recv(source=rank-1)
        u_upwind = upwind_scheme(u_upwind, recv_left_upwind)

        
        
        # For the QUICK scheme
        recv_left_QUICK = []
        recv_right_QUICK = []
        send_right_QUICK = [u_QUICK[-2],u_QUICK[-1]]
        send_left_QUICK = [u_QUICK[0]]
        if rank < size-1:
            comm.send(send_right_QUICK, dest=rank+1)
        if rank > 0:
            comm.send(send_left_QUICK, dest=rank-1)
        if rank > 0:
            recv_left_QUICK = comm.recv(source=rank-1)
        if rank < size-1:
            recv_right_QUICK = comm.recv(source=rank+1)
        u_QUICK = QUICK_scheme(u_QUICK, recv_left_QUICK, recv_right_QUICK)
    
        t1 += dt

    

    u_upg = comm.gather(u_upwind,root=0)
    u_QUICKg = comm.gather(u_QUICK,root=0)


    if rank ==0 :
        
        fig, axs = plt.subplots(figsize=(8, 12))
        u = [i for item in u_analytical for i in item] 
        axs.plot(tx, u, label='Analytical Solution')
        
        u1 = [i for item in u_upg for i in item]
        axs.plot(tx, u1, label='Upwind Scheme', color = 'black')
        
        u2 = [i for item in u_QUICKg for i in item]
        axs.plot(tx, u2, label='QUICK Scheme')
        axs.set_xlabel('x')
        axs.set_ylabel('u at time t = ' + str(t))
        axs.legend()

        plt.tight_layout()
        plt.show()