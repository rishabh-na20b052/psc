import numpy as np
import math
import matplotlib.pyplot as plt
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

dx = dy = 0.01
n = int(2/dx) + 1 # number of points in the domain

if rank == 0:
    tx = np.arange(-1, 1 + dx, dx)

llb = (n*rank)//size 
lrb = (n*(rank+1))//size - 1

x = np.linspace(llb*dx -1, lrb*dx - 1, lrb-llb+1) #local x
y = np.arange(-1, 1 + dy, dy)

# Function to calculate the value of phi at each point
def gs_update(phi, left, right, q, alpha):
    change = 0
    for i in range(1, len(phi) - 1):
        for j in range(1, len(phi[0]) - 1):
            if (i+j)%2 == alpha:
                continue
            x = phi[i,j]
            phi[i, j] = 0.25 * (phi[i+1, j] + phi[i-1, j] + phi[i, j+1] + phi[i, j-1] + (dx**2)*q[i, j])
            change = max(change, abs(phi [i, j] - x))

    if rank == size-1:
        for j in range(len(phi[-1])):
            if (j + len(phi) - 1 )%2 == alpha:
                continue
            x = phi[-1, j]
            phi[-1,j] = (4*phi[-2,j] - phi[-3,j])/3
            change = max(change, abs(phi[-1, j] - x))

    else :
        for j in range(1, len(phi[-1])-1):
            if (j + len(phi) - 1 )%2 == alpha:
                continue
            x = phi[-1, j]
            phi[-1, j] = 0.25 * (right[j] + phi[-2, j] + phi[-1, j+1] + phi[-1, j-1] + (dx**2)*q[-1, j])
            change = max(change, abs(phi[-1, j] - x))

    if rank != 0:
        for j in range(1, len(phi[0])-1):
            if (j + len(phi) - 1 )%2 == alpha:
                continue
            x = phi[0, j]
            phi[0, j] = 0.25 * (phi[1, j] + left[j] + phi[0, j+1] + phi[0, j-1] + (dx**2)*q[0, j])
            change = max(change, abs(phi[0, j] - x))
    return change

# Function to calculate the value of q at each point
def q_value(x, y):
    q = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            q[i, j] = x[i]**2 + y[j]**2
    return q    


q = q_value(x, y) # local q


phi = np.zeros((len(x), len(y)), dtype=np.float64) # Initial guess of phi is  0  everywhere
if rank == 0:
    for i in range(len(y)):
        phi[0,i] = math.sin(2*math.pi*y[i]) # Boundary condition at x = -1

change_red = 1
change_black = 1
tolerance = 1e-4

iterations = 0


left =  np.zeros(len(y), dtype=np.float64)
right =  np.zeros(len(y), dtype=np.float64)


while True:    
    if rank > 0:
        comm.send(phi[0, :], dest=rank-1, tag=1)
    if rank < size-1:
        right = comm.recv(source=rank+1, tag=1)
    if rank < size-1:
        comm.send(phi[-1, :], dest=rank+1, tag=2)
    if rank > 0:    
        left = comm.recv(source=rank-1, tag=2)
    

    change_red = gs_update(phi, left, right, q, 0)
    change_red = comm.allreduce(change_red, op=MPI.MAX)
    comm.Barrier()

    if rank > 0:
        comm.send(phi[0, :], dest=rank-1, tag=1)
    if rank < size-1:
        right = comm.recv(source=rank+1, tag=1)
    if rank < size-1:
        comm.send(phi[-1, :], dest=rank+1, tag=2)
    if rank > 0:    
        left = comm.recv(source=rank-1, tag=2)

    change_black = gs_update(phi, left, right, q, 1)
    change_black = comm.allreduce(change_black, op=MPI.MAX)
    comm.Barrier()


    iterations += 1

    if max(change_red,change_black) < tolerance :
        print(rank, change_red, iterations)
        break


comm.Barrier()
phixx = np.zeros((len(x)), dtype=np.float64)
for i in range(len(x)):
    phixx[i] = phi[i, 100]
phig = comm.gather(phixx, root=0)

if rank == 0:
    print("Number of iterations for the solution to converge to 10^-4 is", iterations)
    # Plotting the numerical solution of phi vs x for y = 0.0
    fig, axs = plt.subplots(figsize=(8, 12))
    axs.plot(tx, [i for phi in phig for i in phi], label="Gauss-Seidel", linewidth=3)
    phix = np.loadtxt("phi_x.txt")
    axs.plot(tx, phix, label="Jacobi", linestyle='--')
    axs.set_xlabel('x')
    axs.set_ylabel('phi at y = 0.0')
    axs.legend()
    plt.tight_layout()
    plt.show()

if rank == 2:
    fig, axs = plt.subplots(figsize=(8, 12))
    axs.plot(y, phi[0], label="Gauss-Seidel", linewidth=3)
    phiy = np.loadtxt("phi_y.txt")
    axs.plot(y, phiy, label="Jacobi", linestyle='--') 
    axs.set_xlabel('y')
    axs.set_ylabel('phi at x = 0.0')
    axs.legend()

    plt.tight_layout()
    plt.show()