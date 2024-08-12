import numpy as np
import math
import matplotlib.pyplot as plt

dx = dy = 0.1
x = np.arange(-1, 1 + dx, dx)
y = np.arange(-1, 1 + dy, dy)


# Function to calculate the value of phi at each point
def jacobi_update(phi, q):
    phi_new = phi.copy()
    change = 0
    for i in range(1, len(phi) - 1):
        for j in range(1, len(phi[0]) - 1):
            phi_new[i, j] = 0.25 * (phi[i+1, j] + phi[i-1, j] + phi[i, j+1] + phi[i, j-1] + (dx**2)*q[i, j])
            change = max(change, abs(phi_new[i, j] - phi[i, j]))
    for j in range(len(phi[-1])):
        phi_new[-1,j] = (4*phi_new[-2,j] - phi_new[-3,j])/3
        change = max(change, abs(phi_new[-1,j] - phi[-1,j]))
        
    return phi_new, change

# Function to calculate the value of q at each point
def q_value(x, y):
    q = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            q[i, j] = x[i]**2 + y[j]**2
    return q    

q = q_value(x, y)

phi = np.zeros((len(x), len(y)), dtype=np.float64) # Initial guess of phi is  0  everywhere
for i in range(len(y)):
    phi[0,i] = math.sin(2*math.pi*y[i]) # Boundary condition at x = -1


tolerance = 1e-4
change = 1 # intial value
iterations = 0

while change > tolerance:
    phi, change = jacobi_update(phi, q)
    iterations += 1
    if iterations % 100 == 0:
        print("Iterations:", iterations, "Change:", change)



print("Number of iterations for the solution between successive iterations to converge to 10^-4 is", iterations)

# Plotting the numerical solution of phi vs x for y = 0.0
fig, axs = plt.subplots(figsize=(8, 12))
axs.plot(x, phi[:, 10])
axs.set_xlabel('x')
axs.set_ylabel('phi at y = 0.0')
plt.tight_layout()
plt.show()
# phix = phi[:, 100]
# phiy = phi[100, :]
# #saving the data
# np.savetxt("phi_x.txt", phix)
# np.savetxt("phi_y.txt", phiy)



# Plotting the numerical solution of phi vs y for x = 0.0
fig, axs = plt.subplots(figsize=(8, 12))
axs.plot(y, phi[10, :])
axs.set_xlabel('y')
axs.set_ylabel('phi at x = 0.0')
plt.tight_layout()
plt.show()
