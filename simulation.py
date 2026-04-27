import numpy as np
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 

# physical constants
hbar = 1
m = 1

# system parameters
L = 1
tfinal = 10

dx = 0.05
dt = dx**2

J = round(2*L/dx) + 1
N = round(tfinal/dt) + 1

x = np.linspace(-L, L, J)

# infinite square well
# a = 1
# V = np.where((x < -a) | (x > a), np.inf, 0)
V = np.zeros(J)

psi = np.zeros((J, N), dtype=complex)

# Gaussian initial state
psi[:,0] = np.exp(-20*x**2)
psi[:,0] /= np.linalg.norm(psi)

# since Dirichlet boundary conditions are 0, only update interior
psi[1,0] = 0
psi[-1,0] = 0

# matrix equation
alpha = hbar**2/(4*m*dx**2)
beta = 1j*hbar/dt

diag = np.full(J-2-1, alpha)

A = np.diag(beta - 2*alpha - V[1:-1]/2) + np.diag(diag, 1) + np.diag(diag, -1)
B = np.diag(beta + 2*alpha + V[1:-1]/2) - np.diag(diag, 1) - np.diag(diag, -1)

# update state
lu, piv = lu_factor(A)
for n in range(N-1):
    psi[1:-1,n+1] = lu_solve((lu, piv), B @ psi[1:-1,n])

# plot animation
fig = plt.figure() 
ax = plt.axes(xlim =(-L, L), ylim =(0, 1)) 

pdf, = ax.plot([], [], label='Probability density')
re, = ax.plot([], [], label='Real part')
im, = ax.plot([], [], label='Imaginary part')

text = ax.text(-0.95, 0.8, '$t = 0.00$')

ax.set_xlabel('$x$')
ax.set_ylabel('$|\\Psi|^2$')
ax.legend()

def update(i):
    y_pdf = np.abs(psi[:,i])**2
    y_re = np.real(psi[:,i])
    y_im = np.imag(psi[:,i])

    pdf.set_data(x, y_pdf)
    re.set_data(x, y_re)
    im.set_data(x, y_im)

    text.set_text(f'$t = {i*dt:.2f}$')

    return pdf, re, im
 
anim = FuncAnimation(fig, update, frames=range(0, N, 10), interval=50, blit=True)
anim.save('anim.gif')#, fps=24)
