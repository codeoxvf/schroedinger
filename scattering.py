import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from schroedinger import Grid, WaveFunction, cn_solve, animate_histories

grid = Grid(x_min=-35, x_max=35, t_final=5, dx=0.01, dt=0.01)

# step potential
E = 5
V = np.zeros(grid.Nx)
V[grid.x > 0] = E

# Gaussian wave packet
p0 = 5
sigma0 = 0.5
x0 = -5
psi0 = np.exp(1j*p0*(grid.x-x0) - (grid.x-x0)**2/(4*sigma0**2))
wf = WaveFunction(grid, psi0)

wfh = cn_solve(wf, V)

anim = animate_histories(wfh, filename='scattering.gif')
plt.show()
