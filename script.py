import numpy as np
import matplotlib.pyplot as plt
from simulation import SolverParams, cn_solve, animate_wavefunction

a = -1
b = -1
tfinal = 10
dx = 0.01
dt = dx
params = SolverParams(a, b, tfinal, dx, dt)

# infinite square well
V = np.zeros(params.J)

# initial state
x = np.linspace(params.a, params.b, params.J)
gaussian = np.exp(-20*x**2)
psi0 = gaussian/np.linalg.norm(gaussian)
psi0[1] = 0
psi0[-1] = 0

psi = cn_solve(params, V, psi0)

anim = animate_wavefunction(psi, params, filename='anim.gif', timescale=0.5,
    axes_kwargs={'ylim': (0, 0.25)})
