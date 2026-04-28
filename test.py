import numpy as np
import matplotlib.pyplot as plt
from simulation import SolverParams, cn_solve, animate_wavefunction

a = -5
b = 5
tfinal = 10
dx = 0.01
dt = dx
params = SolverParams(a, b, tfinal, dx, dt)
x = np.linspace(params.a, params.b, params.J)

# harmonic oscillator
V = x**2

# initial state
gaussian = np.exp(-20*x**2) * np.exp(1j*x)
psi0 = gaussian/np.linalg.norm(gaussian)
psi0[1] = 0
psi0[-1] = 0

psi = cn_solve(params, V, psi0)

anim = animate_wavefunction(psi, params, filename='oscillator.gif', timescale=0.5,
    axes_kwargs={'ylim': (0, 0.25)})
plt.show()
