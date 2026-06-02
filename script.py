import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from simulation import Grid, WaveFunction, WaveFunctionHistory, cn_solve, animate_histories

grid = Grid(x_min=0, x_max=1, t_final=5, dx=0.01, dt=0.01)

# infinite square well
V = np.zeros(grid.Nx)

# quadratic initial state
psi0 = WaveFunction(grid, (grid.x_min - grid.x) * (grid.x_max - grid.x))

# numerical solver
history_num = cn_solve(psi0, V)

# anim = animate_histories(history)
# plt.show()
# exit()

# exact solutions for eigenstates
def psi_n(n, x, t):
    return np.sin(n*np.pi*x) * np.exp(-1j*n**2 * np.pi**2 * t/2)

n_max = 5

# overlap integral to find c_n
coeffs = np.array([np.sum(np.vdot(psi_n(n, grid.x, 0), psi0.psi)) * grid.dx
    for n in range(1, n_max+1)])

X, T = np.meshgrid(grid.x, grid.t, indexing='ij')
eigenstates = np.array([coeffs[n-1] * psi_n(n, X, T) for n in range(1, n_max+1)])
history_exact = WaveFunctionHistory(grid, np.sum(eigenstates, axis=0))
eigenstates /= history_exact.norm()

history_eigen = []
for n in range(n_max):
    history_eigen.append(WaveFunctionHistory(grid, eigenstates[n]))

# adjust phase to look matched on plot
phase = np.angle(np.vdot(history_exact.at_time(0), history_num.at_time(0)))
history_exact.psi *= np.exp(-1j*phase)

# rmse = np.std(solver.psi.psi, mean=psi_exact.psi)
# print(f'rmse: {rmse}')

# rmse_t = np.std(solver.psi.psi, mean=psi_exact.psi, axis=0)
# plt.scatter(t, rmse_t, marker='.')
# plt.xlabel('$t$')
# plt.ylabel('rmse')
# plt.title('RMS error')
# plt.tight_layout()
# plt.savefig('rmse.png')
# plt.show()

# anim1 = animate_wavefunction(solver)
# solver.psi.psi = psi_exact
# anim2 = animate_wavefunction(solver)
# plt.show()

ns = []
for n in range(1, n_max+1):
    if n % 2 == 1:
        ns.append(n)

eigenlabels = []
for n in ns:
    eigenlabels.append(f'$n = {n}$ eigenstate')
anim = animate_histories([history_num, history_exact] + history_eigen[0::2],
    labels=['Numerical solution', 'Exact solution'] + eigenlabels)
plt.show()
