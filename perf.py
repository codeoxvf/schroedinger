#%%
import numpy as np
import pandas as pd

from quantum import Grid, WaveFunction
from solver import cn_solve

from timeit import timeit

import matplotlib.pyplot as plt

#%%
def benchmark_harmonic_osc(dx, dt, methods, number=10):
    grid = Grid(x_min=-5, x_max=5, t_final=5, dx=dx, dt=dt)

    omega = 1
    V = omega**2 * grid.x**2

    # Gaussian wave packet
    sigma0 = 0.5
    psi0 = np.exp(-grid.x**2/(4*sigma0**2))
    wf = WaveFunction(grid, psi0)

    tot_times = np.array([timeit(lambda: cn_solve(wf, V, method=method),
        number=number) for method in methods])
    return tot_times/number

#%%
methods = ['lu', 'banded-naive-mult', 'banded']
rows = []
for k in range(1, 7):
    dx = 2**(-k) / 10
    dt = 0.01
    times = benchmark_harmonic_osc(dx=dx, dt=dt, methods=methods)
    times_dict = dict(zip([f't_{method}' for method in methods], times))
    rows.append(times_dict | { 'dx': dx, 'dt': dt })

df = pd.DataFrame(rows)

#%%
plt.plot(df['dx'], df['t_lu'], 'o', label='LU')
plt.plot(df['dx'], df['t_banded-naive-mult'], 'o', label='Banded (naive mult)')
plt.plot(df['dx'], df['t_banded'], 'o', label='Banded')
plt.xlabel('$dx$')
plt.ylabel('$t$')
plt.legend()
plt.savefig('figures/perf.png')
plt.show()

#%%
df['speedup_naive'] = df['t_lu'] / df['t_banded-naive-mult']
df['speedup_banded'] = df['t_lu'] / df['t_banded']

#%%
df['N'] = round(10/df['dx']) + 1

plt.plot(df['N'], df['speedup_naive'], 'o', label='LU / naive banded')
plt.plot(df['N'], df['speedup_banded'], 'o', label='LU / banded')
plt.xlabel('$N$')
plt.ylabel('Speedup factor')
plt.legend()
plt.savefig('figures/speedup.png')
plt.show()

#%%
df
