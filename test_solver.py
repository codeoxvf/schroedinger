import numpy as np
from schroedinger import Grid, WaveFunction, WaveFunctionHistory, cn_solve, animate_histories
import matplotlib.pyplot as plt

def test_accuracy(
        history_exact: WaveFunctionHistory, history_num: WaveFunctionHistory,
        rtol=1e-5, atol=1e-8):
    assert np.allclose(history_num.pdf(), history_exact.pdf(), rtol, atol)
    return True

def test_norm(history_exact: WaveFunctionHistory, rtol=1e-5, atol=1e-8):
    for t in range(history_exact.grid.Nt):
        assert np.isclose(history_exact.at_time(t).norm(), 1.0)
    return True

grid = Grid(x_min=0, x_max=1, t_final=10, dx=0.01, dt=0.01)

# infinite square well
V = np.zeros(grid.Nx)

# exact solutions for eigenstates
def psi_n(n, x, t):
    return np.sin(n*np.pi*x) * np.exp(-1j*n**2 * np.pi**2 * t/2)

X, T = np.meshgrid(grid.x, grid.t, indexing='ij')

# even eigenstates are zero so only test odd ones
for n in range(1, 12, 2):
    history_exact = WaveFunctionHistory(grid, psi_n(n, X, T))
    history_num = cn_solve(history_exact.at_time(0), V)

    if test_accuracy(history_exact, history_num):
        print(f'PDF accuracy test passed for n={n} eigenstate')
    if test_norm(history_exact):
        print(f'Normalisation test passed for n={n} eigenstate')
