import numpy as np
from quantum import Grid, WaveFunction, WaveFunctionHistory
from solver import cn_solve

def test_accuracy(
        wfh_exact: WaveFunctionHistory, wfh_num: WaveFunctionHistory,
        rtol=1e-5, atol=1e-8):
    # pdf
    assert np.allclose(wfh_num.pdf(), wfh_exact.pdf(), rtol, atol)

    # TODO: Hamiltonian

    return True

def test_norm(wfh_exact: WaveFunctionHistory, rtol=1e-5, atol=1e-8):
    for t in range(wfh_exact.grid.Nt):
        assert np.isclose(wfh_exact.at_time(t).norm(), 1.0)

    return True

grid = Grid(x_min=0, x_max=1, t_final=10, dx=0.01, dt=0.01)

# exact solutions for eigenstates
def psi_n(n, x, t):
    return np.sin(n*np.pi*x) * np.exp(-1j*n**2 * np.pi**2 * t/2)

X, T = np.meshgrid(grid.x, grid.t, indexing='ij')

# even eigenstates are zero so only test odd ones
for n in range(1, 12, 2):
    wfh_exact = WaveFunctionHistory(grid, psi_n(n, X, T))
    wfh_num = cn_solve(wfh_exact.at_time(0))

    if test_accuracy(wfh_exact, wfh_num):
        print(f'PDF accuracy test passed for n={n} eigenstate')
    if test_norm(wfh_exact):
        print(f'Normalisation test passed for n={n} eigenstate')
