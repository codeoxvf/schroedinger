import numpy as np
from scipy.linalg import lu_factor, lu_solve, solve_banded
from numpy.typing import ArrayLike
from quantum import Grid, WaveFunction, WaveFunctionHistory

def _schrod_matrix_dense(grid: Grid, V: ArrayLike):
    # set up matrix equation
    # hbar = m = 1
    alpha = 1/(4*grid.dx**2)
    beta = 1j/grid.dt

    # since Dirichlet boundary conditions are 0, only update interior
    d = np.full(grid.Nx-2-1, alpha)
    a = np.diag(beta - 2*alpha - V[1:-1]/2) + np.diag(d, 1) + np.diag(d, -1)
    b = np.diag(beta + 2*alpha + V[1:-1]/2) - np.diag(d, 1) - np.diag(d, -1)

    lu, piv = lu_factor(a)
    return lu, piv, b

def _get_step_dense(grid: Grid, V: ArrayLike):
    lu, piv, b = _schrod_matrix_dense(grid, V)

    def step(dest, psi):
        b_psi = b @ psi[1:-1]
        dest[1:-1] = lu_solve((lu, piv), b_psi)

    return step

def _schrod_matrix_banded_naive_mult(grid: Grid, V: ArrayLike):
    # set up matrix equation
    # hbar = m = 1
    alpha = 1/(4*grid.dx**2)
    beta = 1j/grid.dt

    # since Dirichlet boundary conditions are 0, only update interior
    ab = np.zeros((3, grid.Nx-2), dtype=complex)
    ab[0,1:] = alpha
    ab[1,:] = beta - 2*alpha - V[1:-1]/2
    ab[2,:-1] = alpha

    d = np.full(grid.Nx-2-1, alpha)
    b = np.diag(beta + 2*alpha + V[1:-1]/2) - np.diag(d, 1) - np.diag(d, -1)

    return ab, b

def _get_step_banded_naive_mult(grid: Grid, V: ArrayLike):
    ab, b = _schrod_matrix_banded_naive_mult(grid, V)

    def step(dest, psi):
        b_psi = b @ psi[1:-1]
        dest[1:-1] = solve_banded((1, 1), ab, b_psi)

    return step

def _schrod_matrix_banded(grid: Grid, V: ArrayLike):
    # set up matrix equation
    # hbar = m = 1
    alpha = 1/(4*grid.dx**2)
    beta = 1j/grid.dt

    # since Dirichlet boundary conditions are 0, only update interior
    ab = np.zeros((3, grid.Nx-2), dtype=complex)
    ab[0,1:] = alpha
    ab[1,:] = beta - 2*alpha - V[1:-1]/2
    ab[2,:-1] = alpha

    gamma = beta + 2*alpha + V[1:-1]/2

    return ab, alpha, gamma

def _get_step_banded(grid: Grid, V: ArrayLike):
    ab, alpha, gamma = _schrod_matrix_banded(grid, V)

    def step(dest, psi):
        b_psi = gamma * psi[1:-1]
        b_psi[1:] -= alpha * psi[1:-2]
        b_psi[:-1] -= alpha * psi[2:-1]
        dest[1:-1] = solve_banded((1, 1), ab, b_psi)

    return step

def cn_solve(wf: WaveFunction, V=None, method='banded', history=False, progress=False):
    '''Numerically solve the Schroedinger equation over time for the given
        initial condtions.

    Parameters
    ----------
    wf : WaveFunction
        Initial state of the wavefunction.
    V : ArrayLike, default None
        Potential for the system. If None, treat as zero potential.
    method: str, default 'banded'
        The method of solving the matrix equation ('banded', 'banded-naive-mult',
        'lu' in order of speed).
    history: bool, default False
        Return the full evolution history.
    progress : bool, default False
        Print progress percentage.

    Returns
    -------
    WaveFunction or WaveFunctionHistory
        Final state or evolution history of the initial state over time.
    '''
    grid = wf.grid

    if history:
        psi = np.zeros((grid.Nx, grid.Nt), dtype=complex)
        psi[:,0] = wf.psi
    else:
        psi = wf.psi.copy()

    if V is None:
        V = np.zeros(grid.Nx)

    if method == 'banded':
        step = _get_step_banded(grid, V)
    elif method == 'banded-naive-mult':
        step = _get_step_banded_naive_mult(grid, V)
    elif method == 'lu':
        step = _get_step_dense(grid, V)
    else:
        raise Exception(f'Unrecognised method {method}')

    # update state
    for n in range(grid.Nt-1):
        if history:
            step(psi[:,n+1], psi[:,n])
        else:
            step(psi, psi)
        if progress and n % ((grid.Nt-1)//10) == 0:
            print(f'{100 * n / (grid.Nt-1):.0f}%')

    if history:
        return WaveFunctionHistory(grid, psi, normalise=False)
    else:
        return WaveFunction(grid, psi, normalise=False)
