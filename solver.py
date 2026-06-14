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

    def step(psi, n):
        b_psi = b @ psi[1:-1,n]
        psi[1:-1,n+1] = lu_solve((lu, piv), b_psi)
    
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

    def step(psi, n):
        b_psi = b @ psi[1:-1,n]
        psi[1:-1,n+1] = solve_banded((1, 1), ab, b_psi)
    
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

    def step(psi, n):
        b_psi = gamma * psi[1:-1,n]
        b_psi[1:] -= alpha * psi[1:-2,n]
        b_psi[:-1] -= alpha * psi[2:-1,n]
        psi[1:-1,n+1] = solve_banded((1, 1), ab, b_psi)
    
    return step

def cn_solve(wf: WaveFunction, V=None, method='banded', progress=False):
    '''Numerically solve the Schroedinger equation over time for the given
        initial condtions.
        
    Parameters
    ----------
    wf : WaveFunction
        Initial state of the wavefunction.
    V : ArrayLike, default None
        Potential for the system. If None, treat as zero potential.
    progress : bool, default False
        Print progress percentage.

    Returns
    -------
    WaveFunctionHistory
        Evolution history of the initial state over time.
    '''
    grid = wf.grid

    psi = np.zeros((grid.Nx, grid.Nt), dtype=complex)
    psi[:,0] = wf.psi

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
        step(psi, n)
        if progress and n % ((grid.Nt-1)//10) == 0:
            print(f'{100 * n / (grid.Nt-1):.0f}%')

    return WaveFunctionHistory(grid, psi)
