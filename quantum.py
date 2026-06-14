import numpy as np
from scipy.linalg import lu_factor, lu_solve

from dataclasses import dataclass, field
from numpy.typing import ArrayLike
from typing import Union
from collections.abc import Callable

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 

@dataclass
class Grid:
    '''Discretised space/time coordinate grid for numerical solvers.

    Parameters
    ----------
    x_min : float
        Lower bound for x-coordinates.
    x_max : float
        Upper bound for x-coordinates.
    t_final : float
        Final time until which to evolve the system.
    dx : float
        Step-size for spatial coordinate grid.
    dt : float
        Step-size for temporal coordinate grid.

    Attributes
    ----------
    x : ArrayLike
        Spatial coordinate grid of the discretised interval [`x_min`, `x_max`].
    t : ArrayLike
        Temporal coordinate grid of the discretised interval [0, `t_final`].
    Nx : int
        Number of spatial grid coordinates.
    Nt : int
        Number of temporal grid coordinates.
    '''

    x_min: float
    x_max: float
    t_final: float
    dx: float
    dt: float
    x: ArrayLike = field(init=False)
    t: ArrayLike = field(init=False)

    def __post_init__(self):
        self.x = np.linspace(self.x_min, self.x_max, self.Nx)
        self.t = np.linspace(0, self.t_final, self.Nt)

    @property
    def Nx(self):
        return round((self.x_max-self.x_min)/self.dx) + 1

    @property
    def Nt(self):
        return round(self.t_final/self.dt) + 1

class WaveFunction:
    '''Position-space discretised wavefunction.

    Parameters
    ----------
    grid : Grid
        Grid on which the wavefunction is defined.
    psi : ArrayLike or Callable
        Wavefunction data.
    normalise : bool, default True
        Normalise the given wavefunction on creation.
    '''

    def __init__(self, grid: Grid, psi: Union(ArrayLike, Callable), normalise=True):
        self.grid = grid

        if callable(psi):
            self.psi = psi(grid)
        else:
            self.psi = psi
        
        assert self.psi.ndim == 1
        assert self.psi.shape[0] == self.grid.Nx

        if normalise:
            self._normalise()

    def pdf(self):
        '''Return the pdf of the wavefunction.'''
        return np.abs(self.psi)**2

    def norm(self):
        '''Return the norm of the wavefunction.'''
        return np.sqrt(np.sum(self.pdf()) * self.grid.dx)

    def _normalise(self):
        self.psi /= self.norm()
    
    def T(self):
        '''Return the kinetic energy of the wavefunction.'''
        d0 = np.full(self.grid.Nx, -2)
        d1 = np.full(self.grid.Nx, 1)
        lap = np.diag(d0) + np.diag(d1, 1) + np.diag(d1, -1)
        return -0.5/self.grid.dx**2 * lap

class WaveFunctionHistory:
    '''Evolution history of a wavefunction over time.

    Parameters
    ----------
    grid : Grid
        Grid on which the wavefunction is defined.
    psi : ArrayLike
        Wavefunction history data.
    normalise : bool, default True
        Normalise the given wavefunction history on creation by the norm at t=0.
    '''

    def __init__(self, grid: Grid, psi: ArrayLike, normalise=True):
        self.grid = grid
        self.psi = psi

        assert self.psi.ndim == 2
        assert self.psi.shape[0] == self.grid.Nx
        assert self.psi.shape[1] == self.grid.Nt

        if normalise:
            self._normalise()

    def at_time(self, t: int):
        '''Return the wavefunction at time index t.'''
        return WaveFunction(self.grid, self.psi[:,t], normalise=False)

    def pdf(self):
        '''Return the pdf of the wavefunction over time.'''
        return np.abs(self.psi)**2

    def norm(self):
        '''Return the norm of the wavefunction over time.'''
        return np.sqrt(np.sum(self.pdf(), axis=0) * self.grid.dx)

    def _normalise(self):
        self.psi /= self.at_time(0).norm()

ylabels = {
    'pdf': '$|\\Psi|^2$',
    'real': '$\\mathrm{Re}(\\Psi)$',
    'imag': '$\\mathrm{Im}(\\Psi)$',
}
transforms = {
    'pdf': lambda psi: np.abs(psi)**2,
    'real': lambda psi: np.real(psi),
    'imag': lambda psi: np.imag(psi),
}

def animate_histories(histories: ArrayLike, V=None, labels=None, filename=None, display='all',
    every=2, timescale=1.0, **kwargs):
    '''Animate the evolution of one or more quantum systems over time.

    Parameters
    ----------
    histories : ArrayLike
        WaveFunctionHistory or list of such to be animated.
    V : ndarray, default None
        Potential for the system.
    labels : ArrayLike, default None
        Labels for the histories. If None, legend is not shown.
    filename : str, default None
        Name of the file to which the animation is saved. If None, animation
        is not saved.
    display : str or list, default 'all'
        Components to plot ('all', 'pdf', 'real', 'imag').
    every : int, default 2
        Number of time steps between each frame.
    timescale : float, default 1.0
        Animation speed (1 for real-time).
    **kwargs
        Extra arguments for subplots.

    Returns
    -------
    FuncAnimation
        The generated animation.
    '''
    # clean up input formats (make iterable)
    if isinstance(histories, WaveFunctionHistory):
        histories = [histories]
    
    showlabels = True
    if not labels:
        labels = list(range(len(histories)))
        showlabels = False
    else:
        assert len(labels) == len(histories)

    grid = histories[0].grid
    for h in histories:
        assert h.grid == grid

    # display plots in correct order
    s = display
    display = []
    if 'pdf' in s or 'all' in s:
        display.append('pdf')
    if 'real' in s or 'all' in s:
        display.append('real')
    if 'imag' in s or 'all' in s:
        display.append('imag')
    
    kwargs['xlim'] = (grid.x_min, grid.x_max)
    kwargs.setdefault('ylim', (-1, 1))

    fig, axes = plt.subplots(len(display), 1, subplot_kw=kwargs)
    if isinstance(axes, plt.Axes):
        axes = [axes]

    # plot lines
    lines = {}
    ys = {}
    for d, ax in zip(display, axes):
        y_max = -np.inf
        y_min = np.inf
        for i, h in enumerate(histories):
            lines[f'{labels[i]} {d}'], = ax.plot([], [], label=labels[i])
            ys[f'{labels[i]} {d}'] = (transforms[d](h.psi))
            y_max = max(y_max, np.max(ys[f'{labels[i]} {d}']))
            y_min = min(y_min, np.min(ys[f'{labels[i]} {d}']))

        ax.set_xlabel('$x$')
        ax.set_ylabel(ylabels[d])
        ax.set_ylim(1.1*y_min, 1.1*y_max)

    ax = axes[0]
    if showlabels:
        fig.legend(*ax.get_legend_handles_labels())
    text = ax.text(0.01, 0.98, '$t = 0.00$', transform=ax.transAxes, va='top')

    # draw potential
    if V is not None and 'pdf' in display:
        V_max = np.max(V)
        if not np.isclose(V_max, 0.0):
            height = axes[0].get_ylim()[1] * 0.9
            axes[0].plot(grid.x, height*V/V_max, color='black',
                linewidth=1.5, alpha=0.7)

    def update(t):
        for d in lines:
            lines[d].set_data(grid.x, ys[d][:,t])

        text.set_text(f'$t = {t*grid.dt:.2f}$')

        return *lines.values(), text

    anim = FuncAnimation(fig, update, frames=range(0, grid.Nt, every),
        interval=1e3*every*grid.dt/timescale, blit=True)
    if filename:
        anim.save(filename, fps=timescale*grid.Nt/(every*grid.t_final))

    return anim
