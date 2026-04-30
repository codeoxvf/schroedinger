from dataclasses import dataclass
import numpy as np
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 

@dataclass
class SolverParams:
    x_min: float
    x_max: float
    t_final: float
    dx: float
    dt: float

    @property
    def Nx(self):
        return round((self.x_max-self.x_min)/self.dx) + 1

    @property
    def Nt(self):
        return round(self.t_final/self.dt) + 1

def cn_solve(params, V, psi0, progress=False):
    x_min, x_max, dx, Nx = params.x_min, params.x_max, params.dx, params.Nx
    t_final, dt, Nt = params.t_final, params.dt, params.Nt

    psi = np.zeros((Nx, Nt), dtype=complex)
    psi[:,0] = psi0

    # set up matrix equation
    # hbar = m = 1
    alpha = 1/(4*dx**2)
    beta = 1j/dt

    # since Dirichlet boundary conditions are 0, only update interior
    diag = np.full(Nx-2-1, alpha)

    A = np.diag(beta - 2*alpha - V[1:-1]/2) + np.diag(diag, 1) + np.diag(diag, -1)
    B = np.diag(beta + 2*alpha + V[1:-1]/2) - np.diag(diag, 1) - np.diag(diag, -1)

    # update state
    lu, piv = lu_factor(A)
    for n in range(Nt-1):
        psi[1:-1,n+1] = lu_solve((lu, piv), B @ psi[1:-1,n])
        if progress and n % ((Nt-1)//10) == 0:
            print(f'{100*n/(Nt-1):.0f}%')

    # TODO: record/analyse pdf integral
    return psi

def animate_wavefunction(psi, params, filename=None, display='all', every=2, timescale=1.0,
    axes_kwargs={'ylim': (-0.25, 1)}):
    display = display.split(' ')

    fig = plt.figure() 
    ax = plt.axes(xlim =(params.x_min, params.x_max), **axes_kwargs) 

    if 'all' in display or 'pdf' in display:
        pdf, = ax.plot([], [], label='Probability density')
    if 'all' in display or 'real' in display:
        re, = ax.plot([], [], label='Real part')
    if 'all' in display or 'imag' in display:
        im, = ax.plot([], [], label='Imaginary part')

    text = ax.text(0.01, 0.98, '$t = 0.00$', transform=ax.transAxes, va='top')

    ax.set_xlabel('$x$')
    ax.set_ylabel('$|\\Psi|^2$')
    ax.legend()

    x = np.linspace(params.x_min, params.x_max, params.Nx)
    if 'all' in display or 'pdf' in display:
        # y_pdf = psi[i].conj() * psi[i]
        y_pdf = np.abs(psi)**2
    if 'all' in display or 'real' in display:
        y_re = np.real(psi)
    if 'all' in display or 'imag' in display:
        y_im = np.imag(psi)

    def update(t):
        if 'all' in display or 'pdf' in display:
            pdf.set_data(x, y_pdf[:,t])
        if 'all' in display or 'real' in display:
            re.set_data(x, y_re[:,t])
        if 'all' in display or 'imag' in display:
            im.set_data(x, y_im[:,t])

        text.set_text(f'$t = {t*params.dt:.2f}$')

        return pdf, re, im, text

    anim = FuncAnimation(fig, update, frames=range(0, params.N, every),
        interval=1e3*every*params.dt/timescale, blit=True)
    if filename:
        anim.save(filename, fps=timescale*params.N/(every*params.t_final))

    return anim
