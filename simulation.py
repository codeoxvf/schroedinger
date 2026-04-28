from dataclasses import dataclass
import numpy as np
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 

@dataclass
class SolverParams:
    a: float
    b: float
    tfinal: float
    dx: float
    dt: float

    @property
    def J(self):
        return round((self.b-self.a)/self.dx) + 1

    @property
    def N(self):
        return round(self.tfinal/self.dt) + 1

def cn_solve(params, V, psi0):
    a, b, dx, J = params.a, params.b, params.dx, params.J
    tfinal, dt, N = params.tfinal, params.dt, params.N

    psi = np.zeros((J, N), dtype=complex)
    psi[:,0] = psi0

    # set up matrix equation
    # hbar = m = 1
    alpha = 1/(4*dx**2)
    beta = 1j/dt

    # since Dirichlet boundary conditions are 0, only update interior
    diag = np.full(J-2-1, alpha)

    A = np.diag(beta - 2*alpha - V[1:-1]/2) + np.diag(diag, 1) + np.diag(diag, -1)
    B = np.diag(beta + 2*alpha + V[1:-1]/2) - np.diag(diag, 1) - np.diag(diag, -1)

    # update state
    lu, piv = lu_factor(A)
    for n in range(N-1):
        psi[1:-1,n+1] = lu_solve((lu, piv), B @ psi[1:-1,n])
        if n % ((N-1)//10) == 0:
            print(f'{100*n/(N-1):.0f}%')

    # TODO: record/analyse pdf integral
    return psi

def animate_wavefunction(psi, params, filename=None, display='all', every=2, timescale=1.0,
    axes_kwargs={'ylim': (0, 1)}):
    display = display.split(' ')

    fig = plt.figure() 
    ax = plt.axes(xlim =(params.a, params.b), **axes_kwargs) 

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

    x = np.linspace(params.a, params.b, params.J)
    if 'all' in display or 'pdf' in display:
        # y_pdf = psi[i].conj() * psi[i]
        y_pdf = np.abs(psi)**2
    if 'all' in display or 'real' in display:
        y_re = np.real(psi)
    if 'all' in display or 'imag' in display:
        y_im = np.imag(psi)

    def update(i):
        if 'all' in display or 'pdf' in display:
            pdf.set_data(x, y_pdf[:,i])
        if 'all' in display or 'real' in display:
            re.set_data(x, y_re[:,i])
        if 'all' in display or 'imag' in display:
            im.set_data(x, y_im[:,i])

        text.set_text(f'$t = {i*params.dt:.2f}$')

        return pdf, re, im, text

    anim = FuncAnimation(fig, update, frames=range(0, params.N, every),
        interval=1e3*every*params.dt/timescale, blit=True)
    if filename:
        anim.save(filename, fps=timescale*params.N/(every*params.tfinal))

    return anim
