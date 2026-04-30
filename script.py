import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from simulation import SolverParams, cn_solve, animate_wavefunction
from multiprocessing import Process

x_min = 0
x_max = 1
t_final = 10
dx = 0.01
dt = dx
params = SolverParams(x_min, x_max, t_final, dx, dt)

x = np.linspace(params.x_min, params.x_max, params.Nx)
t = np.linspace(0, params.t_final, params.Nt)

# infinite square well
V = np.zeros(params.Nx)

# quadratic initial state
psi0 = (x_min-x) * (x_max-x)
psi0 /= np.linalg.norm(psi0)

# numerical solver
psi_num = cn_solve(params, V, psi0)

# exact solutions for eigenstates
def psi_n(n, x, t):
    return np.sin(n*np.pi*x) * np.exp(-1j*n**2 * np.pi**2 * t/2)

n_max = 5

# overlap integral to find c_n
coeffs = np.array([np.sum(np.vdot(psi_n(n, x, 0), psi0)) * dx
    for n in range(n_max+1)])

X, T = np.meshgrid(x, t, indexing='ij')
eigenstates = np.array([coeffs[n] * psi_n(n, X, T) for n in range(n_max+1)])
psi_exact = np.sum(eigenstates, axis=0)

# normalise
A = np.linalg.norm(psi_exact[:,0])
psi_exact /= A
eigenstates /= A

# adjust phase to look matched on plot
phase = np.angle(np.vdot(psi_exact[:,0], psi_num[:,0]))
psi_exact *= np.exp(-1j*phase)

rmse = np.std(psi_num, mean=psi_exact)
print(f'rmse: {rmse}')

rmse_t = np.std(psi_num, mean=psi_exact, axis=0)
plt.scatter(t, rmse_t, marker='.')
plt.xlabel('$t$')
plt.ylabel('rmse')
plt.title('RMS error')
plt.tight_layout()
plt.savefig('rmse.png')
# plt.show()

# pdf, real, imag
def generate_data(psi):
    return np.array([np.abs(psi)**2, np.real(psi), np.imag(psi)])

ns = []
for n in range(n_max+1):
    if n % 2 == 1:
        ns.append(n)

data = {
    'num': generate_data(psi_num),
    'exact': generate_data(psi_exact),
}
for i in range(len(ns)):
    data[f'n{ns[i]}'] = generate_data(eigenstates[i])

labels = {
    'num': 'Numerical solution',
    'exact': 'Exact solution',
}
for i in range(len(ns)):
    labels[f'n{ns[i]}'] = f'$n = {ns[i]}$ eigenstate'

# animation
fig, axs = plt.subplots(1, 3)

titles = ['Probability density', 'Real part', 'Imaginary part']
ylims = [(0, 0.1), (-0.25, 0.25), (-0.25, 0.25)]
plots = []
for i, ax in enumerate(axs):
    lines = {}
    for key in data:
        lines[key], = ax.plot([], [], label=labels[key])
    plots.append(lines)

    ax.set_xlim(params.x_min, params.x_max)
    ax.set_ylim(*ylims[i])
    ax.set_title(titles[i])
    ax.set_xlabel('$x$')

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower right')

text = axs[0].text(0.01, 0.98, '$t = 0.00$', transform=axs[0].transAxes, va='top')

def update(t):
    artists = []
    for i, plot in enumerate(plots):
        for key in data:
            plot[key].set_data(x, data[key][i,:,t])
            artists.append(plot[key])

    text.set_text(f'$t = {t*params.dt:.2f}$')
    return *artists, text

timescale = 4
anim = FuncAnimation(fig, update, frames=range(0, params.Nt),
    interval=timescale*1e3*params.dt, blit=True)
# anim.save('anim.gif', fps=params.Nt/(params.t_final*timescale))
plt.show()
