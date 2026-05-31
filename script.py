import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from wavefunction import norm, normalise
from simulation import SystemParams, cn_solve, animate_wavefunction

params = SystemParams(x_min=0, x_max=1, t_final=5, dx=0.01, dt=0.01)

# infinite square well
V = np.zeros(params.Nx)

# quadratic initial state
psi0 = (params.x_min - params.x) * (params.x_max - params.x)
psi0 = normalise(params, psi0)

# numerical solver
result = cn_solve(params, V, psi0)

# exact solutions for eigenstates
def psi_n(n, x, t):
    return np.sin(n*np.pi*x) * np.exp(-1j*n**2 * np.pi**2 * t/2)

n_max = 5

# overlap integral to find c_n
coeffs = np.array([np.sum(np.vdot(psi_n(n, params.x, 0), psi0)) * params.dx
    for n in range(n_max+1)])

X, T = np.meshgrid(params.x, params.t, indexing='ij')
eigenstates = np.array([coeffs[n] * psi_n(n, X, T) for n in range(n_max+1)])
psi_exact = np.sum(eigenstates, axis=0)

# normalise
A = norm(params, psi_exact[:,0])
psi_exact /= A
eigenstates /= A

# adjust phase to look matched on plot
phase = np.angle(np.vdot(psi_exact[:,0], result.psi[:,0]))
psi_exact *= np.exp(-1j*phase)

rmse = np.std(result.psi, mean=psi_exact)
print(f'rmse: {rmse}')

rmse_t = np.std(result.psi, mean=psi_exact, axis=0)
# plt.scatter(t, rmse_t, marker='.')
# plt.xlabel('$t$')
# plt.ylabel('rmse')
# plt.title('RMS error')
# plt.tight_layout()
# plt.savefig('rmse.png')
# plt.show()

# anim1 = animate_wavefunction(result)
# result.psi = psi_exact
# anim2 = animate_wavefunction(result)
# plt.show()

# exit()

# pdf, real, imag
def generate_data(psi):
    return np.array([np.abs(psi)**2, np.real(psi), np.imag(psi)])

ns = []
for n in range(n_max+1):
    if n % 2 == 1:
        ns.append(n)

data = {
    'num': generate_data(result.psi),
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
ylims = [(0, 2.5), (-1.5, 1.5), (-1.5, 1.5)]
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
            plot[key].set_data(params.x, data[key][i,:,t])
            artists.append(plot[key])

    text.set_text(f'$t = {t*params.dt:.2f}$')
    return *artists, text

timescale = 4
anim = FuncAnimation(fig, update, frames=range(0, params.Nt),
    interval=timescale*1e3*params.dt, blit=True)
# anim.save('anim.gif', fps=params.Nt/(params.t_final*timescale))
plt.show()
