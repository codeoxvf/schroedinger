import numpy as np
from simulation import SystemParams

def norm(params: SystemParams, psi: np.ndarray):
    return np.sqrt(np.sum(np.abs(psi)**2) * params.dx)

def normalise(params: SystemParams, psi: np.ndarray):
    return psi/norm_coeff(params, psi)
