import numpy as np

def generate_colored_noise(z_array, L_c=200.0, sigma=0.2, seed=None):
    """
    Genera ruido coloreado en altura.
    z_array: array de alturas
    L_c: longitud de correlación [m]
    sigma: desviación estándar del ruido
    """
    if seed is not None:
        np.random.seed(seed)

    dz = np.mean(np.diff(z_array))
    n = len(z_array)

    white_noise = np.random.normal(0, 1, n)
    noise = np.zeros(n)

    noise[0] = white_noise[0]
    for i in range(1, n):
        noise[i] = noise[i-1] * np.exp(-dz/L_c) + white_noise[i] * np.sqrt(1-np.exp(-2*dz/L_c))

    return sigma * noise
