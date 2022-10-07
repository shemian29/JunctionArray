import numpy as np
from tqdm.notebook import tqdm

# Functions to obtain Wigner distributions


def Wmatrix(phi, n, m, mp, N):
    return (1 / (2 * np.pi)) * np.exp(1j * (m - mp) * phi / N) * np.sin((n - (m + mp) / 2) * np.pi / N + 0.0000001) / (
                (n - (m + mp) / 2) * np.pi / N + 0.0000001)


def WignerPoint(n, phi, Ncut, rho, N):
    xaxis = np.arange(-Ncut, Ncut + 1)
    yaxis = np.arange(-Ncut, Ncut + 1)
    return np.real(np.sum(rho * Wmatrix(phi, n, xaxis[:, None], yaxis[None, :], N)))


def WignerArray(rho, N):
    Ncut = int((len(rho) - 1) / 2)
    #     print(Ncut)
    n_list = np.arange(-Ncut, Ncut + 1)
    #     print(len(n_list))
    phi_list = 2 * np.pi * N * np.arange(0, 2 * Ncut + 1) / (2 * Ncut + 1)

    scanT = []
    for ph in tqdm(phi_list):
        scan = []
        for n in n_list:
            scan.append(WignerPoint(n, ph, Ncut, rho, N))
        scanT.append(scan)

    return np.array(scanT)


def cartesian(arrays, out=None):
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
    return out
