import os

import numpy as np
import qutip as qt
from scipy.sparse import csr_matrix
from tqdm.notebook import tqdm


# Setup of the coupler Hamiltonian


# Funnctions for symmetry-resolved system


# def Index2Ket(n, b, N):
#     if n == 0:
#         return [-int((b-1)/2) for r in range(N)]
#     digits = []
#     while n:
#         digits.append(int(n % b))
#         n //= b

#     digits = (np.array(digits[::-1])-int((b-1)/2)).tolist()
#     for r in range(N-len(digits)):
#         digits.insert(0,-int((b-1)/2))

#     return digits

# def Ket2Index(st,b):
#     return int(''.join(map(str, st+int((b-1)/2))),b)

def T1(v, Ket2Index):
    return Ket2Index[str(np.roll(v, -1))]
    # return np.roll(v, -1)


def Ket(ns, Ncut):
    return qt.tensor([qt.basis(2 * Ncut + 1, ns[r] + Ncut) for r in range(len(ns))])


def ToKet(coefs, sts, bs, Ncut):
    sm = 0
    for r in range(len(coefs)):
        sm = sm + coefs[r] * Ket(bs[sts[r]], Ncut)

    return sm


def ind2occ(s, r, N, Ncut):
    return int((s // ((2 * Ncut + 1) ** (N - r - 1))) % (2 * Ncut + 1))


def ind2state(s, N, Ncut):
    return np.array([ind2occ(s, r, N, Ncut) for r in range(N)]) - Ncut


def state2ind(state, N, Ncut):
    sps = (2 * Ncut + 1) ** (N - np.arange(N) - 1)
    return np.dot(state + Ncut, sps)


def TransInd(s, N, Ncut):
    return int(
        ind2occ(s, N - 1, N, Ncut) * ((2 * Ncut + 1) ** (N - 1)) - (ind2occ(s, N - 1, N, Ncut) / (2 * Ncut + 1)) + s / (
                    2 * Ncut + 1))



