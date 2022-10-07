

import os

import numpy as np
import qutip as qt
from scipy.sparse import csr_matrix
from tqdm.notebook import tqdm
import JJArray as jja


def GenerateMomentumBasis(N, Ncut):
    dms = (2 * Ncut + 1) ** N
    lst = list(range(dms))
    vbs = []

    while len(lst) > 0:


        bas = [lst[0]]
        tmp = jja.TransInd(bas[-1], N, Ncut)

        while (tmp - bas[0]) != 0:
            bas.append(tmp)
            tmp = jja.TransInd(bas[-1], N, Ncut)

        vbs.append(bas)

        lst = list(set(lst) - set(bas))

    return vbs


def amps(cycle, k):
    N0 = len(cycle)
    seq = np.arange(N0)
    return (1 / np.sqrt(N0)) * np.exp(-1j * 2 * np.pi * (k) * seq)


def ChargeToTranslation(N, Ncut):
    flnms = os.listdir()
    flag = 0
    for fname in flnms:
        if 'vbs_' + str((N, Ncut)) + '.npz' in fname:
            flag = 1
    if flag == 0:
        vbs = np.array(GenerateMomentumBasis(N, Ncut), dtype=object)
        np.savez_compressed('vbs_' + str((N, Ncut)), vbs=vbs)
    else:
        loaded = np.load('vbs_' + str((N, Ncut)) + '.npz', allow_pickle=True)
        vbs = loaded['vbs'].tolist()

    DIMS = [[2 * Ncut + 1 for r in range(N)], [2 * Ncut + 1 for r in range(N)]]

    mms = np.array([n / N for n in range(N)])
    lns = np.unique(list(map(len, vbs)))

    m_seqs = [[] for r in range(N)]
    for N0 in lns:
        tmp = np.array([n / N0 for n in range(N0)])
        for k in range(N):
            if mms[k] in tmp:
                m_seqs[k].append(N0)

    V = []
    sm = 0
    print('Full Hilbert space dimension = ', (2 * Ncut + 1) ** N)
    for n in tqdm(range(N)):
        row = []
        col = []
        data = []
        ind = 0
        for state in tqdm(range(len(vbs))):
            if len(vbs[state]) in m_seqs[n]:
                vec = amps(vbs[state], n / N)
                row.append(vbs[state])
                col.append((ind * np.ones(len(vec), dtype=int)).tolist())
                data.append(vec.tolist())
                ind = ind + 1
        col = [item for sublist in col for item in sublist]
        row = [item for sublist in row for item in sublist]
        data = [item for sublist in data for item in sublist]
        print('Sector ' + str(n) + '=', ind)
        sm = sm + ind
        V.append(qt.Qobj(csr_matrix((data, (row, col)),
                                    shape=((2 * Ncut + 1) ** N, ind)), dims=DIMS))
    print('Sum of sector dimensions = ', sm)
    if sm != (2 * Ncut + 1) ** N:
        np.savetxt('WARNING_' + str([N, Ncut]) + '.txt', [1])
    return V


def SectorDiagonalization(H, V, Nvals):
    data = []
    for k in range(len(V)):
        HF0 = V[k].dag() * H * V[k]
        evals, evecs = HF0.eigenstates(eigvals=Nvals, sparse=True)
        data.append([evals, [V[k] * evecs[r] for r in range(len(evecs))]])

    return data


def SortedDiagonalization(H, V, Nvals):
    tst = SectorDiagonalization(H, V, Nvals)

    vecs = []
    eigs = []
    for k in range(len(tst)):
        tmp = np.array(tst[k][1])
        tmpe = np.array(tst[k][0])
        for r in range(len(tmp)):
            vecs.append(tmp[r].T[0])
            eigs.append(tmpe[r])
    eigs = np.array(eigs)
    vecs = np.array(vecs)
    vecs = vecs[np.argsort(eigs)]
    eigs = eigs[np.argsort(eigs)]

    return eigs, vecs, tst


def SectorDiagonalization_Energies(H, V, Nvals):

    data = []
    for k in range(len(V)):
        HF0 = V[k].dag() * H * V[k]
        evals = HF0.eigenenergies(eigvals=Nvals, sparse=True)
        data.append(evals)

    return data


def SortedDiagonalization_Energies(H, V, Nvals):
    tst = SectorDiagonalization_Energies(H, V, Nvals)

    eigs = np.sort(np.concatenate(tst))

    return eigs, tst