

import os

import numpy as np
import qutip as qt
from scipy.sparse import csr_matrix
from tqdm.notebook import tqdm
import JJArray as jja

import scipy.sparse

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


    # Check if vbs file has been generated, if not then generate it
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





# ------------------------------------------------------------------

def Hk(hamiltonian, basis, basis_ind,size, check_symm = False, check_spect = False):


    # print("Setup momentum basis")

    k_list = np.arange(0, size) / size
    scan = {}
    for kk in k_list:
        scan[kk] = []
    vbs = GenerateMomentumBasis(size, basis)
    # for n in tqdm(range(len(vbs))):
    for n in range(len(vbs)):

        ktemp_list = np.arange(0, len(vbs[n])) / len(vbs[n])
        inds = list(map(basis_ind.get, vbs[n]))

        for kk in ktemp_list:
            phs = np.exp(-2 * np.pi * kk * 1j * np.arange(0, len(inds))) / np.sqrt(len(inds))
            scan[kk].append([inds, phs])


    # print("Calculating k-basis transformations")
    U = {}
    # for kk in tqdm(k_list):
    for kk in k_list:
        # print('---------------------------')
        U[kk] = 0
        for n in range(len(scan[kk])):
            row = n * np.ones(len(scan[kk][n][0]))
            col = np.array(scan[kk][n][0])
            data = np.array(scan[kk][n][1])

            U[kk] = U[kk] + scipy.sparse.csr_matrix((data, (row, col)), shape=(len(scan[kk]), \
                                                                               len(basis))).T

    # print("Calculate momentum Hamiltonians")
    Hs = {}
    scan = []
    # for k in tqdm(k_list):
    for k in k_list:
        U[k] = qt.Qobj(U[k])
        Hs[k]=(U[k].dag()*hamiltonian*U[k])
        scan.append(Hs[k].eigenstates()[0])
    scan = [x for xs in scan for x in xs]


    if (check_symm == True):
        if hamiltonian.shape[0]<4000:
            print("Run symmetry check:")
            Trn = 0
            for n in range(len(vbs)):
                inds = list(map(basis_ind.get, vbs[n]))
                row = inds
                col = np.roll(inds, 1)
                data = np.ones(len(np.roll(inds, 1)))

                Trn = Trn + scipy.sparse.csr_matrix((data, (row, col)), shape=(len(basis), \
                                                                               len(basis))).T
            Trn = qt.Qobj(Trn)
            commutator =  np.sum(np.abs((Trn*hamiltonian-hamiltonian*Trn).full()))
            if commutator == 0.0:
                print(" -> Passed: Hamiltonian is translationally invariant")
            else:
                print("WARNING: Hamiltonian is NOT translationally invariant, failure amount: ",commutator)
        else:
            print("Hilbert space is too big to perform symmetry check")

    if check_spect == True:
        if hamiltonian.shape[0]<4000:
            print("Run spectrum check:")
            de = hamiltonian.eigenenergies() - np.sort(scan)
            dE = np.sum(np.abs(de))
            if dE < 10**(-10):
                print(" -> Passed: Spectra of full H and the H(k) match")
            else:
                print("WARNING: Spectra of full H and the H(k) do NOT match, failure amount: ",dE)
        else:
            print("Hilbert space is too big to perform spectrum check")


    return Hs, U