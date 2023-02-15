import scqubits as scq
import numpy as np
import qutip as qt
from tqdm.notebook import tqdm
import os
from scipy.sparse import csr_matrix
from matplotlib import pyplot as plt

class JJCircuit(scq.Circuit):

    def __init__(self, EJs, ECs, converge=False, Ncut=5):
        self.EJs = EJs
        self.ECs = ECs
        self.Ncut = None
        self.N = len(EJs) - 1
        self.symmetric_basis_transformation = None
        JJcirc_yaml = "branches:\n"

        N_junctions = self.N + 1
        for n in range(N_junctions):
            JJcirc_yaml = JJcirc_yaml + \
                          '- ["JJ", ' + str(n) + ', ' + str((n + 1) % N_junctions) + \
                          ", EJ" + str(n % N_junctions) + str((n + 1) % N_junctions) + " = " + str(
                EJs[n % N_junctions]) + \
                          ", EC" + str(n % N_junctions) + str((n + 1) % N_junctions) + " = " + str(
                ECs[n % N_junctions]) + "]\n"

        super().__init__(JJcirc_yaml, from_file=False)

        closure_branches = [self.branches[0]]
        trans_mat = np.triu(np.ones((self.N, self.N)), 0) * (-1)

        self.configure(transformation_matrix=trans_mat, closure_branches=closure_branches)

        if converge:
            self.Ncut = self.ConvergeCutoff()
        else:
            self.Ncut = Ncut
            self.SetCutoff(Ncut)

    def SetCutoff(self, Ncut):
        for nc in self.cutoff_names:
            self.__dict__["_" + nc] = Ncut

    def ConvergeCutoff(self):

        Ncut = 1
        for nc in self.cutoff_names:
            self.__dict__["_" + nc] = Ncut

        nvals = np.min([5, (2*Ncut+1)**self.N])
        ntries = 10
        grid = np.linspace(0,1,5)
        grid = grid[0:len(grid)-1]
        sample = np.concatenate((self.cartesian(tuple([grid for r in range(self.N + 1)])),
                                 np.random.rand(ntries, self.N + 1)))

        # sample = np.concatenate((self.cartesian(tuple([[0, 0.25, 0.5, 0.75] for r in range(self.N + 1)])),
        #                          np.random.rand(ntries, self.N + 1)))
        print()
        print("Initiating calculation of converged Ncut. Number of samples:", len(sample))

        for smp in tqdm(range(len(sample))):

            prms = sample[smp]
            print(smp, prms)
            self.__dict__['_Φ1'] = prms[0]
            for r in range(1, self.N + 1):
                self.__dict__['_ng' + str(r)] = prms[r]

            eps = 1
            eigs_new = qt.Qobj(self.hamiltonian()).eigenenergies(sparse=True, sort='low', eigvals=nvals)
            while eps > 10 ** (-10):
                for nc in self.cutoff_names:
                    self.__dict__["_" + nc] = Ncut + 1
                eigs_old = eigs_new
                eigs_new = qt.Qobj(self.hamiltonian()).eigenenergies(sparse=True, sort='low', eigvals=nvals)
                eps = np.mean(np.abs(eigs_old - eigs_new))
                if eps > 10 ** (-10):
                    Ncut = Ncut + 1
                    print("Changed at sample:", (Ncut, sample[smp]))

        print("Final Ncut value:", Ncut)

        return Ncut

    def cartesian(self, arrays, out=None):
        arrays = [np.asarray(x) for x in arrays]
        dtype = arrays[0].dtype

        n = np.prod([x.size for x in arrays])
        if out is None:
            out = np.zeros([n, len(arrays)], dtype=dtype)

        m = int(n / arrays[0].size)
        out[:, 0] = np.repeat(arrays[0], m)
        if arrays[1:]:
            self.cartesian(arrays[1:], out=out[0:m, 1:])
            for j in range(1, arrays[0].size):
                out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
        return out

    def GenerateMomentumBasis(self):
        dms = (2 * self.Ncut + 1) ** self.N
        lst = list(range(dms))
        vbs = []

        while len(lst) > 0:

            bas = [lst[0]]
            tmp = self.TransInd(bas[-1], self.N, self.Ncut)

            while (tmp - bas[0]) != 0:
                bas.append(tmp)
                tmp = self.TransInd(bas[-1], self.N, self.Ncut)

            vbs.append(bas)

            lst = list(set(lst) - set(bas))

        return vbs

    def amps(self, cycle, k):
        N0 = len(cycle)
        seq = np.arange(N0)
        return (1 / np.sqrt(N0)) * np.exp(-1j * 2 * np.pi * (k) * seq)

    def ChargeToTranslation(self):

        # Check if vbs file has been generated, if not then generate it
        flnms = os.listdir()
        flag = 0
        for fname in flnms:
            if 'vbs_' + str((self.N, self.Ncut)) + '.npz' in fname:
                flag = 1
        if flag == 0:
            vbs = np.array(self.GenerateMomentumBasis(), dtype=object)
            np.savez_compressed('vbs_' + str((self.N, self.Ncut)), vbs=vbs)
        else:
            loaded = np.load('vbs_' + str((self.N, self.Ncut)) + '.npz', allow_pickle=True)
            vbs = loaded['vbs'].tolist()

        DIMS = [[2 * self.Ncut + 1 for r in range(self.N)], [2 * self.Ncut + 1 for r in range(self.N)]]

        mms = np.array([n / self.N for n in range(self.N)])
        lns = np.unique(list(map(len, vbs)))

        m_seqs = [[] for r in range(self.N)]
        for N0 in lns:
            tmp = np.array([n / N0 for n in range(N0)])
            for k in range(self.N):
                if mms[k] in tmp:
                    m_seqs[k].append(N0)

        V = []
        sm = 0
        print('Full Hilbert space dimension = ', (2 * self.Ncut + 1) ** self.N)
        for n in tqdm(range(self.N)):
            row = []
            col = []
            data = []
            ind = 0
            for state in tqdm(range(len(vbs))):
                if len(vbs[state]) in m_seqs[n]:
                    vec = self.amps(vbs[state], n / self.N)
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
                                        shape=((2 * self.Ncut + 1) ** self.N, ind)), dims=DIMS))
            # print(V[-1].shape)
        print('Sum of sector dimensions = ', sm)
        if sm != (2 * self.Ncut + 1) ** self.N:
            np.savetxt('WARNING_' + str([self.N, self.Ncut]) + '.txt', [1])
        self.symmetric_basis_transformation = V

    def SectorDiagonalization(self, Nvals):
        self.ChargeToTranslation()
        DIMS = ((2 * self.Ncut + 1) * np.ones((2, self.N), dtype=int)).tolist()
        H = qt.Qobj(self.hamiltonian(), dims=DIMS)
        # print(H.shape)

        data = []
        for k in range(len(self.symmetric_basis_transformation)):
            # print(self.symmetric_basis_transformation[k].dims)
            HF0 = self.symmetric_basis_transformation[k].dag() * H * self.symmetric_basis_transformation[k]
            evals, evecs = HF0.eigenstates(eigvals=Nvals, sparse=True)
            data.append([evals, [self.symmetric_basis_transformation[k] * evecs[r] for r in range(len(evecs))]])

        return data

    def SortedDiagonalization(self, Nvals):

        tst = self.SectorDiagonalization(Nvals)

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

    def SectorDiagonalization_Energies(self, Nvals):

        data = []
        for k in range(len(V)):
            HF0 = V[k].dag() * H * V[k]
            evals = HF0.eigenenergies(eigvals=Nvals, sparse=True)
            data.append(evals)

        return data

    def SortedDiagonalization_Energies(self, H, V, Nvals):
        tst = self.SectorDiagonalization_Energies(H, V, Nvals)

        eigs = np.sort(np.concatenate(tst))

        return eigs, tst

    def TransInd(self, s, N, Ncut):
        return int(
            self.ind2occ(s, N - 1, N, Ncut) * ((2 * Ncut + 1) ** (N - 1)) - (
                        self.ind2occ(s, N - 1, N, Ncut) / (2 * Ncut + 1)) + s / (
                    2 * Ncut + 1))

    def ind2occ(self, s, r, N, Ncut):
        return int((s // ((2 * Ncut + 1) ** (N - r - 1))) % (2 * Ncut + 1))

    def hamiltonian_one_mode_model(self, ph_points = 200):

        ph_list = np.linspace(-self.N * np.pi, self.N * np.pi, ph_points)
        dphi = ph_list[1]-ph_list[0]
        d2d2phi = (qt.tunneling(ph_points, 1) - 2 * qt.identity(ph_points)).full()
        d2d2phi[0, ph_points - 1] = 1
        d2d2phi[ph_points - 1, 0] = 1
        d2d2phi = qt.Qobj(d2d2phi) / (dphi * dphi)



        EC = self.ECs[0]/(1+self.ECs[0]/(np.mean(self.ECs[1:])*self.N))
        h = qt.Qobj(-4 * EC * d2d2phi - (self.N) * np.mean(self.EJs[1:]) * np.diag(np.cos(ph_list / (self.N))) - self.EJs[0] * np.diag(
                np.cos(ph_list + 2 * np.pi * self.Φ1)))

        return h


    def paramSweep(self, param):
        ng_list = np.linspace(-2, 2, 220)
        self.plot_evals_vs_paramvals(param, ng_list, evals_count=7, num_cpus=1, subtract_ground=True)
        plt.title(fr'$N= {self.N} \quad E_j^b = {self.ECs[0]} \quad E_j^a = {self.EJs[1:]} \quad  E_C^b = {self.ECs[0]} \quad E_C^a = {self.ECs[1:]}$')
