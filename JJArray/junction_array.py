import scqubits as scq
import numpy as np
import qutip as qt
from tqdm.notebook import tqdm
import os
from scipy.sparse import csr_matrix
from matplotlib import pyplot as plt
scq.settings.T1_DEFAULT_WARNING=False

class junction_array(scq.Circuit):
    def __init__(self, ejs: list, ecs: list, converge: bool = False, ncut: int = 5,
                 monitor=True) -> None:
        """


        Parameters
        ----------
        ejs :
            list of Josephson energies
        ecs :
            list of charging energies
        converge :
            boolean option to converge ncut for a sample of the flux and ngs parameter space
        ncut :
            cutoff value, the same for all junctions

        """

        self.EJs = EJs
        self.ECs = ECs
        self.N = len(EJs) - 1

        self.ncut = None
        self.symmetric_basis_transformation = None

        self.circuit_setup(ncut, converge, monitor)

    def circuit_setup(self, ncut: int, converge: bool, monitor: object = True) -> None:
        """


        :param ncut: manually set cut-off in the charge basis for each degree of freedom
        :param converge: determine whether to automatically search for optimal cut-off
        :param monitor: track convergence of cut-off
        :rtype: Circuit class object
        """
        jj_circ_yaml = "branches:\n"

        n_junctions = self.N + 1
        for n in range(n_junctions):
            jj_circ_yaml = (
                    jj_circ_yaml
                    + '- ["JJ", '
                    + str(n)
                    + ", "
                    + str((n + 1) % n_junctions)
                    + ", EJ"
                    + str(n % n_junctions)
                    + str((n + 1) % n_junctions)
                    + " = "
                    + str(self.EJs[n % n_junctions])
                    + ", EC"
                    + str(n % n_junctions)
                    + str((n + 1) % n_junctions)
                    + " = "
                    + str(self.ECs[n % n_junctions])
                    + "]\n"
            )

        super().__init__(jj_circ_yaml, from_file=False)

        closure_branches = [self.branches[0]]
        transformation_matrix = np.triu(np.ones((self.N, self.N)), 0) * (-1)

        self.configure(
            transformation_matrix=        transformation_matrix, closure_branches=closure_branches
        )

        if converge:
            self.ncut = self.converge_cutoff(monitor)
        else:
            self.ncut = ncut
            self.set_cutoff(ncut)

    def set_cutoff(self, ncut: int) -> None:
        """
        Parameters
        ----------
        ncut :
            cutoff value, the same for all junctions
        """
        for nc in self.cutoff_names:
            self.__dict__["_" + nc] = ncut

    def converge_cutoff(self, monitor: object = True) -> object:
        """
        Parameters
        ----------
        :param monitor: 
        :return: 
        """
        ncut = 1
        for nc in self.cutoff_names:
            self.__dict__["_" + nc] = ncut

        nvals = np.min([5, (2 * ncut + 1) ** self.N])
        ntries = 2
        grid = np.linspace(0, 1, 5)
        grid = grid[0: len(grid) - 1]
        sample = np.concatenate(
            (
                self.cartesian(tuple([grid for _ in range(self.N + 1)])),
                np.random.rand(ntries, self.N + 1),
            )
        )

        if monitor:
            print()
            print(
                "Initiating calculation of converged ncut. Number of samples:", len(sample)
            )
            for smp in tqdm(range(len(sample))):

                prms = sample[smp]
                print(smp, prms)
                self.__dict__["_Φ1"] = prms[0]
                for r in range(1, self.N + 1):
                    self.__dict__["_ng" + str(r)] = prms[r]

                eps = 1
                eigs_new = qt.Qobj(self.hamiltonian()).eigenenergies(
                    sparse=True, sort="low", eigvals=nvals
                )
                while eps > 10 ** (-10):
                    for nc in self.cutoff_names:
                        self.__dict__["_" + nc] = ncut + 1
                    eigs_old = eigs_new
                    eigs_new = qt.Qobj(self.hamiltonian()).eigenenergies(
                        sparse=True, sort="low", eigvals=nvals
                    )
                    eps = np.mean(np.abs(eigs_old - eigs_new))
                    if eps > 10 ** (-10):
                        ncut = ncut + 1
                        print("Changed at sample:", (ncut, sample[smp]))
            print("Final ncut value:", ncut)
        elif not monitor:
            for smp in range(len(sample)):

                prms = sample[smp]
                # print(smp, prms)
                self.__dict__["_Φ1"] = prms[0]
                for r in range(1, self.N + 1):
                    self.__dict__["_ng" + str(r)] = prms[r]

                eps = 1
                eigs_new = qt.Qobj(self.hamiltonian()).eigenenergies(
                    sparse=True, sort="low", eigvals=nvals
                )
                while eps > 10 ** (-10):
                    for nc in self.cutoff_names:
                        self.__dict__["_" + nc] = ncut + 1
                    eigs_old = eigs_new
                    eigs_new = qt.Qobj(self.hamiltonian()).eigenenergies(
                        sparse=True, sort="low", eigvals=nvals
                    )
                    eps = np.mean(np.abs(eigs_old - eigs_new))
                    if eps > 10 ** (-10):
                        ncut = ncut + 1
                        # print("Changed at sample:", (ncut, sample[smp]))

        return ncut

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
                out[j * m: (j + 1) * m, 1:] = out[0:m, 1:]
        return out

    def generate_momentum_basis(self):
        """

        :return:
        :rtype: object
        """
        dms = (2 * self.ncut + 1) ** self.N
        lst = list(range(dms))
        vbs = []

        while len(lst) > 0:

            bas = [lst[0]]
            tmp = self.trans_ind(bas[-1], self.N, self.ncut)

            while (tmp - bas[0]) != 0:
                bas.append(tmp)
                tmp = self.trans_ind(bas[-1], self.N, self.ncut)

            vbs.append(bas)

            lst = list(set(lst) - set(bas))

        return vbs

    def charge_to_translation(self):

        # Check if vbs file has been generated, if not then generate it
        flnms = os.listdir()
        flag = 0
        for fname in flnms:
            if "vbs_" + str((self.N, self.ncut)) + ".npz" in fname:
                flag = 1
        if flag == 0:
            vbs = np.array(self.generate_momentum_basis(), dtype=object)
            np.savez_compressed("vbs_" + str((self.N, self.ncut)), vbs=vbs)
        else:
            loaded = np.load(
                "vbs_" + str((self.N, self.ncut)) + ".npz", allow_pickle=True
            )
            vbs = loaded["vbs"].tolist()

        dims = [
            [2 * self.ncut + 1 for _ in range(self.N)],
            [2 * self.ncut + 1 for _ in range(self.N)],
        ]

        mms = np.array([n / self.N for n in range(self.N)])
        lns = np.unique(list(map(len, vbs)))

        m_seqs = [[] for _ in range(self.N)]
        for N0 in lns:
            tmp = np.array([n / N0 for n in range(N0)])
            for k in range(self.N):
                if mms[k] in tmp:
                    m_seqs[k].append(N0)

        v_op = []
        sm = 0
        print("Full Hilbert space dimension = ", (2 * self.ncut + 1) ** self.N)
        for n in tqdm(range(self.N)):
            row = []
            col = []
            data = []
            ind = 0
            for state in tqdm(range(len(vbs))):
                if len(vbs[state]) in m_seqs[n]:
                    vec = _amps(vbs[state], n / self.N)
                    row.append(vbs[state])
                    col.append((ind * np.ones(len(vec), dtype=int)).tolist())
                    data.append(vec.tolist())
                    ind = ind + 1
            col = [item for sublist in col for item in sublist]
            row = [item for sublist in row for item in sublist]
            data = [item for sublist in data for item in sublist]
            print("Sector " + str(n) + "=", ind)
            sm = sm + ind
            v_op.append(
                qt.Qobj(
                    csr_matrix(
                        (data, (row, col)), shape=((2 * self.ncut + 1) ** self.N, ind)
                    ),
                    dims=dims,
                )
            )
            # print(V[-1].shape)
        print("Sum of sector dimensions = ", sm)
        if sm != (2 * self.ncut + 1) ** self.N:
            np.savetxt("WARNING_" + str([self.N, self.ncut]) + ".txt", [1])
        self.symmetric_basis_transformation = v_op

    def sector_diagonalization(self, nvals):
        self.charge_to_translation()
        dims = ((2 * self.ncut + 1) * np.ones((2, self.N), dtype=int)).tolist()
        h = qt.Qobj(self.hamiltonian(), dims=dims)
        # print(H.shape)

        data = []
        for k in range(len(self.symmetric_basis_transformation)):
            # print(self.symmetric_basis_transformation[k].dims)
            hf0 = (
                    self.symmetric_basis_transformation[k].dag()
                    * h
                    * self.symmetric_basis_transformation[k]
            )
            evals, evecs = hf0.eigenstates(eigvals=nvals, sparse=True)
            data.append(
                [
                    evals,
                    [
                        self.symmetric_basis_transformation[k] * evecs[r]
                        for r in range(len(evecs))
                    ],
                ]
            )

        return data

    def sorted_diagonalization(self, nvals):

        tst = self.sector_diagonalization(nvals)

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

    def trans_ind(self, s, number_junctions, ncut):
        return int(
            _ind2occ(s, number_junctions - 1, number_junctions, ncut) * ((2 * ncut + 1) ** (number_junctions - 1))
            - (_ind2occ(s, number_junctions - 1, number_junctions, ncut) / (2 * ncut + 1))
            + s / (2 * ncut + 1)
        )

    def hamiltonian_one_mode_model(self, ph_points=200):

        ph_list = np.linspace(-self.N * np.pi, self.N * np.pi, ph_points)
        dphi = ph_list[1] - ph_list[0]
        d2d2phi = (qt.tunneling(ph_points, 1) - 2 * qt.identity(ph_points)).full()
        d2d2phi[0, ph_points - 1] = 1
        d2d2phi[ph_points - 1, 0] = 1
        d2d2phi = qt.Qobj(d2d2phi) / (dphi * dphi)

        ec = self.ECs[0] / (1 + self.ECs[0] / (np.mean(self.ECs[1:]) * self.N))
        h = qt.Qobj(
            -4 * ec * d2d2phi
            - self.N * np.mean(self.EJs[1:]) * np.diag(np.cos(ph_list / self.N))
            - self.EJs[0] * np.diag(np.cos(ph_list + 2 * np.pi * self.Φ1))
        )

        return h

    def param_sweep(self, param):
        ng_list = np.linspace(-2, 2, 220)
        self.plot_evals_vs_paramvals(
            param, ng_list, evals_count=7, num_cpus=1, subtract_ground=True
        )
        plt.title(
            rf"$N= {self.N} \quad E_j^b = {self.ECs[0]} \quad E_j^a = {self.EJs[1:]} \quad  E_C^b = {self.ECs[0]} \quad E_C^a = {self.ECs[1:]}$"
        )

    def get_effective_coherence(self, param: object, state0: object, state1: object, converge: object = True) -> object:

        self.generate_all_noise_methods()
        fract_sweep = np.linspace(0.5, 1.5, 20)
        if param[0] == "EJ":
            t1_eff = []
            t2_eff = []
            ej_temp = self.EJs[param[1]]
            prm_sweep = fract_sweep * ej_temp
            for prm in tqdm(prm_sweep):
                self.EJs[param[1]] = prm
                self.circuit_setup(self.ncut, converge=converge, monitor=False)

                t1_eff.append(self.t1_effective(i=int(state1), j=int(state0), total=False))
                t2_eff.append(self.t2_effective())

            self.EJs[param[1]] = ej_temp

        return prm_sweep, t1_eff, t2_eff


def _ind2occ(s, r, number_junctions, ncut):
    return int((s // ((2 * ncut + 1) ** (number_junctions - r - 1))) % (2 * ncut + 1))


def _amps(cycle, k):
    n0 = len(cycle)
    seq = np.arange(n0)
    return (1 / np.sqrt(n0)) * np.exp(-1j * 2 * np.pi * k * seq)
