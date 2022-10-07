import numpy as np
import qutip as qt


def Op(Os, N, r):
    ops = []
    Id = qt.identity(Os.shape[0])

    if (r > 0) and (r < N - 1):
        [ops.append(Id) for r in range(r)]
        ops.append(Os)
        [ops.append(Id) for r in range(N - r - 1)]
        return qt.tensor(ops)

    elif r == 0:
        ops.append(Os)
        [ops.append(Id) for r in range(N - 1)]
        return qt.tensor(ops)

    elif r == N - 1:
        [ops.append(Id) for r in range(N - 1)]
        ops.append(Os)
        return qt.tensor(ops)


def ChargingEnergy_array(N, EC, ECb):
    EE = np.linalg.inv((1 / 8) * np.diag(1 / EC) + (1 / 8) * (1 / ECb) * np.ones((N, N)))

    return EE


def H_array(phi_ext, N, Ncut, EJ, EC, EJb, ECb):
    Ncharge = qt.charge(Ncut)
    Tp = qt.Qobj(np.diag(np.ones(2 * Ncut + 1 - 1), 1))
    Tm = Tp.dag()
    CosPhi = (Tp + Tm) / 2

    EE = np.linalg.inv((1 / 8) * np.diag(1 / EC) + (1 / 8) * (1 / ECb) * np.ones((N, N)))

    # Charging energies
    ENN = 0
    for r2 in range(0, N):
        for r1 in range(0, N):
            ENN = ENN + 0.5 * EE[r1, r2] * Op(Ncharge, N, r1) * Op(Ncharge, N, r2)

    # Josephson inductive energies
    EJJ = 0
    for r in range(N):
        EJJ = EJJ - EJ[r] * Op(CosPhi, N, r)

    # Interacting cosine term
    if N > 0:
        smExpPPhi = 1
        for r in range(0, N):
            smExpPPhi = smExpPPhi * Op(Tp, N, r)

        phase = np.exp(2 * np.pi * 1j * phi_ext)
        sum_CosPhi = EJb * (phase * smExpPPhi + np.conjugate(phase) * smExpPPhi.dag()) / 2
        HF = ENN + EJJ - sum_CosPhi
    else:
        HF = ENN + EJJ
    return (HF)


# Basic operators of the system


def Phase_j(r1, N, Ncut):
    M = np.diag(np.ones(2 * Ncut + 1 - 1), k=1)

    M[0, 2 * Ncut + 1 - 1] = 1
    M[2 * Ncut + 1 - 1, 0] = 1

    return (Op(qt.Qobj(M), N, r1))


def Charge_j(r1, N, Ncut):
    M = qt.charge(Ncut, -Ncut)

    return (Op(qt.Qobj(M), N, r1))


def NN(r1, r2, N, Ncut):
    Ncharge = qt.charge(Ncut)

    return (Op(Ncharge, N, r1) * Op(Ncharge, N, r2))


def Cos_phi12(r1, r2, phi_ext, N, Ncut):
    Tp = qt.Qobj(np.diag(np.ones(2 * Ncut + 1 - 1), 1))
    Tm = Tp.dag()
    CosPhi = (Tp + Tm) / 2

    if N > 0:
        smExpPPhi = 1
        smExpPPhi = Op(Tp, N, r1) * Op(Tp, N, r2)

        phase = np.exp(2 * np.pi * 1j * phi_ext)
        smCosPhi = (phase * smExpPPhi + np.conjugate(phase) * smExpPPhi.dag()) / 2
        HF = - smCosPhi
    else:
        HF = 0

    return (HF)


def TensorArrange(O, DIMS):
    return qt.Qobj(O.full(), dims=DIMS)
