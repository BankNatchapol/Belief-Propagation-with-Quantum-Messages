import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.library import save_probabilities_dict

import cvxpy as cp
import time

from bpqm import *
from cloner import *
from linearcode import *
from cvxpy_partial_trace import *


def TP(exprs):
    out = exprs[0]
    for mat in exprs[1:]:
        out = np.kron(out, mat)
    return out


def decode_bpqm(
    code,
    theta,
    cloner,
    height,
    mode,
    bit=None,
    order=None,
    only_zero_codeword=True,
    debug=False,
    coset=None,
):
    assert mode in ["bit", "codeword"]
    if mode == "bit":
        order = [bit]

    # 1) build computation graphs
    cgraphs = [code.get_computation_graph(f"x{b}", height) for b in order]

    # 2) determine qubit counts
    n_data_qubits = max(sum(occ.values()) for _, occ, _ in cgraphs)
    n_data_qubits = max(n_data_qubits, code.n)
    n_qubits = n_data_qubits + len(order) - 1

    # 3) generate main circuit
    qc = QuantumCircuit(n_qubits)
    for i, (graph, occ, root) in enumerate(cgraphs):
        # qubit mapping
        leaves = [n for n in graph.nodes() if graph.nodes[n]["type"] == "output"]
        leaves = sorted(leaves, key=lambda s: int(s.split("_")[1]))
        qm = {f"y{j}_0": j for j in range(code.n)}
        idx = code.n
        for node in leaves:
            if int(node.split("_")[1]) > 0:
                qm[node] = idx
                idx += 1

        # annotate graph
        cloner.mark_angles(graph, occ)
        for node in leaves:
            graph.nodes[node]["qubit_idx"] = qm[node]

        # build BPQM + cloner
        qc_bpqm = QuantumCircuit(n_qubits)
        meas_idx, angles = tree_bpqm(graph, qc_bpqm, root=root)
        qc_cloner = cloner.generate_cloner_circuit(graph, occ, qm, n_qubits)

        # append & uncompute with compose
        qc.compose(qc_cloner, inplace=True)
        qc.barrier()
        qc.compose(qc_bpqm, inplace=True)
        qc.barrier()
        if i < len(order) - 1:
            qc.h(meas_idx)
            qc.cx(meas_idx, n_data_qubits + i)
            qc.h(meas_idx)
            qc.barrier()
            qc.compose(qc_bpqm.inverse(), inplace=True)
            qc.barrier()
            qc.compose(qc_cloner.inverse(), inplace=True)
            qc.barrier()
        else:
            qc.h(meas_idx)

    # snapshot as dict
    cw_qubits = list(range(n_data_qubits, n_data_qubits + len(order) - 1)) + [meas_idx]
    qc.save_probabilities_dict(label="prob", qubits=cw_qubits)

    # simulate
    backend = AerSimulator(method="statevector")
    codewords = [[0] * code.n] if only_zero_codeword else code.get_codewords()
    if coset is not None:
        cos = np.array(coset, dtype=int)
        codewords = [list((np.array(cw, dtype=int) + cos) % 2) for cw in codewords]
    prob = 0.0
    for cw in codewords:
        qc_init = QuantumCircuit(n_qubits)
        plus = np.array([np.cos(0.5 * theta), np.sin(0.5 * theta)])
        minus = np.array([np.cos(0.5 * theta), -np.sin(0.5 * theta)])
        for j, v in enumerate(cw):
            qc_init.initialize(plus if v == 0 else minus, [j])

        combined = qc_init.compose(qc)
        full_qc = transpile(combined, backend)
        result = backend.run(full_qc).result()

        probs = result.data()["prob"]
        key = int("".join(str(cw[i]) for i in reversed(order)), 2)
        prob += probs.get(key, 0.0) / len(codewords)

    return prob


def decode_bit_optimal_quantum(code, theta, index):
    rho0 = np.zeros((2**code.n, 2**code.n), complex)
    rho1 = np.zeros((2**code.n, 2**code.n), complex)
    vecs = [
        np.array([[np.cos(0.5 * theta)], [np.sin(0.5 * theta)]]),
        np.array([[np.cos(0.5 * theta)], [-np.sin(0.5 * theta)]]),
    ]
    codewords = code.get_codewords()

    for cw in [c for c in codewords if c[index] == "0"]:
        psi = TP([vecs[int(b)] for b in cw])
        rho0 += psi @ psi.T / (0.5 * len(codewords))
    for cw in [c for c in codewords if c[index] == "1"]:
        psi = TP([vecs[int(b)] for b in cw])
        rho1 += psi @ psi.T / (0.5 * len(codewords))

    eigs = np.linalg.eigvals(rho0 - rho1)
    return 0.5 + 0.25 * np.sum(np.abs(eigs))


def decode_codeword_PGM(code, theta):
    vecs = [
        np.array([[np.cos(0.5 * theta)], [np.sin(0.5 * theta)]]),
        np.array([[np.cos(0.5 * theta)], [-np.sin(0.5 * theta)]]),
    ]
    codewords = code.get_codewords()

    rho = sum(
        (TP([vecs[int(b)] for b in cw]) @ TP([vecs[int(b)] for b in cw]).T)
        for cw in codewords
    ) / len(codewords)
    vals, vecs_mat = np.linalg.eig(rho)
    inv_sqrt = vecs_mat @ np.diag(vals ** (-0.5)) @ np.linalg.inv(vecs_mat)

    return sum(
        abs(
            (
                TP([vecs[int(b)] for b in cw]).T
                @ (inv_sqrt @ TP([vecs[int(b)] for b in cw]))
            )
        ).item()
        ** 2
        for cw in codewords
    ) / len(codewords)


def decode_codeword_optimal_quantum(code, theta):
    sigma = cp.Variable((2**code.n, 2**code.n), PSD=True)
    codewords = code.get_codewords()
    vecs = [
        np.array([[np.cos(0.5 * theta)], [np.sin(0.5 * theta)]]),
        np.array([[np.cos(0.5 * theta)], [-np.sin(0.5 * theta)]]),
    ]

    constraints = []
    for cw in codewords:
        psi = TP([vecs[int(b)] for b in cw])
        constraints.append(sigma >> (psi @ psi.T) / len(codewords))

    prob = cp.Problem(cp.Minimize(cp.trace(sigma)), constraints).solve(solver=cp.SCS)
    return float(prob)


def decode_bit_optimal_classical(code, theta, index):
    if theta < 1e-8:
        return 0.5
    p_r = 0.5 * (1 + np.sin(theta))
    p_w = 0.5 * (1 - np.sin(theta))
    codewords = code.get_codewords()

    success = 0.0
    for m in range(2**code.n):
        y = list(map(float, bin(m)[2:].zfill(code.n)))

        def like(c):
            return p_w ** sum(abs(ci - yi) for ci, yi in zip(c, y)) * p_r ** sum(
                ci == yi for ci, yi in zip(c, y)
            )

        P0 = sum(like(c) for c in codewords if c[index] == "0")
        P1 = sum(like(c) for c in codewords if c[index] == "1")
        out = 0 if P0 > P1 else 1
        success += sum(like(c) for c in codewords if int(c[index]) == out) / len(
            codewords
        )
    return success


def decode_codeword_optimal_classical(code, theta):
    if theta < 1e-8:
        return 1.0 / (2**code.k)
    p_r = 0.5 * (1 + np.sin(theta))
    p_w = 0.5 * (1 - np.sin(theta))
    codewords = code.get_codewords()

    success = 0.0
    for m in range(2**code.n):
        y = list(map(float, bin(m)[2:].zfill(code.n)))
        best = max(
            codewords,
            key=lambda c: (
                p_w ** sum(abs(ci - yi) for ci, yi in zip(c, y))
                * p_r ** sum(ci == yi for ci, yi in zip(c, y))
            ),
        )
        success += (
            p_w ** sum(abs(int(b) - yi) for b, yi in zip(best, y))
            * p_r ** sum(int(b) == yi for b, yi in zip(best, y))
        ) / len(codewords)
    return success


def decode_syndrome_bpqm(
    H, syndrome, theta, cloner, height, mode, bit=None, order=None, debug=False
):
    """Decode an error pattern from a given syndrome using BPQM."""
    G = generator_from_parity_check(H)
    code = LinearCode(G, H)
    offset = solve_syndrome(H, syndrome)
    return decode_bpqm(
        code,
        theta,
        cloner,
        height,
        mode,
        bit=bit,
        order=order,
        only_zero_codeword=False,
        debug=debug,
        coset=offset,
    )
