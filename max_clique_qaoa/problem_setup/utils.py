import itertools
from quri_parts.core.operator import Operator, pauli_label
import numpy as np


def get_hamiltonian_from_QUBO(Q):
    # Convert QUBO to Hamiltonian operator
    hamiltonian = Operator()
    max_i = max(i for i, j in Q.keys())
    max_j = max(j for i, j in Q.keys())
    for i in range(max_i):
        for j in range(i, max_j):
            if abs(Q[i, j]) > 1e-10:
                if i == j:
                    # Diagonal terms
                    hamiltonian.add_term(pauli_label(f"Z{i}"), Q[i, j] / 2)
                else:
                    # Off-diagonal terms
                    hamiltonian.add_term(pauli_label(f"Z{i} Z{j}"), Q[i, j] / 4)

    # # Convert QUBO to Ising
    # h = {}  # Linear terms
    # J = {}  # Quadratic terms
    # offset = 0
    # # Calculate h, J and offset
    # for (i, j), value in Q.items():
    #     if i == j:  # Diagonal terms
    #         h[i] = value / 2
    #         offset += value / 4
    #     else:  # Off-diagonal terms
    #         J[(i, j)] = value / 4
    #         h[i] = h.get(i, 0) + value / 4
    #         h[j] = h.get(j, 0) + value / 4
    #         offset += value / 4
    # return h, J, offset

    return hamiltonian


def get_QUBO_from_graph(G, weights):
    # Convert max clique to QUBO
    n = len(G.nodes())
    Q = {}
    for i in range(n):
        for j in range(i + 1, n):
            if not G.has_edge(i, j):
                Q[(i, j)] = 10  # Penalty for non-adjacent nodes
            else:
                Q[(i, i)] = weights[(i, j)] / 2  # Add edge weight contribution
                Q[(j, j)] = weights[(i, j)] / 2
                Q[(i, j)] = -weights[(i, j)]
    return Q