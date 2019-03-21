#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy import sparse
from entropart import base

np.seterr(all='raise')


def entropy(array):
    """Compute entropy."""
    # if array is a scalar
    if isinstance(array, (float, np.float32, np.float64)):
        if array <= 0.0:
            return 0.0
        return - array * np.log2(array)

    # if it is a vector or matrix
    try:
        # if it store plogp
        return array.entropy()
    except AttributeError:
        # otherwise use numpy
        array = np.array(array)
        array = array[array > 0]
        return - np.sum(array * np.log2(array))


def delta(h1old, h2old, h1new, h2new, alpha=0.0):
    return (2 - alpha) * (h1new - h1old) - h2new + h2old


def value(pgraph, alpha=0.0):
    h1, h2 = pgraph.entropies()
    return (2 - alpha) * h1 - h2


def get_probabilities(edges, node_num,
                      symmetric=False,
                      return_transition=False):
    """Compute p_ij and p_i at the steady state"""

    graph = edgelist2csr_sparse(edges, node_num)
    steadystate = graph.sum(0)

    if symmetric and not return_transition:
        return (
            graph / graph.sum(),
            np.array(steadystate).flatten() / graph.sum()
        )

    diag = sparse.spdiags(1.0 / steadystate, [0], node_num, node_num)
    transition = graph @ diag

    diff = 1.0
    count = 0
    steadystate = np.array(steadystate).reshape(-1, 1) / steadystate.sum()
    while diff > 1e-10:
        old_ss = steadystate.copy()
        steadystate = transition @ steadystate
        diff = np.abs(np.max(steadystate - old_ss))
        count += 1
        if count > 1e5:
            break

    diag = sparse.spdiags(steadystate.reshape((1, -1)), [0], node_num, node_num)
    if return_transition:
        return transition, diag, np.array(steadystate).flatten()
    else:
        return transition @ diag, np.array(steadystate).flatten()


def edgelist2csr_sparse(edgelist, node_num):
    """Edges as [(i, j, weight), …]"""
    graph = sparse.coo_matrix(
        (
            # data
            [e[2] for e in edgelist],
            # i and j
            ([e[1] for e in edgelist], [e[0] for e in edgelist])
        ),
        shape=(node_num, node_num)
    )
    return sparse.csr_matrix(graph)


def partition2coo_sparse(part):
    """from dict {(i, j, k, …): weight, …}"""
    n_n = len(part)
    n_p = len(np.unique(list(part.values())))
    parts = {v: k for k, v in enumerate(np.unique(list(part.values())))}
    parts = [parts[v] for k, v in part.items()]
    try:
        return sparse.coo_matrix(
            (np.ones(n_n), (list(part.keys()), parts)),
            shape=(n_n, n_p),
            dtype=float,
        )
    except ValueError:
        print(n_p)
        print(n_n)
        raise


def kron(A, B):
    dok = {}
    for n in range(A.shape[0]):
        for pA in A.paths_through_node(n, position=-1):
            for pB in B.paths_through_node(n, position=0):
                dok[tuple(list(pA) + list(pB))] = A[pA] * B[pB]

    return base.SparseMat(dok)


def zeros(node_num):
    return base.SparseMat({}, node_num=0, normalize=False)


def zeros_like(sparsemat):
    return base.SparseMat(
        {},
        node_num=sparsemat.nn,
        normalize=1.0,
        plength=sparsemat._dim
    )

