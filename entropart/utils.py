#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse


def entropy(array):
    """Compute entropy."""
    # if array is a scalar
    if isinstance(array, (float, np.float32)):
        if np.isclose(float(array), 0.0):
            return 0.0
        return - array * np.log2(array)
    # if it is a vector
    try:
        # if it store plogp
        return - np.sum([p.plogp for p in array])
    except AttributeError:
        # otherwise use numpy
        array = np.array(array)
        return - np.sum(array * np.log2(array))


def delta(h1pre, h2pre, h1post, h2post, alpha=0.0, beta=1.0):
    return (2 - alpha) * (h1post - h1pre) - h2post + h2pre


def value(pgraph, alpha=0.0):
    h1, h2 = pgraph.entropies()
    return (2 - alpha) * h1 - h2


def get_probabilities(edges, node_num, symmetric=False):
    """Compute p_ij and p_i at the steady state"""
    graph = sparse.coo_matrix(
        (
            # data
            [e[2] for e in edges],
            # i and j
            ([e[1] for e in edges], [e[0] for e in edges])
        ),
        shape=(node_num, node_num)
    )
    graph = sparse.csr_matrix(graph)
    steadystate = graph.sum(0)

    if symmetric:
        return (
            graph / graph.sum(),
            np.array(steadystate).flatten() / graph.sum()
        )

    diag = sparse.spdiags(1.0 / steadystate, [0], node_num, node_num)
    transition = graph @ diag

    diff = 1.0
    count = 0
    steadystate = np.array(steadystate).reshape(4, 1) / steadystate.sum()
    while diff > 1e-10:
        old_ss = steadystate.copy()
        steadystate = transition @ steadystate
        diff = np.abs(np.max(steadystate - old_ss))
        count += 1
        if count > 1e5:
            break

    diag = sparse.spdiags(steadystate.reshape((1, 4)), [0], node_num, node_num)
    return transition @ diag, np.array(steadystate).flatten()
