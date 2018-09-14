#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
from scipy import sparse
from collections import Counter
import logging
import sys
from . import utils

try:
    import tqdm
except ModuleNotFoundError:
    pass


FORMAT = '%(asctime)-15s || %(message)s'
logging.basicConfig(format=FORMAT)
log = logging.getLogger("EntroLog")
log.setLevel(logging.WARNING)
SYMBOLS = '0123456789ABCDEFGHJKLMNOPQRSTUVWXYZabcdefghjklmnopqrstuvwxyz'


class PGraph(object):
    """A Graph with partition."""

    def __init__(self, graph, compute_steady=True, init_part=None):
        """A graph with partition

        :graph: nx.[Di]Graph()
        :compute_steady: bool
        :init_part: dict {node: part, ...}

        """

        if isinstance(graph, nx.DiGraph):
            self._isdirected = True
        elif isinstance(graph, nx.Graph):
            self._isdirected = False
        else:
            raise ValueError('Need nx.[Di]Graph, not ' + str(type(graph)))
        _graph = nx.DiGraph(graph)
        nn = _graph.number_of_nodes()

        # node to index map
        self._n2i = {n: i for i, n in enumerate(_graph.nodes())}
        # index to node map
        self._i2n = {i: n for n, i in self._n2i.items()}

        # set up partition
        if init_part is None:
            # every node in its own partition
            _part = sparse.eye(
                len(_graph.nodes()),
                dtype=float,
                format='coo'
            )
        else:
            _part = utils.partition2coo_sparse(
                {self._n2i[k]: v for k, v in init_part.items()}
            )

        # save the partition as a double dict
        self._i2p = {}
        self._p2i = {}
        for n, p in zip(_part.row, _part.col):
            self._i2p[n] = p
            if p not in self._p2i:
                self._p2i[p] = set([])
            self._p2i[p].add(n)
        _part = sparse.csr_matrix(_part)
        self._nn, self._np = _part.shape

        # compute the probabilities p(i, j) and p(i)
        self.__st_computed = compute_steady
        if compute_steady:
            edges = [
                (self._n2i[i], self._n2i[j], w)
                for i, j, w in _graph.edges.data('weight', default=1.0)
            ]
            pij, pi = utils.get_probabilities(
                edges, nn,
                symmetric=not self._isdirected,
                return_transition=False
            )

            # projected probabilities
            p_pij = _part.transpose() @ pij @ _part
            # self._pij = SparseMat(self._pij)
            p_pi = _part.transpose() @ pi
        else:
            pij = {
                (self._n2i[i], self._n2i[j]): w
                for i, j, w in _graph.edges.data('weight', default=1.0)
            }
            pi = np.zeros(self._nn)
            p_pij = {}
            for (i, j), w in pij.items():
                pi[i] += w
                ppi, ppj = self._i2p[i], self._i2p[j]
                if (ppi, ppj) in p_pij:
                    p_pij[(ppi, ppj)] += w
                else:
                    p_pij[(ppi, ppj)] = w

            p_pi = np.zeros(self._np)
            for (i, j), w in p_pij.items():
                p_pi[i] += w

        self._pij = SparseMat(pij, normalize=True)
        self._pi = pi / pi.sum()
        self._ppij = SparseMat(p_pij, normalize=True)
        self._ppi = p_pi / p_pi.sum()

        self._reset()

    @property
    def np(self):
        return self._np

    @property
    def nn(self):
        return self._nn

    @property
    def st_computed(self):
        return self.__st_computed

    def _get_random_move(self, inode=None, algorithm='correl'):
        if inode is None:
            inode = np.random.randint(self._nn)
        old_part = self._i2p[inode]

        if algorithm == 'random':
            return inode, np.random.randint(self._np), 1.0

        elif algorithm == 'peixoto':
            n_ego = self._pij.get_egonet(inode)
            n_ego = self._pij.get_egonet(inode, direction='out')
            if n_ego is None:
                return inode, old_part, 1.0
            link, link_prob = n_ego.get_random_entry()

            # get all partitions connected to p1
            # indx represents:
            # 0: out link
            # -1: in link
            indx = 0
            if link[0] == link[-1]:
                # self loop
                p1 = self._i2p[link[-1]]
                p1_ego = self._ppij.get_egonet(p1)
            elif link[0] == inode:
                # outgoing link
                p1 = self._i2p[link[-1]]
                p1_ego = self._ppij.get_egonet(p1, direction='in')
            else:
                # incoming link
                p1 = self._i2p[link[0]]
                p1_ego = self._ppij.get_egonet(p1, direction='out')
                indx = -1
            if p1_ego is None:
                return inode, old_part, 1.0

            link, p_prob, probs = p1_ego.get_random_entry(return_all_probs=True)

            R_t = self._np * np.min(probs) / (self._np * np.min(probs) + 1.0)
            if np.random.rand() < R_t:
                return inode, np.random.randint(self._np), 1 / self._np

            return inode, link[indx], p_prob / (p_prob + link_prob)

        else:

            n_ego = self._pij.get_egonet(inode, direction='out')
            if n_ego is None:
                return inode, old_part, 1.0
            # prob of getting out of inode and arriving to any partition.
            n2p_arr = np.zeros(len(self._ppi), dtype=float)
            for i, v in n_ego:
                n2p_arr[self._i2p[i[-1]]] += v

            probs = self._ppij.dot(n2p_arr, indx=-1)
            probs /= probs.sum()

            new_part = np.random.choice(np.arange(self._np), p=probs)
            return inode, new_part, probs[old_part] / probs[new_part]

    def _get_best_merge(self, **kwargs):
        best = {
            'parts': (None, None),
            'delta': -np.inf
        }
        for p1 in range(self._np):
            for p2 in range(p1 + 1, self._np):
                d = self._try_merge(p1, p2, **kwargs)
                if d > best['delta']:
                    best = {
                        'parts': (p1, p2),
                        'delta': d
                    }
        return best['parts']

    def _try_merge(self, p1, p2, **kwargs):
        p1_ego = self._ppij.get_egonet(p1)
        p2_ego = self._ppij.get_egonet(p2)
        p12 = p1_ego | p2_ego
        H2pre = utils.entropy(p12)

        p12 = p12.merge_colrow(p1, p2)
        H2post = utils.entropy(p12)

        h1pre = utils.entropy(self._ppi[p1]) + utils.entropy(self._ppi[p2])
        h1post = utils.entropy(self._ppi[p1] + self._ppi[p2])
        return utils.delta(h1pre, H2pre, h1post, H2post, **kwargs)

    def _try_move_node(self, inode, partition, bruteforce=False, **kwargs):
        """deltaH"""

        if (inode, partition) in self._tryed_moves:
            return self._tryed_moves[(inode, partition)]

        # check if we are moving to the same partition
        if self._i2p[inode] == partition:
            return None

        # check if starting partition has just one node
        if len(self._p2i[self._i2p[inode]]) == 1:
            return None

        old_part = self._i2p[inode]
        if bruteforce:
            self._move_node(inode, partition)
            fnew = utils.value(self, **kwargs)
            self._move_node(inode, old_part)
            fold = utils.value(self, **kwargs)
            return fnew - fold

        # node ego nets (projected to partitions)
        ego_node = self._pij.get_egonet(inode)
        proj_ego_org = ego_node.project(self._i2p)
        proj_ego_dst = ego_node.project(self._i2p,
                                        move_node=(inode, partition))

        # partition ego-nets (origin)
        # just get the values that would be changed removing the node
        part_orig = self._ppij.get_submat([old_part, partition])
        part_dest = part_orig + proj_ego_dst
        part_dest -= proj_ego_org

        # part_ego_org = self._ppij.get_from_sparse(
        #     proj_ego_org | proj_ego_dst
        # )

        # partition ego-nets (destination)
        # part_ego_dst = part_ego_org.copy()
        # part_ego_dst += proj_ego_dst
        # part_ego_dst -= proj_ego_org

        h1org = utils.entropy(self._ppi[old_part]) +\
            utils.entropy(self._ppi[partition])
        # if self._pi[inode] > self._ppi[old_part]:
        #     raise ValueError('noooo {} - {}'.format(self._ppi[old_part],
        #                                             self._pi[inode]))
        h1dst = utils.entropy(self._ppi[old_part] - self._pi[inode]) +\
            utils.entropy(self._ppi[partition] + self._pi[inode])

        # H2org = utils.entropy(part_ego_org)
        # H2dst = utils.entropy(part_ego_dst)
        H2org = utils.entropy(part_orig)
        H2dst = utils.entropy(part_dest)

        d = utils.delta(h1org, H2org, h1dst, H2dst, **kwargs)
        self._tryed_moves[(inode, partition)] = d
        return d

    def _move_node(self, inode, partition):
        if self._i2p[inode] == partition:
            return None
        pnode = self._pi[inode]
        old_part = self._i2p[inode]
        self._ppi[old_part] -= pnode
        self._ppi[partition] += pnode

        ego_node = self._pij.get_egonet(inode)
        proj_ego_org = ego_node.project(self._i2p)
        proj_ego_dst = ego_node.project(
            self._i2p, move_node=(inode, partition))
        self._ppij += proj_ego_dst
        self._ppij -= proj_ego_org

        self._i2p[inode] = partition
        self._p2i[old_part].remove(inode)
        self._p2i[partition].add(inode)

        self._reset()

    def _project_dict(self, dcts, move_node=None):
        """Get the projected column and row of a given node.

        :dict: tuple of (dict, int, dict);
        :move_node: tuple (node, new partition)
        """
        if move_node is not None:
            old_part = self._i2p[move_node[0]]
            self._i2p[move_node[0]] = move_node[1]
        data = (
            Counter([(self._i2p[k], v) for k, v in dcts[0].items()]),
            dcts[1],
            Counter([(self._i2p[k], v) for k, v in dcts[2].items()])
        )
        if move_node is not None:
            self._i2p[move_node[0]] = old_part
        return data

    def project(self, node):
        try:
            out = (self._i2p[i] for i in node)
        except TypeError:
            out = self._i2p[node]
        return out

    def merge_partitions(self, part1, part2):
        part1, part2 = sorted([part1, part2])

        # self._ppi
        self._ppi[part1] += self._ppi[part2]
        self._ppi = np.array(
            [self._ppi[i] for i in range(self._np) if i != part2]
        )

        # self.__ppij
        self._ppij = self._ppij.merge_colrow(part1, part2)

        # self._i2p
        for node in self._p2i[part2]:
            self._i2p[node] = part1

        # self._p2i
        self._p2i[part1] = self._p2i[part1] | self._p2i[part2]
        del self._p2i[part2]
        for part in range(part2 + 1, self._np):
            for node in self._p2i[part]:
                self._i2p[node] = part - 1

            self._p2i[part - 1] = self._p2i.pop(part)

        self._np -= 1
        self._reset()

    def sum(self):
        return np.sum([float(n) for n in self._ppi])

    def _reset(self):
        self._tryed_moves = {}

    def nodes(self):
        for n in self._n2i.keys():
            yield n

    def parts(self):
        for p in self._p2i.keys():
            yield p

    def __repr__(self):
        return "Graph with {} nodes {} edges and {} partitions".format(
            self._nn, len(self._pij._dok), self._np
        )

    def print_partition(self):
        try:
            strng = ''.join([SYMBOLS[self._i2p[i]] for i in range(self._nn)])
        except IndexError:
            return 'Too many symbols!'
        if len(strng) > 80:
            return strng[:78] + '…'
        else:
            return strng

    def entropies(self):
        return utils.entropy(self._ppi), utils.entropy(self._ppij)

    def partition(self):
        return {self._i2n[i]: p for i, p in self._i2p.items()}


class Prob(object):
    """A class to store the probability p and the plogp value."""
    def __init__(self, value):
        """Given a float or a Prob, store p and plogp."""
        if float(value) < 0.0:
            raise ValueError('Must be non-negative.')
        self.__p = float(value)
        if isinstance(value, Prob):
            self.__plogp = value.plogp
        else:
            self.__update_plogp()

    def __update_plogp(self):
        if self.__p > 0.0:
            self.__plogp = self.__p * np.log2(self.__p)
        else:
            self.__plogp = 0.0

    @property
    def plogp(self):
        return self.__plogp

    @property
    def p(self):
        return self.__p

    def copy(self):
        return Prob(self)

    def __float__(self):
        return self.__p

    def __iadd__(self, other):
        self.__p += float(other)
        self.__update_plogp()
        return self

    def __add__(self, other):
        new = self.copy()
        new += other
        return new

    def __isub__(self, other):
        self.__p -= float(other)
        self.__update_plogp()
        return self

    def __sub__(self, other):
        new = self.copy()
        new -= other
        return new

    def __imul__(self, other):
        p = self.__p
        self.__p *= float(other)
        if isinstance(other, Prob):
            self.__plogp = other.p * self.plogp + p * other.plogp
        else:
            self.__update_plogp()
        return self

    def __mul__(self, other):
        new = self.copy()
        new *= other
        return new

    def __itruediv__(self, other):
        self.__p /= float(other)
        self.__update_plogp()
        return self

    def __truediv__(self, other):
        new = self.copy()
        new /= other
        return new

    def __repr__(self):
        return '{:g} [{:g}]'.format(self.__p,
                                      self.__plogp)

    def __eq__(self, other):
        # TODO: add approx?
        return self.__p == float(other)

    # set inverse operators
    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__
    __rtruediv__ = __truediv__


class SparseMat(object):
    """A sparse matrix with column and row slicing capabilities"""

    def __init__(self, mat, node_num=None, normalize=False):
        """Initiate the matrix

        :mat: scipy sparse matrix or
                list of ((i, j, ...), w) or
                dict (i, j, ...): w
        """
        if isinstance(mat, sparse.spmatrix):
            mat = sparse.coo_matrix(mat)
            self._dok = {
                (i, j): Prob(d) for i, j, d in zip(mat.col, mat.row, mat.data)
            }
            if node_num is None:
                self._nn = mat.shape[0]
            self._dim = 2
        elif isinstance(mat, dict):
            self._dok = {k: Prob(v) for k, v in mat.items()}
            if node_num is None:
                self._nn = np.max([dd for d in self._dok for dd in d]) + 1
            # get the first key of the dict
            val = next(iter(self._dok.keys()))
            self._dim = len(val)
        else:
            self._dok = {i: Prob(d) for i, d in mat}
            if node_num is None:
                self._nn = np.max([dd for d in self._dok for dd in d])
            self._dim = len(mat[0][0])

        if node_num is not None:
            self._nn = node_num

        if isinstance(normalize, (float, Prob)):
            self._norm = Prob(normalize)
        elif normalize:
            vsum = np.sum([float(v) for v in self._dok.values()])
            self._norm = Prob(vsum)
        elif not normalize:
            self._norm = Prob(1.0)
        else:
            raise ValueError()

        if self._norm == 0.0:
            raise ValueError('This is a zero matrix')

        self.__update_all_paths()
        # self.checkme()

    def __update_all_paths(self):
        # for each node, all paths that go through it
        self.__p_thr = [set() for _ in range(self._nn)]
        for path in self._dok.keys():
            for i in path:
                try:
                    self.__p_thr[i].add(path)
                except IndexError:
                    print(path, self._nn)
                    raise

    def entropy(self):
        sum_plogp = np.sum([p.plogp for p in self._dok.values()])
        return (self._norm.plogp - sum_plogp) / self._norm.p

    @property
    def shape(self):
        return tuple([self._nn] * self._dim)

    def checkme(self):
        log.info('{} -- NN {}; NL {}'.format(
            self.__class__.__name__,
            self._nn,
            len(self._dok)
        ))

    def size(self):
        return len(self._dok)

    def project(self, part, move_node=None):
        """Returns a new SparseMat projected to part"""

        # if a node should be reassigned
        if move_node is not None:
            old_part = part[move_node[0]]
            part[move_node[0]] = move_node[1]

        new_dok = {}
        for path, val in self._dok.items():
            new_indx = tuple(part[i] for i in path)
            if new_indx in new_dok:
                new_dok[new_indx] += val.copy()
            else:
                new_dok[new_indx] = val.copy()

        # fix partition before returning
        if move_node is not None:
            part[move_node[0]] = old_part

        return SparseMat(
            new_dok,
            node_num=max(list(part.values())) + 1,
            normalize=self._norm,
        )

    def copy(self):
        return SparseMat(
            {path[:]: w.copy() for path, w in self._dok.items()},
            node_num=self._nn,
            normalize=self._norm,
        )

    def dot(self, other, indx):
        if not isinstance(other, np.ndarray):
            raise TypeError(
                'other should be numpy.ndarray, not {}'.format(type(other)))
        out = np.zeros_like(other, dtype=float)
        for path, w in self._dok.items():
            out[path[indx]] += float(w) * other[path[1 + indx]]
        return out / self._norm.p

    def get_egonet(self, node, direction='both'):
        """Return the adjacency matrix of the ego net of node node."""
        if direction == 'both':
            slist = [(p, self._dok[p]) for p in self.__p_thr[node]]
        elif direction == 'out':
            slist = [
                (p, self._dok[p]) for p in self.__p_thr[node] if p[0] == node
            ]
        elif direction == 'in':
            slist = [
                (p, self._dok[p]) for p in self.__p_thr[node] if p[-1] == node
            ]
        if len(slist) < 1:
            return None
        return SparseMat(
            slist,
            node_num=self._nn,
            normalize=self._norm,
        )

    def get_submat(self, nodelist):
        nodelist = set(nodelist)
        plist = [p for p in self._dok if any(n in p for n in nodelist)]
        return SparseMat(
            {p: self._dok[p] for p in plist},
            node_num=self._nn,
            normalize=self._norm,
        )

    def get_random_entry(self, return_all_probs=False):
        probs = np.array([float(n) for n in self._dok.values()])
        probs /= probs.sum()

        # choose one neighbour based on probs
        link_id = np.random.choice(len(self._dok), p=probs)
        link_prob = probs[link_id]
        link = list(self._dok.keys())[link_id]
        if return_all_probs:
            return link, link_prob, probs
        return link, link_prob

    def paths_through_node(self, node, position=0):
        return [p for p in self.__p_thr[node] if p[position] == node]

    def __iter__(self):
        for k, v in self._dok.items():
            yield k, v / self._norm

    def __getitem__(self, item):
        return self._dok[item] / self._norm

    def __iadd__(self, other):
        ratio = self._norm / other._norm
        for p, d in other._dok.items():
            if p in self._dok:
                self._dok[p] += d * ratio
            else:
                self._dok[p] = d * ratio
            for i in p:
                self.__p_thr[i].add(p)
        return self

    def __add__(self, other):
        new = self.copy()
        new += other
        return new

    def __isub__(self, other):
        ratio = self._norm / other._norm
        """Can provide negative values."""
        for p, d in other._dok.items():
            d_norm = d * ratio
            if np.isclose(float(self._dok[p]), float(d_norm)):
                for i in p:
                    self.__p_thr[i].discard(p)
                del self._dok[p]
            else:
                # no need to update __p_thr
                self._dok[p] -= d_norm
        return self

    def __sub__(self, other):
        new = self.copy()
        new -= other
        return new

    def set_path(self, path, weight):
        self._dok[path] = Prob(weight) * self._norm
        for i in path:
            self.__p_thr[i].add(path)

    def __or__(self, other):
        new = self.copy()
        for p, v in other:
            new.set_path(p, v)
        return new

    def get_from_sparse(self, other, normalize=False):
        return SparseMat(
            {p: self._dok[p] for p, _ in other if p in self._dok},
            node_num=self._nn,
            normalize=normalize
        )

    def get_from_paths(self, paths, normalize=False):
        return SparseMat(
            {p: self._dok[p] for p in paths if p in self._dok},
            node_num=self._nn,
            normalize=normalize
        )

    def merge_colrow(self, indx1, indx2):
        indx1, indx2 = sorted([indx1, indx2])
        new_dict = {}
        for path, value in self._dok.items():
            if indx2 in path:
                path = tuple(i if i != indx2 else indx1 for i in path)
            path = tuple(i - int(i > indx2) for i in path)
            if path in new_dict:
                new_dict[path] += value
            else:
                new_dict[path] = value

        return SparseMat(
            new_dict,
            node_num=self._nn - 1,
            normalize=self._norm,
        )

    def kron(self, other):
        dok = {}
        for n in range(self._nn):
            for pA in self.paths_through_node(n, position=-1):
                for pB in other.paths_through_node(n, position=0):
                    dok[pA[:-1] + pB] = self._dok[pA] * other._dok[pB]

        return SparseMat(
            dok,
            node_num=self._nn,
            normalize=self._norm * other._norm
            # normalize=True
        )

    def sum(self):
        return (
            np.sum([float(p) for p in self._dok.values()]) / float(self._norm)
        )


def entrogram(graph, partition, depth=3):
    """TODO: Docstring for entrogram.

    :graph: TODO
    :partition: TODO
    :depth: TODO
    :returns: TODO

    """
    # node to index map
    n2i = {n: i for i, n in enumerate(graph.nodes())}
    i2p = {n2i[n]: p for n, p in partition.items()}
    n_n = len(n2i)
    n_p = len(np.unique(list(i2p.values())))

    symmetric = True if isinstance(graph, nx.Graph) else False
    edges = [
        (n2i[i], n2i[j], w)
        for i, j, w in graph.edges.data('weight', default=1.0)
    ]
    if symmetric:
        edges += [
            (j, i, w) for i, j, w in edges
        ]

    transition, diag, pi = utils.get_probabilities(
        edges, n_n,
        symmetric=symmetric,
        return_transition=True)

    pij = transition @ diag
    pij = SparseMat(pij, normalize=True)
    # transition[i, j] = p(j| i)
    transition = SparseMat(transition)

    p_pij = pij.project(i2p)
    p_pi = np.zeros(n_p)
    for (i, j), w in p_pij:
        p_pi[i] += w
    Hs = [utils.entropy(p_pi), utils.entropy(p_pij)]

    for step in range(1, depth + 1):
        # pij = utils.kron(pij, transition)
        pij = pij.kron(transition)
        p_pij = pij.project(i2p)
        Hs.append(utils.entropy(p_pij))

    entrogram = np.array(Hs)
    entrogram = entrogram[1:] - entrogram[:-1]
    Hks = Hs[-1] - Hs[-2]
    return Hks, entrogram[:-1] - Hks


def best_partition(
        graph,
        kmin=2,
        kmax=None,
        beta=1.0,
        probNorm=1.0,
        compute_steady=True,
        save_partials=False,
        partials_flnm='net_{:03}.npz',
        tsteps=4000,
        partname=None,
        **kwargs):
    """TODO: Docstring for best_partition.

    :graph: nx.Graph or nx.DiGraph
    :returns: TODO

    """

    if kmax is None:
        kmax = graph.number_of_nodes()
        initp = {
            n: i for i, n in enumerate(graph.nodes())
        }
    elif isinstance(kmax, dict):
        initp = kmax
        kmax = len(np.unique(list(initp.values())))
    else:
        initp = {
            n: i % kmax for i, n in enumerate(graph.nodes())
        }

    pgraph = PGraph(graph, init_part=initp, compute_steady=compute_steady)

    # for k in range(kmax, max(kmin - 1, 1), -1):
    results = {}
    log.info("Optimization with {} parts, alpha {}, beta {}, probNorm {}"
             .format(pgraph._np, kwargs.get('alpha', 0.0), beta, probNorm))
    best = optimize(pgraph, beta, probNorm, tsteps, **kwargs)
    results[pgraph._np] = dict(best)
    pgraph = PGraph(graph, init_part=best,
                    compute_steady=compute_steady)
    val = utils.value(pgraph, **kwargs)
    if save_partials:
        np.savez_compressed(
            partials_flnm.format(pgraph.np),
            partition=best,
            value=val,
            **kwargs,
        )
    log.info('{} -- {} '.format(pgraph._np, pgraph.print_partition()))
    log.info('   -- {}'.format(val))

    while pgraph._np > kmin:
        p1, p2 = pgraph._get_best_merge()
        pgraph.merge_partitions(p1, p2)

        best = optimize(pgraph, beta, probNorm, tsteps, **kwargs)
        results[pgraph._np] = dict(best)
        pgraph = PGraph(graph, init_part=best,
                        compute_steady=compute_steady)
        val = utils.value(pgraph, **kwargs)
        if save_partials:
            np.savez_compressed(
                partials_flnm.format(pgraph.np),
                partition=best,
                value=val,
                **kwargs,
            )
        log.info('{} -- {} '.format(pgraph._np, pgraph.print_partition()))
        log.info('   -- {}'.format(val))
    return (results)


def optimize(
        pgraph, beta, probNorm, tsteps,
        partname=None,
        **kwargs
):
    bestp = pgraph.partition()
    cumul = 0.0
    moves = [0, 0, 0]
    if 'tqdm' in sys.modules:
        tsrange = tqdm.trange(tsteps)
    else:
        tsrange = range(tsteps)
    for _ in tsrange:
        r_node, r_part, p = pgraph._get_random_move(algorithm='correl')
        delta = pgraph._try_move_node(
            r_node,
            r_part,
            bruteforce=False,
            **kwargs
        )
        log.debug(
            '         {}[{}] -> {}'.format(r_node, pgraph._i2p[r_node],
                                           r_part))

        if delta is None:
            continue

        if delta >= 0.0:
            # log.info('good ' + partname[pgraph._i2p[r_node]] +
            #          '=>' + partname[r_part])
            pgraph._move_node(r_node, r_part)
            cumul += delta
            moves[0] += 1
        else:
            rand = np.random.rand()
            threshold = np.exp(beta * delta) * p * probNorm
            log.debug('delta {:6f}, rand {:6f}, threshold {:6f}, p {:6f}'
                      .format(delta, rand, threshold, p))
            if rand < threshold:
                pgraph._move_node(r_node, r_part)
                cumul += delta
                moves[1] += 1
        if cumul > 0:
            bestp = pgraph.partition()
            cumul = 0.0
            moves[2] += 1
    log.info('good {}, not so good {}, best {}'.format(*moves))
    return bestp
