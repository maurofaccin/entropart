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

    def __init__(self, graph,
                 compute_steady=True, init_part=None):
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

        assert np.isclose(self._pij.sum(), 1.0)
        assert np.isclose(self._ppij.sum(), 1.0)

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

    def _move_probability(self, inode, part_probs=None, part=None):
        """Compute the probability to move to a partition.

        :inode: node index to move
        :part_probs: optional prob distribution to compute from
        :part: optional partition to move the node to
        """

        if part_probs is None:
            part_probs = self._ppij
        n_ego_out = self._pij.get_egonet(inode, axis=0).project(self._i2p)
        n_ego_in = self._pij.get_egonet(inode, axis=1).project(self._i2p)

        # compute probability to choose a partition to mode to
        if part is None:
            probs = np.zeros(part_probs.nn)
            rng = range(part_probs.nn)
        else:
            probs = np.array([0.0, ])
            rng = [part]

        for prt in rng:

            probs[prt] += np.sum([
                float(v) * float(part_probs[(prt, p[1])])
                for p, v in n_ego_out
            ])

            probs[prt] += np.sum([
                float(v) * float(part_probs[(p[0], prt)])
                for p, v in n_ego_in])

        probs += 1.0 / (part_probs.nn + 1)
        return probs / probs.sum()

    def _get_random_move(
            self,
            inode=None,
            algorithm='correl',
            kmax=None,
            kmin=None,
            **kwargs):
        """Select one node and a partition to move to.
        Returns the probability of the move and the delta energy.
        """
        if inode is None:
            inode = np.random.randint(self._nn)
        old_part = self._i2p[inode]

        if kmax is not None and self._np == kmax:
            # do not go for a new partition
            delta = 0.0
        else:
            delta = 1.0 / (self._np + 1)

        if kmin is not None:
            if len(self._p2i[old_part]) == 1 and self._np == kmin:
                return inode, None, None, None

        if algorithm == 'random':
            new_part = np.random.randint(self._np)
            prob_ratio = 1.0
            delta_obj = self._try_move_node(
                inode,
                new_part,
                bruteforce=False,
                **kwargs
            )

        elif algorithm == 'new':
            n_ego_full = self._pij.get_egonet(inode)
            n_ego = n_ego_full.project(self._i2p)
            p1_ego = self._ppij.get_egonet(old_part)

            if np.random.random() < delta:
                # inode is going to start a new partition
                n_ego.add_colrow()
                p1_ego.add_colrow()
                p2_ego = utils.zeros_like(p1_ego)

                new_part = self._np

                prob_move = delta
                act = 'split'
            else:
                # move inode to anothere partition
                probs_go = self._move_probability(inode)
                new_part = np.random.choice(
                    np.arange(self._np),
                    p=probs_go
                )
                p2_ego = self._ppij.get_egonet(new_part)
                prob_move = probs_go[new_part]
                act = 'move'

            if (inode, new_part) in self._tryed_moves:
                return (inode,
                        new_part,
                        self._tryed_moves[(inode, new_part)][0],
                        self._tryed_moves[(inode, new_part)][1])

            n_ego_post = n_ego_full.project(
                self._i2p,
                move_node=(inode, new_part)
            )

            p12_post = (p1_ego | p2_ego) + n_ego_post - n_ego
            probs_back = self._move_probability(
                inode,
                part_probs=p12_post
            )

            prob_ratio = probs_back[old_part] / prob_move

            if new_part == self._np:
                h1org = utils.entropy(self._ppi[old_part])
                h1dst = utils.entropy([
                    self._ppi[old_part] - self._pi[inode],
                    self._pi[inode]
                ])
            else:
                h1org = utils.entropy([
                    self._ppi[old_part],
                    self._ppi[new_part]
                ])
                h1dst = utils.entropy([
                    self._ppi[old_part] - self._pi[inode],
                    self._ppi[new_part] + self._pi[inode]
                ])

            H2org = utils.entropy(p1_ego | p2_ego)
            H2dst = utils.entropy(p12_post)

            delta_obj = self.delta(
                h1org, H2org, h1dst, H2dst, action=act, **kwargs
            )
            self._tryed_moves[(inode, new_part)] = (prob_ratio, delta_obj)

        elif algorithm == 'correl':
            n_ego = self._pij.get_egonet(inode, axis=0)
            if n_ego is None:
                return inode, old_part, 1.0
            # prob of getting out of inode and arriving to any partition.
            n2p_arr = np.zeros(len(self._ppi), dtype=float)
            for i, v in n_ego:
                n2p_arr[self._i2p[i[-1]]] += v

            probs = self._ppij.dot(n2p_arr, indx=-1)
            if probs.sum() > 0:
                probs /= probs.sum()
            else:
                probs = np.ones_like(probs) / len(probs)

            new_part = np.random.choice(np.arange(self._np), p=probs)
            prob_ratio = probs[old_part] / probs[new_part]

            delta_obj = self._try_move_node(inode, new_part,
                                            bruteforce=False, **kwargs)
        else:
            raise NotImplementedError('Algorithm not known ' + algorithm)

        return inode, new_part, prob_ratio, delta_obj

    def _try_move_node(self, inode, partition, bruteforce=False, **kwargs):
        """deltaH"""

        if (inode, partition) in self._tryed_moves:
            return self._tryed_moves[(inode, partition)]

        # check if we are moving to the same partition
        old_part = self._i2p[inode]
        if old_part == partition:
            return None

        # check if starting partition has just one node
        if len(self._p2i[old_part]) == 0:
            return None

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

        h1org = utils.entropy(self._ppi[old_part]) +\
            utils.entropy(self._ppi[partition])
        h1dst = utils.entropy(self._ppi[old_part] - self._pi[inode]) +\
            utils.entropy(self._ppi[partition] + self._pi[inode])

        H2org = utils.entropy(part_orig)
        H2dst = utils.entropy(part_dest)

        d = self.delta(h1org, H2org, h1dst, H2dst, action='move', **kwargs)
        self._tryed_moves[(inode, partition)] = d
        return d

    def _move_node(self, inode, partition):
        if self._i2p[inode] == partition:
            return None

        old_part = self._i2p[inode]
        if len(self._p2i[old_part]) == 1:
            self.merge_partitions(partition, old_part)
            return

        pnode = self._pi[inode]
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
        # union
        p12 = p1_ego | p2_ego
        H2pre = utils.entropy(p12)

        p12 = p12.merge_colrow(p1, p2)
        H2post = utils.entropy(p12)

        h1pre = utils.entropy(self._ppi[p1]) + utils.entropy(self._ppi[p2])
        h1post = utils.entropy(self._ppi[p1] + self._ppi[p2])
        return self.delta(
            h1pre, H2pre, h1post, H2post, action='merge', **kwargs
        )

    def _try_split(self, inode, **kwargs):
        # before splitting
        old_part = self._i2p[inode]
        if len(self._p2i[old_part]) <= 1:
            return None

        h1pre = utils.entropy(self._ppi[old_part])

        part_ego = self._ppij.get_egonet(old_part)
        H2pre = utils.entropy(part_ego)

        # after splitting
        h1post = utils.entropy(self._ppi[old_part] - self._pi[inode]) +\
            utils.entropy(self._pi[inode])
        # node ego nets (projected to partitions)
        new_part = part_ego.add_colrow()
        new_i2p = self._i2p.copy()
        new_i2p[inode] = new_part

        ego_node = self._pij.get_egonet(inode)\
            .project(new_i2p, move_node=(inode, new_part))
        part_ego_post = part_ego - ego_node
        H2post = utils.entropy(ego_node) + utils.entropy(part_ego_post)

        return self.delta(
            h1pre, H2pre, h1post, H2post, action='split', **kwargs
        )

    def _split(self, inode):
        old_part = self._i2p[inode]
        log.debug('Splitting node {}'.format(inode))
        if len(self._p2i[old_part]) == 1:
            return

        # self._ppi
        self._ppi = np.append(self._ppi, [0])
        self._ppi[old_part] -= self._pi[inode]
        self._ppi[-1] = self._pi[inode]

        # self.__ppij
        new_part = self._ppij.add_colrow()

        ego_node = self._pij.get_egonet(inode)
        en_pre = ego_node.project(self._i2p)
        # self._i2p
        self._i2p[inode] = new_part
        en_post = ego_node.project(self._i2p)
        self._ppij += en_post - en_pre

        # self._p2i
        self._p2i[old_part].remove(inode)
        self._p2i[len(self._ppi) - 1] = set([inode])

        # final updates
        self._np += 1
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
        log.debug('Merging partitions {} and {}.'.format(part1, part2))
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

    def delta(
            self,
            h1old,
            h2old,
            h1new,
            h2new,
            alpha=0.0,
            gamma=None,
            action='move'):

        if gamma is not None:
            if action == 'move':
                dgamma = 0
            elif action == 'merge':
                dgamma = gamma * np.log(self.np / (self.np - 1))
            elif action == 'split':
                dgamma = gamma * np.log(self.np / (self.np + 1))
            else:
                raise ValueError(
                    'action should be either `move`, `merge` or `split`'
                )
        else:
            dgamma = 0


        return (2 - alpha) * (h1new - h1old) - h2new + h2old - dgamma


class Prob(object):
    """A class to store the probability p and the plogp value."""
    __slots__ = ['__p', '__plogp']

    def __init__(self, value):
        """Given a float or a Prob, store p and plogp."""
        # if float(value) < 0.0:
        #     raise ValueError('Must be non-negative.')
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

    __slots__ = ['_dok', '_nn', '_dim', '_norm', '__p_thr']

    def __init__(self, mat, node_num=None, normalize=False, plength=None):
        """Initiate the matrix

        :mat: scipy sparse matrix or
                list of ((i, j, ...), w) or
                dict (i, j, ...): w
        :node_num: number of nodes
        :normalize: (bool) whether to normalize entries or not
        :plenght: lenght of each path, to use only if len(mat) == 0
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
            self._dok = {tuple(k): Prob(v) for k, v in mat.items()}
            if node_num is None:
                self._nn = np.max([dd for d in self._dok for dd in d]) + 1
            # get the first key of the dict
            if plength is None:
                val = next(iter(self._dok.keys()))
                self._dim = len(val)
            else:
                self._dim = plength
        else:
            self._dok = {tuple(i): Prob(d) for i, d in mat}
            if node_num is None:
                self._nn = np.max([dd for d in self._dok for dd in d]) + 1
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

        if self._norm == 0.0 and len(self._dok) > 0:
            raise ValueError('This is a zero matrix')

        self.__update_all_paths()
        # self.checkme()

    def __update_all_paths(self):
        """ For each node, all paths that go through it."""
        self.__p_thr = [set() for _ in range(self._nn)]
        for path in self._dok.keys():
            for i in path:
                try:
                    self.__p_thr[i].add(path)
                except IndexError:
                    print(path, self._nn)
                    raise

    def entropy(self):
        if self._nn == 0:
            return 0.0
        sum_plogp = np.sum([p.plogp for p in self._dok.values()])
        return (self._norm.plogp - sum_plogp) / self._norm.p

    @property
    def shape(self):
        return tuple([self._nn] * self._dim)

    @property
    def nn(self):
        return self._nn

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

        new_n_part = max(list(part.values())) + 1

        # fix partition before returning
        if move_node is not None:
            part[move_node[0]] = old_part

        return SparseMat(
            new_dok,
            node_num=new_n_part,
            normalize=self._norm,
        )

    def copy(self):
        return SparseMat(
            {path[:]: w.copy() for path, w in self._dok.items()},
            node_num=self._nn,
            normalize=self._norm,
            plength=self._dim,
        )

    def dot(self, other, indx):
        if not isinstance(other, np.ndarray):
            raise TypeError(
                'other should be numpy.ndarray, not {}'.format(type(other)))
        out = np.zeros_like(other, dtype=float)
        for path, w in self._dok.items():
            out[path[indx]] += float(w) * other[path[1 + indx]]
        return out / self._norm.p

    def get_egonet(self, inode, axis=None):
        """Return the adjacency matrix of the ego net of node node."""
        if axis is None:
            slist = [
                (p, self._dok[p]) for p in self.__p_thr[inode]
            ]
        else:
            slist = [
                (p, self._dok[p]) for p in self.__p_thr[inode]
                if p[axis] == inode
            ]
        if len(slist) < 1:
            return None
        return SparseMat(
            slist,
            node_num=self._nn,
            normalize=self._norm,
        )

    def slice(self, axis=0, n=0):
        if axis == 0:
            vec = [self._dok.get((n, nn), 0.0) for nn in range(self._nn)]
        else:
            vec = [self._dok.get((nn, n), 0.0) for nn in range(self._nn)]
        return np.array(vec)

    def get_submat(self, nodelist):
        nodelist = set(nodelist)
        plist = [p for p in self._dok if any(n in p for n in nodelist)]
        try:
            sm = SparseMat(
                {p: self._dok[p] for p in plist},
                node_num=self._nn,
                normalize=self._norm,
            )
        except StopIteration:
            print(nodelist)
            raise
        return sm

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

    def paths(self, axis=None, node=None):
        if axis is None or node is None:
            return self.__iter__()

        for p, v in self._dok.items():
            if p[axis] == node:
                yield p, v / self._norm

    def __getitem__(self, item):
        try:
            return self._dok[item] / self._norm
        except KeyError:
            return 0.0

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
            if np.isclose(float(self._dok.get(p, 0.0)), float(d_norm)):
                for i in p:
                    self.__p_thr[i].discard(p)
                del self._dok[p]
            else:
                # no need to update __p_thr
                self._dok[p] = self._dok.get(p, Prob(0.0)) - d_norm
        return self

    def __sub__(self, other):
        new = self.copy()
        new -= other
        return new

    def __imul__(self, other):
        if self._nn != other._nn:
            raise ValueError(
                'Impossible to multiply matrices of different sizes {} and {}'
                .format(self._nn, other._nn)
            )
        self._norm *= other._norm
        if self.size() < other.size():
            keys = [k for k in self._dok.keys() if k in other._dok]
        else:
            keys = [k for k in other._dok.keys() if k in self._dok]
        new_dok = {k: other._dok[k] * self._dok[k] for k in keys}

        self._dok = new_dok

        # TODO: check if there is a better way to update paths
        self.__update_all_paths()
        return self

    def __mul__(self, other):
        new = self.copy()
        new *= other
        return new

    def set_path(self, path, weight):
        """ Overwrite path weight. """
        self._dok[path] = Prob(weight) * self._norm
        for i in path:
            self.__p_thr[i].add(path)

    def __or__(self, other):
        """ Return a SparseMat with entries from both self and other.
        Local entries will be overwritten by other's.
        """
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

    def add_colrow(self):
        self._nn += 1
        self.__p_thr.append(set())
        return self._nn - 1

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

    def sum(self, axis=None):
        # return the sum of all entries
        if axis is not None:
            probs = np.zeros(self._nn)
            for p, v in self._dok.items():
                probs[p[axis]] += float(v)
            return probs / float(self._norm)
        if self._nn == 0:
            return 0.0
        return (
            np.sum([float(p) for p in self._dok.values()]) / float(self._norm)
        )


class Partition(dict):
    """A bidirectional dictionary to store partitions."""

    def __init__(self, *args, **kwargs):
        super(Partition, self).__init__(*args, **kwargs)
        self.part = {}
        for key, value in self.items():
            self.part.setdefault(value, []).append(key)

    def __setitem__(self, key, value):
        if key in self:
            self.part[self[key]].remove(key)
        super(Partition, self).__setitem__(key, value)
        self.part.setdefault(value, []).append(key)

    def __delitem__(self, key):
        self.part.setdefault(self[key], []).remove(key)
        if self[key] in self.part and not self.part[self[key]]:
            del self.part[self[key]]
        super(Partition, self).__delitem__(key)


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
        init_part=None,
        kmin=None,
        kmax=None,
        beta=1.0,
        compute_steady=True,
        partials=None,
        tsteps=4000,
        **kwargs):
    """TODO: Docstring for best_partition.

    :graph: nx.Graph or nx.DiGraph
    :returns: TODO

    """

    if kmax is None:
        kmax = graph.number_of_nodes()

    if kmin is None:
        kmin = 2

    if init_part is None:
        if kmax < graph.number_of_nodes():
            # start from a random partition with partitions in [kmin, kmax]
            k = int((kmax + kmin) / 2)
            part = [i % k for i in range(graph.number_of_nodes())]
            np.random.shuffle(part)
            initp = {
                n: i for i, n in zip(part, graph.nodes())
            }
        else:
            # start from N partitions
            initp = {
                n: i for i, n in enumerate(graph.nodes())
            }
    elif isinstance(init_part, dict):
        initp = init_part
    else:
        raise ValueError('init_part should be a dict not {}'
                         .format(type(init_part)))

    pgraph = PGraph(graph, compute_steady=compute_steady, init_part=initp)

    log.info("Optimization with {} parts, alpha {}, beta {}"
             .format(pgraph._np, kwargs.get('alpha', 0.0), beta))
    best = optimize(
        pgraph,
        beta,
        tsteps,
        kmin,
        kmax,
        partials=partials,
        **kwargs
    )

    results = dict(best)
    val = utils.value(pgraph, **kwargs)
    if partials is not None:
        np.savez_compressed(
            partials.format(pgraph.np),
            partition=best,
            value=val,
            **kwargs,
        )

    pgraph = PGraph(graph, compute_steady=compute_steady, init_part=results)
    log.info('final: num part {}'.format(pgraph.np))
    log.info('{} -- {} '.format(pgraph._np, pgraph.print_partition()))
    log.info('   -- {}'.format(val))

    return results


def optimize(
        pgraph, beta,
        tsteps,
        kmin,
        kmax,
        partials=None,
        **kwargs):

    bestp = pgraph.partition()
    cumul = 0.0
    moves = [
        0,  # delta > 0
        0,  # delta < 0 accepted
        0,  # best
        0  # changes since last move
    ]
    if 'tqdm' in sys.modules and log.level >= 20:
        tsrange = tqdm.trange(tsteps)
    else:
        tsrange = range(tsteps)
    for i in tsrange:
        r_node, r_part, p, delta = pgraph._get_random_move(
            algorithm='new',
            kmin=kmin,
            kmax=kmax,
            **kwargs
        )
        if r_part is None:
            continue

        log.debug('proposed move: n {:5}, p {:5}, p() {:5.3f}, d {}'
                  .format(r_node, r_part, p, delta))

        log.debug('CUMUL {}'.format(cumul))
        if delta is None:
            continue

        if delta >= 0.0:
            if r_part == pgraph.np:
                pgraph._split(r_node)
            else:
                pgraph._move_node(r_node, r_part)
            cumul += delta
            moves[0] += 1
            moves[3] = 0
            log.debug('accepted move')
            if 'tqdm' in sys.modules and log.level >= 20:
                tsrange.set_description('{} [{}]'.format(moves[2],
                                                         pgraph.np))
        else:
            rand = np.random.rand()
            if rand == 0.0:
                continue
            threshold = beta * delta + np.log(p)
            if np.log(rand) < threshold:
                if r_part == pgraph.np:
                    pgraph._split(r_node)
                else:
                    pgraph._move_node(r_node, r_part)
                cumul += delta
                moves[1] += 1
                moves[3] = 0
                log.debug('accepted move {} < {}'.format(rand, threshold))
                if 'tqdm' in sys.modules and log.level >= 20:
                    tsrange.set_description('{} [{}]'.format(moves[2],
                                                             pgraph.np))
            else:
                log.debug('rejected move')
                moves[3] += 1

        if cumul > 0:
            log.debug('BEST move +++ {} +++'.format(cumul))
            bestp = pgraph.partition()
            cumul = 0.0
            moves[2] += 1
            if partials is not None:
                np.savez_compressed(
                    partials.format(pgraph.np),
                    partition=bestp,
                    value=cumul,
                    **kwargs,
                )
        if moves[3] > 1000:
            break
    log.info('good {}, not so good {}, best {}'.format(*moves))
    return bestp

