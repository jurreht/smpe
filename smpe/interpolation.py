import abc
import itertools

import numpy as np


class InterpolatedFunction(abc.ABC):
    @abc.abstractmethod
    def nodes(self):
        pass

    @abc.abstractmethod
    def __call__(self, x):
        pass

    @abc.abstractmethod
    def update(self, values):
        pass


class DifferentiableInterpolatedFuncion(InterpolatedFunction):
    @abc.abstractmethod
    def derivative(self, x):
        pass


class ChebyshevInterpolatedFunction(DifferentiableInterpolatedFuncion):
    def __init__(
        self, nodes_per_state, degree, node_min, node_max,
        complete=True
    ):
        if degree < 1:
            raise ValueError(
                'Chebyshev polynomial degree must be at least one')
        self.degree = degree

        if nodes_per_state <= degree:
            raise ValueError(
                'The number of nodes per state must be larger than the ' +
                'degree of a Chebyshev polynomial')
        self.nodes_per_state = nodes_per_state

        if len(node_min) != len(node_max):
            raise ValueError(
                'The number of node lower bounds must be equal to the '
                'number of node upper bounds')
        if np.any(node_min >= node_max):
            raise ValueError('Node lower bounds must be below upper bounds')
        self.node_min = node_min
        self.node_max = node_max

        self.dim_state = len(node_min)
        self.n_nodes = nodes_per_state ** self.dim_state
        self.complete = complete

        self.cheb_nodes = self._make_cheb_nodes(nodes_per_state)

        self._nodes = []
        for i in range(self.dim_state):
            scale = .5 * (self.node_max[i] - self.node_min[i])
            scaled_nodes = (self.cheb_nodes + 1) * scale + self.node_min[i]
            self._nodes.append(scaled_nodes)

        self.n_coefs = (degree + 1) ** self.dim_state
        self.coefs = np.zeros(self.n_coefs)

    def __call__(self, x):
        ret = 0.
        x = np.array(x)
        x = self._normalize_state(x)
        inds = np.zeros(self.dim_state, dtype=np.int_)

        pols = np.array([self._cheby_pols(s) for s in np.asarray(x)])
        for i in range(self.n_coefs):
            if self.complete and np.sum(inds) > self.degree:
                continue

            ind = self._coef_ind(inds)
            add = self.coefs[ind]
            for j in range(self.dim_state):
                add *= pols[j][inds[j]]
            ret += add

            self._update_inds(inds)

        return ret

    def nodes(self):
        return itertools.product(*self._nodes)

    def chebyshev_nodes(self):
        return itertools.product(self.cheb_nodes, repeat=self.dim_state)

    def update(self, values):
        coef_num = np.zeros(self.n_coefs)
        coef_denum = np.zeros((self.n_coefs, self.dim_state))

        for node_ind, (node, norm_node) in enumerate(
            zip(self.nodes(), self.chebyshev_nodes())
        ):
            pols = np.zeros((self.dim_state, self.degree + 1))
            for k in range(self.dim_state):
                pols[k] = self._cheby_pols(norm_node[k])

            inds = np.zeros(self.dim_state, dtype=np.int_)
            for ind in range(self.n_coefs):
                if self.complete and np.sum(inds) > self.degree:
                    continue

                coef_num_add = values[node_ind]
                for k in range(self.dim_state):
                    coef_num_add *= pols[k, inds[k]]
                coef_num[ind] += coef_num_add
                self._update_inds(inds)

        for j in range(self.nodes_per_state):
            inds = np.zeros(self.dim_state, dtype=np.int_)
            for ind in range(self.n_coefs):
                if self.complete and np.sum(inds) > self.degree:
                    continue

                pol = self._cheby_pols(self.cheb_nodes[j])
                for k in range(self.dim_state):
                    coef_denum[ind, k] += pol[inds[k]]**2
                self._update_inds(inds)

        inds = np.zeros(self.dim_state, dtype=np.int_)
        for ind in range(self.n_coefs):
            if self.complete and np.sum(inds) > self.degree:
                continue

            self.coefs[ind] = coef_num[ind]
            for k in range(self.dim_state):
                self.coefs[ind] /= coef_denum[ind, k]
            self._update_inds(inds)

    def derivative(self, x):
        x = np.array(x)
        x = self._normalize_state(x)
        inds = np.zeros(self.dim_state, dtype=np.int_)
        deriv = np.zeros(self.dim_state)

        pols = np.array([self._cheby_pols(s) for s in x])
        pols_2nd = np.array([self._cheby_pols_2nd_kind(s) for s in x])
        for i in range(self.n_coefs):
            if self.complete and np.sum(inds) > self.degree:
                continue

            ind = self._coef_ind(inds)
            add_deriv = np.full(self.dim_state, self.coefs[ind])
            for j in range(self.dim_state):
                for k in range(self.dim_state):
                    if k == j:
                        if inds[j] == 0:
                            add_deriv[j] = 0
                        else:
                            add_deriv[j] *= inds[k] * pols_2nd[k][inds[k] - 1]
                    else:
                        add_deriv[j] *= pols[k][inds[k]]

            for j in range(self.dim_state):
                deriv[j] += add_deriv[j]

            self._update_inds(inds)

        # Chain rule: account for normalization of state variables
        for j in range(self.dim_state):
            deriv[j] *= 2 / (self.node_max[j] - self.node_min[j])

        return deriv

    @staticmethod
    def _make_cheb_nodes(nodes_per_state):
        cheb_nodes = np.zeros(nodes_per_state)
        for j in range(nodes_per_state):
            cheb_nodes[j] = -1 * np.cos(
                (2 * (j + 1) - 1) / (2 * nodes_per_state) * np.pi)
        return cheb_nodes

    def _cheby_pols(self, x):
        ret = np.zeros(self.degree + 1)
        ret[0] = 1
        ret[1] = x
        for i in range(2, self.degree + 1):
            ret[i] = 2 * x * ret[i - 1] - ret[i - 2]
        return ret

    def _cheby_pols_2nd_kind(self, x):
        ret = np.zeros(self.degree + 1)
        ret[0] = 1
        ret[1] = 2 * x
        for i in range(2, self.degree + 1):
            ret[i] = 2 * x * ret[i - 1] - ret[i - 2]
        return ret

    def _normalize_state(self, state):
        normalized_state = state.copy()
        for i in range(self.dim_state):
            normalized_state[i] = (
                2 * (state[i] - self.node_min[i]) /
                (self.node_max[i] - self.node_min[i]) - 1)

        return normalized_state

    def _update_inds(self, inds):

        for j in range(self.dim_state - 1, -1, -1):
            if inds[j] < self.degree:
                inds[j] += 1
                inds[(j + 1):] = 0
                break

    def _coef_ind(self, inds):
        ind = 0
        for j in range(self.dim_state):
            ind += inds[j] * (self.degree + 1)**(self.dim_state - j - 1)
        return ind
