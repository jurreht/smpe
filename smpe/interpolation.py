import abc
import itertools
import math

import numba
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

    @property
    @abc.abstractmethod
    def num_nodes(self):
        pass

    @property
    @abc.abstractmethod
    def dim_state(self):
        pass

    @property
    @abc.abstractmethod
    def __getitem__(self, key):
        pass

    @property
    def dtype(self):
        # Always return double since the main purpose implementing the Numpy
        # array interface is to interface with Dask. The nodes will be combined
        # into a single array with actions anyway, the latter of which are
        # always double.
        return np.double

    def numpy_nodes(self):
        return np.array(list(iter(self.nodes())))

    @property
    def shape(self):
        return (self.num_nodes, self.dim_state)


class DifferentiableInterpolatedFunction(InterpolatedFunction):
    @abc.abstractmethod
    def derivative(self, x):
        pass


class TwiceDifferentiableInterpolatedFunction(
    DifferentiableInterpolatedFunction
):
    @abc.abstractmethod
    def second_derivative(self, x):
        pass


class ShareableInterpolatedFunction(InterpolatedFunction):
    @abc.abstractmethod
    def share(self):
        pass


class MultivariateInterpolatedFunction:
    def __init__(self, func: ShareableInterpolatedFunction, num_y: int):
        self.vf = [func]
        if num_y > 1:
            for i in range(num_y - 1):
                self.vf.append(func.share())

    def nodes(self):
        return self.vf[0].nodes()

    @property
    def dim_state(self):
        return self.vf[0].dim_state

    @property
    def num_nodes(self):
        return self.vf[0].num_nodes

    def numpy_nodes(self):
        return self.vf[0].numpy_nodes()

    @property
    def shape(self):
        return self.vf[0].shape

    @property
    def dtype(self):
        return self.vf[0].dtype

    def __getitem__(self, key):
        return self.vf[0][key]

    def __len__(self):
        return len(self.vf[0])


class ChebyshevInterpolatedFunction(
    TwiceDifferentiableInterpolatedFunction, ShareableInterpolatedFunction
):
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
                'The number of nodes per state must be larger than the '
                'degree of a Chebyshev polynomial')
        self.nodes_per_state = nodes_per_state

        self.node_min = np.array(node_min)
        self.node_max = np.array(node_max)
        if len(self.node_min) != len(self.node_max):
            raise ValueError(
                'The number of node lower bounds must be equal to the '
                'number of node upper bounds')
        if np.any(self.node_min >= self.node_max):
            raise ValueError('Node lower bounds must be below upper bounds')

        self._dim_state = len(node_min)
        self.n_nodes = nodes_per_state ** self.dim_state
        self.complete = complete

        self.cheb_nodes = _cheby_make_cheb_nodes(self.nodes_per_state)

        self._nodes = np.zeros((self.dim_state, self.nodes_per_state))
        for i in range(self.dim_state):
            scale = .5 * (self.node_max[i] - self.node_min[i])
            scaled_nodes = (self.cheb_nodes + 1) * scale + self.node_min[i]
            self._nodes[i] = scaled_nodes

        self.n_coefs = (degree + 1) ** self.dim_state
        self.coefs = np.zeros(self.n_coefs)

    def share(self):
        cls = type(self)
        copy = cls.__new__(cls)
        copy.degree = self.degree
        copy.nodes_per_state = self.nodes_per_state
        copy.node_min = self.node_min
        copy.node_max = self.node_max
        copy._dim_state = self._dim_state
        copy.n_nodes = self.n_nodes
        copy.complete = self.complete
        copy.cheb_nodes = self.cheb_nodes
        copy._nodes = self._nodes
        copy.n_coefs = self.n_coefs
        copy.coefs = np.copy(self.coefs)
        return copy

    def __call__(self, x):
        x = np.array(x)
        x = _cheby_normalize_state(
            x, self._dim_state, self.node_min, self.node_max)
        inds = np.zeros(self._dim_state, dtype=np.int_)

        pols = np.array([
            _cheby_cheby_pols(s, self.degree) for s in np.asarray(x)])
        return _cheby_call_opt(
            self._dim_state, self.degree, self.complete, self.n_coefs,
            self.coefs, inds, pols)

    def nodes(self):
        return itertools.product(*self._nodes)

    @property
    def num_nodes(self):
        return self.n_nodes

    @property
    def dim_state(self):
        return self._dim_state

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._get_one_item(key, slice(None, None, None))
        elif isinstance(key, slice):
            return self._get_range_items(key, slice(None, None, None))
        elif isinstance(key, tuple):
            if isinstance(key[0], int):
                return self._get_one_item(key[0], key[1])
            elif isinstance(key[0], slice):
                ret = self._get_range_items(key[0], key[1])

                if isinstance(key[1], int):
                    return ret[:, 0]
                else:
                    return ret
            else:
                raise TypeError()

        else:
            raise TypeError()

    def _get_one_item(self, key, y_ind):
        ret = np.empty(self._dim_state)
        for i in range(self.dim_state):
            ind = self._dim_state - i - 1
            remainder = (
                (key // self.nodes_per_state**i) %
                self.nodes_per_state)
            ret[ind] = self._nodes[ind, remainder]
            key -= remainder
        return ret[y_ind]

    def _get_range_items(self, key, y_ind):
        start, stop, step = self._slice_to_range(key, self.n_nodes)
        x_els = math.ceil((stop - start) / step)

        if isinstance(y_ind, int):
            y_els = 1
        else:
            # Calculate the range of y_ind as well to allocate the right amount
            # of memory
            y_start, y_stop, y_step = self._slice_to_range(
                y_ind, self._dim_state)
            y_els = max(math.ceil((y_stop - y_start) / y_step), 0)

        if x_els <= 0:
            return np.zeros((0, y_els))

        ret = np.empty((x_els, y_els))
        for ind, i in enumerate(range(start, stop, step)):
            ret[ind] = self._get_one_item(i, y_ind)
        return ret

    def _slice_to_range(self, key, max_el):
        if key.start is None:
            start = 0
        elif key.start >= 0:
            start = key.start
        else:
            start = max_el + key.start

        if key.stop is None:
            stop = max_el
        elif key.stop >= 0:
            stop = key.stop
        else:
            stop = max_el + key.stop

        step = 1 if key.step is None else key.step

        if step < 0:
            if key.stop is None and key.start is not None:
                stop = -1
            elif key.start is None and key.stop is not None:
                start = max_el - 1
                if key.stop >= 0:
                    stop = key.stop
                else:
                    stop = max_el + key.stop
            elif key.start is None and key.stop is None:
                start, stop = stop, start
                stop -= 1
                start -= 1

        return start, stop, step

    def chebyshev_nodes(self):
        return itertools.product(self.cheb_nodes, repeat=self._dim_state)

    def update(self, values):
        coef_num = np.zeros(self.n_coefs)

        for node_ind, (node, norm_node) in enumerate(
            zip(self.nodes(), self.chebyshev_nodes())
        ):
            _cheby_update_step1(
                self._dim_state, self.degree, self.complete,
                self.n_coefs, node_ind, norm_node, coef_num, values)

        _cheby_update_step2(
            self.nodes_per_state, self._dim_state, self.degree,
            self.complete, self.n_coefs, self.cheb_nodes,
            coef_num, self.coefs)

    def derivative(self, x):
        x = np.array(x)
        x = _cheby_normalize_state(
            x, self.dim_state, self.node_min, self.node_max)
        inds = np.zeros(self._dim_state, dtype=np.int_)

        pols = np.array([_cheby_cheby_pols(s, self.degree) for s in x])
        pols_2nd = np.array([
            _cheby_cheby_pols_2nd_kind(s, self.degree) for s in x])
        return _cheby_derivative_opt(
            self._dim_state, self.degree, self.complete, self.n_coefs,
            self.coefs, inds, pols, pols_2nd, self.node_min, self.node_max)

    def second_derivative(self, x):
        x = np.array(x)
        x = _cheby_normalize_state(
            x, self._dim_state, self.node_min, self.node_max
        )
        inds = np.zeros(self._dim_state, dtype=np.int_)

        pols = np.array([_cheby_cheby_pols(s, self.degree) for s in x])
        pols_2nd = np.array([
            _cheby_cheby_pols_2nd_kind(s, self.degree) for s in x])
        return _cheby_second_derivative_opt(
            self._dim_state, self.degree, self.complete, self.n_coefs,
            self.coefs, inds, pols, pols_2nd, self.node_min, self.node_max, x)


#
# The Numba functions starting with _cheby_* belong to the
# ChebyshevInterpolatedFunction class. Since Numba does not really
# support attribute access on compiled functions. this is the best
# solution.
#

@numba.jit(cache=True, nopython=True)
def _cheby_make_cheb_nodes(nodes_per_state):
    cheb_nodes = np.zeros(nodes_per_state)
    for j in range(nodes_per_state):
        cheb_nodes[j] = -1 * np.cos(
            (2 * (j + 1) - 1) / (2 * nodes_per_state) * np.pi)
    return cheb_nodes


@numba.jit(cache=True, nopython=True)
def _cheby_call_opt(dim_state, degree, complete, n_coefs, coefs, inds, pols):
    ret = 0.
    for i in range(n_coefs):
        if complete and np.sum(inds) > degree:
            continue

        ind = _cheby_coef_ind(inds, dim_state, degree)
        add = coefs[ind]
        for j in range(dim_state):
            add *= pols[j][inds[j]]
        ret += add

        _cheby_update_inds(inds, dim_state, degree)

    return ret


@numba.jit(cache=True, nopython=True)
def _cheby_derivative_opt(
    dim_state, degree, complete, n_coefs, coefs, inds, pols, pols_2nd,
    node_min, node_max
):
    deriv = np.zeros(dim_state)

    for i in range(n_coefs):
        if complete and np.sum(inds) > degree:
            continue

        ind = _cheby_coef_ind(inds, dim_state, degree)
        add_deriv = np.full(dim_state, coefs[ind])
        for j in range(dim_state):
            for k in range(dim_state):
                if k == j:
                    if inds[j] == 0:
                        add_deriv[j] = 0
                    else:
                        add_deriv[j] *= inds[k] * pols_2nd[k][inds[k] - 1]
                else:
                    add_deriv[j] *= pols[k][inds[k]]

        for j in range(dim_state):
            deriv[j] += add_deriv[j]

        _cheby_update_inds(inds, dim_state, degree)

    # Chain rule: account for normalization of state variables
    for j in range(dim_state):
        deriv[j] *= 2 / (node_max[j] - node_min[j])

    return deriv


@numba.jit(cache=True, nopython=True)
def _cheby_second_derivative_opt(
    dim_state, degree, complete, n_coefs, coefs, inds, pols, pols_2nd,
    node_min, node_max, x
):
    deriv2 = np.zeros((dim_state, dim_state))

    for i in range(n_coefs):
        if complete and np.sum(inds) > degree:
            continue

        ind = _cheby_coef_ind(inds, dim_state, degree)
        add_deriv2 = np.full((dim_state, dim_state), coefs[ind])
        for j in range(dim_state):
            for k in range(dim_state):
                for m in range(dim_state):
                    n = inds[m]
                    if m == j and m == k:
                        if inds[m] <= 1:
                            add_deriv2[j, k] = 0
                        elif x[m] == 1:
                            add_deriv2[j, k] *= (n**4 - n**2) / 3
                        elif x[m] == -1:
                            add_deriv2[j, k] *= ((-1)**n) * (n**4 - n**2) / 3
                        else:
                            add_deriv2[j, k] *= (
                                n * ((n + 1) * pols[m][n] - pols_2nd[m][n]) /
                                (x[m]**2 - 1))
                    elif m == j or m == k:
                        if inds[m] == 0:
                            add_deriv2[j, k] = 0
                        else:
                            add_deriv2[j, k] *= n * pols_2nd[m][n - 1]
                    else:
                        add_deriv2[j, k] *= pols[m][n]
        for j in range(dim_state):
            for k in range(dim_state):
                deriv2[j, k] += add_deriv2[j, k]

        _cheby_update_inds(inds, dim_state, degree)

    # Chain rule: account for normalization of state variables
    for j in range(dim_state):
        for k in range(dim_state):
            deriv2[j, k] *= 4 / ((node_max[j] - node_min[j]) *
                                 (node_max[k] - node_min[k]))

    return deriv2


@numba.jit(cache=True, nopython=True)
def _cheby_update_step1(
    dim_state, degree, complete, n_coefs, node_ind, norm_node, coef_num, values
):
    pols = np.zeros((dim_state, degree + 1))
    for k in range(dim_state):
        pols[k] = _cheby_cheby_pols(norm_node[k], degree)

    inds = np.zeros(dim_state, dtype=np.int_)
    for ind in range(n_coefs):
        if complete and np.sum(inds) > degree:
            continue

        coef_num_add = values[node_ind]
        for k in range(dim_state):
            coef_num_add *= pols[k, inds[k]]
        coef_num[ind] += coef_num_add
        _cheby_update_inds(inds, dim_state, degree)


@numba.jit(cache=True, nopython=True)
def _cheby_update_step2(
    nodes_per_state, dim_state, degree, complete, n_coefs, cheb_nodes,
    coef_num, coefs
):
    coef_denum = np.zeros((n_coefs, dim_state))
    for j in range(nodes_per_state):
        inds = np.zeros(dim_state, dtype=np.int_)
        for ind in range(n_coefs):
            if complete and np.sum(inds) > degree:
                continue

            pol = _cheby_cheby_pols(cheb_nodes[j], degree)
            for k in range(dim_state):
                coef_denum[ind, k] += pol[inds[k]]**2
            _cheby_update_inds(inds, dim_state, degree)

    inds = np.zeros(dim_state, dtype=np.int_)
    for ind in range(n_coefs):
        if complete and np.sum(inds) > degree:
            continue

        coefs[ind] = coef_num[ind]
        for k in range(dim_state):
            coefs[ind] /= coef_denum[ind, k]
        _cheby_update_inds(inds, dim_state, degree)


@numba.jit(cache=True, nopython=True)
def _cheby_cheby_pols(x, degree):
    ret = np.zeros(degree + 1)
    ret[0] = 1
    ret[1] = x
    for i in range(2, degree + 1):
        ret[i] = 2 * x * ret[i - 1] - ret[i - 2]
    return ret


@numba.jit(cache=True, nopython=True)
def _cheby_cheby_pols_2nd_kind(x, degree):
    ret = np.zeros(degree + 1)
    ret[0] = 1
    ret[1] = 2 * x
    for i in range(2, degree + 1):
        ret[i] = 2 * x * ret[i - 1] - ret[i - 2]
    return ret


@numba.jit(cache=True, nopython=True)
def _cheby_normalize_state(state, dim_state, node_min, node_max):
    normalized_state = state.copy()
    for i in range(dim_state):
        normalized_state[i] = (
            2 * (state[i] - node_min[i]) /
            (node_max[i] - node_min[i]) - 1)

    return normalized_state


@numba.jit(cache=True, nopython=True)
def _cheby_update_inds(inds, dim_state, degree):
    for j in range(dim_state - 1, -1, -1):
        if inds[j] < degree:
            inds[j] += 1
            inds[(j + 1):] = 0
            break


@numba.jit(cache=True, nopython=True)
def _cheby_coef_ind(inds, dim_state, degree):
    ind = 0
    for j in range(dim_state):
        ind += inds[j] * (degree + 1)**(dim_state - j - 1)
    return ind
