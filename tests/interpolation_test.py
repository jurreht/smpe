import collections

from hypothesis import given
import hypothesis.strategies as st
import numpy as np
from numpy.testing import assert_allclose
import pytest
import scipy.optimize

from smpe.interpolation import (
    InterpolatedFunction,
    DifferentiableInterpolatedFunction,
    TwiceDifferentiableInterpolatedFunction,
    ChebyshevInterpolatedFunction,
)


class CallNotImplementedMock(InterpolatedFunction):
    def update(self, values):
        pass

    def nodes(self):
        pass

    @property
    def num_nodes(self):
        pass

    def __getitem__(self, key):
        pass


def test_call_not_implemented():
    """
    Subclasses of InterpolatedFunction must implement __call__().
    """
    with pytest.raises(TypeError):
        CallNotImplementedMock()


class UpdateNotImplementedMock(InterpolatedFunction):
    def __call__(self, x):
        pass

    def nodes(self):
        pass

    @property
    def num_nodes(self):
        pass

    def __getitem__(self, key):
        pass


def test_update_not_implemented():
    """
    Subclasses of InterpolatedFunction must implement update().
    """
    with pytest.raises(TypeError):
        UpdateNotImplementedMock()


class NodesNotImplementedMock(InterpolatedFunction):
    def __call__(self, x):
        pass

    def update(self, values):
        pass

    @property
    def num_nodes(self):
        pass

    def __getitem__(self, key):
        pass


def test_nodes_not_implemented():
    """
    Subclasses of InterpolatedFunction must implement nodes().
    """
    with pytest.raises(TypeError):
        NodesNotImplementedMock()


class NumNodesNotImplementedMock(InterpolatedFunction):
    def __call__(self, x):
        pass

    def update(self, values):
        pass

    def nodes(self):
        pass

    def __getitem__(self, key):
        pass


def test_num_nodes_not_implemented():
    """
    Subclasses of InterpolatedFunction must implement .num_nodes.
    """
    with pytest.raises(TypeError):
        NumNodesNotImplementedMock()


class DimStateNotImplementedMock(InterpolatedFunction):
    def __call__(self, x):
        pass

    def update(self, values):
        pass

    def nodes(self):
        pass

    @property
    def num_nodes(self):
        pass

    def __getitem__(self, key):
        pass


def test_dim_state_not_implemented():
    """
    Subclasses of InterpolatedFunction must implement .dim_state.
    """
    with pytest.raises(TypeError):
        DimStateNotImplementedMock()


class GetItemNotImplementedMock(InterpolatedFunction):
    def __call__(self, x):
        pass

    def update(self, values):
        pass

    def nodes(self):
        pass

    @property
    def num_nodes(self):
        pass

    @property
    def dim_state(self):
        pass


def test_get_item_not_implemented():
    """
    Subclasses of InterpolatedFunction must implement __get_item__().
    """
    with pytest.raises(TypeError):
        GetItemNotImplementedMock()


class NumpyNodesInterpolatedFunction(InterpolatedFunction):
    def __call__(self, x):
        pass

    def update(self, values):
        pass

    def nodes(self):
        yield [0.0, 0.0]
        yield [1.0, 2.0]

    def num_nodes(self):
        pass

    @property
    def dim_state(self):
        pass

    def __getitem__(self, key):
        pass


def test_numpy_nodes():
    """
    numpy_nodes() should return a Numpy array of all the nodes.
    """
    func = NumpyNodesInterpolatedFunction()
    assert np.all(func.numpy_nodes() == np.array([[0.0, 0.0], [1.0, 2.0]]))


class DerivativeNotImplementedMock(DifferentiableInterpolatedFunction):
    def __call__(self, x):
        pass

    def update(self, values):
        pass

    def nodes(self):
        pass


def test_derivative_not_implemented():
    """
    Subclasses of DifferentiableInterpolatedFuncion must implement
    derivative().
    """
    with pytest.raises(TypeError):
        DerivativeNotImplementedMock()


class SecondDerivativeNotImplementedMock(TwiceDifferentiableInterpolatedFunction):
    def __call__(self, x):
        pass

    def update(self, values):
        pass

    def nodes(self):
        pass

    def derivative(self, x):
        pass


def test_second_derivative_not_implemented():
    """
    Subclasses of TwiceDifferentiableInterpolatedFuncion must implement
    second_derivative().
    """
    with pytest.raises(TypeError):
        SecondDerivativeNotImplementedMock()


# low max_value to speed up the test
@given(st.integers(max_value=4))
def test_chebyshev_nodes_per_state_check(nodes_per_state):
    """
    THe Chebyshev constructor must raise an error if the nodes per state
    is not >= 2, otherwise not.
    """
    degree = nodes_per_state - 1
    if nodes_per_state < 2:
        with pytest.raises(ValueError):
            ChebyshevInterpolatedFunction(
                nodes_per_state, degree, np.zeros(3), np.ones(3)
            )
    else:
        ChebyshevInterpolatedFunction(nodes_per_state, degree, np.zeros(3), np.ones(3))


# low max_value speeds up the test
@given(st.integers(max_value=4))
def test_chebyshev_degree_check(degree):
    """
    The Chebyshev constructor must raise an error if the degree is < 1,
    otherwise not.
    """
    if degree < 1:
        with pytest.raises(ValueError):
            ChebyshevInterpolatedFunction(degree + 1, degree, np.zeros(3), np.ones(3))
    else:
        ChebyshevInterpolatedFunction(degree + 1, degree, np.zeros(3), np.ones(3))


@given(
    # low max_value speeds up the test
    st.integers(min_value=1, max_value=4),
    st.integers(min_value=1, max_value=4),
)
def test_chebyshev_nodes_per_state_degree_check(nodes_per_state, degree):
    """
    The Chebyshev constructor must raise an exception if the number of
    nodes per state is <= than degree of the polynomial, and no exception
    otherwise.
    """
    if nodes_per_state <= degree:
        with pytest.raises(ValueError):
            ChebyshevInterpolatedFunction(
                nodes_per_state, degree, np.zeros(3), np.ones(3)
            )
    else:
        ChebyshevInterpolatedFunction(nodes_per_state, degree, np.zeros(3), np.ones(3))


def test_chebyshev_node_bounds_length():
    """
    The Chebyshev constructor must raise an error if the length of the
    node lower bound vector is different form the node upper bound vector.
    """
    with pytest.raises(ValueError):
        ChebyshevInterpolatedFunction(3, 2, np.zeros(2), np.ones(3))


def test_chebyshev_node_incorrect_bounds():
    """
    The Chebyshev constructor must raise an exception if one of the
    lower bounds is >= one of the upper bounds.
    """
    with pytest.raises(ValueError):
        ChebyshevInterpolatedFunction(3, 2, np.array([0, 1]), np.array([1, 0.5]))


def test_chebyshev_nodes():
    """
    The Chebyshev nodes() function should return an iterable of
    the Cartesian product of the nodes of every dimension.
    """
    # Use [-1, 1] as domain so that for every variable, the nodes
    # equal the standard Chebyshev nodes
    func = ChebyshevInterpolatedFunction(2, 1, -1 * np.ones(2), np.ones(2))
    nodes = func.nodes()
    assert isinstance(nodes, collections.Iterable)
    products = set(nodes)  # The order of the nodes does not matter
    assert products == {
        (func.cheb_nodes[0], func.cheb_nodes[0]),
        (func.cheb_nodes[0], func.cheb_nodes[1]),
        (func.cheb_nodes[1], func.cheb_nodes[0]),
        (func.cheb_nodes[1], func.cheb_nodes[1]),
    }


def test_chebyshev_nodes_stable_order():
    """
    ChebyshevInterpolatedFunction.nodes() must return the same order on
    every invocation.
    """
    func = ChebyshevInterpolatedFunction(2, 1, np.array([0, -1]), np.array([1, 2]))
    assert list(func.nodes()) == list(func.nodes())


@st.composite
def node_bounds(draw):
    dim_state = draw(st.integers(min_value=1, max_value=3))
    node_min = draw(
        st.lists(
            st.floats(
                allow_nan=False,
                allow_infinity=False,
                # Bound values to prevent float overflow issues
                min_value=-1000,
                max_value=1000,
            ),
            min_size=dim_state,
            max_size=dim_state,
        )
    )
    node_max = []
    for i in range(dim_state):
        node_max.append(
            draw(
                st.floats(
                    min_value=node_min[i] + 1, allow_nan=False, allow_infinity=False
                )
            )
        )
    return node_min, node_max


@given(
    node_bounds=node_bounds(), nodes_per_state=st.integers(min_value=2, max_value=10)
)
def test_chebyshev_nodes_in_range(nodes_per_state, node_bounds):
    """
    ChebyshevInterpolatedFunction nodes must lie within the specified range.
    """
    node_min, node_max = node_bounds
    func = ChebyshevInterpolatedFunction(nodes_per_state, 1, node_min, node_max)
    nodes = func.numpy_nodes()
    for i in range(func.dim_state):
        assert np.all(node_min[i] <= nodes[:, i])
        assert np.all(nodes[:, i] <= node_max[i])


def test_chebyshev_getitem_int():
    """
    The ChebyshevInterpolatedFunction __getitem__() implementation must
    return the i'th element of numpy_nodes() when passed i as a key.
    """
    func = ChebyshevInterpolatedFunction(4, 1, [-1, 0, 10], [1, 3, 20])
    nodes = func.numpy_nodes()
    for i in range(func.num_nodes):
        assert np.all(func[i] == nodes[i])


def slices(ir):
    """
    Returns a slice where all elements are within the index range `ir'.
    """
    return st.tuples(
        st.one_of(st.none(), st.integers(min_value=-ir, max_value=ir - 1)),
        st.one_of(st.none(), st.integers(min_value=-ir, max_value=ir - 1)),
        st.one_of(
            st.none(),
            st.integers(min_value=-ir, max_value=-1),
            st.integers(min_value=1, max_value=ir),
        ),
    ).map(lambda x: slice(*x))


@given(slices(64))
def test_chebyshev_getitem_slice(ind):
    """
    The ChebyshevInterpolatedFunction __getitem__() implementation must
    work with slice indices.
    """
    func = ChebyshevInterpolatedFunction(4, 1, [-1, 0, 10], [1, 3, 20])
    nodes = func.numpy_nodes()
    assert np.all(func[ind] == nodes[ind])


def indices(ir):
    """
    Returns an index or slice where all elements are within the index range
    `ir'.
    """
    return st.one_of(slices(ir), st.integers(min_value=-ir, max_value=ir - 1))


@given(indices(64), indices(3))
def test_chebyshev_getitem_numpy_slice(ind1, ind2):
    """
    The ChebyshevInterpolatedFunction __getitem__() implementation must
    work with NumPy slices.
    """
    func = ChebyshevInterpolatedFunction(4, 1, [-1, 0, 10], [1, 3, 20])
    nodes = func.numpy_nodes()
    assert np.all(func[ind1, ind2] == nodes[ind1, ind2])


interpolation_tests = [
    (10, 2, np.array([0]), np.array([1]), lambda x: x),
    (10, 2, np.array([0, 1]), np.array([1, 10]), lambda x, y: x - y ** 2),
    # The following 4 test cases are from Franke (1979), "A Critical
    # Comparison of Some Methods for Interpolation of Scattered Data"
    # (http://hdl.handle.net/10945/35052)
    (
        100,
        3,
        np.array([0, 0]),
        np.array([1, 1]),
        lambda x, y: (1.25 + np.cos(5.4 * y)) / (6 * (1 + (3 * x - 1) ** 2)),
    ),
    (
        100,
        3,
        np.array([0, 0]),
        np.array([1, 1]),
        lambda x, y: (np.tanh(9 * y - 9 * x) + 1) / 9,
    ),
    (
        100,
        3,
        np.array([0, 0]),
        np.array([1, 1]),
        lambda x, y: np.exp(-81 * ((x - 0.5) ** 2 + (y - 0.5) ** 2 / 16)) / 3,
    ),
    (
        100,
        3,
        np.array([0, 0]),
        np.array([1, 1]),
        lambda x, y: np.exp(-81 * ((x - 0.5) ** 2 + (y - 0.5) ** 2 / 4)) / 3,
    ),
    (
        100,
        3,
        np.array([0, 0]),
        np.array([1, 1]),
        lambda x, y: np.sqrt(64 - 81 * ((x - 0.5) ** 2 + (y - 0.5) ** 2)) / 9 - 0.5,
    ),
]


@given(
    nodes_per_state=st.integers(min_value=2, max_value=5),
    n_states=st.integers(min_value=1, max_value=4),
)
def test_chebyshev_num_nodes(nodes_per_state, n_states):
    func = ChebyshevInterpolatedFunction(
        nodes_per_state, 1, np.zeros(n_states), np.ones(n_states)
    )
    assert func.num_nodes == nodes_per_state ** n_states


@pytest.mark.parametrize(
    "nodes_per_state, degree, node_min, node_max, func", interpolation_tests
)
def test_chebyshev_fit(nodes_per_state, degree, node_min, node_max, func):
    """
    The Chebyshev approximation should be close to the true function value.
    """
    approx = ChebyshevInterpolatedFunction(nodes_per_state, degree, node_min, node_max)
    vals = np.zeros(approx.n_nodes)
    for i, node in enumerate(approx.nodes()):
        vals[i] = func(*node)
    approx.update(vals)
    approx_vals = np.array([approx(n) for n in approx.nodes()])
    assert np.linalg.norm(approx_vals - vals) / vals.shape[0] <= 1e-3


@pytest.mark.parametrize(
    "nodes_per_state, degree, node_min, node_max, func", interpolation_tests
)
def test_chebyshev_derivative(nodes_per_state, degree, node_min, node_max, func):
    """
    The calculated derivative of the Chebyshev should be close to a
    numerically calculated one.
    """
    approx = ChebyshevInterpolatedFunction(nodes_per_state, degree, node_min, node_max)
    vals = np.zeros(approx.n_nodes)
    for i, node in enumerate(approx.nodes()):
        vals[i] = func(*node)
    approx.update(vals)

    for node in approx.nodes():
        assert_allclose(
            scipy.optimize.check_grad(approx, approx.derivative, node), 0, atol=1e-6
        )


@pytest.mark.parametrize(
    "nodes_per_state, degree, node_min, node_max, func", interpolation_tests
)
def test_chebyshev_second_derivative(nodes_per_state, degree, node_min, node_max, func):
    """
    The calculated second derivative of the Chebyshev should be close to a
    numerically calculated one.
    """
    approx = ChebyshevInterpolatedFunction(nodes_per_state, degree, node_min, node_max)
    vals = np.zeros(approx.n_nodes)
    for i, node in enumerate(approx.nodes()):
        vals[i] = func(*node)
    approx.update(vals)

    dim_state = len(node_min)
    for node in approx.nodes():
        # Calculate the Hessian only once and use a constant Lambda function
        # below for speed
        hess = approx.second_derivative(node)
        for i in range(dim_state):
            assert_allclose(
                scipy.optimize.check_grad(
                    lambda x: approx.derivative(x)[i],
                    lambda x: hess[i],
                    node,
                    epsilon=np.sqrt(np.finfo(float).eps),
                ),
                0,
                atol=1e-6,
            )


def test_chebyshev_share():
    """
    share() should generate an exacpt copy, where all elements except
    for coefs are shared in memory with the original.
    """
    original = ChebyshevInterpolatedFunction(3, 2, np.array([0]), np.array([1]))
    copy = original.share()
    for k in set(original.__dict__).union(copy.__dict__):
        if k != "coefs":
            assert original.__dict__[k] is copy.__dict__[k]
        else:
            assert original.__dict__[k] is not copy.__dict__[k]
            assert np.all(original.__dict__[k] == copy.__dict__[k])
