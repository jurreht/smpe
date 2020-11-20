import logging

import numpy as np
import quadprog


def aqlr_stat(W, eps):
    n, k = W.shape
    omega = np.cov(W, rowvar=False, bias=True)

    omega_adj = np.max(eps - np.linalg.det(omega), 0) * np.eye(k) + omega

    omega_inv = np.linalg.inv(omega_adj)
    W_mean = np.mean(W, axis=0)
    W_scaled = np.sqrt(n) * W_mean
    max_val = quadprog.solve_qp(
        omega_inv,
        W_scaled.T @ omega_inv,
        -1 * np.eye(k),
        np.zeros(k)
    )[1]
    return 2 * max_val + W_scaled.T @ omega_inv @ W_scaled


def max_stat(W):
    n, k = W.shape
    return np.max(np.sqrt(n) * W.mean(axis=0) / np.std(W, axis=0))


def test_inequalities(
    W, alpha=.05, beta=0.005, B=499, stat_fun=None, p_value=False,
    seed=12345
):
    """
    Tests whether the inequalities using Romano, Shaik and Wolf (2014)
    two-stage bootstrap.

    The function tests whether E[W[:, j]] <= 0 for all j. The notation
    in this function tries to mirror that of the paper as close as possible.

    Params:
    W (np.ndarray[N, K]): N samples of K inequalities
    alpha (float): size of main test
    beta (float): size of first stage bootstrap. RSM recommend
        to take beta = alpha / 10
    B (int, optional): Number of bootstrap samples to use
    stat_fun (callable, optional): Function to calculate test statistic
        as function of W. The module provides `aqlr_stat` and `max_stat`.
        If none, `aqlr_stat` with epsilon=1e-7 is used.
    seed (float): Seed for random number generator

    Returns:
    bool: whether the null can be rejected (True=rejection)
    """
    np.random.seed(seed)
    logger = logging.getLogger(__name__)

    if stat_fun is None:
        stat_fun = lambda W: aqlr_stat(W, 1e-7)

    n, k = W.shape

    # Precompute some important numbers
    W_mean = np.mean(W, axis=0)
    omega = np.cov(W, rowvar=False, bias=True)
    S2 = np.diag(omega)
    S = np.sqrt(S2)

    K_inv = _bootstrap_quantile(W, W_mean, beta, B)
    logger.info('Bootstrapped moment quantile: {}'.format(K_inv))

    # Upper limits of the bootstrap confidence region
    max_mu = W_mean + S * K_inv / np.sqrt(n)

    if np.all(max_mu <= 0):
        # The bootstrapped confidence region is completely
        # below 0 on all axes => we're done here (cannot reject)
        logger.info("Can't reject since the upper bound on all moments is negative")
        return False

    # Test statistic
    T = stat_fun(W)
    logger.info('Test statistic: {}'.format(T))

    test_stat = _bootstrap_test_stats(W, W_mean, alpha, beta, B, max_mu, stat_fun)
    if p_value:
        return (T < test_stat).mean() + beta
    else:
        c = np.percentile(test_stat, 100 * (1 - alpha + beta),
                          interpolation='higher')
        logger.info('Bootstrapped critical value: {}'.format(c))

        return T > c


def _bootstrap_quantile(W, W_mean, beta, B):
    n, k = W.shape

    test_stat = np.zeros(B)
    for b in range(B):
        # Make bootstrap sample
        W_bs = W[np.random.randint(n, size=n)]

        t_stats = np.sqrt(n) * (W_mean - W_bs.mean(axis=0)) / np.std(W_bs, axis=0)
        test_stat[b] = np.max(t_stats)

    return np.percentile(test_stat, 100 * (1 - beta), interpolation='higher')


def _bootstrap_test_stats(W, W_mean, alpha, beta, B, max_mu, stat_fun):
    n, k = W.shape
    lambda_ = np.minimum(max_mu, 0)

    test_stat = np.zeros(B)
    for b in range(B):
        # Make bootstrap sample
        W_bs = W[np.random.randint(n, size=n)]

        # W_bs_mean = np.mean(W_bs, axis=0)
        test_stat[b] = stat_fun(W_bs - W_mean + lambda_)

    return test_stat
