import dask.distributed
import pytest


def pytest_addoption(parser):
    parser.addoption(
        '--singlethread', action='store_true', dest='singlethread')


@pytest.fixture(scope='session')
def dask_client(request):
    # Start one Dask session for all tests, and close it on shutdown
    # This is more efficient than starting a Client every time a DynamicGame
    # is instantiated.

    if request.config.getoption('singlethread'):
        # Run on a single thread
        cluster = dask.distributed.LocalCluster(
            n_workers=1, processes=False, threads_per_worker=1)
    else:
        # Run on the default local cluster
        cluster = dask.distributed.LocalCluster()

    client = dask.distributed.Client(cluster)
    yield client
    client.close()
