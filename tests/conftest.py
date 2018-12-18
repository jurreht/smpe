import dask.distributed
import pytest


@pytest.fixture(scope='session')
def dask_client():
    # Start one Dask session for all tests, and close it on shutdown
    # This is more efficient than starting a Client every time a DynamicGame
    # is instantiated.
    client = dask.distributed.Client()
    yield client
    client.close()
