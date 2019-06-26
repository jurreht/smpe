from distutils.core import setup

setup(
    name="SMPE",
    version="0.1.0",
    author="Jurre H. Thiel",
    author_email="j.h.thiel@vu.nl",
    packages=["smpe"],
    license="LICENSE",
    description="A library to efficiently compute (Sparse) Markov"
    "Perfect Equilibria of dynamic games",
    long_description=open("README.md").read(),
    install_requires=[
        "dask >= 1.0.0",
        "numpy >= 1.15.0",
        "scipy >= 1.2.0",
        "numba >= 0.41",
        "xxhash >= 1.3.0",
    ],
)
