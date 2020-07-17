"""A setuptools based setup module.
"""

from setuptools import find_packages, setup

setup(
    name="prediction_utils",
    version="0.6.0",
    description="Extractors and Models for predictions on OMOP CDM datasets",
    url="https://github.com/som-shahlab/fairness_benchmark",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas>=1.0.0",
        "matplotlib",
        "torch>=1.4",
        "pyyaml",
        "pyarrow>=0.17.0",
        "sqlalchemy",
        "dask>=2.14.0",
        "scipy",
        "sklearn",
        "torchcontrib",
        "configargparse",
        "google-cloud-bigquery",
        "google-cloud-bigquery-storage",
        "pandas-gbq",
        "pytest",
        "tqdm",
    ],
)
