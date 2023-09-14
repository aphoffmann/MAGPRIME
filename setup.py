# Purpose: Setup file for MAGPRIME package
from setuptools import setup

REQUIREMENTS = [
    'pandas',
    'numpy',
    'pytest',
    'scipy',
    'numba',
    'tqdm',
    'matplotlib',
    'toolz',
    'nsgt',
    'cvxpy',
    'scikit-learn',
    'tqdm',
    'hdbscan',
    'git+https://github.com/aphoffmann/wavelets.git',
    'git+https://github.com/aphoffmann/pymssa.git'
]

setup(
    name='magprime',
    version='0.1',
    description="Multivariate Singular Spectrum Analysis (MSSA)",
    author="Alex Paul Hoffmann",
    author_email='aphoff@umich.edu',
    url='https://github.com/aphoffmann/MAGPRIME',
    packages=['magprime'],
    install_requires=REQUIREMENTS,
)