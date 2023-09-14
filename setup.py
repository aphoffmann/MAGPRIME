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
    'cvxpy',
    'scikit-learn',
    'tqdm',
    'nsgt',
    'hdbscan',
    'magpylib',
    'wavelets @ git+https://github.com/aphoffmann/wavelets.git',
    'pymssa @ git+https://github.com/aphoffmann/pymssa.git',
]

DEPENDENCY_LINKS = [    
    'git+https://github.com/aphoffmann/wavelets.git',
    'git+https://github.com/aphoffmann/pymssa.git'

    
]


setup(
    name='magprime',
    version='0.1',
    description="Magnetic signal PRocessing, Interference Mitigation, and Enhancement (MAGPRIME)",
    author="Alex Paul Hoffmann",
    author_email='aphoff@umich.edu',
    url='https://github.com/aphoffmann/MAGPRIME',
    packages=['magprime'],
    install_requires=REQUIREMENTS,
    dependency_links=DEPENDENCY_LINKS
)