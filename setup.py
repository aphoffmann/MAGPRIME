# Purpose: Setup file for MAGPRIME package
from setuptools import setup, find_packages

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
    'magpylib',
    'keyboard',
    'wavelets @ git+https://github.com/aphoffmann/wavelets.git',
    'pymssa @ git+https://github.com/aphoffmann/pymssa.git',
    'nsgt @ git+https://github.com/aphoffmann/nsgt.git',
    'invertiblewavelets @ git+https://github.com/aphoffmann/invertiblewavelets.git'
]

DEPENDENCY_LINKS = [    
    'git+https://github.com/aphoffmann/wavelets.git',
    'git+https://github.com/aphoffmann/pymssa.git',
    'git+https://github.com/aphoffmann/invertiblewavelets.git',
    'git+https://github.com/aphoffmann/invertiblewavelets.git'

    
]


setup(
    name='magprime',
    version='1.8.0.3',
    description="Magnetic signal PRocessing, Interference Mitigation, and Enhancement (MAGPRIME)",
    author="Alex Paul Hoffmann",
    author_email='aphoff@umich.edu',
    url='https://github.com/aphoffmann/MAGPRIME',
    packages=find_packages(),
    package_data={'magprime.examples': ['*.ipynb'],
                  'magprime.utility.SPACE_DATA': ['*.dat', '*.csv'],},
    install_requires=REQUIREMENTS,
    dependency_links=DEPENDENCY_LINKS
)
