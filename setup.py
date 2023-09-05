try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
        'description': 'MAGnetic signal PRocessing, Interference Mitigation, and Enhancement',
        'author': "Alex Paul Hoffmann",
        'url': 'https://github.com/aphoffmann/MAGPRIME',
        'author_email': 'aphoff@umich.edu',
        'version': '0.1',
        'packages': ['magprime'],
        'name': 'magprime',
        }

setup(**config)