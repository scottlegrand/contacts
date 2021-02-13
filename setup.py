#!/usr/bin/env python
from setuptools import setup, find_packages

kw = {}
try:
    import cupy
except ImportError:
    # declare dependency
    kw['install_requires'] = ['cupy']

setup(name='contacts',
      packages=['contacts'],
      version=0.1,
      description='GPU contact calculation',
      author='Scott LeGrand',
      author_email='slegrand@nvidia.com',
      url='https://github.com/scottlegrand/contacts',
      license='BSD',
      keywords=['contact calculation', 'RFScore', 'GPU', 'virtual screening', 'docking'],
      **kw
      )
