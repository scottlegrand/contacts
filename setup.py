#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='contacts',
      packages=['contacts'],
      version=0.1,
      description='GPU contact calculation',
      author='Scott LeGrand',
      author_email='slegrand@nvidia.com',
      url='https://github.com/scottlegrand/contacts',
      license='BSD',
      setup_requires=['cupy'],
      install_requires=open('requirements.txt', 'r').readlines(),
      keywords=['contact calculation', 'RFScore', 'GPU', 'virtual screening', 'docking']
      )
