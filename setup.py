#!/usr/bin/env python

import logging
import sys
import pprint
import platform
from setuptools import setup, find_packages
from setuptools.extension import Extension

# Set up the logging environment
logging.basicConfig()
log = logging.getLogger()

# Handle the -W all flag
if 'all' in sys.warnoptions:
    log.level = logging.DEBUG

with open('VERSION.txt', 'r') as f:
    version = f.read()

# Use Cython if available
try:
    from Cython.Build import cythonize
except:
    log.critical(
        'Cython.Build.cythonize not found. '
        'Cython is required to build from a repo.')
    sys.exit(1)

# Use README.rst as the long description
with open('README.rst') as f:
    readme = f.read()

# Extension options
include_dirs = []
try:
    import numpy
    include_dirs.append(numpy.get_include())
except ImportError:
    log.critical('Numpy and its headers are required to run setup(). Exiting')
    sys.exit(1)

opts = dict(
    include_dirs=include_dirs,
)
log.debug('opts:\n%s', pprint.pformat(opts))
pyx_path = 'feerci.pyx'

if 'test' in sys.argv and platform.python_implementation() == 'CPython':
    from Cython.Build import cythonize
    ext_modules = cythonize(Extension(
        "feerci",
        [pyx_path],**opts,
        define_macros=[('CYTHON_TRACE', '1')]
    ), compiler_directives={
        'linetrace': True,
        'binding': True
    })
else:
    from Cython.Build import cythonize
    ext_modules = cythonize(Extension(
        "feerci",
        [pyx_path],**opts,
        extra_compile_args=['-O3']
    ))

# Dependencies
install_requires = [
    'numpy>=1.7',
    'Cython',
]
python_requires = '>=3.5'

setup_args = dict(
    name='feerci',
    version=version,
    description='FEERCI: A python package for EER confidence intervals ',
    long_description=readme,
    url='https://github.com/feerci/feerci',
    # author='FEERCI Dev',
    # author_email='feerci@example.com',
    # license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
    ],
    keywords=['EER', 'Confidence Interval', 'feerci','feer','biometrics'],
    ext_modules=ext_modules,
    install_requires=install_requires,
    packages=find_packages(exclude='tests'),
    tests_require=['cython', 'pytest', 'coverage'],
    test_suite='tests'
)

setup(**setup_args)