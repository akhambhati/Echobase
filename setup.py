import io
import os
import re

from setuptools import find_packages, setup

import version

HERE = os.path.abspath(os.path.dirname(__file__))

# ==============================================================================
# Variables
# ==============================================================================

NAME = "Echobase"
VERSION = version.get_version()
DESCRIPTION = "Echobase: A (dynamic) graph construction and analysis package for neuronal data."
with open(os.path.join(HERE, 'README.rst'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()
PACKAGES = find_packages()
AUTHOR = "Ankit N. Khambhati"
AUTHOR_EMAIL = "akhambhati@gmail.com"
DOWNLOAD_URL = 'http://github.com/akhambhati/echobase/'
LICENSE = 'GNU General Public License (GPL)'
INSTALL_REQUIRES = ['numpy', 'scipy', 'mtspec', 'statsmodels',
                    'matplotlib', 'mayavi', 'vtk', 'ipython']

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
    ],
    packages=PACKAGES,
    install_requires=INSTALL_REQUIRES,
)
