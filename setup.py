#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'biopython ~= 1.79',
    'cobra >= 0.22',
    'joblib >= 1.0',
    'matplotlib ~= 3.4',
    'numpy ~= 1.23',
    'pandas ~= 1.2',
    'pickleshare ~= 0.7',
    'scipy ~= 1.6',
    'jinja2<3.1.0',
]

test_requirements = ['pytest>=3', ]

# getting the latest version from the __init__.py file in src/silvio
import re
import os
VERSIONFILE = os.path.join('src', 'silvio', '__init__.py')
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__=['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

setup(
    name='silvio',
    version=verstr, # update version string in src/silvio/__init__.py
    url='https://git.rwth-aachen.de/ulf.liebal/silvio.git',
    author="Ulf Liebal",
    author_email='ulf.liebal@rwth-aachen.de',
    python_requires='>=3.9', 
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Typing :: Typed',
    ],
    description="silvio is an environment for Simulation of Virtual Organisms. silvio contains several linked microbial models.",
    entry_points={},
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='biotechnology, microbiology, virtual cell, systems biology',
    package_dir={'': 'src'},
    packages=find_packages(where="./src"),
    test_suite='tests',
    tests_require=test_requirements,
    zip_safe=False,
)
