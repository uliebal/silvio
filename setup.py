#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'biopython ~= 1.79',
    'cobra ~= 0.22',
    'joblib ~= 1.0',
    'matplotlib ~= 3.4',
    'numpy ~= 1.18',
    'pandas ~= 1.2',
    'pickleshare ~= 0.7',
    'scipy ~= 1.6',
]

test_requirements = ['pytest>=3', ]

setup(
    name='silvio',
    version='0.1.2',
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
