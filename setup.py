import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='silvio',
    version='0.0.4',
    author='Ulf Liebal, Rafael Schimassek',
    author_email='ulf.liebal@rwth-aachen.de',
    description='silvio is an environment for Simulation of Virtual Organisms. silvio contains several linked microbial models.',
    keywords='biotechnology, microbiology, virtual cell, systems biology',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://git.rwth-aachen.de/ulf.liebal/silvio.git',
#    include_package_data = True,
    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    package_data = {'': ['*.csv','*.pkl','*.xml']},
    python_requires='>=3.9',
    install_requires=[
        'biopython >= 1.79',
        'joblib >= 1.0.1',
        'matplotlib >= 3.3.4',
        'numpy >= 1.18.0',
        'pandas >= 1.2.1',
        'pickleshare >= 0.7.5',
        'scipy >= 1.6.0',
    ],
)
