
Developer Notes
===============

This file contains notes for the developers.


Setup silvio for development
----------------------------

* Clone the repository and enter it.
* Install developer dependencies with :code:`pip install -r requirements_dev.txt`
* Install silvio as an editable module with :code:`pip install -e .`


Try PyPi Upload in Test Server
------------------------------

.. code-block:: bash

        # Bump the source code to the next version.
        make bump-patch
        # Build the distribution files and latest docs
        make docs
        make dist
        # Upload the package to the test PyPi server.
        python3 -m twine upload --repository testpypi dist/*

.. code-block:: bash

        # Test installation:
        pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ silvio


Also check out these notes: https://cookiecutter-pypackage.readthedocs.io/en/latest/pypi_release_checklist.html

Setup project for local development
-----------------------------------
.. code-block:: bash

	# Setup the python virtual environment next to it. (use Python 3.8)
	python3.8 -m venv py38-env
	# Activate your environment. (Broad topic that depends on what software and OS is used)
	source py38-env/bin/activate
	# Clone the repository to a nearby folder.
	git clone https://git.rwth-aachen.de/ulf.liebal/biolabsim.git repo-biolabsim
	# Enter the newly cloned repository.
	cd repo-biolabsim
	# Install all required python libraries.
	pip install -r requirements.txt

# See the Notebook for examples on how to use the library.


Generate Sphinx documentation
-----------------------------

Sphinx is not very automatic on how documentation is extracted from the code. We use
[sphinx-apidoc](https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html) to periodically
generate the documentation `.rst` files.

.. code-block:: bash

	# Assuming you start at the project root directory.
	# Enter the documentation directory.
	cd docs
	# Remove the old API documentation.
	rm -ri ./api
	# Generate the new reStructuredText files for the API documentation.
	sphinx-apidoc --module-first -d 4 -o api ../biolabsim
	# Generate the HTML from all documentation files.
	make html





Writing a Module
----------------

Prefer the use of [standardized code style](https://pep8.org/).

Make use of [python type hints](https://docs.python.org/3/library/typing.html) whenever possible.
When specifying types for variables and methods, your IDE will help you with organizing the inputs,
outputs and arguments that you may use.

.. code-block:: python

	# Initial definition of a variable to store a probability
	some_probability: float = 0
	some_probability = 0.4      # Will work. The variable may receive fractional numbers.
	some_probability = 0        # Will work. Integers are also numbers.
	some_probability = "a lot"  # Error! The IDE will notify us about this bad assignment.
	some_probability = "0.3"    # Error! This is still a string. No more conversion problems.
	some_probability = -1.4     # Unfortunately this still works. Typing only defines simple types.

When writing classes, keep all properties (variables inside a class) at the top of the class definition,
outside of the constructor. The constructor should only perform the initial assignment.

.. code-block:: python
	class BayesianNetworkNode :
	    """
	    Each class should document what it does. Ideally, it should have a single purpose.
	    """
	    # Probability that this node is true.
	    true_prob: float
	    # Probability that the node is false. Should be inverse of true probability.
	    false_prob: float


	    def __init__ ( self, true_prob: float ) :
	        # Notice that constructor arguments may have the same name as properties.
        	self.true_prob = true_prob
	        # The constructor only uses necessary arguments to initialize all properties.
        	self.false_prob = 1 - true_prob

How to name things is a very debated topic in many languages. When in doubt, follow the conventions
that have been laid by the [python standard](https://www.python.org/dev/peps/pep-0008/#naming-conventions).
Some common examples are.

.. code-block:: python

	# Use lower_case with underscores. Prefer distinct names to single letters.
	num_strands = 2
	# Constants are values embedded into the code. Use UPPER_CASE with underscores.
	GOLDEN_RATIO = 1.6180
	# Module names use lower_case and avoids underscore when possible.
	import biolabsim.sequencing.evaluation
	# Custom types use PascalCase.
	from typing import Tuple, Literal
	GeneBase = Literal['A','T','C','G']
	# Functions use lower_case and typically start with a verb.
	def complement_base ( base:GeneBase ) -> GeneBase :  # (input) -> output
	# Include most initilization on top of the method.
	orig_bases: List[GeneBase] = ['A','T','C','G']  # Common words may be shortened. orig = original
	comp_bases: List[GeneBase] = ['T','A','G','C']  # But spell it out in comments.  comp = complementary
	# Split your code into blocks of related operations. Provide a small summary of each block.
    	# Comments should help outsiders to skim through the code and to explain programming decisions.
    	found_orig_index = orig_bases.index(base)  # Avoid one-liners. Variable names provide context.
    	return comp_bases[found_orig_index]
	# Use simple types to construct more complex ones.
	Codon = Tuple[ GeneBase, GeneBase, GeneBase ]

	# Classes use PascalCase as well.
	class AminoAcid :
	    # Class properties use lower_case as well.
	    gene_triplet : Codon
    	# Constructors initialize the properties.
	def __init__ ( self, base1:GeneBase, base2:GeneBase, base3:GeneBase ) :
        	self.gene_triplet = ( base1, base2, base3 )

	# Leave enough space between method definitions.
	def complement_triplet (self) -> Codon :
        return (                                       # Use multiple lines and more spacing if the
            complement_base( self.gene_triplet[0] ),   # code becomes too bulky.
            complement_base( self.gene_triplet[1] ),
            complement_base( self.gene_triplet[2] ),
        )




Contacts
--------

*Ulf Liebal, Lars Blank*

Institute of Applied Microbiology-iAMB, Aachen Biology and Biotechnology-ABBT, RWTH Aachen University, Worringerweg 1, 52074 Aachen Germany



Last update: 1 April, 2022

Contact: ulf.liebal@rwth-aachen.de

Licence: See LICENCE file @https://git.rwth-aachen.de/ulf.liebal/biolabsim, or @https://github.com/uliebal/BioLabSim


Click the Jupyter icon to start BioLabSim at the Jupyter instance RWTH Aachen University:

[![](https://jupyter.pages.rwth-aachen.de/documentation/images/badge-launch-rwth-jupyter.svg)](https://jupyter.rwth-aachen.de/hub/spawn?profile=biolabsim)
