
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
