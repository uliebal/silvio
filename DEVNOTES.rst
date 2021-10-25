===
Developer Notes
===

This file contains notes for the developers.


Setup silvio for development
---

* Clone the repository and enter it.
* Install developer dependencies with :code:`pip install -r requirements_dev.txt`
* Install silvio as an editable module with :code:`pip install -e .`


Try PyPi Upload in Test Server
--------

.. code-block:: bash
        python3 -m pip install build
        python3 -m build --sdist
        # make sure only the most recent version is inside dist/
        python3 -m twine upload --repository testpypi dist/*

Also check out these notes: https://cookiecutter-pypackage.readthedocs.io/en/latest/pypi_release_checklist.html


.. _credits:

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage


