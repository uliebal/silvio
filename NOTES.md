# Notes

This document contains development notes.

## Try PyPi Upload in Test Server

```bash
python3 -m pip install build
python3 -m build --sdist
# make sure only the most recent version is inside dist/
python3 -m twine upload --repository testpypi dist/*
```
