
import os
from pathlib import Path


# Global variable to see the directory where all data files are stored.
# TODO: In future, use an approach that works with setuptools and environment.
# DATADIR = Path(os.path.dirname(os.path.abspath(__file__))) / ".." / "data"
DATADIR = Path(os.path.dirname(os.path.abspath(__file__))) / "data"
