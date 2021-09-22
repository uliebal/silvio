"""
Common datatypes between the different extensions.
"""

from typing import List, Literal, Optional


# One of the simple 4 nucleic bases, plus the unknown base.
Base = Literal['A','C','G','T']


# Well distinguished promoter sites.
PromoterSite = Literal[ '-10', '-35' ]


# Read methods usable by the sequencer.
ReadMethod = Literal[ 'single-read', 'paired-end' ]
