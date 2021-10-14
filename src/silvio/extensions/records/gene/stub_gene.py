"""
StubGene is a gene with no information about its actual sequence.
It only has properties about its name.
"""

from __future__ import annotations

from Bio.Seq import Seq

from .gene import Gene



class StubGene (Gene) :

    _name: str

    def __init__ ( self, name:str ) :
        super().__init__()
        self._name = name

    @property
    def name ( self ) -> str :
        return self._name

    @property
    def orf ( self ) -> Seq :
        return Seq("Z")

    @property
    def prom ( self ) -> Seq :
        return Seq("Z")
