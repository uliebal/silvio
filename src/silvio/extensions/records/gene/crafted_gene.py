"""
Gene crafted by the user itself.
"""

from __future__ import annotations

from Bio.Seq import Seq

from .gene import Gene



class CraftedGene (Gene) :

    _name: str

    _orf: Seq

    _prom: Seq

    def __init__ ( self, name:str, orf:Seq, prom:Seq ) :
        super().__init__()
        self._name = name
        self._orf = orf
        self._prom = prom

    # def clone ( self ) -> CraftedGene :
    #     return CraftedGene(
    #         name=self._name,
    #         orf=self._orf, # Can pass because Seq is immutable.
    #         prom=self._prom, # Can pass because Seq is immutable.
    #     )

    @property
    def name ( self ) -> str :
        return self._name

    @property
    def orf ( self ) -> Seq :
        return self._orf

    @property
    def prom ( self ) -> Seq :
        return self._prom


    def make_variant ( self, orf_loc:int, orf_sub:Seq ) -> CraftedGene :
        """ TODO: Maybe its better to clone-then-mutate. """
        new_orf = self._orf[:orf_loc-1] + orf_sub + self._orf[orf_loc:] # orf_loc starts at 1
        return CraftedGene(
            name=self._name,
            orf=new_orf,
            prom=self._prom, # Can pass because Seq is immutable.
        )
