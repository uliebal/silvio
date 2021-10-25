
from __future__ import annotations

from Bio.Seq import Seq

from .gene import Gene



class LocalizedGene (Gene) :
    """
    Gene that stores its sequence in a shared sequence.

    .. code-block::

        StartLoc       + PromLen       + OrfLen
            |----------------|--------------|

    TODO: Simple for now. Only has promoter and ORF which are adjacent to each other.
    """

    _name: str

    _seq: Seq

    _start_loc: int

    _prom_len: int

    _orf_len: int

    def __init__ ( self, name:str, seq:Seq, start_loc:int, prom_len:int, orf_len:int ) :
        super().__init__()
        self._name = name
        self._seq = seq
        self._start_loc = start_loc
        self._prom_len = prom_len
        self._orf_len = orf_len

    # def clone ( self ) -> LocalizedGene :
    #     return LocalizedGene(
    #         name=self.name,
    #         seq=self.seq, # Can pass because Seq is immutable.
    #         start_loc=self.start_loc,
    #         prom_len=self.prom_len,
    #         orf_len=self.orf_len,
    #     )

    @property
    def name ( self ) -> str :
        return self._name

    @property
    def orf ( self ) -> Seq :
        start = self._start_loc + self._prom_len
        end = start + self._orf_len
        return self._seq[start:end] # TODO: Test is end is include or exclusive

    @property
    def prom ( self ) -> Seq :
        start = self._start_loc
        end = start + self._prom_len
        return self._seq[start:end] # TODO: Test is end is include or exclusive



    @property
    def seq ( self ) -> Seq :
        return self._seq

    @property
    def start_loc ( self ) -> str :
        return self._start_loc

    @property
    def end_loc ( self ) -> str :
        return self._start_loc + self._prom_len + self._orf_len

    @property
    def prom_len ( self ) -> str :
        return self._prom_len

    @property
    def orf_len ( self ) -> str :
        return self._orf_len


    @seq.setter
    def seq ( self, value ) -> None:
        self._seq = value

    @start_loc.setter
    def start_loc ( self, value ) -> None:
        self._start_loc = value
