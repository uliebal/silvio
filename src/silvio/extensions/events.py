
from dataclasses import dataclass

from ..events import Event
from .records.gene.gene import Gene


@dataclass
class InsertGeneEvent ( Event ) :
    gene: Gene
    locus: int


@dataclass
class RemoveGeneEvent ( Event ) :
    gene: Gene
