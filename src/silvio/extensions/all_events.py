
from typing import Optional
from dataclasses import dataclass

from cobra.core import Model as CobraModel
from Bio.Seq import Seq

from ..events import Event
from .records.gene.gene import Gene


@dataclass
class InsertGeneEvent ( Event ) :
    gene: Gene
    locus: Optional[int]


@dataclass
class RemoveGeneEvent ( Event ) :
    gene: Gene


@dataclass
class AlterGenePromoterEvent ( Event ) :
    gene: Gene
    new_promoter: Seq
    # TODO: Not implemented in GeneList and GeneLibrary.


@dataclass
class AlterGeneExpressionEvent ( Event ) :
    gene: Gene
    new_expression: float

