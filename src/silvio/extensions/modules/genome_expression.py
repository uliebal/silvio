"""
TODO: GenomeExpression needs to be reviewed.
"""

from typing import Union
from copy import copy

from Bio.Seq import Seq

from ...host import Host
from ...module import Module
from ...events import EventEmitter, EventLogger
from ..all_events import InsertGeneEvent, AlterGenePromoterEvent, AlterGeneExpressionEvent
from ..records.gene.gene import Gene
from ..utils.misc import Help_PromoterStrength
from .genome_library import GenomeLibrary
from .genome_list import GenomeList


CompatGenome = Union[GenomeLibrary,GenomeList]


class GenomeExpression ( Module ) :

    # Dependent module Genome Library holding the genes to express.
    genome: CompatGenome

    # Optimal primer length.
    opt_primer_len: int

    # Factor which influences the range of the promoter strength.
    # TODO: These characteristics maybe go elsewhere.
    infl_prom_str: float
    species_prom_str: float

    # Path to parameter files. TODO: use Paths.
    regressor_file: str
    addparams_file: str



    def make ( self,
        opt_primer_len:int,
        infl_prom_str:float,
        species_prom_str:float,
        regressor_file:str,
        addparams_file:str,
    ) -> None :
        self.opt_primer_len = opt_primer_len
        self.infl_prom_str = infl_prom_str
        self.species_prom_str = species_prom_str
        self.regressor_file = regressor_file
        self.addparams_file = addparams_file


    def copy ( self, ref:'GenomeExpression' ) -> None :
        self.opt_primer_len = ref.opt_primer_len
        self.infl_prom_str = ref.infl_prom_str
        self.species_prom_str = ref.species_prom_str
        self.regressor_file = ref.regressor_file
        self.addparams_file = ref.addparams_file



    def bind ( self, host:Host, genome:CompatGenome ) -> None :
        self.genome = genome
        host.observe( AlterGenePromoterEvent, self.listen_alter_gene_promoter )



    def sync ( self, emit:EventEmitter, log:EventLogger ) -> None :
        pass # Nothing to sync.



    def listen_alter_gene_promoter ( self, event:AlterGenePromoterEvent, emit:EventEmitter, log:EventLogger ) -> None :
        """ When a gene promoter is altered, we recalculate the expression and update it. """
        expr = self.calc_fast_wrong_prom_expr(event.new_promoter)
        log( "GenomeExpression: Changed gene={} to expression={}.".format(event.gene.name,expr) )
        emit(AlterGeneExpressionEvent( event.gene, expr ))



    def calc_fast_wrong_prom_expr ( self, prom:Seq ) -> float :
        """
        This method executes a wrong calculation of the promoter expression. It serves to
        demonstrate how editing a promoter can affect the expression value. That expression value
        can then change the metabolic models.
        TODO: Only for demonstration purpose of chained events.
        """
        # Use sum of deviations. Perfectly balanced counts will return an expression of 1.
        [a,c,g,t] = [ prom.count(base) for base in ["A","C","G","T"] ]
        avg = ( a + c + g + t ) / 4
        if avg == 0 : # On zero average, assume the sequence is undefined and always return 1.
            return 1.0
        dev = abs(a-avg) + abs(c-avg) + abs(g-avg) + abs(t-avg)
        expr = ( len(prom) * 2 - dev ) / ( len(prom) * 2 )
        return expr



    def calc_prom_str ( self, gene:Gene, ref_prom:str ) -> float :
        final_prom_str = float('NaN') # 0

        if gene in self.genome.genes :
            prom_str = Help_PromoterStrength(
                PromSequence=gene.prom,
                RefPromoter=ref_prom,
                Scaler=1,
                Similarity_Thresh=.4,
                Regressor_File=self.regressor_file,
                AddParams_File=self.addparams_file,
            )
            final_prom_str = round(prom_str * self.infl_prom_str, 2)

        return final_prom_str
