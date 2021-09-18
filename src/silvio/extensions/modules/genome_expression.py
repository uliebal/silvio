"""
TODO: GenomeExpression needs to be reviewed.
"""

from typing import Union

from ...host import Host
from ...module import Module
from ..records.gene.gene import Gene
from ..utils.misc import Help_PromoterStrength
from .genome_library import GenomeLibrary
from .genome_list import GenomeList



class GenomeExpression ( Module ) :

    # Dependent module Genome Library holding the genes to express.
    genome: Union[GenomeLibrary,GenomeList]

    # Optimal primer length.
    opt_primer_len: int

    # Factor which influences the range of the promoter strength.
    # TODO: These characteristics maybe go elsewhere.
    infl_prom_str : float
    species_prom_str : float

    # Path to parameter files. TODO: use Paths.
    regressor_file:str
    addparams_file:str



    def __init__ (
        self, host:Host, genome:Union[GenomeLibrary,GenomeList],
        opt_primer_len:int,
        infl_prom_str:float,
        species_prom_str:float,
        regressor_file:str,
        addparams_file:str,
    ) :
        super().__init__(host)

        self.genome = genome

        self.opt_primer_len = opt_primer_len
        self.infl_prom_str = infl_prom_str
        self.species_prom_str = species_prom_str
        self.regressor_file = regressor_file
        self.addparams_file = addparams_file

        # self.host.observe( InsertGeneEvent, self.listen_insert_gene )
        # self.host.observe( RemoveGeneEvent, self.listen_remove_gene )



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
