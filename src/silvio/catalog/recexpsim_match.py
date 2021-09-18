"""
TODO: This catalog is currently outdated. When `recexpsim_original` is deemed ready, then copy-paste
  it and apply the changes this file currently has on `clone_with_recombination`.

An alternative version of the RecExpSim Catalog with some differences:
  - The host has a full sequence (of genes and background)
  - Gene insertion is done via cutting sites and primer matching
"""

from __future__ import annotations
from typing import Optional, List, Tuple
import os

import pandas as pd
from Bio.Seq import Seq

from ..random import pick_integer
from ..utils import Help_Progressbar
from ..experiment import Experiment
from ..registry import Registry
from ..host import Host, HostException

from ..config import DATADIR
from ..extensions.modules.growth_behaviour import GrowthBehaviour
from ..extensions.modules.genome_library import GenomeLibrary
from ..extensions.modules.genome_expression import GenomeExpression
from ..extensions.records.gene.gene import Gene
from ..extensions.utils import check_primer_integrity



class RecExperiment (Experiment) :

    def __init__ ( self ) :
        super().__init__()



class RecHost (Host) :

    growth: GrowthBehaviour

    genome: GenomeLibrary

    genexpr: GenomeExpression



    def __init__ (
        self, exp:Experiment,
        genome:Optional[GenomeLibrary] = None,
            bg_size:Optional[int] = 80,
            bg_gc_content:Optional[float] = 0.4,
        growth:Optional[GrowthBehaviour] = None,
            opt_growth_temp:Optional[int] = None,
            max_biomass:Optional[int] = None,
        genexpr:Optional[GenomeExpression] = None,
            infl_prom_str = None,
            opt_primer_len = None,
            regressor_file = None,
            addparams_file = None
    ) :
        super().__init__(exp=exp)

        # Initialize GenomeLibrary module.
        if genome is not None :
            self.genome = genome
        elif all( arg is not None for arg in [ bg_size, bg_gc_content ] ) :
            self.genome = GenomeLibrary( host=self, bg_size=bg_size, bg_gc_content=bg_gc_content )
        else :
            raise HostException("GenomeLibrary module cannot be initialized for this Host")

        # Initialize GrowthBehaviour module.
        if growth is not None :
            self.growth = growth
        elif all( arg is not None for arg in [ opt_growth_temp, max_biomass ] ) :
            self.growth = GrowthBehaviour( host=self,
                opt_growth_temp=opt_growth_temp, max_biomass=max_biomass
            )
        else :
            raise HostException("GrowthBehaviour module cannot be initialized for this Host")

        # Initialize GenomeExpression module.
        if genexpr is not None :
            self.genexpr = genexpr
        elif all( arg is not None for arg in [ bg_size, bg_gc_content ] ) :
            self.genexpr = GenomeExpression( host=self, genlib=self.genome,
                opt_primer_len=opt_primer_len, infl_prom_str=infl_prom_str,
                regressor_file=regressor_file, addparams_file=addparams_file
            )
        else :
            raise HostException("GenomeExpression module cannot be initialized for this Host")


    def clone ( self ) -> RecHost :
        """
        TODO: Not sure how I like this cloning. I need to get into deep properties like
            `self.growth.opt_growth_temp` to re-create the modules. I want to call `module.clone`.
        """
        return RecHost(
            exp=self.exp,
            genome=self.genome.clone(self),
            growth=self.growth.clone(self),
            genexpr=self.genexpr.clone(self),
        )



    def print_status ( self, width:int=1000 ) -> None :
        print("Host Information:")

        print("  opt_growth_temp = {}".format( self.growth.opt_growth_temp ))
        print("  max_biomass = {}".format( self.growth.max_biomass ))
        print("  opt_primer_len = {}".format( self.genexpr.opt_primer_len ))

        print("  Gene Library: {} genes".format(len(self.genome.locgenes)))
        for lg in self.genome.locgenes :
            print("  - {} = @{} : {} * {}".format(lg.name, lg.start_loc, lg.prom, lg.orf))

        # The gene sequence will printed as a paginated mix of bases and gene annotations.
        print("  Gene Sequence: size {}".format(len(self.genome.sequence)))
        genlen = len(self.genome.sequence)
        print("-" * min(width,genlen))
        for pos in range(0,genlen,width) :
            linew = min(width,genlen-pos) # Actual width of this line.
            # The header holds the base counts.
            div = linew // 10
            rem = linew - (div * 10)
            forms = [ "{:<10}" for _ in range(div-1) ]
            forms.append( "{:>" + str(10+rem) + "}" )
            nums = [ pos+i*10 for i in range(div-1) ]
            nums.append( pos+div*10+rem )
            # The sequence is output as is.
            seq = self.genome.sequence[ pos:min(pos+width,genlen) ]
            # The gene annotations are below if there are genes.
            annots = []
            for locgene in self.genome.locgenes :
                if pos <= locgene.start_loc < min(pos+width,genlen) :
                    annot = "{}[{}| {}]".format(
                        " " * (locgene.start_loc - pos), # left padding
                        " " * (locgene.prom_len - 1), # promoter space
                        locgene.name + " " * (locgene.orf_len - 3 - len(locgene.name)) # orf + name
                    )
                    annots.append(annot)
            # Print out all necessary lines
            print("".join(forms).format(*nums))
            print(seq)
            for annot in annots :
                print(annot)
            print("-" * linew)







    def sim_growth ( self, temps:List[int] ) -> pd.DataFrame :
        """ Simulate a growth under multiple temperatures and return expected biomasses over time. """
        ( df, pauses ) = self.growth.Make_TempGrowthExp(CultTemps=temps)

        wait = 0.001 # has to be adjusted, waiting time for loading bar
        for pause in pauses :
            loading_time = wait * pause.loading_len
            Help_Progressbar(45, loading_time, pause.exp)

        return df



    def clone_with_recombination ( self, primer:Seq, gene:Gene, tm:int ) -> Tuple[Host,str] :
        """
        Clone the host while trying to insert a gene. The Clone might or might not contain
        the added Gene.
        Returns: ( cloned_host, outcome_message )
        """

        # A clone is always made.
        cloned_host = self.clone()

        ref_prom = 'GCCCATTGACAAGGCTCTCGCGGCCAGGTATAATTGCACG'
        primer_integrity = check_primer_integrity(gene.prom, primer, tm, ref_prom, self.genexpr.opt_primer_len)
        if primer_integrity.success == False :
            return ( cloned_host, "Primer Failed: " + primer_integrity.error )

        matches = cloned_host.genome.calc_primer_matches(primer)
        if len(matches) == 0 :
            return ( cloned_host, "No insertion sites matched." )

        # Both primer integrity and recombination succeeded. Insert the new gene into all
        # insertion sites with good enough matching.
        for match in matches :
            if match.success > 0.5 :
                cloned_host.genome.insert_gene(gene, match.loc_end)

        return ( cloned_host, "Cloning with recombination succeeded." )



    def calc_promoter_strength ( self, gene:Gene ) -> float :
        ref_prom = 'GCCCATTGACAAGGCTCTCGCGGCCAGGTATAATTGCACG'
        return self.genexpr.calc_prom_str( gene, ref_prom )





class Ecol (RecHost) :
    def __init__ ( self, exp ) :
        opt_growth_temp = pick_integer(25,40)  # unit: degree celsius, source: https://application.wiley-vch.de/books/sample/3527335153_c01.pdf
        infl_prom_str = pick_integer(30,50) # explanation see Plot_ExpressionRate
        opt_primer_len = pick_integer(16,28) # unit: nt, source: https://link.springer.com/article/10.1007/s10529-013-1249-8
        max_biomass = pick_integer(30,100) # unit: in gDCW/l, source (german): https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=2ahUKEwjzt_aJ9pzpAhWGiqQKHb1jC6MQFjABegQIAhAB&url=https%3A%2F%2Fwww.repo.uni-hannover.de%2Fbitstream%2Fhandle%2F123456789%2F3512%2FDissertation.pdf%3Fsequence%3D1&usg=AOvVaw2XfGH11P9gK2F2B63mY4IM
        regressor_file = DATADIR / 'expression_predictor' / 'Ecol-Promoter-predictor.pkl'
        addparams_file = DATADIR / 'expression_predictor' / 'Ecol-Promoter-AddParams.pkl'
        super().__init__(
            exp=exp,
            opt_growth_temp=opt_growth_temp,
            infl_prom_str=infl_prom_str,
            opt_primer_len=5, # NOTE: Need to reduce to hit them. #opt_primer_len,
            max_biomass=max_biomass,
            regressor_file=regressor_file,
            addparams_file=addparams_file,
        )



class Pput (RecHost) :
    def __init__ ( self, exp ) :
        opt_growth_temp = pick_integer(25,40)  # unit: degree celsius, source: https://application.wiley-vch.de/books/sample/3527335153_c01.pdf
        infl_prom_str = pick_integer(30,50) # explanation see Plot_ExpressionRate
        opt_primer_len = pick_integer(16,28) # unit: nt, source: https://link.springer.com/article/10.1007/s10529-013-1249-8
        max_biomass = pick_integer(45,145) # unit: in gDCW/l, source 1: https://onlinelibrary.wiley.com/doi/pdf/10.1002/bit.25474, source 2: https://link.springer.com/article/10.1385/ABAB:119:1:51
        regressor_file = DATADIR / 'expression_predictor' / 'Ptai-Promoter-predictor.pkl'
        addparams_file = DATADIR / 'expression_predictor' / 'Ptai-Promoter-AddParams.pkl'
        super().__init__(
            exp=exp,
            opt_growth_temp=opt_growth_temp,
            infl_prom_str=infl_prom_str,
            opt_primer_len=opt_primer_len,
            max_biomass=max_biomass,
            regressor_file=regressor_file,
            addparams_file=addparams_file,
        )
