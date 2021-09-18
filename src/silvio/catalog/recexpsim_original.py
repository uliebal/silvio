"""
An experiment that uses hosts which implements all currently existing modules.
"""

from __future__ import annotations
from typing import Optional, List, Tuple, Literal
from copy import copy

import numpy as np
import matplotlib.pyplot as plt
from Bio.Seq import Seq
import pandas as pd

from ..config import DATADIR
from ..experiment import Experiment, ExperimentException
from ..host import Host, HostException
from ..utils import alldef, coalesce, first
from ..random import Generator
from ..outcome import DataOutcome, DataWithPlotOutcome, combine_data
from ..extensions.events import InsertGeneEvent
from ..extensions.modules.growth_behaviour import GrowthBehaviour
from ..extensions.modules.genome_list import GenomeList
from ..extensions.modules.genome_expression import GenomeExpression
from ..extensions.records.gene.gene import Gene
from ..extensions.utils.misc import check_primer_integrity_and_recombination
from ..extensions.utils.visual import Help_Progressbar
from ..extensions.utils.laboratory import ErrorRate



class RecExperiment (Experiment) :
    """
    Recombinant Expression Experiment with culture growth and genetic recombination.
    Has a budget system to limit bruteforcing.
    """

    budget: int

    suc_rate: float

    # Keep track of how many hosts were created
    host_counter: int



    def __init__ ( self, seed:Optional[int] = None, equipment_investment:int = 0, max_budget:int = 10000 ) :

        if equipment_investment > max_budget :
            raise Exception("Investment cost is higher than maximal budget.")

        super().__init__(seed=seed)
        self.budget = max_budget - equipment_investment
        self.suc_rate = ErrorRate(equipment_investment, max_budget)
        self.host_counter = 0



    def create_host ( self, species:Literal['ecol','pput'], name:Optional[str] ) -> RecHost:
        self.host_counter += 1
        seed = self.rnd_gen.pick_seed() # The experiment provides stable seed generation for hosts.
        new_host = None

        chosen_name = coalesce( name, species + str(self.host_counter) )
        if species == 'ecol' :
            new_host = Ecol( name=chosen_name, seed=seed )
        elif species == 'pput' :
            new_host = Pput( name=chosen_name, seed=seed )

        if species is None :
            raise Exception("Invalid species provided. The possible species are: 'ecol', 'pput'")

        self.bind_host(new_host)
        return new_host



    def find_host_or_abort ( self, host_name:str ) -> RecHost :
        """
        Find a host by a given name or abort with an exception.
        Throws: ExperimentException
        """
        host:Optional[RecHost] = first( self.hosts, lambda h: h.name == host_name )
            # Find the first host matching the name.
        if host is None :
            raise ExperimentException("Experiment has not found the host '{}'".format(host_name))
        return host



    def spend_budget_or_abort ( self, amount:int ) -> None :
        """
        Each operation may use up a certain amount of the budget.
        When the budget is not sufficient those operations will all fail.
        By using error handling via raised exceptions we don't need to check each method for a
        success flag on their return value.
        Throws: ExperimentException
        """
        if amount > self.budget :
            raise ExperimentException("Experiment has surpassed its budget. No more operations are allowed.")
        self.budget -= amount



    def simulate_growth ( self, host_name:str, temps:List[int] ) -> GrowthOutcome :
        """ Simulate a growth under multiple temperatures and return expected biomasses over time. """
        self.spend_budget_or_abort(100)
        host = self.find_host_or_abort(host_name)
        ( df, pauses ) = host.growth.Make_TempGrowthExp(CultTemps=temps, exp_suc_rate=self.suc_rate)

        wait = 0.001 # has to be adjusted, waiting time for loading bar
        for pause in pauses :
            loading_time = wait * pause.loading_len
            Help_Progressbar(45, loading_time, pause.exp)

        return GrowthOutcome(value=df)



    def clone_with_recombination ( self, host_name:str, primer:Seq, gene:Gene, tm:int ) -> Tuple[RecHost,str] :
        """
        Clone the host while trying to insert a gene. The Clone might or might not contain
        the added Gene.
        Returns: ( new_host, outcome_message )
        """
        self.spend_budget_or_abort(200)
        host = self.find_host_or_abort(host_name)
        rnd = host.make_generator()

        # A clone is always made.
        # TODO: Maybe there is a failure to clone and the return should be `Optional[RecHost]`
        new_host = RecHost( ref=host ) # Clone the host but with different seed.
        self.bind_host(new_host)
        if rnd.pick_uniform(0,1) < self.suc_rate: # experiment failure depending on investment to equipment
            return( new_host, 'Cloning failed: Bad Equipment' )

        primer_integrity = check_primer_integrity_and_recombination(
            gene.prom, primer, tm, RecHost.ref_prom, host.genexpr.opt_primer_len
        )
        if not primer_integrity.value :
            return ( new_host, "Primer Failed: " + primer_integrity.error )

        # Both primer integrity and recombination succeeded. Insert the new gene into the clone.
        new_host.insert_gene(gene)
        return ( new_host, "Cloning with recombination succeeded." )



    def measure_promoter_strength ( self, host_name:str, gene_name:str ) -> DataOutcome :
        """
        Measure the promoter strength of gene `gene_name` inside host `host_name`.
        """
        self.spend_budget_or_abort(100)
        host = self.find_host_or_abort(host_name)

        # Calculate the cumulative promoter strength of all genes with the same name.
        matching_genes = [g for g in host.genome.genes if g.name == gene_name]
        prom_str = 0
        for mg in matching_genes :
            prom_str += host.genexpr.calc_prom_str( gene=mg, ref_prom=RecHost.ref_prom )

        return DataOutcome( value=pd.Series({
            'Host': host.name,
            'GeneName': gene_name,
            'GenePromoter': str(matching_genes[0].prom if len(matching_genes) > 0 else None),
                # TODO: Only the promoter from the first matching gene is output. This might be
                #   slightly incorrect if multiple genes with same name have different promoter seqs.
            'PromoterStrength': prom_str
        }) )



    def simulate_vaccine_production (
        self, host_names:List[str], gene_name:Gene, cult_temp:int, growth_rate:float, biomass:int
    ) -> DataOutcome :
        """
        Simulate the protein production of multiple host cultures.
        """
        self.spend_budget_or_abort(500)

        # Equipment failure can prematurely end the simulation.
        if self.rnd_gen.pick_uniform(0,1) < self.suc_rate :
            return DataOutcome( None, 'Experiment failed, bad equipment.' )

        all_outcomes:List[DataOutcome] = []
        for host_name in host_names :

            # Skip this host if it does not exist.
            host = first( self.hosts, lambda h: h.name == host_name )
            if host is None :
                outcome = DataOutcome( pd.Series({ 'Host': host_name }), 'Host does not exist.' )
                all_outcomes.append(outcome)
                continue

            # Skip this host if no matching gene is found.
            gene = first( host.genome.genes, lambda g: g.name == gene_name )
            if gene is None :
                outcome = DataOutcome( pd.Series({ 'Host': host.name, 'Gene_Name': gene_name }), 'Gene does not exist.' )
                all_outcomes.append(outcome)
                continue

            # Simulate the production for this host.
            production = host.growth.Make_ProductionExperiment(
                gene=gene,
                CultTemp=cult_temp,
                GrowthRate=growth_rate,
                Biomass=biomass,
                ref_prom=RecHost.ref_prom,
                accuracy_Test=.9
            )
            outcome = DataOutcome(
                value=pd.Series({
                    'Host': host.name,
                    'Gene_Name': gene.name,
                    'Promoter_Sequence': str(gene.prom),
                    'Promoter_GC-content': (gene.prom.count('C') + gene.prom.count('G')) / len(gene.prom),
                    'Expression_Temperature': cult_temp,
                    'Expression_Biomass': biomass,
                    'Expression_Rate': production.value,
                }),
                error=production.error
            )
            all_outcomes.append(outcome)

        return combine_data(all_outcomes)



    def print_status ( self ) -> None :
        print("Experiment:")
        print("  budget = {}".format( self.budget ))
        print("  failure rate = {}".format( self.suc_rate ))
        print("  hosts = [ {} ]".format( " , ".join([h.name for h in self.hosts]) ))
        # Could display the status of each host if wanted.



class RecHost (Host) :

    growth: GrowthBehaviour

    genome: GenomeList

    genexpr: GenomeExpression

    ref_prom:str = 'GCCCATTGACAAGGCTCTCGCGGCCAGGTATAATTGCACG'



    def __init__ (
        self, ref:Optional[RecHost] = None, name:Optional[str] = None, seed:Optional[int] = None,
        opt_growth_temp = None,
        max_biomass = None,
        infl_prom_str = None,
        species_prom_str = None,
        opt_primer_len = None,
        regressor_file = None,
        addparams_file = None
    ) :
        super().__init__(ref=ref, name=name, seed=seed)

        # Init Clone
        if ref is not None :
            self.genome = GenomeList( host=self, genes=copy(ref.genome.genes) )
            self.genexpr = GenomeExpression(
                host=self,
                genome=self.genome,
                opt_primer_len=ref.genexpr.opt_primer_len,
                infl_prom_str=ref.genexpr.infl_prom_str,
                species_prom_str=ref.genexpr.species_prom_str,
                regressor_file=ref.genexpr.regressor_file,
                addparams_file=ref.genexpr.addparams_file
            )
            self.growth = GrowthBehaviour(
                host=self,
                genexpr=self.genexpr,
                opt_growth_temp=ref.growth.opt_growth_temp,
                max_biomass=ref.growth.max_biomass
            )

        # Init New
        elif alldef( opt_growth_temp, max_biomass, infl_prom_str, species_prom_str, opt_primer_len, regressor_file, addparams_file ) :
            self.genome = GenomeList( host=self )
            self.genexpr = GenomeExpression(
                host=self,
                genome=self.genome,
                opt_primer_len=opt_primer_len,
                infl_prom_str=infl_prom_str,
                species_prom_str=species_prom_str,
                regressor_file=regressor_file,
                addparams_file=addparams_file
            )
            self.growth = GrowthBehaviour(
                host=self,
                genexpr=self.genexpr,
                opt_growth_temp=opt_growth_temp,
                max_biomass=max_biomass
            )

        # Failed Init
        else :
            raise HostException("Host not initialized. Reason: incomplete arguments.")



    def find_gene_or_abort ( self, gene_name:str ) -> Gene :
        """
        Find a gene by a given name or abort with an exception.
        Throws: HostException
        """
        gene:Optional[Gene] = first( self.genome.genes, lambda g: g.name == gene_name )
            # Find the first gene matching the name.
        if gene is None :
            raise HostException("Host has not found the gene '{}'".format(gene_name))
        return gene



    def insert_gene ( self, gene:Gene ) -> None :
        self.emit( InsertGeneEvent( gene=gene, locus=0 ) )



    def print_status ( self ) -> None :
        print("Host [{}]:".format( self.name ))
        print("  seed plus counter = {} + {}".format( self.rnd_seed, self.rnd_counter ))
        print("  optimal growth temperature = {}".format( self.growth.opt_growth_temp ))
        print("  max biomass = {}".format( self.growth.max_biomass ))
        print("  optimal primer length = {}".format( self.genexpr.opt_primer_len ))
        print("  Gene List: {} genes".format(len(self.genome.genes)))
        for gene in self.genome.genes :
            print("  - {} = {} * {}".format(gene.name, gene.prom, gene.orf))
        print("  Event History: {} events".format(len(self.event_log)))
        for el in self.event_log :
            print("  - {}".format(el))




class Ecol (RecHost) :
    def __init__ ( self, name, seed ) :
        gen = Generator( seed )
        opt_growth_temp = gen.pick_integer(25,40)  # unit: degree celsius, source: https://application.wiley-vch.de/books/sample/3527335153_c01.pdf
        infl_prom_str = gen.pick_integer(30,50) # explanation see Plot_ExpressionRate
        opt_primer_len = gen.pick_integer(16,28) # unit: nt, source: https://link.springer.com/article/10.1007/s10529-013-1249-8
        max_biomass = gen.pick_integer(30,100) # unit: in gDCW/l, source (german): https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=2ahUKEwjzt_aJ9pzpAhWGiqQKHb1jC6MQFjABegQIAhAB&url=https%3A%2F%2Fwww.repo.uni-hannover.de%2Fbitstream%2Fhandle%2F123456789%2F3512%2FDissertation.pdf%3Fsequence%3D1&usg=AOvVaw2XfGH11P9gK2F2B63mY4IM
        regressor_file = DATADIR / 'expression_predictor' / 'Ecol-Promoter-predictor.pkl'
        addparams_file = DATADIR / 'expression_predictor' / 'Ecol-Promoter-AddParams.pkl'
        super().__init__(
            name=name,
            seed=seed,
            opt_growth_temp=opt_growth_temp,
            infl_prom_str=infl_prom_str,
            opt_primer_len=opt_primer_len,
            max_biomass=max_biomass,
            regressor_file=regressor_file,
            addparams_file=addparams_file,
            species_prom_str=0.057,
        )



class Pput (RecHost) :
    def __init__ ( self, name, seed ) :
        gen = Generator( seed )
        opt_growth_temp = gen.pick_integer(25,40)  # unit: degree celsius, source: https://application.wiley-vch.de/books/sample/3527335153_c01.pdf
        infl_prom_str = gen.pick_integer(30,50) # explanation see Plot_ExpressionRate
        opt_primer_len = gen.pick_integer(16,28) # unit: nt, source: https://link.springer.com/article/10.1007/s10529-013-1249-8
        max_biomass = gen.pick_integer(45,145) # unit: in gDCW/l, source 1: https://onlinelibrary.wiley.com/doi/pdf/10.1002/bit.25474, source 2: https://link.springer.com/article/10.1385/ABAB:119:1:51
        regressor_file = DATADIR / 'expression_predictor' / 'Ptai-Promoter-predictor.pkl'
        addparams_file = DATADIR / 'expression_predictor' / 'Ptai-Promoter-AddParams.pkl'
        super().__init__(
            name=name,
            seed=seed,
            opt_growth_temp=opt_growth_temp,
            infl_prom_str=infl_prom_str,
            opt_primer_len=opt_primer_len,
            max_biomass=max_biomass,
            regressor_file=regressor_file,
            addparams_file=addparams_file,
            species_prom_str=0.04
        )



class GrowthOutcome ( DataWithPlotOutcome ) :

    def make_plot ( self ) -> plt.Figure :
        """
        Plotting with pyplot is unfortunately unintuitive. You cannot display a single figure by
        using the object-oriented API. When you do `plt.subplots` (or create a plot by any other
        means) it will be stored in the global context. You can only display things from the glbbal
        context, and displaying it will remove it from there.
        """
        Time, Biomass = self.value.iloc[:,0], self.value.iloc[:,1:]
        LnBiomass = np.log(Biomass)

        fig, ax = plt.subplots()
        for Exp,X in LnBiomass.iteritems() :
            ax.scatter(Time, X, label=Exp)
        ax.legend()
        return fig
