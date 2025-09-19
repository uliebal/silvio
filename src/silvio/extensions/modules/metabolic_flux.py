"""
GenomeLibrary is a more complex version of GenomeList:
  - support for genome sequences
  - genes are located inside the sequence
"""


from __future__ import annotations
from typing import Callable, Optional, List, NamedTuple
from math import copysign
from copy import copy

# from numpy.random import Generator
from ...random import Generator
from cobra.core import Model as CobraModel
from cobra.core.solution import Solution as CobraSolution
from cobra.manipulation import delete_model_genes

from ...host import Host
from ...module import Module, ModuleException
from ...events import EventEmitter, EventLogger
from ..records.gene.stub_gene import StubGene
from ..all_events import InsertGeneEvent, RemoveGeneEvent, AlterGeneExpressionEvent
from ...extensions.utils.misc import Help_setDeactivateCExchanges, Help_getCarbonExchange, Help_CalcRate



class MetabolicFlux (Module) :
    """
    MetabolicFlux can handle cobrapy models and interfaces them with module events.

    This module can start with an non-existing model, which can then be integrated as an event.

    TODO: Its probably not elegant to allow non-existing models, but this makes it easier to add
    all genes in that model later on (as events). If we would be really pedantic and needed a
    strictly-existinging model we could implement an initialization step where a module calls
    multiple events on the host in order to "initialize" the module properly. To achieve that,
    modules themselves need to be able to generate events (right now only a Host can send events
    down to modules, events never go up the chain) and prevent infinite event loops. But maybe
    there are better alternatives altogether.
    """
    # Creator of random number generators. Host will bind this.
    make_generator: Callable[[],Generator]

    model: CobraModel

    # Since cobra models have limited methods to manage knocked-out genes, we need to keep track
    # of them in this list. A set of gene names.
    koed_genes: Set[str]


    def make ( self, model:CobraModel ) -> None :
        self.model = model
        self.model_tmp = self.model.copy()
        self.koed_genes = set()


    def copy ( self, ref:'MetabolicFlux' ) -> None :
        self.model = ref.model.copy()
        self.koed_genes = ref.koed_genes.copy()


    def bind ( self, host:Host ) -> None :
        self.make_generator = host.make_generator
        # host.observe( InsertGeneEvent, self.listen_insert_gene )
        # host.observe( RemoveGeneEvent, self.listen_remove_gene )
        # host.observe( AlterGeneExpressionEvent, self.listen_alter_gene_expression )


    def sync ( self, emit:EventEmitter, log:EventLogger ) -> None :
        log("MetabolicFlux: share model genes")
        for gene in self.model.genes : # Add all genes from the metabolic model.
            stub = StubGene( name=gene.name )
            emit( InsertGeneEvent(stub,locus=None) )

    def set_KinParameters ( self ) -> None :
        """
        Set kinetic parameters in the model. Each excahnge reaction is assigned with a random Vmax and Km.
        """

        CarbonExchanges = Help_getCarbonExchange(self.model)
        for ExRxn in CarbonExchanges:
            Vmax = round(self.make_generator().pick_uniform(low=2, high=20),1)
            Km = round(self.make_generator().pick_uniform(low=0.05, high=1),2)
            setattr(self.model.reactions.get_by_id(ExRxn), 'Vmax', Vmax)
            setattr(self.model.reactions.get_by_id(ExRxn), 'Km', Km)
            setattr(self.model_tmp.reactions.get_by_id(ExRxn), 'Vmax', Vmax)
            setattr(self.model_tmp.reactions.get_by_id(ExRxn), 'Km', Km)

    def reset( self ) -> None :
        self.model_tmp = self.model.copy()

    def optimize ( self ) -> CobraSolution :
        return self.model_tmp.optimize()
    
    def slim_optimize( self ) -> float :
        return round(self.model_tmp.slim_optimize(),2)
    
    def summary ( self ) -> str :
        return self.model_tmp.summary()

    def set_resetCarbonExchanges ( self , CarbonSubstrate:dict, Test:bool=False) -> None :
        """
        Set all carbon exchange reactions to zero uptake (lower_bound=0). Activate exchange reactions for input carbon sources.

        Inputs:
            CarbonSubstrate: dict, dictionary with carbon substrate exchange reactions as keys and their concentration in mmol/L as values
        Outputs:

        """
        if Test:
            print("MetabolicFlux: Test mode - not changing carbon exchange reactions")
            return
        # Deactivate all carbon exchange reactions 
        # using the original model to reset all changes
        self.model_tmp = Help_setDeactivateCExchanges(self.model_tmp) 
        # Activate exchange reactions for input carbon sources
        for ExRxn, Conc in CarbonSubstrate.items():
            Vmax = getattr(self.model_tmp.reactions.get_by_id(ExRxn), 'Vmax')
            Km = getattr(self.model_tmp.reactions.get_by_id(ExRxn), 'Km')
            UptakeRate = round(Help_CalcRate(Conc, Vmax, Km, Law='Michaelis-Menten'),1)
            setattr(self.model_tmp.reactions.get_by_id(ExRxn), 'lower_bound', -abs(UptakeRate))  # set uptake rate

  
    def listen_insert_gene ( self, event:InsertGeneEvent, emit:EventEmitter, log:EventLogger ) -> None :
        if event.gene.name in self.koed_genes :
            self.koed_genes.remove( event.gene.name )
            delete_model_genes( self.model, list(self.koed_genes), cumulative_deletions=False )
            log( "MetabolicFlux: reinserted gene={}".format(event.gene.name) )



    def listen_remove_gene ( self, event:RemoveGeneEvent, emit:EventEmitter, log:EventLogger ) -> None :
        """
        When a gene is removed we perform a knockout in the model.
        """
        found_genes = [ gene for gene in self.model.genes if gene.name == event.gene.name ]
        if len(found_genes) > 0 :
            self.koed_genes.add( event.gene.name )
            delete_model_genes( self.model, list(self.koed_genes), cumulative_deletions=False )
            log( "MetabolicFlux: knocked out gene={}".format(event.gene.name) )



    def listen_alter_gene_expression ( self, event:AlterGeneExpressionEvent, emit:EventEmitter, log:EventLogger ) -> None :
        """
        When a gene expression changes, we adapt the bounds.
        TODO: This method should be wrong. Please rewrite logic on how gene expression affects the
          metabolic model.
        """
        found_genes = [ gene for gene in self.model.genes if gene.name == event.gene.name ]
        if len(found_genes) > 0 :
            for gene in found_genes :
                for rct in gene.reactions :
                    rct.upper_bound = event.new_expression * copysign(1,rct.upper_bound)
                    log( "MetabolicFlux: changed upper_bound on reaction={} to {}".format(rct.id,rct.upper_bound) )
