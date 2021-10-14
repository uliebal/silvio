"""
GenomeList is a module that stores the multiple genes an Host may have.

TODO: Add support for adding the same gene twice (probably multisets)
"""

from __future__ import annotations
from copy import copy

from ...host import Host
from ...module import Module
from ...events import EventEmitter, EventLogger
from ..records.gene.gene import Gene
from ..all_events import InsertGeneEvent, RemoveGeneEvent, AlterGenePromoterEvent



class GenomeList ( Module ) :

    genes: set[Gene]



    def make ( self, genes: set[Gene] = set() ) -> None :
        self.genes = genes



    def copy ( self, ref:'GenomeList' ) -> None :
        self.genes = ref.genes.copy()



    def bind ( self, host:Host ) -> None :
        host.observe( InsertGeneEvent, self.listen_insert_gene )
        host.observe( RemoveGeneEvent, self.listen_remove_gene )
        host.observe( AlterGenePromoterEvent, self.listen_alter_gene_promoter )



    def sync ( self, emit:EventEmitter, log:EventLogger ) -> None :
        pass # Nothing to sync.



    def listen_insert_gene ( self, event:InsertGeneEvent, emit:EventEmitter, log:EventLogger ) -> None :
        self.genes.add( event.gene )
        log( "GenomeLibrary: added gene={}".format(event.gene.name) )



    def listen_remove_gene ( self, event:RemoveGeneEvent, emit:EventEmitter, log:EventLogger ) -> None :
        self.genes.remove( event.gene )
        log( "GenomeLibrary: removed gene={}".format(event.gene.name) )



    def listen_alter_gene_promoter ( self, event:AlterGenePromoterEvent, emit:EventEmitter, log:EventLogger ) -> None :
        """ When a gene promoter is altered, we recalculate the expression and update it. """
        found_gene = first( self.locgenes, lambda lg : lg == event.gene )
        if found_gene is not None :
            self.alter_sequence( found_gene.start_loc, found_gene.prom_len, event.new_promoter )
            log( "GenomeLibrary: Changed promoter of gene={} in sequence.".format(event.gene.name) )



    def get_genes_by_name ( self, name:str ) -> List[Gene] :
        found_genes = [ gene for gene in self.locgenes if name == gene.name ]
        return found_genes
