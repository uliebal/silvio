"""
GenomeList is a module that stores the multiple genes an Host may have.

TODO: Add support for adding the same gene twice (probably multisets)
"""

from __future__ import annotations

from ...host import Host
from ...module import Module
from ..records.gene.gene import Gene
from ..events import InsertGeneEvent, RemoveGeneEvent



class GenomeList ( Module ) :

    genes: set[Gene]


    def __init__ ( self, host:Host, genes: set[Gene] = set() ) :
        super().__init__(host)

        self.genes = genes

        self.host.observe( InsertGeneEvent, self.listen_insert_gene )
        self.host.observe( RemoveGeneEvent, self.listen_remove_gene )



    def listen_insert_gene ( self, event:InsertGeneEvent ) -> None :
        self.genes.add( event.gene )
        return ( "Added gene={} to the GenomeLibrary.".format(event.gene.name) )



    def listen_remove_gene ( self, event:RemoveGeneEvent ) -> None :
        self.genes.remove( event.gene )
        return ( "Removed gene={} from the GenomeLibrary.".format(event.gene.name) )
