"""
This is a TESTING module to showcase Host events.
The number inside this module (example: a size of a cell) will increase with the number of genes.
"""

from ...host import Host
from ...module import Module
from ...events import EventEmitter, EventLogger
from ..all_events import InsertGeneEvent, RemoveGeneEvent



class PhenotypeSize (Module) :

    size: int



    def make ( self, size:int = 0 ) -> None :
        self.size = 0



    def copy ( self, ref:'PhenotypeSize' ) -> None :
        self.size = ref.size



    def bind ( self, host:Host ) -> None :
        host.observe( InsertGeneEvent, self.listen_insert_gene )
        host.observe( RemoveGeneEvent, self.listen_remove_gene )



    def sync ( self, emit:EventEmitter, log:EventLogger ) -> None :
        pass # Nothing to sync.



    def listen_insert_gene ( self, event:InsertGeneEvent, emit:EventEmitter, log:EventLogger ) -> None :
        self.size += 1
        log("PhenotypeSize: incremented size by 1")



    def listen_remove_gene ( self, event:RemoveGeneEvent, emit:EventEmitter, log:EventLogger ) -> None :
        self.size -= 1
        log("PhenotypeSize: decremented size by 1")
