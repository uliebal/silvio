"""
Registries provide an efficient way to store Records that are used by multiple Hosts.
That way, each host only needs to store a reference to each registered object (for example
a gene) instead to storing and copying the entire gene.
When editing an object in the registry, the editor should make sure to practice copy-on-write.

Entries of a registry must be copy-able.
"""

from typing import Set

from .record import Record



class Registry :
    """ A Registry holds multiple Entries. """

    records : Set[Record]

    def __init__ ( self ) :
        self.records = set()


    def insert ( self, rec: Record ) -> None :
        """ Insert a new record into the Registry. """
        self.records.add(rec)


    def clone ( self, rec: Record ) -> Record :
        """ Clone a Record inside the Registry and return it. """
        if rec in self.records :
            cloned_rec = rec.clone()
            self.insert(cloned_rec)
            return cloned_rec
        raise Exception("Tried to clone a non-registered record.")

