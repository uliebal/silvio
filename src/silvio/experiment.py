"""
Experiment is the top-most scope. It includes all hosts and registries and acts as a global
namespace.
"""

from typing import Set, Optional, Union
from abc import ABC

from numpy.random import SeedSequence

from .host import Host
from .random import Generator



class ExperimentException (Exception) :
    """ Exception that is triggered by an experiment. """
    pass



class Experiment (ABC) :

    hosts: Set[Host]

    # The experiment will hold an internal random generator to allow repeatability.
    rnd_gen: Generator



    def __init__ ( self, seed:Optional[Union[int,SeedSequence]] = None ) :
        self.hosts = set()
        self.rnd_gen = Generator(seed)



    def bind_host ( self, host:Host ) -> None :
        self.hosts.add(host)



    # def create_host ( self, host_class:Type[Host], **kwargs ) -> None :
    #     host = host_class( exp=self, **kwargs )
    #     self.hosts.append( host )
