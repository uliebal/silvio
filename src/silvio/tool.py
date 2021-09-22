"""
Tools are long-standing objects that have to be initialized and that perform calculations on
other low-level values, such as sequences, matches, etc.

This class provides some common facilities such as stable randomization.
"""

from typing import List, Callable, NamedTuple, Optional, Type

from .events import Event
from .random import Generator, pick_seed
from .utils import coalesce



class ToolException (Exception) :
    """ Exception that is triggered when a Tool has a problem. """
    pass



class Tool :

    name: str

    # By specifying a seed, the same code will produce the same results.
    rnd_seed: Optional[int]

    # A seed with an incremental counter provides a stable randomization were experiments can be
    # run multiple times with different results.
    rnd_counter: int



    def __init__ ( self, name:str = None, seed:Optional[int] = None ) :
        """
        Init is responsible to initialize the stable randomization.

        Code from derived classes should be:

            super().__init__( name, seed )
        """
        self.name = coalesce( name, 'unnamed' )
        self.rnd_seed = coalesce( seed, pick_seed() ) # by default, use a random seed
        self.rnd_counter = 0



    def make_generator ( self ) -> Generator :
        """ Construct a random number generator with the same seed stored in the host. """
        self.rnd_counter += 1
        return Generator( self.rnd_seed + self.rnd_counter )
