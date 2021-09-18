"""
All operations that generate random numbers may come from this globally defined random generator.

This allows the user to provide a seed and generate the same sequences on every run as long as
requests are made in the same order.

The methods it exposes are the same of those from a numpy Generator object, but with typing and
slightly altered parameters.
https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.Generator
"""

import sys
from typing import Any, List, Optional

from numpy.random import Generator as NumpyGenerator, PCG64



# Initialize the global generator with a random seed first.
_randgen = NumpyGenerator(PCG64())



def set_seed ( new_seed:int ) -> None :
    """
    Sets the seed of the global random generator so that all future random picks follow the same
    pseudo-random sequence.
    """
    global _randgen
    _randgen = NumpyGenerator(PCG64(new_seed))



def pick_choice ( choices:List[Any], weights:List[float] = None ) -> Any :
    """
    Pick a single value out of a provided list. Can receive a non-normalized weights.
    """
    probs = None
    if weights is not None :
        sum_w = sum(list(weights))
        probs = [ w / sum_w for w in list(weights) ]
    return _randgen.choice( choices, p=probs )


def pick_sample ( choices:List[Any], amount:int ) -> List[Any] :
    """
    Pick a sample from choices.
    """
    return _randgen.choice( choices, size=amount, replace=False )


def pick_integer ( low:int, high:int ) -> int :
    """
    Pick a single integer from a range from low (inclusive) to high (exclusive).
    """
    return _randgen.integers( low, high, endpoint=False )


def pick_uniform ( low:float = 0, high:float = 1 ) -> float :
    """
    Pick a single float from an uniform distribution between low (inclusive) and high (exclusive).
    """
    return _randgen.uniform( low, high )


def pick_normal ( mean:float = 0, sd:float = 0 ) -> float :
    """
    Pick a single number from a normal distribution.
    """
    return _randgen.normal( mean, sd )


def pick_exponential ( beta:float ) -> float :
    """
    Pick a single number from an exponential distribution.
    """
    return _randgen.exponential( beta )


def pick_seed () -> int :
    """
    Pick a new number that is usable as a seed.
    """
    return _randgen.integers(0)



class Generator () :
    """ All random methods also exist inside a local generator object. """

    gen: NumpyGenerator

    def __init__ ( self, seed:Optional[int] = None ) :
        self.gen = NumpyGenerator(PCG64(seed))

    def pick_choice ( self, choices:List[Any], weights:List[float] = None ) -> Any :
        probs = None
        if weights is not None :
            sum_w = sum(list(weights))
            probs = [ w / sum_w for w in list(weights) ]
        return self.gen.choice( choices, p=probs )

    def pick_sample ( self, choices:List[Any], amount:int ) -> List[Any] :
        return self.gen.choice( choices, size=amount, replace=False )

    def pick_integer ( self, low:int, high:int ) -> int :
        return self.gen.integers( low, high, endpoint=False )

    def pick_uniform ( self, low:float = 0, high:float = 1 ) -> float :
        return self.gen.uniform( low, high )

    def pick_normal ( self, mean:float = 0, sd:float = 0 ) -> float :
        return self.gen.normal( mean, sd )

    def pick_exponential ( self, beta:float ) -> float :
        return self.gen.exponential( beta )

    def pick_seed ( self ) -> int :
        return int(self.gen.integers(sys.maxsize))
