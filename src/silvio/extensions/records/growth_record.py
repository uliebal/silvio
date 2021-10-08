"""
The Growth record stores a possible growth outcome.

TODO: Currently not using as GrowthBehaviour.grow is outputting a DataFrame.
  Think about it, but I believe using Records (and maybe Registries) could be better.
  For example, a GrowthRegistry could be a specialized table that can be queried for
  aggregated records or plots.
"""

from __future__ import annotations
from abc import ABC


from ...record import Record



class GrowthOutcome ( Record, ABC ) :

    def __init__ ( self ) :
        super().__init__()

    def clone ( self ) -> GrowthOutcome :
        return GrowthOutcome()
