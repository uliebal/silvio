"""
A Module is an extension of behaviour that can be attached to an Host.
"""

from __future__ import annotations
from abc import ABC
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..base import Host



class Module (ABC) :

    host: 'Host'



    def __init__ ( self, host:'Host', ref:Optional[Module] = None ) :
        """
        Init is resposible to build a new Module, whether it is created from scratch or created
        by using another equally-typed Module as a reference. After building, the Module should
        also register itself on the Host.

        Arguments are usually in the order of:
            host : The host this module belongs to.
            deps* : Zero or more dependent modules this module is using.
            params* : Zero or more params to build this module.

        Extending Modules have the following structure on their __init__ :

            super().__init__(host,ref)         # Call parent init.
            self.depmodule = depmodule         # Add dependent modules.

            self.param1 = param1               # Set the params from the arguments.

            self.host.bind( 'event_type', self.listen_event_type )   # Register listeners.

        """
        self.host = host



class ModuleException (Exception) :
    """ Exception that is triggered when a module cannot be created. """
    pass
