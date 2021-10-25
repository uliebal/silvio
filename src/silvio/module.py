"""
A Module is an extension of behaviour that can be attached to an Host.
"""

from __future__ import annotations
from abc import ABC
from typing import Optional, TYPE_CHECKING
from copy import copy

if TYPE_CHECKING:
    from ..base import Host

from .events import EventEmitter, EventLogger



class Module (ABC) :
    """
    A module usually contains 3 types of content:

      - host : The host this module belongs to.
      - deps* : Zero or more dependent modules this module is using.
      - params* : Zero or more params to build this module.

    Module creation happens can happen in 2 variations of steps:

      - make > bind > sync
      - copy > bind

    The steps are responsible for:

      - make : The initial params are set for a module created from scratch.
      - copy : The complete (copies have no sync step) params are set using another module as reference.
      - bind : The host and deps are set.
      - sync : The module executes a startup method that may include sending out events and
               using the dependent modules to initiate some data.
    """



    def __init__ ( self ) :
        """
        Will initialize the module by running either:
        - the make step if no reference object is given,
        - the copy step if a ref is given.

        This `__init__` should not be overriden. Instead, a derived class should override all of
        the `make`, `copy`, `bind` and `start` methods.
        """
        pass



    def make ( self ) -> None :
        """
        Make a new module from scratch with the help of arguments. This will only set the params.

        Extending Modules should have the following structure on their method:

        .. code-block:: python

            def make ( self, param_1, param_2, ... ) -> None :
                self.param_1 = param_1
                self.model_1 = load_model(param1)
                self.param_2 = param_2
        """
        raise ModuleException("Module does not implement `make`.")



    def copy ( self, ref:Module ) -> None :
        """
        Make a copy of the module params by using another similar Module as a reference.
        The code inside this method should provide a good copy where shallow and deep copies are
        used appropriately.

        Extending Modules should have the following structure on their method:

        .. code-block:: python

            def copy ( self, ref ) -> None :
                self.simple_param_1 = ref.simple_param_1
                self.complex_model_1 = copy(ref.complex_model_1)
        """
        raise ModuleException("Module does not implement `copy`.")



    def bind ( self, host:Host ) -> None :
        """
        Bind this module to its host and deps (dependent modules). Here, the event listeners should
        also be set.

        Extending Modules should have the following structure on their method:

        .. code-block:: python

            def bind ( self, host, req_mod_1, req_mod_2, ... ) -> None :
                self.host = host
                self.req_mod_1 = req_mod_1
                self.req_mod_2 = req_mod_2
                self.host.observe( EventTypeA, self.listen_event_a )
                self.host.observe( EventTypeB, self.listen_event_b )
        """
        raise ModuleException("Module does not implement `bind`.")



    def sync ( self, emit:EventEmitter, log:EventLogger ) -> None :
        """
        Run the sync procedure on this module. This usually means dispatching some events after the
        module is made and all other modules have been bound and are listening to event emitters.

        Extending Modules should have the following structure on their method:

        .. code-block:: python

            def sync ( self, emit, log ) -> None :
                log("ModuleX: start sync")
                emit( EventTypeA( event_args ) )
                for i in self.items :
                    emit( EventTypeB( event_args_i ) )

        """
        raise ModuleException("Module does not implement `sync`.")



class ModuleException (Exception) :
    """ Exception that is triggered when a module cannot be created. """
    pass
