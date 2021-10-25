"""
An host is a unit that can take in any number of host modules. It serves as the container
of all modules and allows for obverser-event communication between the modules. In addition, it is
capable of properly copying itself and it's modules. It also provides some facilities for
pseudo-random generation to allow deterministic generation.

In technical terms, the Host uses composition-over-inheritance to define all behaviours that
Modules may add to it. In addition to the usual forwarding methods, the Host is also an
observable that can be used by each module to communicate with other modules.
"""

from typing import List, Callable, NamedTuple, Optional, Type, TYPE_CHECKING
from abc import ABC

from .events import Event, EventEmitter, EventLogger
from .random import Generator, pick_seed
from .utils import coalesce



class HostException (Exception) :
    """ Exception that is triggered when a Host cannot be created. """
    pass



# The callback will execute provide an event to the listener an expect a message to add to
# the event log.
ListenerCallback = Callable[[Event,EventEmitter,EventLogger],Optional[str]]



class ListenerEntry ( NamedTuple ) :
    evtype: Type[Event]
    run: ListenerCallback



class Host (ABC) :
    """
    A host usually contains 3 types of content:

    - core params : Parameters that are used by all hosts.
    - modules : Zero or more modules inside the host.
    - extra params : Zero or more extra params used by the host extension.

    Host creation happens can happen in 2 variations of steps:

      - make > sync
      - copy

    The steps are responsible for:

      - make : The initial params are set for a host created from scratch. The extending host
      - copy : The complete (copies have no sync step) params are set using another host as reference.
      - sync : The host call out each module to sync.
    """

    name: str

    listeners: List[ListenerEntry]

    # The event_log holds small messages of all events that occured to the host.
    event_log: List[str]

    # By specifying a seed, the same code will produce the same results.
    rnd_seed: Optional[int]

    # A seed with an incremental counter provides a stable randomization were experiments can be
    # run multiple times with different results.
    rnd_counter: int

    # Number of clones made. This is used by the cloning method to generate new names.
    clone_counter: int



    def __init__ (
        self, ref:Optional['Host'] = None, name:str = None, seed:Optional[int] = None, **kwargs
    ) :
        """
        Init is responsible to initialize the base properties of an host, may it be new from
        scratch or by using another host as a reference.

        Extended Hosts should not override the __init__ method.
        """
        self.listeners = []
        self.event_log = []
        self.clone_counter = 0
        self.name = 'unnamed' # start unnamed and get a name later in this constructor
        self.rnd_seed = pick_seed() # by default, use a random seed
        self.rnd_counter = 0


        # Creating with ref puts some defaults on hierarchical names and stable seeds.
        if ref is not None :
            self.name, self.rnd_seed = ref.build_clone_attrs()

        # Constructor attributes always override all other values.
        if name is not None :
            self.name = name
        if seed is not None :
            self.rnd_seed = seed

        # Creation via "make > sync" is made if no ref is given. If a ref is given we perform "copy".
        # TODO: This is currently deactivated. Its a shorter style but for now prefer using the
        #   longer style because its very similar to how Modules are initialized.
        # if ref is None :
        #     self.make(**kwargs)
        #     self.sync()
        # else :
        #     self.copy(ref=ref)



    def make ( self ) -> None :
        """
        Make a new host from scratch with the help of arguments. Set possible params and modules.

        Extending Host should have the following structure on their method:

        .. code-block python::

            def make ( self, param_1, ... ) -> None :
                self.module_1 = ModuleTypeA()
                self.module_1.make( module_param_1=300, module_param_2=param_1 )
                self.module_1.bind( host=self )
        """
        raise HostException("Host does not implement `make`.")



    def copy ( self, ref:'Host' ) -> None :
        """
        Make a copy of the host params and modules by using another similar Host as a reference.
        The code inside this method should provide a good copy where shallow and deep copies are
        used appropriately.

        Extending Hosts should have the following structure on their method:

        .. code-block python::

            def copy ( self, ref ) -> None :
                self.module_1 = ModuleTypeA()
                self.module_1.copy( ref=self.genome )
                self.module_1.bind( host=self )
        """
        raise HostException("Host does not implement `copy`.")



    def sync ( self ) -> None :
        """
        Run the sync procedure on this host. It usually only calls the sync of each module using
        a helper method for that.

        Extending Hosts should have the following structure on their method:

        .. code-block python::

            def sync ( self ) -> None :
                self.sync_modules([ self.module_1, self.module_2, ... ])

        """
        raise HostException("Host does not implement `sync`.")



    def build_clone_attrs ( self ) -> [ str, int ] :
        """
        Generate attributes for a possible clone.

        Returns
        -------
        [ name, seed ]
            name: A hierarchical name is generated.
            seed: A stable seed is generated.
        """
        self.clone_counter += 1
        gen = self.make_generator()
        hierarchical_name = self.name + "." + str(self.clone_counter)
        stable_seed = gen.pick_seed() + self.clone_counter
        return [ hierarchical_name, stable_seed ]



    def emit ( self, event: Event, chain: List[ListenerEntry] = [] ) -> None :
        """
        Trigger all listeners for a given Event.

        Uses a chain of listener entries to keep track of which has already been called
        and to abort loops that may result. Each listener callback may only execute once for each
        event chain.

        The Host also provides an EventEmitter to each listener callback in order to hide the functionality
        of listener chains from the modules. If they want to emit events they need to use the method
        provided by the host.
        """
        for le in self.listeners :
            if type(event) == le.evtype and le not in chain :
                # Run the listener with the emitted event and append to chain to keep track.
                new_chain = chain + [le]
                emit = lambda event : self.emit( event, new_chain )
                log = lambda text : self.event_log.append( "- " * len(new_chain) + text )
                le.run( event, emit, log )



    def observe ( self, evtype: Type[Event], run: ListenerCallback ) -> None :
        """ Bind a new listener for a type of event. """
        self.listeners.append( ListenerEntry( evtype=evtype, run=run ) )



    def sync_modules ( self, modules:List['Module'] ) -> None :
        """
        Sync all modules as modules may emit some events or do load up calculations when they start.
        After creating all modules you should call this method with those modules.
        This cannot be included inside the module __init__ because all listeners have to be set.
        """
        emit = lambda event : self.emit( event, [] )
        log = lambda text : self.event_log.append( text )
        for mod in modules :
            mod.sync( emit, log )



    def make_generator ( self ) -> Generator :
        """ Construct a random number generator with the same seed stored in the host. """
        self.rnd_counter += 1
        return Generator( self.rnd_seed + self.rnd_counter )



    def print_event_log ( self ) -> None :
        for ev in self.event_log :
            print(ev)
