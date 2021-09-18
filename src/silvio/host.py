"""
An host is a unit that can take in any number of host modules. It serves as the container
of all modules and allows for obverser-event communication between the modules. In addition, it is
capable of properly copying itself and it's modules. It also provides some facilities for
pseudo-random generation to allow deterministic generation.

In technical terms, the Host uses composition-over-inheritance to define all behaviours that
Modules may add to it. In addition to the usual forwarding methods, the Host is also an
observable that can be used by each module to communicate with other modules.
"""

from typing import List, Callable, NamedTuple, Optional, Type

from .events import Event
from .random import Generator
from .utils import coalesce



class HostException (Exception) :
    """ Exception that is triggered when a Host cannot be created. """
    pass



# The callback will execute provide an event to the listener an expect a message to add to
# the event log.
ListenerCallback = Callable[[Event],Optional[str]]



class ListenerEntry ( NamedTuple ) :
    evtype: Type[Event]
    run: ListenerCallback



class Host :

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



    def __init__ ( self, ref:Optional['Host'] = None, name:str = None, seed:Optional[int] = None ) :
        """
        Init is responsible to initialize all modules of an host, may it be new from scratch or
        by using another host as a reference.

        Code from derived classes should be:

            super().__init__( ref, name, seed )

            self.moduleA = Module( host=self )
            self.moduleB = Module( host=self, depModule=self.moduleA )
        """
        self.listeners = []
        self.event_log = []
        self.clone_counter = 0
        self.name = 'unnamed' # start unnamed and get a name later in this constructor
        self.rnd_seed = None
        self.rnd_counter = 0


        # Creating with ref puts some defaults on hierarchical names and stable seeds.
        if ref is not None :
            self.name, self.rnd_seed = ref.make_clone_attrs()

        # Constructor attributes always override all other values.
        if name is not None :
            self.name = name
        if seed is not None :
            self.rnd_seed = seed



    def make_clone_attrs ( self ) -> [ str, int ] :
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



    def emit ( self, event: Event ) -> None :
        """ Trigger all listeners for a given Event. """
        for le in self.listeners :
            if type(event) == le.evtype :
                log_entry = le.run(event) # run the listener with the emitted event.
                self.event_log.append(coalesce( log_entry, "no description" ))



    def observe ( self, evtype: Type[Event], run: ListenerCallback ) -> None :
        """ Bind a new listener for a type of event. """
        self.listeners.append( ListenerEntry( evtype=evtype, run=run ) )



    def make_generator ( self ) -> Generator :
        """ Construct a random number generator with the same seed stored in the host. """
        self.rnd_counter += 1
        return Generator( self.rnd_seed + self.rnd_counter )
