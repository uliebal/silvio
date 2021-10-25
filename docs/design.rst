======
Design
======

The Silvio reactive programming system that aims to simulate how different concepts inside a
biological host interact with each other.

A **module** embodies a concept of a biological unit: for example the list of genes it contains,
the associated metabolic model or a measurement of gene expressions.

The **host** contains multiple modules and allows modules to send events to each other. For example,
when a gene is deleted from genome sequence, an event is sent to the metabolic model that accomodate
for that delete gene.

The user of ``silvio`` can create its own host and include only those modules that are relevant for
the simulation. Hosts can suffer changes (by altering the modules inside it) and the reactive events
keep the entire system state in sync.

Then, **processes** can be run on the host. They use the current state of the host and generate
data in the form of outcomes. The **outcomes** are packages of data (series, dataframes, values)
that can be analyzed or stored.

The **experiment** is all-encompassing entity that contains all other concepts and which can dictate
the random generation. Random number generation is stable in a sense that the same code will
produce the same results.


---------------
Creating a Host
---------------

The best place to see examples of using ``silvio`` is to check the ``biolabsim`` workflows. The
**workflow** usually defines a tailor-made host and experiment, which then get used in the
notebooks.

This section will show how to create a host for direct use of module features. Workflows prefer to
close off hosts in a way that only student-facing methods are available for use, and most methods
simply redirect to the direct module features. Here we create a host that allows free usage of the
modules.

To have a Host for personal use you have to create a new class that derives from the abstract Host.
Then, you will have to define the following 3 methods:

* ``make ( self ) -> None``

The **make** method is responsible for building a new host out of nothing but arguments. Your
implementation of ``make`` should add all necessary arguments, then create the modules while
respecting their dependencies, and bind them with the host to allow event listening. The way a
module is initialized follows a specific recipe which is very similar to creating hosts.

* ``copy ( self, ref:Host ) -> None``

The **copy** method will make a clone of a host. It uses the ``ref`` host to retrieve all necessary
data for cloning. When all modules themselves implement ``copy``, then this method consists usually
of boilerplate code that copies all the modules and binds them to this host.

* ``sync ( self ) -> None``

The **sync** method starts up all the modules and the host itself. It may contain code that should
run after a host is made, but it is not run if the host is copied. Usually we just call the sync of
each module to allow each module to send some initializing events before we use the most. For
example, the MetabolicModel module can import a CobraPy model and will use the sync step to share
all of the contained genes with the GenomeList module.

What follows is an example of a minimally created module:

.. code-block:: python

    from silvio import Host, GenomeLibrary, sync_modules

    class CustomHost (Host) :

        genome: GenomeLibrary  # Store the modules as properties.

        # Make takes additional arguments to be able to create a GenomeLibrary module.
        # Inside make, use the (new > make > bind) style to create each module.
        # The host also has extra method for setting the random number generator.
        def make ( self, bg_size:int, bg_gc_content:float ) -> None :
            self.genome = GenomeLibrary()
            self.genome.make( bg_size=bg_size, bg_gc_content=bg_gc_content, bg_rnd=self.make_generator() )
            self.genome.bind( host=self )

        # Copy has a single reference host argument that is used to copy all other modules.
        # Inside copy, use the (new > copy > bind) style to create each module.
        def copy ( self, ref:CustomHost ) -> None :
            self.genome = GenomeLibrary()
            self.genome.copy( ref=ref.genome )
            self.genome.bind( host=self )

        # Most sync methods will only run the sync on each module. We use a helper method for that.
        def sync ( self ) -> None :
            self.sync_modules([ self.genome ])

Now to use this host, we can create a new one and call its methods:

.. code-block:: python

    from silvio import CraftedGene
    from Bio.Seq import Seq

    # Similar modules, we use a (new > make > sync) style for creation
    my_host = CustomHost( name="origin", seed=1885 )
    my_host.make( bg_size=100, bg_gc_content=0.45 )
    my_host.sync()

    # Now we can freely use the module methods.
    new_gene = CraftedGene( name="Custom1", orf=Seq("ATGCAAAGGTAA"), prom=Seq("TATAAATGTGTTC") )
    my_host.genome.insert_gene( new_gene )
    print( my_host.genome.sequence )

In the example above we initialize our host and add a handcrafted Gene to it. That addition will
alter the sequence, which we then read on the last line. Adding the gene will also send a "gene has
been added" event to all other modules. If another module for GenomeExpression would exist, then it
would calculate the promoter strength of it and send an event of "a promoter strength for a gene
has been altered". Then, if a MetabolicFlux module would exist, it would listen to that event and
alter the metabolic model in accordance to the new promoter strength. This is the chain of events
where each module can listen to events that are important to it.


------------------------------------
Creating an Experiment to hold Hosts
------------------------------------

An **experiment** can hold all the hosts and be central entity set the possible actions and to
dictate stable randomness.

An example follows:

.. code-block:: python

    class CustomExperiment (Experiment) :

        def __init__ ( self, seed:Optional[int] = None ) :
            super().__init__(seed=seed)

        def create_host ( self, name:str, bg_size:int, bg_gc_content:float ) -> CustomHost:
            seed = self.rnd_gen.pick_seed() # The experiment provides stable seed generation for hosts.
            new_host = CustomHost( name=name, seed=seed )
            new_host.make( bg_size=bg_size, bg_gc_content=bg_gc_content )
            new_host.sync()
            self.bind_host(new_host)  # Call this to access it from the experiment.
            return new_host

Defining an experiment is very useful if you want to handpick all possible methods the end user
may or may not invoke. For internal usage it is not really necessary.


---------------------
Creating a new Module
---------------------

The creation of a new **module** follows a similar style to the host, but with an added step. You
will have to extend the ``Module`` class and implement the following methods:


* ``make ( self ) -> None``

The **make** method implements for a module is created from arguments. You may add additional
arguments to the method signature.

* ``copy ( self, ref:Module ) -> None``

The **copy** method will build the module to be an exact copy of the reference module.

* ``bind ( self, host:Host ) -> None``

The **bind** will register the event listeners on the holding host.

* ``sync ( self, emit:EventEmitter, log:EventLogger ) -> None``

And **sync** may run code after executing make.

Apart from implementing these 4 methods, the module should store all properties and establish all
methods that it needs to perform its intent. When the module listens to events, it is usual to
create listener methods that receive the event, an internal logger and an event emitter that chain
more event calls. The event emission system will prevent infinite loops.

Here is a very simple example of a module that simply counts the number of genes it has. Every time
an event is emitted for a gene addition, it will listen and update its internal state:

.. code-block:: python

    class PhenotypeSize (Module) :

        size: int  # Internal properties of the module.

        def make ( self, size:int = 0 ) -> None :
            self.size = 0

        def copy ( self, ref:PhenotypeSize ) -> None :
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
