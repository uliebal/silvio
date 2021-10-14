# Make these deep classes and methods available on the top-level.
# By convention, these are the public methods available for users of the library.

# We use a written `__init__.py` file at the top level which makes deep references into sub-modules.
# By doing this, all logic is inside this file and other __init__ files in the sub-folders can
# stay empty.

from .config import DATADIR
from .experiment import Experiment, ExperimentException
from .host import Host, HostException
from .tool import Tool, ToolException
from .outcome import SimulationException, Outcome, DataOutcome, DataWithPlotOutcome, combine_data
from .random import Generator
from .utils import alldef, coalesce, first

from .extensions.common import Base, PromoterSite, ReadMethod

from .extensions.all_events import (
    InsertGeneEvent, RemoveGeneEvent, AlterGenePromoterEvent, AlterGeneExpressionEvent
)

from .extensions.modules.genome_expression import GenomeExpression
from .extensions.modules.genome_library import GenomeLibrary
from .extensions.modules.genome_list import GenomeList
from .extensions.modules.growth_behaviour import GrowthBehaviour
from .extensions.modules.metabolic_flux import MetabolicFlux
from .extensions.modules.phenotype_size import PhenotypeSize

from .extensions.records.gene.gene import Gene
from .extensions.records.gene.crafted_gene import CraftedGene

from .extensions.tools.shotgun_sequencing import (
    ContigAssembler, GreedyContigAssembler, RandomContigAssembler, ShotgunSequencer
)

from .extensions.utils.shotgun_sequencing import (
    get_consensus_from_overlap, estimate_from_overlap, calc_total_score, calc_sequence_score,
    evaluate_sequence, write_scaffolds_to_file, print_scaffold_as_fastq, print_scaffold,
    print_assembly_evaluation, print_estimation_evaluation
)