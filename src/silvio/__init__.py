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

from .extensions.events import InsertGeneEvent, RemoveGeneEvent

from .extensions.modules.genome_expression import GenomeExpression
from .extensions.modules.genome_library import GenomeLibrary
from .extensions.modules.genome_list import GenomeList
from .extensions.modules.growth_behaviour import GrowthBehaviour
from .extensions.modules.phenotype_size import PhenotypeSize

from .extensions.records.gene.gene import Gene
from .extensions.records.gene.crafted_gene import CraftedGene

from .extensions.tools.shotgun_sequencing.assembly import ContigAssembler, GreedyContigAssembler, RandomContigAssembler
from .extensions.tools.shotgun_sequencing.datatype import Scaffold, LocalizedSequence, EstimatedSequence, get_consensus_from_overlap, estimate_from_overlap
from .extensions.tools.shotgun_sequencing.evaluation import calc_total_score, calc_sequence_score, evaluate_sequence
from .extensions.tools.shotgun_sequencing.sequencing import ShotgunSequencer
from .extensions.tools.shotgun_sequencing.storage import write_scaffolds_to_file
from .extensions.tools.shotgun_sequencing.visualization import print_scaffold_as_fastq, print_scaffold, print_assembly_evaluation, print_estimation_evaluation
