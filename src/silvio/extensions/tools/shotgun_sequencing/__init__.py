"""
This sub-module contains methods to simulate second-generation shotgun sequencing, and tools to
visualize and evaluate fragment assembly.
"""

from .assembly import ContigAssembler, GreedyContigAssembler, RandomContigAssembler

from .datatype import Scaffold, LocalizedSequence, EstimatedSequence, \
    get_consensus_from_overlap, estimate_from_overlap

from .evaluation import calc_total_score, calc_sequence_score, evaluate_sequence

from .sequencing import ShotgunSequencer

from .storage import write_scaffolds_to_file

from .visualization import print_scaffold_as_fastq, print_scaffold, \
    print_assembly_evaluation, print_estimation_evaluation
