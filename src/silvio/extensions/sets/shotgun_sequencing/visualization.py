"""
This file contains helper methods to visualize data structures and aid in debugging.
"""

from math import floor, ceil
from typing import Dict, List, Tuple

import numpy as np
from Bio.Seq import Seq

from .datatype import Scaffold, EstimatedSequence, LocalizedSequence, estimate_from_overlap
from .evaluation import calc_total_score, calc_sequence_score



def print_scaffold_as_fastq ( scaffold:Scaffold ) -> None :
    """
    Print the scaffold content as FASTQ.
    """
    print(scaffold.r1_seqrecord.format("fastq"))
    if scaffold.r2_seqrecord is not None :
        print(scaffold.r2_seqrecord.format("fastq"))



def print_scaffold ( scaffold:Scaffold ) -> None :
    """
    Visually print a scaffold as text.
    """
    r2_exists = scaffold.r2_seqrecord is not None

    # Calculate the gap size.
    gap = 0
    if scaffold.r2_seqrecord is None : # only R1 sequence [XX]---
        gap = scaffold.expected_len - len(scaffold.r1_seqrecord.seq)
    else : # both R1 and R2 sequence [XX]---[XX]
        gap = scaffold.expected_len - len(scaffold.r1_seqrecord.seq) - len(scaffold.r2_seqrecord)

    print( "| {r1s}~{lspad}(ca.{gap}){rspad}~{r2s}".format(
        r1s="".join(scaffold.r1_seqrecord.seq),
        r2s="".join(reversed(scaffold.r2_seqrecord)) if r2_exists else "",
        lspad="~" * ceil(gap / 2),
        rspad="~" * floor(gap / 2),
        gap=gap,
    ))



def numtochar ( num:int ) -> str :
    """
    Convert a number to a visual character.
    """
    if num == 0 :
        return "·"
    elif num >= 1 and num <= 9 :
        return str(round(num))
    elif num > 9 :
        return "#"
    return "?"



def print_evaluation_alignment (
    score:float, genome_start:int, genome:Seq,
    estseq:EstimatedSequence, locseqs:List[LocalizedSequence] = None
) -> None :
    """
    Print a visual representation of how alignment was evaluated.
    """

    # Calculate some parameters for nice formatting.
    bundle_start = 0
    bundle_end = len(estseq)
    window_start = min( bundle_start, genome_start )
    window_end = max( bundle_end, genome_start + len(genome) )

    # Print the heatmap.
    print( "-" * (13 + window_end - window_start) )
    for base in ['A','C','G','T'] :
        print("[heatmap:{}] {}{}{}".format(
            base,
            numtochar(0) * (bundle_start - window_start),
            "".join([ numtochar(floor(bp*10)) for bp in estseq.base_probs[base] ]),
            numtochar(0) * (window_end - bundle_end),
        ))

    # If localized sequences are given, print the coverage and each sequence.
    if locseqs is not None :

        locseq_start = np.amin([ locseq.locus for locseq in locseqs ])
        coverage = ""
        for pos in range( locseq_start, bundle_end + locseq_start ) :
            pos_cov = 0
            for locseq in locseqs :
                if locseq.locus <= pos < locseq.locus + len(locseq.sequence) :
                    pos_cov += 1
            coverage += numtochar(pos_cov)

        print( "-" * (13 + window_end - window_start) )
        print(" [coverage] {}{}{}".format(
            numtochar(0) * (bundle_start - window_start),
            coverage,
            numtochar(0) * (window_end - bundle_end),
        ))

        seq_scores = calc_sequence_score( locseqs, genome, genome_start )

        print( "-" * (13 + window_end - window_start) )
        for l_i, locseq in enumerate(locseqs) :
            print("  [seq:{:03}] {}{}{} ({:3.0f}%)".format(
                l_i,
                " " * (locseq.locus - locseq_start - window_start),
                str(locseq.sequence),
                " " * (window_end - (locseq.locus - locseq_start) - len(locseq.sequence)),
                seq_scores[l_i] * 100
            ))

    # Print the consensus sequence.
    consensus = estseq.as_consensus_sequence()
    print( "-" * (13 + window_end - window_start) )
    print("[consensus] {}{}".format(
        " " * (bundle_start - window_start),
        str(consensus)
    ))

    # Print the real genome.
    print( "-" * (13 + window_end - window_start) )
    print(" [real-gen] {}{}".format(
        " " * (genome_start - window_start),
        str(genome)
    ))

    # And at last, the final score.
    print( "-" * (13 + window_end - window_start) )
    print( "Final Score: {:.4f} ({:.1f}%)".format( score, score * 100 ) )



def print_assembly_evaluation ( loc_sequences:List[LocalizedSequence], genome:Seq ) -> None :
    """
    Given a list of localized sequences, print a visual representation of the evaluation.
    """
    est_sequence = estimate_from_overlap(loc_sequences)
    ( best_score, best_genome_start ) = calc_total_score( est_sequence, genome )
    print_evaluation_alignment(
        score=best_score, genome_start=best_genome_start, genome=genome,
        estseq=est_sequence, locseqs=loc_sequences
    )



def print_estimation_evaluation ( est_sequence:EstimatedSequence, genome:Seq ) -> None :
    """
    Given only an estimated sequence, print a visual representation of the evaluation.
    """
    ( best_score, best_genome_start ) = calc_total_score( est_sequence, genome )
    print_evaluation_alignment(
        score=best_score, genome_start=best_genome_start, genome=genome,
        estseq=est_sequence, locseqs=None
    )
