
from typing import Dict, List, Tuple
from math import inf

import numpy as np
from Bio.Seq import Seq

from ...common import Base
from .datatype import BaseProbDict, LocalizedSequence, EstimatedSequence



def calc_total_score ( estseq:EstimatedSequence, genome:Seq ) -> Tuple[float,int] :
    """
    Shift the genome around the heatmap and get the best overall match.
    Score is a simple addition of all ratios of the correct base, normalized as percentage.
    """

    gen_len = len(genome)
    est_len = len(estseq)
    best_score = -inf
    best_genome_start = 0
    for genome_start in range( -gen_len, est_len ) :
        score = 0
        for g_i in range(gen_len) : # g_i = index on genome sequence
            b_i = genome_start + g_i # b_i = map to an index in the bundle
            if 0 <= b_i < est_len : # only add score if it overlaps with the heatmap, 0 otherwise.
                gen_base = genome[g_i]
                hm_total = sum([ estseq.base_probs[base][b_i] for base in ['A','C','G','T'] ])
                if hm_total > 0 :
                    score += ( estseq.base_probs[gen_base][b_i] / hm_total )

        if score > best_score :
            best_score = score
            best_genome_start = genome_start

    return ( best_score / len(genome), best_genome_start )



def calc_sequence_score ( locseqs:List[LocalizedSequence], genome:Seq, genome_start:int ) -> List[float] :
    """
    Calculate the matching score of each singular localized sequence.
    """

    # genome_start is relative to min(locseqs.locus) when that is shifted to zero.
    locseq_start = int(np.amin([ locseq.locus for locseq in locseqs ]))
    rel_genome_start = genome_start + locseq_start

    seq_scores = [ 0 for _ in locseqs ]
    for l_i, locseq in enumerate(locseqs) :
        score = 0
        for s_i in range(len(locseq.sequence)) :
            g_i = locseq.locus + s_i - rel_genome_start
            if 0 <= g_i < len(genome) :
                if locseq.sequence[s_i] == genome[g_i] :
                    score += 1
        seq_scores[l_i] = score / len(locseq.sequence)

    return seq_scores



def evaluate_sequence ( estimated_sequence:EstimatedSequence, genome:Seq ) -> float :
    """
    Compares an estimated sequence with the real genome and returns the true positive rate.
    Return value is a number between 0 (worst score) and 1 (best score).
    """
    return calc_total_score( estimated_sequence, genome )[0]
