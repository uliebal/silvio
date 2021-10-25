
from typing import Dict, List, Optional, NamedTuple

import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from ...common import Base, ReadMethod





BaseProbDict = Dict[Base,List[float]]



class Scaffold ( NamedTuple ) :
    """
    A sequence of bases, with added information about read quality and relative distances.
    Read quality is expressed in a ratio between 0 (worst) and 1 (best).
    Can either have a single contig or 2 contigs with paired ends.
    The R2 contig has a reversed sequence.
    The expected expected_len includes both contigs and the gap.

        |  R1 contig  |            gap            |  R2 contig  |
        [0123=========] - - - - - - - - - - - - - [=========3210]
         ---> order                                   order <---
    """
    expected_len : float # Just an expectation and not necessarily integer.
    r1_seqrecord : SeqRecord
    r2_seqrecord : Optional[SeqRecord]



class LocalizedSequence ( NamedTuple ) :
    """ A sequence with a starting position. """
    sequence : Seq
    locus : int



class EstimatedSequence ( ) :
    """
    An EstimatedSequence can hold a base probability for each of the positions in the gene sequence.
    Assemblers that split their certainty of a base call between two or more different bases can
    express that with these probabilities.
    Deterministic sequences can also be converted to estimated sequences where each corresponding
    base call is rated with a full probability.
    """

    # Accessed via: base_probs[base][position] = probability-of-base-in-position
    # Values will be normalized, so it does not need to be necessarily a probability.
    base_probs : BaseProbDict



    def __init__ ( self, base_props:BaseProbDict ) :

        # Check that at least one base has been given as probabilities.
        if len(base_props.keys()) == 0 :
            raise Exception("To create an EstimatedSequence the probabilities of at least one base"
                " have to be specified.")
        first_base = next(iter(base_props.keys()))

        # Check if each sequence for a base has the same length.
        for base in base_props.keys() :
            if len(base_props[base]) != len(base_props[first_base]) :
                raise Exception("The probabilities of all bases need to have the same size.")

        self.base_probs = base_props



    def __len__ ( self ) :
        first_base = next(iter(self.base_probs.keys()))
        return len(self.base_probs[first_base])



    def calc_shannon_entropy ( self ) -> float :
        """
        Calculate the Shannon Entropy over the own estimated sequence.
        The Shannon Entropy will be calculated for each position and the average will be output.
        """
        shannon_sum = 0

        first_base = next(iter(self.base_probs.keys()))
        for i in range(len(self.base_probs[first_base])) :
            bpli = np.array([ self.base_probs[base][i] for base in self.base_probs.keys() ])
                # bpli -> base probabilities on location i
            nbpli = bpli / bpli.sum()
                # nbpli -> normalized base probabilities on location i
            with np.errstate(divide='ignore') : # Disable warning to allow log2 calculation on zero
                shannon_i = -np.sum(nbpli * np.nan_to_num( np.log2(nbpli), neginf=0.0 ) )
                    # shannon_i -> shannon entropy on location i
                    # Additional conversion from NaN to 0 is required because: 0 * log(0) == NaN
            shannon_sum += shannon_i

        shannon_avg = shannon_sum / len(self.base_probs[first_base])
        return shannon_avg



    def as_consensus_sequence ( self ) -> Seq :
        """
        Outputs the consensus sequence by choosing the base with highest probability for each location.
        """
        consensus:str = ""
        first_base = next(iter(self.base_probs.keys()))
        for i in range(len( self.base_probs[first_base] )) :
            quorum = dict([ (base,self.base_probs[base][i]) for base in self.base_probs.keys() ])
            max_base = max( quorum, key=quorum.get, default='-' ) # Get Key with maximum value. Get first if equal.
            consensus += max_base
        return Seq(consensus)



def get_consensus_from_overlap ( locseqs:List[LocalizedSequence] ) -> LocalizedSequence :
    """
    Output the consensus sequence of multiple localized sequences.
    Returns: ( min starting locus, max ending locus, consensus sequence )
    """

    # Pay attention that indexes start at zero while locus could be negative.
    min_start = min([ locseq.locus for locseq in locseqs ])
    max_end = max([ locseq.locus + len(locseq.sequence) for locseq in locseqs ])
    conseq:List[str] = [ "-" for _ in range(min_start,max_end) ]

    # For each position in the consensus sequence, get the value of each localized sequence in that
    # same position. Could be out of bounds.
    for pos in range( min_start, max_end ) :
        quorum:Dict[Base,int] = {}
        for locseq in locseqs :
            if pos >= locseq.locus and pos < locseq.locus + len(locseq.sequence) : # inside bounds

                # Add this base to the possible bases in the consensus. Initialize if required.
                cur_base = str(locseq.sequence[ pos - locseq.locus ])
                if cur_base not in quorum :
                    quorum[cur_base] = 0
                quorum[cur_base] += 1

        # Based on the quorum, select the best base for this position.
        max_base = max( quorum, key=quorum.get, default='-' ) # Get Key with maximum value. Get first if equal.
        conseq[ pos - min_start ] = max_base

    return LocalizedSequence( sequence= Seq("".join(conseq)), locus= min_start )



def estimate_from_overlap ( locseqs:List[LocalizedSequence] ) -> EstimatedSequence :
    """
    Given a list of localized sequences, overlap them and measure the probability of bases their
    sum returns.
    Similar to the `get_consensus_from_overlap` method, but the probabilities are kept instead
    of the best base.
    """

    # Pay attention that indexes start at zero while locus could be negative.
    min_start = min([ locseq.locus for locseq in locseqs ])
    max_end = max([ locseq.locus + len(locseq.sequence) for locseq in locseqs ])
    quorum:BaseProbDict = {
        'A': [ 0 for _ in range(min_start,max_end) ],
        'C': [ 0 for _ in range(min_start,max_end) ],
        'G': [ 0 for _ in range(min_start,max_end) ],
        'T': [ 0 for _ in range(min_start,max_end) ],
    }

    # For each position in the consensus sequence, get the value of each localized sequence in that
    # same position. Could be out of bounds.
    for pos in range( min_start, max_end ) :
        for locseq in locseqs :
            if pos >= locseq.locus and pos < locseq.locus + len(locseq.sequence) : # inside bounds
                cur_base = str(locseq.sequence[ pos - locseq.locus ])
                quorum[cur_base][ pos - min_start ] += 1

    # Convert the counts in the quorum into a probability. If the quorum is completely empty,
    # then the probability for each base is equalized.
    for pos in range( max_end - min_start ) :
        sum_counts = sum([ quorum[base][pos] for base in ['A','C','G','T'] ])
        if sum_counts == 0 : # Special case to set equal probabilities.
            for base in ['A','C','G','T'] :
                quorum[base][pos] = 1/4
        else : # Normal case to normalize into probabilities.
            for base in ['A','C','G','T'] :
                quorum[base][pos] = quorum[base][pos] / sum_counts

    return EstimatedSequence(quorum)

