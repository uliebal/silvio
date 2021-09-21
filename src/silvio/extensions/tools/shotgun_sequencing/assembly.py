
from abc import ABC, abstractmethod
from typing import List, NamedTuple, Dict, Any, Optional
from math import inf

import numpy as np
from Bio import pairwise2
from Bio.Seq import Seq
from Bio.Blast.Record import Alignment

from ....utils import coalesce
from ....tool import Tool, ToolException
from ....random import Generator
from ...common import Base
from .datatype import (
    Scaffold, LocalizedSequence, EstimatedSequence,
    get_consensus_from_overlap, estimate_from_overlap
)




class ContigAssembler (Tool, ABC) :
    """
    Abstract class that defines the interface of all assemblers.
    """

    def __init__ ( self, name:str, seed:Optional[int] = None ) :
        Tool.__init__( self, name=name, seed=seed )

    @abstractmethod
    def apply_internal ( self, scaffolds:List[Scaffold] ) -> List[LocalizedSequence] :
        pass

    @abstractmethod
    def apply ( self, scaffolds:List[Scaffold] ) -> EstimatedSequence :
        pass



class PairwiseScore (NamedTuple) :
    i : int
    k : int
    score : float



def match ( sa:Seq, sb:Seq ) -> Alignment :
    """ Internal method for pairwise matching. """
    return pairwise2.align.localms( sa, sb, 1, -1, -100, -100, gap_char='-' )



class GreedyContigAssembler (ContigAssembler) :
    """
    This assembler will try to assemble the DNA scaffolds by using greedy pairwise matching.
    It will start with the pair of sequences with highest pairwise matching and from there one
    iteratively add one of the remaining sequences by matching it with the consensus sequence of
    all already clustered sequences.

    Attention: It will extract both sequences of paired-end reads, but is not able to use their
    pairing relationship during assembly.
    """

    def __init__ ( self, name:Optional[str] = None, seed:Optional[int] = None ) :
        super().__init__( name=coalesce(name,"GreedyAssembler"), seed=seed )



    def apply_internal ( self, scaffolds:List[Scaffold] ) -> List[LocalizedSequence] :

        # Extract all sequences from the scaffolds. If its 'paired-end' then we also extract the
        # R2 sequences but discard their paired relationship.
        sequences:List[Seq] = []
        for scaf in scaffolds :
            sequences.append( scaf.r1_seqrecord.seq )
            if scaf.r2_seqrecord is not None :
                sequences.append( scaf.r2_seqrecord.seq.reverse_complement().complement() )
                    # Strange how there is no simple `Seq.reverse()` in biopython.

        # Calculate best pairwise matching score to decide on the first sequences to match.
        fbest = PairwiseScore( 0, 0, -inf )
        for i in range(len(sequences)) :
            for k in range( i+1, len(sequences) ) :
                cur_align = match( sequences[i], sequences[k] )[0]
                if cur_align.score > fbest.score :
                    fbest = PairwiseScore( i, 0, cur_align.score ) # 0 because k is not used.

        # To simplify coding (at the expense of an additional loop cycle) we will select the fbest.i
        # as the starting sequence and then recalculate the best pairwise match. It should lead to
        # fbest.k again, but we avoid having to write an extra case for this first match.

        # Keep track of all sequences that are left to cluster.
        available:List[bool] = [ True for _ in range(len(sequences)) ]
        cluster:List[LocalizedSequence] = []

        # Add the first sequence to start the cluster.
        available[fbest.i] = False
        cluster.append( LocalizedSequence( sequence=sequences[fbest.i], locus=0 ) )
        #print("First: " + "".join(sequences[fbest.i]))

        # While sequences are still left unclustered.
        while sum(available) > 0 :
            min_start = min([ locseq.locus for locseq in cluster ])
            consensus = get_consensus_from_overlap(cluster).sequence
            #print("Consensus: " + "".join(consensus))
            best = PairwiseScore( 0, 0, -inf )
            best_align:Alignment = None

            # Get the best matching for the cluster consensus.
            for k in range(len(sequences)) :
                if available[k] :
                    cur_align = match( consensus, sequences[k] )[0]
                    #print("      Try: {} ({}) [{}]".format( "".join(sequences[k]), cur_align.score, k ))
                    #print("         : {}".format("".join(cur_align.seqA)))
                    #print("         : {}".format("".join(cur_align.seqB)))

                    if cur_align.score > best.score :
                        best = PairwiseScore( 0, k, cur_align.score ) # 0 because i is not used.
                        best_align = cur_align

            # Somehow break the entire method if this is None, should never happen.
            if best_align is None :
                raise ToolException("GreedyContigAssembler is not working as intended. Check the code.")

            # From the alignment it is not known if the sequence A starts before or after the
            # sequence B. Unfortunately the `best_align.start` is always positive. Therefore peek
            # which of the sequences starts with the gap character. The one that does will start
            # after the other sequence. All locus is fixed to the locus of the cluster.
            best_locus = min_start
            if best_align.seqB[0] == '-' : # new sequence starts after cluster.
                best_locus = min_start + best_align.start
            elif best_align.seqA[0] == '-' : # new sequence starts before the cluster
                best_locus = min_start - best_align.start
            else : # new sequence start at the same position as the cluster
                best_locus = min_start

            cluster.append( LocalizedSequence( sequence=sequences[best.k], locus=best_locus ))
            available[best.k] = False
            #print("    Added: {} [{}] <{},{}>".format(
            #    "".join(sequences[best.k]), best.k, min_start, best_locus
            #))

        return cluster



    def apply ( self, scaffolds:List[Scaffold] ) -> EstimatedSequence :
        return estimate_from_overlap( self.apply_internal(scaffolds) )



class RandomContigAssembler (ContigAssembler) :
    """
    The random assembler will place all contigs in the scaffolds in random locations.
    Good assemblers should aspire to at least be better than this random assembler.
    """

    expected_genome_size : int



    def __init__ ( self, expected_genome_size:int, name:Optional[str] = None, seed:Optional[int] = None ) :
        self.expected_genome_size = expected_genome_size
        super().__init__( name=coalesce(name,"RandomAssembler"), seed=seed )



    def apply_internal ( self, scaffolds:List[Scaffold] ) -> List[LocalizedSequence] :
        rnd = self.make_generator()

        # Extract all sequences from the scaffolds. If its 'paired-end' then we also extract the
        # R2 sequences but discard their paired relationship.
        sequences:List[Seq] = []
        for scaf in scaffolds :
            sequences.append( scaf.r1_seqrecord.seq )
            if scaf.r2_seqrecord is not None :
                sequences.append( scaf.r2_seqrecord.seq.reverse_complement().complement() )
                    # Strange how there is no simple `Seq.reverse()` in biopython.

        # Randomly decide the start position of each sequence.
        loc_sequences:List[LocalizedSequence] = []
        for seq in sequences :
            start_pos = rnd.pick_integer( 0, self.expected_genome_size - len(seq) )
            loc_sequences.append( LocalizedSequence( sequence=seq, locus=start_pos ) )

        return loc_sequences



    def apply ( self, scaffolds:List[Scaffold] ) -> EstimatedSequence :
        """
        Attributes
        ----------
        scaffolds
            Scaffolds that were returned from fragmentation.

        Returns
        -------
        An estimation of bases for each position, based on random placement.
        """
        return estimate_from_overlap( self.apply_internal(scaffolds) )
