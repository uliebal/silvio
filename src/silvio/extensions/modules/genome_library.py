"""
GenomeLibrary is a more complex version of GenomeList:
 - support for genome sequences
 - genes are located inside the sequence
"""


from __future__ import annotations
from typing import Optional, List, NamedTuple

from numpy.random import Generator
from Bio.Seq import Seq
from Bio.Align import PairwiseAligner # Bio.Blast.Record.Alignment

from ...host import Host
from ...module import Module
from ...events import EventEmitter, EventLogger
from ...utils import alldef, coalesce, first
from ..records.gene.gene import Gene
from ..records.gene.localized_gene import LocalizedGene
from ..all_events import InsertGeneEvent, RemoveGeneEvent, AlterGenePromoterEvent



class PrimerMatch (NamedTuple) :
    loc_start: int # Match goes from start (inclusive) to end (exclusive).
    loc_end: int
    success: float # (0,1)



class GenomeLibrary (Module) :
    """
    GenomeLibrary is a module that models the genome in a simplified way to ease interpretation.
    It stores a sequence of base pairs and provides a Gene interface that reads from the sequence
    at the gene's locus.

    The Sequence is stored as a base-pair list, whereas genes are stored as locations inside
    that sequence.
    """

    # bg_size: int # Amount of background gene bases.

    # bg_gc_content: float # Ratio of GC-content in the background genome. Range: [0,1]

    sequence: Seq

    locgenes: List[LocalizedGene]



    def make ( self,
        sequence:Optional[Seq] = None,
        bg_size:Optional[int] = None,
        bg_gc_content:Optional[float] = None,
        bg_rnd:Optional[Generator] = None,
        locgenes:List[LocalizedGene] = []
    ) -> None :

        # Init 1: pre-defined sequence
        if alldef( sequence ) :
            self.sequence = sequence

        # Init 2: new sequence
        elif alldef( bg_rnd, bg_size, bg_gc_content ) :
            self.sequence = make_background_seq( size=bg_size, gc_content=bg_gc_content, rnd=bg_rnd )

        # Failed Init
        else :
            raise Exception ("Creating GenomeLibrary module without either pre-defined or generated sequence.")

        # Add the locgenes and check if they are matching.
        self.locgenes = []
        for lg in locgenes :
            new_loc_gene = LocalizedGene(
                name=lg.name, seq=self.sequence,
                start_loc=lg._start_loc, prom_len=lg.prom_len, orf_len=lg.orf_len
            )
            self.locgenes.append( new_loc_gene )



    def copy ( self, ref:'GenomeLibrary' ) -> None :
        self.sequence = ref.sequence
        self.locgenes = ref.locgenes.copy()



    def bind ( self, host:Host ) -> None :
        host.observe( InsertGeneEvent, self.listen_insert_gene )
        host.observe( RemoveGeneEvent, self.listen_remove_gene )
        host.observe( AlterGenePromoterEvent, self.listen_alter_gene_promoter )



    def sync ( self, emit:EventEmitter, log:EventLogger ) -> None :
        pass # Nothing to sync.



    @property
    def genes ( self ) -> List[Gene] :
        return self.locgenes



    def listen_insert_gene ( self, event:InsertGeneEvent, emit:EventEmitter, log:EventLogger ) -> None :

        # If no locus is specified, then append it to the sequence.
        loc:int = coalesce( event.locus, len(self.sequence) )

        self.insert_gene( gene=event.gene, loc=loc )
        log( "GenomeLibrary: added gene={} loc={}".format(
            event.gene.name, event.locus
        ))



    def listen_remove_gene ( self, event:RemoveGeneEvent, emit:EventEmitter, log:EventLogger ) -> None :
        """ Remove a gene from the library and sequence. """
        del self.genes[event.gene]
        log( "GenomeLibrary: removed gene={}".format(event.gene.name) )



    def listen_alter_gene_promoter ( self, event:AlterGenePromoterEvent, emit:EventEmitter, log:EventLogger ) -> None :
        """
        When a gene promoter is altered, we replace the associated part in the sequence.
        TODO: No knock-outs are being made on this operation.
        """
        found_gene = first( self.locgenes, lambda lg : lg == event.gene )
        if found_gene is not None :
            self.alter_sequence( found_gene.start_loc, found_gene.prom_len, event.new_promoter )
            log( "GenomeLibrary: Changed promoter of gene={} in sequence.".format(event.gene.name) )



    def get_genes_by_name ( self, name:str ) -> List[Gene] :
        found_genes = [ gene for gene in self.locgenes if name == gene.name ]
        return found_genes



    def insert_gene ( self, gene:Gene, loc:int ) :
        """
        Insert a gene and its sequence at a specific location in the sequence.
        """
        # Insert the sequence at the insertion point.
        self.sequence = self.sequence[0:loc] + gene.prom + gene.orf + self.sequence[loc:None]

        # Remove genes that start before the insertion and end afterwards (knockout)
        is_knocked_out = lambda lg : lg.start_loc < loc and lg.end_loc > loc
        self.locgenes[:] = [ lg for lg in self.locgenes if not is_knocked_out(lg) ]
            # Replace the list of locgenes with the filtered locgenes, while keeping the list reference.

        # Shift the location of all genes after the insertion point.
        new_gene_len = len(gene.prom) + len(gene.orf)
        for lg in self.locgenes :
            lg.seq = self.sequence # Update to use this new sequence (since they are immutable)
            if lg.start_loc >= loc :
                lg.start_loc = lg.start_loc + new_gene_len

        # Annotate the new sequence as a localized gene.
        new_loc_gene = LocalizedGene(
            name=gene.name, seq=self.sequence,
            start_loc=loc, prom_len=len(gene.prom), orf_len=len(gene.orf)
        )
        self.locgenes.append( new_loc_gene )



    def alter_sequence ( self, old_loc:int, old_len:int, new_seq:Seq ) :
        self.sequence = self.sequence[0:old_loc] + new_seq + self.sequence[old_loc+old_len:None]



    def calc_primer_matches ( self, primer:Seq ) -> List[PrimerMatch] :
        """
        Return the insertion sites a primer can have, alongside with success rate.
        The insertion site is located right after a primer match.
        """
        matches: List[PrimerMatch] = []
        targets: List[Seq] = [ self.sequence ] # List of sequences to check.
        for target in targets :
            if len(target) > 0 :
                match = find_best_primer_match( target, primer )
                if match is not None :
                    matches.append( match )
                    targets.append( target[None:match.loc_start] ) # Also search left of match.
                    targets.append( target[match.loc_end:None] ) # Also search right of match.
        return matches



def make_background_seq ( size:int, gc_content:float, rnd:Generator ) -> Seq :
    gc = gc_content
    at = 1 - gc_content
    seq_array = rnd.pick_choices( choices=["A","C","G","T"], amount=size, weights=[at/2,gc/2,gc/2,at/2] ) # 'ATCG' is ArrayLike
    return Seq( "".join(seq_array) )



def check_primer_integrity ( primer:Seq ) -> bool :
    """ Non-deterministic check if the primer itself is well built. """
    return True



def find_best_primer_match ( sequence:Seq, primer:Seq ) -> Optional[PrimerMatch] :
    """ Find best match for a template sequence.
    TODO: Primer matching is dependent on host. algo_params:Optional[Dict] = None
    """

    # The primer matches with the complement sequence.
    comp_seq = sequence.complement()

    # Build the aligner for matching.
    aligner = PairwiseAligner()
    #aligner = extend( aligner, algo_params )
    aligner.mode = 'global' # target_end_gap_score, query_end_gap_score
    aligner.match_score = 1
    aligner.mismatch_score = -1
    aligner.gap_score = -1 # open_gap_score, extend_gap_score
    aligner.query_end_gap_score = 0

    # Get the alignment with the best score, anywhere on the sequence.
    all_alignments = aligner.align( comp_seq, primer )
    best_alignment = next(iter(sorted(all_alignments)))
    #print("score:" + str(best_alignment.score) + "\n" + str(best_alignment))
    if best_alignment is not None and best_alignment.score > 0 :
        return PrimerMatch(
            loc_start= best_alignment.aligned[0][0][0], # start of first chunk in target
            loc_end= best_alignment.aligned[0][-1][1], # end of last chunk in target
            success= best_alignment.score / len(primer)
        )
    return None



# TODO: Unused method that would create a full sequence by taking a list of genes and filling the
#   spaces in between with background bases. Since the background sequence is created at the
#   beginning I am not using this method.
# def build_sequence_from_gene_list ( self ) -> Seq :
#     rnd = self.host.make_generator()
#     gc = self.bg_gc_content
#     at = 1 - self.bg_gc_content
#
#     # Write the sequence from begin to end. Genes have their sequence copied and random bases
#     # are used to fill non-coding positions. Overlapping genes have undefined behaviour.
#     # TODO: When gene positions overlap it actually writes both gene sequences in sequence but
#     # in future this behaviour should be improved. This deals with validity of genes in
#     # overlapping regions.
#     locgenes = self.genes.values()
#     locgenes.sort( key=lambda el: el.loc )
#
#     seq = Seq()
#     cur = 0
#     bg_size = 0 # Current amount of background bases.
#     for locgene in locgenes :
#
#         # If the gene starts later, then fill the space with random (background) bases.
#         filler = locgene.loc - cur
#         if filler > 0 :
#             new_bases = rnd.choice( 'ATCG', size=filler, p=[at,at,gc,gc] )
#             bg_size += filler
#             seq += "".join(new_bases)
#
#         # Copy the gene sequence.
#         seq = seq + locgene.gene.get_prom() + locgene.gene.get_orf()
#
#     # At the end, if the minimum of background bases was not filled, to that now.
#     filler = min( 0, self.bg_min_size - bg_size )
#     if filler > 0 :
#         new_bases = rnd.choice( 'ATCG', size=filler, p=[at,at,gc,gc] )
#         seq += "".join(new_bases)
#
#     return seq
