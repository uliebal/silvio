
from typing import List, Optional, NamedTuple, Literal
from recordclass import RecordClass
from math import floor

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from ....utils import coalesce
from ....tool import Tool, ToolException
from ...common import ReadMethod, Base
from .datatype import Scaffold



# Maximal Phred Quality a base call can possibly achieve.
PHRED_MAX = 40



class RatedSequence (RecordClass) :
    """
    Internal structure to store sequences and their quality score.
    """
    name : str
    seq : Optional[List[Base]]
    qual : Optional[List[float]]



class ShotgunSequencer (Tool) :
    """
    Function that, after setup, can take in a genome and perform whole-genome sequencing on it.
    The result will simulate a sequencing machine and provide a set of fragmented genome libraries
    that can later be assembled together to reconstruct the real genome of the host.

    Sequencing machines can be highly customizable and allow a multitude of parameters. Some
    considerations about the type of parameters and their default values are given below:

      - Usually a library size selection is done and this simulator will perform similar effects to
        those described under "Double-sided size selection and bead clean-up".
        A simplification is done and the blue graph in the link is converted to a normal distribution.
        https://emea.support.illumina.com/bulletins/2020/07/library-size-selection-using-sample-purification-beads.html

      - The read length of each fragment can also be configured, but a limit at roughly 150bp is set
        for each library, with some versions achieving up to 300bp. This limit is for each read.
        For example, with a value of 150bp, running the simulation on single-read method will yield
        libraries of 150bp each, whereas a paired-end method will yield two linked sequences
        of 150bp each.

      - Coverage of the extracted libraries can be controlled by reagents and number of cycles in
        the cloning step. The distribution of covered regions, however, depends on how the fragments
        attach to the flow cell prior to cluster forming. Attachment to the flow cell will follow
        random sampling from the available fragments. Given the amount of fragments this can be
        sampling to "sampling with reposition". The average coverage and read length; the genome
        size is known before-hand; and the number of reads will be calculated from all these values
        by using the Lander/Waterman equation.
        https://www.illumina.com/documents/products/technotes/technote_coverage_calculation.pdf

      - Quality of each call is difficult to simulate as it depends on multiple factors, such as:
        luminosity of dNTPs, signal-to-noise ratio, hardware of the sequencer and chemistry used.
        It defined by Illumina through empirical observations. For the simulation, an
        exponenetial distribution is used such, under default values, a Q30 score is achieved at
        97% of the times.
        https://www.illumina.com/Documents/products/technotes/technote_Q-Scores.pdf

    The sequences that are output will only contain the target region of each library.
    Adapter, index and tag regions are discarded.
    """

    # Average number of base pairs each resulting library will have.
    library_size_mean : float

    # Standard deviation in the length of resulting OpenReadFrames.
    library_size_sd : float

    # Method used when reading fragments.
    read_method : ReadMethod

    # Read length of each fragment.
    read_length : int

    # Average coverage of each base. Influences amount of frames that are returned.
    average_coverage : float

    # Beta scale parameter for the Exponential Distribution that determines the error rate.
    # It is the inverse of the rate parameter Lambda. (Beta = 1 / Lambda)
    # Higher values mean higher error rates.
    call_error_beta : float



    def __init__ (
        self,
        library_size_mean:float = 400,
        library_size_sd:float = 75,
        read_method:ReadMethod = 'single-read',
        read_length:int = 150,
        average_coverage:float = 10,
        call_error_beta:float = 2.85, # Estimation based on MiSeq graph in the TechNotes.
        name:Optional[str] = None,
        seed:Optional[int] = None
    ) :
        super().__init__( name=coalesce(name,"ShotgunSequencer"), seed=seed )
        self.library_size_mean = library_size_mean
        self.library_size_sd = library_size_sd
        self.read_method = read_method
        self.read_length = read_length
        self.average_coverage = average_coverage
        self.call_error_beta = call_error_beta



    def apply ( self, genome:Seq ) -> List[Scaffold] :
        rnd = self.make_generator()

        # Go for the approach of selecting randomly a start point and length, then check if it is
        # possible. Repeat if not.
        out_scaffolds:List[Scaffold] = []
        num_scaffolds = self.calc_num_scaffolds_obtained(genome)
        while len(out_scaffolds) < num_scaffolds :
            start = rnd.pick_integer( low=0, high=len(genome) ) # [low,high)
            library_size = int(rnd.pick_normal( self.library_size_mean, self.library_size_sd ))
            end = start + library_size

            # Repeat this sampling if it surpassed the possible range.
            if not ( end <= len(genome) ) :
                continue

            # Define the extracted the slices. R2 is only extracted if paired-end method is used.
            name_t = "BioLabSim.Sequencer id={} dir={}"
            r1 = RatedSequence( name=name_t.format(len(out_scaffolds),"R1"), seq=None, qual=None )
            r2 = RatedSequence( name=name_t.format(len(out_scaffolds),"R2"), seq=None, qual=None )

            # Extract the slices.
            read_size = min( library_size, self.read_length ) # Read is restricted by max read length.
            r1.seq = list(genome[ start : start + read_size ])
            if self.read_method == 'paired-end' :
                r2.seq = list(reversed(genome[ end - read_size : end ])) # R2 read in opposite direction

            # Apply quality estimation and substitution errors for both.
            for r in [ r1, r2 ] : # Same code for both possible reads.
                if r.seq != None : # Catch non-existent R2 reads.
                    r.qual = [ PHRED_MAX for _ in range(len(r.seq)) ] # Initialize quality.
                    for i in range(len(r.seq)) :

                        # Sample the error rate for each position from the exponential distribution.
                        q_error = rnd.pick_exponential( self.call_error_beta )
                        q_error = min( q_error, PHRED_MAX )
                        r.qual[i] = round(PHRED_MAX - q_error)

                        # Given the Phred Score, there is a probability a substitution error occurs.
                        if rnd.pick_uniform(0,1) < phred_to_prob(r.qual[i]) :
                            err_bases = [ b for b in ['A','C','G','T'] if b != r.seq[i] ]
                                # Always substitute for a base that is not the original.
                            sel_base = err_bases[ rnd.pick_integer(0,len(err_bases)) ]
                            r.seq[i] = sel_base

            # Create the scaffold and add it to the final results.
            cur_scaffold = Scaffold(
                expected_len=self.library_size_mean, # use `library_size` to store actual values
                r1_seqrecord=convert_to_seqrecord(r1),
                r2_seqrecord=convert_to_seqrecord(r2) if r2.seq is not None else None,
            )
            out_scaffolds.append(cur_scaffold)

        return out_scaffolds



    def calc_num_scaffolds_obtained ( self, genome:Seq ) -> int :
        """
        Number of frames that will be obtained. Calculated based on Lander/Waterman equation with
        regards to the current parameters and the host genome.
        This will account for the read method used. Paired-end methods will output half of it.
        """
        method_factor = 2 if self.read_method == 'paired-end' else 1
        return round(
            ( len(genome) * self.average_coverage )
            /
            ( self.read_length * method_factor )
        )



def phred_to_prob ( phred_score:float ) -> float :
    """ Convert a Phred score into a probability. """
    return 10 ** ( -phred_score / 10 )



def convert_to_seqrecord ( rated_seq:RatedSequence ) -> SeqRecord :
    """
    Convert an internal RatedSequence to the Seq that is commonly used throughout the library.
    """
    return SeqRecord(
        Seq( "".join(rated_seq.seq) ),
        id=rated_seq.name,
        name=rated_seq.name,
        description="",
        letter_annotations={ 'phred_quality': rated_seq.qual }
    )
