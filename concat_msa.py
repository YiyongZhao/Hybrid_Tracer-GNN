# USAGE: python concat_msa.py <dir> - Processes .fas, .fasta, or .fa files and concatenates them into a supermatrix.

import glob
from Bio import SeqIO
import sys
import os

def main():
    if len(sys.argv) != 2:
        print("USAGE: python concat_msa.py <dir>")
        sys.exit(1)

    # Get input directory from command line arguments
    input_dir = sys.argv[1]

    # Get a sorted list of all Fasta files in the input directory
    fas_list = sorted(glob.glob(os.path.join(input_dir, "*.fas")) + glob.glob(os.path.join(input_dir, "*.fa")) + glob.glob(os.path.join(input_dir, "*.fasta")))
    len_dic = {}
    seq_dic = {}
    sample_set = set()

    # Read sequences from each file and store them in dictionaries
    for fas_name in fas_list:
        seq_dic[fas_name] = {}
        for fas in SeqIO.parse(fas_name, 'fasta'):
            sample_name = str(fas.id.strip())
            seq = str(fas.seq.strip())
            seq_dic[fas_name][sample_name] = seq
            sample_set.add(sample_name)
        if fas_name not in len_dic:
            len_dic[fas_name] = len(seq)

    # Sort sample names
    sample_set = sorted(sample_set)
    tmp = []

    # Concatenate sequences for each sample name
    for sample in sample_set:
        concated_seq = "".join(
            seq_dic[key][sample] if sample in seq_dic[key] else "-" * len_dic[key]
            for key in fas_list
        )
        tmp.append(f"{sample}\t{concated_seq}\n")

    # Output the concatenated sequences to a file in the current directory
    sequence_length = len(concated_seq)
    output_filename = f"{len(sample_set)}samples_{len(fas_list)}genes_{sequence_length}bp_concatenate.phy"
    with open(output_filename, "w") as o:
        o.write("".join(tmp))

    print("run successful!")
    print("The concatenated supermatrix with phylip foramt was generated: "+output_filename)

if __name__ == "__main__":
    main()

