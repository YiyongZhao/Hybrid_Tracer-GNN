"""
    _utils.py_
    define some functions for saving the simulated data into different formats
    to be used in different validation programs
"""

# import packages
import msprime
import numpy as np
import matplotlib.pyplot as plt
import tskit
# import IPython.display as display
import os
import sys
import demesdraw
from tabulate import tabulate
import pandas as pd

# utils

# rename fasta files


def change_fasta_ind_names(input_file, output_file, individual_names):
    """
    Change the headers of the FASTA file sequences.

    :param input_file: Path to the input FASTA file.
    :param output_file: Path to the output FASTA file.
    :param individual_names: List of new ind names to replace the old ones.
    """
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        header_index = 0
        for line in infile:
            if line.startswith('>'):
                if header_index < len(individual_names):
                    outfile.write(f">{individual_names[header_index]}\n")
                    header_index += 1
                else:
                    raise ValueError(
                        "More headers provided than sequences found in the FASTA file.")
            else:
                outfile.write(line)

# obtain the phylip format MSA sequence from renamed fasta


def fasta_to_txt(input_fasta, output_txt):
    """
    Convert a FASTA file to a TXT file with each line containing the sequence name and the sequence.

    :param input_fasta: Path to the input FASTA file.
    :param output_txt: Path to the output TXT file.
    """
    with open(input_fasta, 'r') as infile, open(output_txt, 'w') as outfile:
        sequence_name = None
        sequence = []

        for line in infile:
            line = line.strip()
            if line.startswith('>'):
                if sequence_name is not None:

                    outfile.write(f"{sequence_name}\t{''.join(sequence)}\n")
                sequence_name = line[1:]
                sequence = []
            else:
                sequence.append(line)
        if sequence_name is not None:
            outfile.write(f"{sequence_name}\t{''.join(sequence)}\n")

# obtain newick trees - group of gene trees


def write_newick_trees(ts, output_file, node_labels=None):
    """
    Convert all trees in a tree sequence to Newick format and save them to a file.

    :param ts: Tree sequence object.
    :param output_file: Path to the output .nwk file.
    :param node_labels: rename the nodes with individual names
    """
    n = 0
    with open(output_file, 'w') as outfile:
        for i, tree in enumerate(ts.trees()):
            newick_tree = tree.as_newick(node_labels=node_labels)
            outfile.write(f"{newick_tree};\n")
            n += 1
        # print(f"There are a total of {n} trees!")

# reorder the renamed txt file lines so another parent can be tested in HyDe


def reorder_renamed_txt_sequences(input_file, output_file):
    """
    reorder the sequences in the phylip format txt file by moving ind15 to the first line

    :param input_file: Path to the input txt file.
    :param output_file: Path to the output reordered txt file.
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()

    reordered_lines = [lines[-1]] + lines[:-1]

    with open(output_file, 'w') as f:
        f.writelines(reordered_lines)


def obtain_genome_length(vcf_file):
    total_length = 0
    with open(vcf_file, 'r') as file:
        for line in file:
            if line.startswith("##contig"):
                # Extract the length value from the line
                parts = line.split(",")
                for part in parts:
                    if part.startswith("length="):
                        length = int(part.split("=")[1].strip(">\n"))
                        total_length += length
    return total_length


def count_vcf_lines(vcf_file_path):
    with open(vcf_file_path, 'r') as file:
        # Skip the header lines
        for line in file:
            if line.startswith("#"):
                continue
            # Count the remaining lines
            content_lines = sum(1 for _ in file) + 1  # +1 for the current line
            return content_lines
