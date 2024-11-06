## Introduction

This folder contains scripts for preprocessing MSA sequences to generate input data files (JSON files) for the machine learning model.

### Example Input Format of Multiple Sequence Alignment （MSA）

The [PHYLIP](https://www.phylo.org/index.php/help/phylip) (Phylogeny Inference Package) format is a widely used text format for storing multiple sequence alignments in bioinformatics.

To get started quickly, follow these steps:

1. **Place Your MSA**:
   
   **MSA File**: Place your MSA file with the ".phy" extension in the "phy" folder. Each line in this file should contain a sequence name followed by its corresponding sequence, separated by a "tab" character.

   **Map File**: In the same "phy" folder, create a mapping file for each MSA file. The map file should have the same base name as the ".phy" file, ending with "_imap" (e.g., if your MSA file is "example.phy", name the map file "example_imap"). The map file should include the correspondence between sequence names and species names, where each line lists a sequence name followed by the species name, separated by a "tab" character.
   ```
   -------------------------example_MSA.phy-----------------------------------------------------------------------------------------

   ind1 GAAGTTAGTA-TGA-ACTGATTAGGTTCCTTGAC-TTAGTACTGATAC-ATTAGGTTCCTCTGAC-TTAGTACTGATAC-ATTAGGTTCCTCGAC-TTAGTACTGA-ACTGA--AGGTTCCTTT
   ind2 GAC-TTAGTACTGA-ACTGA--AGGTTCCTTGAC-TTAGTACTGA-ACTGA--AGGTTCCTTGAC-TTAGTACTGATAC-ATTAGGTTCCTCGAC-TTAGTACTGA-ACTGA--AGGTTCCTTT
   ind3 GAC-TTAGT-CTGATACTGATGAGGTTCCTTGAC-TTAGTACTGA-ACTGA--AGGTTCCTTGAC-TTAGTACTGATAC-ATTAGGTTCCTCGAC-TTAGTACTGA-ACTGA--AGGTTCCTTT
   ind4 GAC-TTAGTACTGATAC-ATTAGGTTCCTCGAC-TTAGTACTGA-ACTGA--AGGTTCCTTTGAC-TTAGTACTGATAC-ATTAGGTTCCTCGAC-TTAGTACTGA-ACTGA--AGGTTCCTTT
   ...
   ...
   indN GAACTGAGTACTGATACTGATTAGGTTCCTTGAC-TTAGTACTGA-ACTGA--AGGTTCCTTGAC-TTAGTACTGATAC-ATTAGGTTCCTCGAC-TTAGTACTGA-ACTGA--AGGTTCCTTT
   ```
   ```
   -------------------------example_MSA_imap-----------------------------------------------------------------------------------------

   ind1 sps1
   ind2 sps1
   ind3 sps2
   ind4 sps2
   ...
   ...
   indN spsN
   ```

2. **Format Your MSA**:

   Ensure your ".phy" file is formatted as shown in the following example:

   - For each species, you can concatenate many orthologous coding genes into a supermatrix with the PHYLIP format.
   - Additionally, you can convert a VCF file to a supermatrix in PHYLIP format from genomic DNA data at the individual level after reference genome alignment using aligners such as [GATK](https://gatk.broadinstitute.org/hc/en-us).

3. **Supermatrix Length**:
   Ideally, the length of an input supermatrix alignment should be longer than 50,000 base pairs to provide sufficient power for significant inference, as estimated by previous phylogenetic invariants arising under the coalescent model with hybridization, using [HyDe](https://github.com/pblischak/HyDe).

### Example Supermatrix with PHYLIP File Format
```
-----------example_MSA_with_sps_level.phy-----------------------------------------------------------------------------------------

sps1	GAAGTTAGTA-TGA-ACTGATTAGGTTCCTTGAC-TTAGTACTGATAC-ATTAGGTTCCTCTGAC-TTAGTACTGATAC-ATTAGGTTCCTCGAC-TTAGTACTGA-ACTGA--AGGTTCCTTT
sps2	GAC-TTAGTACTGA-ACTGA--AGGTTCCTTGAC-TTAGTACTGA-ACTGA--AGGTTCCTTGAC-TTAGTACTGATAC-ATTAGGTTCCTCGAC-TTAGTACTGA-ACTGA--AGGTTCCTTT
sps3	GAC-TTAGT-CTGATACTGATGAGGTTCCTTGAC-TTAGTACTGA-ACTGA--AGGTTCCTTGAC-TTAGTACTGATAC-ATTAGGTTCCTCGAC-TTAGTACTGA-ACTGA--AGGTTCCTTT
sps4	GAC-TTAGTACTGATAC-ATTAGGTTCCTCGAC-TTAGTACTGA-ACTGA--AGGTTCCTTTGAC-TTAGTACTGATAC-ATTAGGTTCCTCGAC-TTAGTACTGA-ACTGA--AGGTTCCTTT
...
...
spsN	GAACTGAGTACTGATACTGATTAGGTTCCTTGAC-TTAGTACTGA-ACTGA--AGGTTCCTTGAC-TTAGTACTGATAC-ATTAGGTTCCTCGAC-TTAGTACTGA-ACTGA--AGGTTCCTTT
```
```
-----------example_MSA_with_pop_level.phy---------------------------------------------------------------------------------------------

sps1_pop1	GAAGTTAGTA-TGA-ACTGATTAGGTTCCTTGAC-TTAGTACTGA-ACTGA--AGGTTCCTTGAC-TTAGTACTGATAC-ATTAGGTTCCTCGAC-TTAGTACTGA-ACTGA--AGGTTCCTTT
sps1_pop2	GAC-TTAGTACTGA-ACTGA--AGGTTCCTTGAC-TTAGTACTGATAC-ATTAGGTTTCCTCGAC-TTAGTACTGATAC-ATTAGGTTCCTCGAC-TTAGTACTGA-ACTGA--AGGTTCCTTT
sps3_pop1	GAC-TTAGT-CTGATACTGATGAGGTTCCTTGAC-TTAGTACTGATAC-ATTAGGTTTCCTCGAC-TTAGTACTGATAC-ATTAGGTTCCTCGAC-TTAGTACTGA-ACTGA--AGGTTCCTTT
sps4_pop1	GAC-TTAGTACTGATAC-ATTAGGTTTCCTCGAC-TTAGTACTGATAC-ATTAGGTTTCCTCGAC-TTAGTACAGATAC-ATTAGGTTTCCTCGAC-TTAGTACTGATAC-ATTAGGTTTCCTC
sps4_pop2	GAC-TTAGTACAGATAC-ATTAGGTTTCCTCGAC-TTAGTACTGATAC-ATTAGGTTTCCTCGAC-TTAGTACAGATAC-ATTAGGTTTCCTCGAC-TTAGTACTGATAC-ATTAGGTTTCCTC
sps4_pop3	GAC-TTAGTACTGGTAC-ATTAGGTTTCCTCGAC-TTAGT-CTGATACTGATGAGGTTCCTTGAC-TTAGTACTGAC-TTAGTACAGATAC-ATTAGGTTTCCTCGAC-TTAGTACTGATAC-A
...
...
spsN_pop1	GAACTGAGTACTGATACTGATTAGGTTCCTTGAC-TTAGTACTGATAC-ATTAGGTTTCCTCGAC-TTAGTACAGATAC-ATTAGGTTTCCTCGAC-TTAGTACTGATAC-ATTAGGTTTCCTC

```
Note that the length of sequence names can vary, but each line must be arranged in the format: "sequence name" + "\t" + "sequence". Additionally, each sequence must have the same length.

How to concatenated orthologous multiple sequnce alignet (each sample with single copy gene) into a supermatirx with phylip format, we provdie a python scirpt:  [``concat_msa.py``](https://github.com/YiyongZhao/Hybrid_Tracer-GNN/blob/main/concat_msa.py)

Place your fasta multiple sequence alignment files with extensions .fas, .fa, or .fasta into a directory, e.g., MSA_dir. Run the script as follows:
This script processes `.fas`, `.fasta`, or `.fa` files in a specified directory and concatenates them into a supermatrix in PHYLIP format. The output file is saved in the current working directory.

```
python concat_msa.py MSA_dir
```
The script will generate a concatenated supermatrix in PHYLIP format and save it in the current directory. The output filename will follow the format: <number_of_samples>samples_<number_of_genes>genes_<sequence_length>bp_concatenate.phy.


### Examples
<p align="justify">
To start it quickly, after input your ".phy" files and "_imap" files into "phy" folder, run "msa_sitepattern_counter.py".

```
python seq_processing/msa_sitepattern_counter.py
```
To specify outgroup species:
```
python seq_processing/msa_sitepattern_counter.py --out_name "Heliconius_hecale_felix"
```
To set the number of CPU cores for parallel calculations (default is None, which utilizes all available cores):
```
python seq_processing/msa_sitepattern_counter.py --num_cores 4
```

### Options
<p align="justify">
The preprocessing of MSA sequences is handled by the `seq_processing/msa_sitepattern_counter.py` script which provides the following command line arguments.</p>

```
-h, --help              Show this help message and exit
--phy_folder            (default "phy")-Folder containing .phy files
--kmer_range            (default [3,4,5])-Range of k-mers to analyze
--depth_range           (default [1,2,3])-Range of depths to analyze
--out_name              (default "out")-The name of the outgroup species in the map file
--num_cores             (default None, use all available)-Number of cores used for parallel processing
--output_folder         (default "output_jsons")-Folder for output JSON files, create folder if not exist
```

### Outputs
**JSON files**: The output JSON files that contains the following content will be saved in the specified output folder:
```
{
    "ged": 2 or 8, 
    # The Hybrid (2) or Non-hybrid (8) label of the quartet sequences, determined by program [HyDe](https://github.com/pblischak/HyDe).

    "labels_1": [], 
    # A list of 15 non-negative real numbers for the counts of 15 site patterns, output by program [HyDe](https://github.com/pblischak/HyDe).

    "labels_2": [], 
    # A list of 256 non-negative real numbers representing the counts of 256 site patterns, combining counts for cases with ignored nucleotides and counts for cases with only "ATCGUatcgu" nucleotides.

    "labels_3": [[]], 
    # A list of lists containing counts of 15 k-mer site patterns for each combination of values in "kmer_range" and "depth_range" (combining counts for cases with ignored nucleotides and counts for cases with only "ATCGUatcgu" nucleotides). The total number of count lists is len(kmer_range) * len(depth_range).

    "labels_4": [], 
    # A list of 256 non-negative real numbers representing the counts of 256 site patterns, only counts for cases with ignored nucleotides.

    "labels_5": [], 
    # A list of 256 non-negative real numbers representing the counts of 256 site patterns, only counts for cases with with "ATCGUatcgu" nucleotides.

    "labels_6": [[]], 
    # A list of lists containing counts of 15 k-mer site patterns for each combination of values in "kmer_range" and "depth_range" (only counts for cases with ignored nucleotides). The total number of count lists is len(kmer_range) * len(depth_range).

    "labels_7": [[]], 
    # A list of lists containing counts of 15 k-mer site patterns for each combination of values in "kmer_range" and "depth_range" (only counts for cases with "ATCGUatcgu" nucleotides). The total number of count lists is len(kmer_range) * len(depth_range).

    "labels_8": [],
    # A list of 15 non-negative real numbers for the counts of 15 k-mer site patterns when k = 4 and d = 3.
}
```
**HyDe outputs**: The HyDe outputs from "run_hyde.py" and "individual_hyde.py" will be saved in the current working directory.