"""_Hybrid_sim_and_verify.py_

The simulation and data-saving process of hybrid data
If need the follow-up data verification steps using iqtree, Dsuite,and HyDe,
and additional D tests of all possible permutations,
please export and prepare the corresponding data files.

I run this script in Ubuntu Linux environment, please modify the data folder paths,
and program execution command based on your own system paths.

Additional D tests R script is from the 6 years ago version of the tutorial by @mmatschiner:
https://github.com/mmatschiner/tutorials/tree/master/analysis_of_introgression_with_snp_data
https://github.com/palc/tutorials-1/blob/master/analysis_of_introgression_with_snp_data/src/calculate_abba_baba.r

Options:
    '-cu', '--coal_units', <float> "Branch scaling in coalescent units", default = 1.0
    '-ori', '--original_fasta_folder', <string>,"The folder for storing the fasta sequences with the original node names", default=original_fasta_folder
    '-tree', '--iqtree_folder', <string>, The folder for storing the renamed fasta sequences for species tree validation by iqtree, default=iqtree_folder
    '-hyde', '--Hyde_folder', <string>, "The folder for storing the renamed phylip format sequences for hybridization validation by HyDe.", default=Hyde_folder
    '-dsuite', '--Dsuite_folder', <string>, The folder for storing the renamed VCF file for admixture validation by Dsuite, default=Dsuite_folder
    '-c', '--current_folder', <string>, The subfolder for storing this type of simulated data, default=current_folder
    '-n', "--run_nums", <int>,The number of simulations, default=5

Output:
in original_fasta_folder: (1) treesequence.trees file, (2) all_newick_gene_trees.nwk file, (3) node_name_fasta.fasta file
                          (4) simulation_configurations.xlsx
in iqtree_folder: (1) renamed_fasta_file.fasta, (2) iqtree outputs
in Hyde_folder: (1) renamed_phylip_format_seqs.txt, (2) reordered_renamed_phylip_format_seqs.txt, (3) HyDe outputs
in Dsuite folder: (1) vcf_format_data.vcf, (2) Dsuite outputs
in genomics_general folder: (1) additional D test outputs and data files
"""

from simulation_models import hybrid
import numpy as np
import tskit
# from random import shuffle
from sys import argv, exit
import argparse
import pandas as pd
import os
from utils import change_fasta_ind_names, fasta_to_txt, write_newick_trees, reorder_renamed_txt_sequences
# from utils import obtain_genome_length, count_vcf_lines
import subprocess

# the destination folder for saving simulated data
original_fasta_folder = "/mnt/c/Users/linta/Temp/msprime/0907_sims"
iqtree_folder = "/mnt/c/Users/linta/Temp/iqtree-2.3.6-Windows/0907_sims"
Hyde_folder = "/mnt/c/Users/linta/Temp/HyDe/0907_sims"
Dsuite_folder = "/mnt/c/Users/linta/Temp/Dsuite/0907_sims"
genomics_general_folder = "/mnt/c/Users/linta/Temp/genomics_general/0907_sims"
current_folder = '0907_hybrid'

# run the script from the command line
if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Options for Hybrid_simulation_and_verify.py", add_help=True)

    additional = parser.add_argument_group("additional arguments")
    additional.add_argument('-cu', '--coal_units', action="store", type=float, default=1.0,
                            metavar='\b', help="Branch scaling in coalescent units")
    additional.add_argument('-ori', '--original_fasta_folder',  type=str,
                            default=original_fasta_folder, metavar='\b',
                            help="The folder for storing the fasta sequences with the original node names."
                            )
    additional.add_argument('-tree', '--iqtree_folder',  type=str,
                            default=iqtree_folder, metavar='\b',
                            help="The folder for storing the renamed fasta sequences for species tree validation by iqtree."
                            )
    additional.add_argument('-hyde', '--Hyde_folder',  type=str,
                            default=Hyde_folder, metavar='\b',
                            help="The folder for storing the renamed phylip format sequences for hybridization validation by HyDe."
                            )
    additional.add_argument('-dsuite', '--Dsuite_folder',  type=str,
                            default=Dsuite_folder, metavar='\b',
                            help="The folder for storing the renamed VCF file for admixture validation by Dsuite."
                            )
    additional.add_argument('-c', '--current_folder',  type=str,
                            default=current_folder, metavar='\b',
                            help="The subfolder for storing this type of simulated data."
                            )
    additional.add_argument('-n', "--run_nums", type=int,
                            default=5, metavar='\b',
                            help="The number of simulations."
                            )

    args = parser.parse_args()
    cu = args.coal_units
    original_fasta_folder = args.original_fasta_folder
    iqtree_folder = args.iqtree_folder
    Hyde_folder = args.Hyde_folder
    Dsuite_folder = args.Dsuite_folder
    current_folder = args.current_folder
    run_nums = args.run_nums

    print("The current simulation is Hybrid data.", flush=True)

    # define the individual names for renaming the output sequences
    # 16 individuals, population order: Out, P1, P2, P3
    individual_names = ["out0",  # outgroup - 1 samples
                        "ind1", "ind2", "ind3", "ind4", "ind5",  # population 1 - 5 samples
                        "ind6", "ind7", "ind8", "ind9", "ind10",  # popualtion 2 - 5 samples
                        "ind11", "ind12", "ind13", "ind14", "ind15"]  # population 3 - 5 samples
    newick_node_labels = {0: 'out0',
                          1: 'ind1', 2: 'ind2', 3: 'ind3', 4: 'ind4', 5: 'ind5',
                          6: 'ind6', 7: 'ind7', 8: 'ind8', 9: 'ind9', 10: 'ind10',
                          11: 'ind11', 12: 'ind12', 13: 'ind13', 14: 'ind14', 15: 'ind15'}

    # define the name parts of the output files
    cu_part = f"_cu{cu}_"
    df_config = pd.DataFrame(columns=['time_units', 'length', 't1','t2', 't3', 
                                      'recomb_rate', 'mutate_rate', 'gamma_1', 'gamma_2',
                                      'Num_of_samples', 'Num_of_sites', "Num_of_trees", "Run_name"])
    for num in range(run_nums):
        num_part = f"{num+1}"
        try:
            config_list, ts = hybrid(coal_units=cu)
            df_temp = pd.DataFrame([config_list], columns=['time_units', 'length', 't1',
                                                        't2', 't3', 'recomb_rate', 'mutate_rate',
                                                        'gamma_1', 'gamma_2'])

            df_temp['Num_of_samples'] = ts.num_samples
            df_temp['Num_of_sites'] = ts.num_sites
            df_temp["Num_of_trees"] = ts.num_trees
            df_temp["Run_name"] = f"cu{cu}_{num_part}"

            df_config = pd.concat(
                [df_config, df_temp], ignore_index=True)

            # save tree sequences, original name fasta file, newick trees
            output_folder = os.path.join(original_fasta_folder, current_folder)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # 1) save tree sequence to .trees file
            ts.dump(os.path.join(output_folder,
                    f"{current_folder}{cu_part}{num_part}.trees"))
            print('Output tree sequence done.')

            # 2) save original name fasta file
            reference = tskit.random_nucleotides(ts.sequence_length)
            ts.write_fasta(file_or_path=os.path.join(output_folder, f"{current_folder}{cu_part}{num_part}.fasta"),
                        reference_sequence=reference,
                        wrap_width=0)  # no wrap
            print('Output original fasta done.')
            # 3) save newick gene trees
            # write_newick_trees(ts,
            #                os.path.join(
            #                    output_folder, f"{current_folder}{cu_part}{num_part}.nwk"),
            #                node_labels=newick_node_labels)
            #print('Output newick trees done.')
            # 4) rename fasta sequence and save file to the iqtree folder
            output_folder = os.path.join(iqtree_folder, current_folder)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            input_fasta = os.path.join(
                original_fasta_folder, current_folder, f"{current_folder}{cu_part}{num_part}.fasta")
            output_fasta = os.path.join(
                output_folder, f"{current_folder}{cu_part}{num_part}_rena.fasta")
            new_names = individual_names
            change_fasta_ind_names(input_fasta, output_fasta, new_names)
            print('Output renamed fasta done.')

            # 5) convert fasta to phylip format and save to HyDe folder
            output_folder = os.path.join(Hyde_folder, current_folder)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            input_fasta = os.path.join(
                iqtree_folder, current_folder, f"{current_folder}{cu_part}{num_part}_rena.fasta")
            output_txt = os.path.join(
                output_folder, f"{current_folder}{cu_part}{num_part}_rena.txt")
            fasta_to_txt(input_fasta, output_txt)
            print('Output renamed phylip txt done.')

            # 6) change renamed.txt sequence order and save to HyDe folder
            input_renamed_txt = os.path.join(
                output_folder, f"{current_folder}{cu_part}{num_part}_rena.txt")
            output_reordered_renamed_txt = os.path.join(
                output_folder, f"2_{current_folder}{cu_part}{num_part}_rena.txt")
            reorder_renamed_txt_sequences(
                input_renamed_txt, output_reordered_renamed_txt)
            print('Output reordered renamed phylip txt done.')

            # 7) write VCF file to Dsuite folder
            output_folder = os.path.join(Dsuite_folder, current_folder)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            with open(os.path.join(output_folder, f"{current_folder}{cu_part}{num_part}.vcf"), "w") as vcf_file:
                ts.write_vcf(vcf_file,
                            individual_names=individual_names)
            print('Output VCF file done.')

            print(f"Simulation {num+1/run_nums} is done.\n")

            print("#################### Run iqtree ####################\n", flush=True)
            #iqtree_fasta_file = os.path.join(
            #    iqtree_folder,
            #    current_folder, f"{current_folder}{cu_part}{num_part}_rena.fasta")
            #subprocess.run([
            #    "iqtree2",
            #    "-s", iqtree_fasta_file,
            #    "-nt", "4",
            #    "-m", "GTR+G",
            #    "-bb", "1000"
            #])
            #print("\niqtree run done.\n")

            print("#################### Run Dsuite ####################\n", flush=True)
            file_path = os.path.join(Dsuite_folder, current_folder,
                                    f"{current_folder}{cu_part}{num_part}.vcf")
            map_path = "map.txt"
            prefix = os.path.join(Dsuite_folder, current_folder,
                                f"{current_folder}{cu_part}{num_part}")
            command = f"""
            Dsuite Dtrios -o {prefix} {file_path} {map_path}"""
            try:
                subprocess.run(command, shell=True, check=True)
                print("Dtrios run successfully.\n")
            except subprocess.CalledProcessError as e:
                print(f"Dtrios command failed with error: {e}")

            print("Dsuite Done.\n")

            print(
                "############### Run Additional D tests ###############\n", flush=True)
            output_folder = os.path.join(
                genomics_general_folder, current_folder)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            output_geno = os.path.join(output_folder,
                                    f"{current_folder}{cu_part}{num_part}.geno")

            permutations = [
                "spc1=sp1; spc2=sp2; spc3=sp3; spc4=Outgroup",
                "spc1=sp1; spc2=sp3; spc3=sp2; spc4=Outgroup",
                "spc1=sp2; spc2=sp1; spc3=sp3; spc4=Outgroup",
                "spc1=sp2; spc2=sp3; spc3=sp1; spc4=Outgroup",
                "spc1=sp3; spc2=sp1; spc3=sp2; spc4=Outgroup",
                "spc1=sp3; spc2=sp2; spc3=sp1; spc4=Outgroup"
            ]

            geno_feq_file_paths = []
            for p in range(1, 7):
                geno_feq_file_paths.append(os.path.join(output_folder,
                                                        f"{current_folder}{cu_part}{num_part}.pops{p}.tsv.gz"))

            seq_length_content = f"1\t{ts.sequence_length}\n2\t{ts.sequence_length}\nNC_031965\t38372991\nUNPLACED\t141274046\n"
            with open(os.path.join(output_folder, "seq_length.txt"), "w") as file:
                file.write(seq_length_content)
            
            command = f"python3 /mnt/c/Users/linta/Temp/genomics_general/VCF_processing/parseVCF.py -i {file_path} --ploidy 1 -o {output_geno}"
            try:
                subprocess.run(command, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                        print(f"VCF fil conversion failed with error: {e}")

            for c_num, c in enumerate(permutations):
                if c_num == 0:
                    spc1, spc2, spc3, spc4 = "sp1", "sp2", "sp3", "Outgroup"
                elif c_num == 1:
                    spc1, spc2, spc3, spc4 = "sp1", "sp3", "sp2", "Outgroup"
                elif c_num == 2:
                    spc1, spc2, spc3, spc4 = "sp2", "sp1", "sp3", "Outgroup"
                elif c_num == 3:
                    spc1, spc2, spc3, spc4 = "sp2", "sp3", "sp1", "Outgroup"
                elif c_num == 4:
                    spc1, spc2, spc3, spc4 = "sp3", "sp1", "sp2", "Outgroup"
                elif c_num == 5:
                    spc1, spc2, spc3, spc4 = "sp3", "sp2", "sp1", "Outgroup"
                abba_baba_out = os.path.join(output_folder, f"{current_folder}{cu_part}{num_part}_pops{c_num+1}.abba_baba.txt")

                commands = [
                    f"python3 /mnt/c/Users/linta/Temp/genomics_general/freq.py -g {output_geno} -p {spc1} -p {spc2} "
                    f"-p {spc3} -p {spc4} --popsFile {map_path} --target derived --ploidy 1 | grep -v nan | gzip > {geno_feq_file_paths[c_num]}",
                    f"Rscript calculate_abba_baba.r {geno_feq_file_paths[c_num]} {abba_baba_out} "
                    f"{spc1} {spc2} {spc3} {spc4} {os.path.join(output_folder, 'seq_length.txt')}",
                ]
                for command in commands:
                    try:
                        subprocess.run(command, shell=True, check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"Additional D test failed with error: {e}")
            
            print("Additional D tests Done.\n")

            print("#################### Run HyDe ####################\n", flush=True)
            file_path = os.path.join(Hyde_folder, current_folder,
                                    f"{current_folder}{cu_part}{num_part}_rena.txt")
            map_path = "map.txt"
            prefix = os.path.join(Hyde_folder, current_folder,
                                f"{current_folder}{cu_part}{num_part}")
            command = f"/mnt/c/Users/linta/Temp/HyDe/scripts/run_hyde.py -i {file_path} -m {map_path} -o Outgroup -n 16 -t 4 -s {int(ts.sequence_length)} --prefix {prefix}"
            try:
                subprocess.run(command, shell=True, check=True)
                print("HyDe1-run_hyDe run successfully.\n")
            except subprocess.CalledProcessError as e:
                print(f"HyDe1-run_hyDe command failed with error: {e}")
            
            file2 = os.path.join(Hyde_folder, current_folder, f"{current_folder}{cu_part}{num_part}-out-filtered.txt")

            with open(file2, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    command = f"/mnt/c/Users/linta/Temp/HyDe/scripts/individual_hyde.py -tr {file2} -i {file_path} -m {map_path} -o Outgroup -n 16 -t 4 -s {int(ts.sequence_length)} --prefix {prefix}"
                    try:
                        subprocess.run(command, shell=True, check=True)
                        print("HyDe1-individual_hyde run successfully.\n")
                    except subprocess.CalledProcessError as e:
                        print(f"HyDe1-individual_hyde command failed with error: {e}")
                elif len(lines) == 1:
                    print("out-filtered file is empty, skip HyDe1-individual_hyde.\n")
            
            # move sp3 at the first place to run HyDe again
            file_path = os.path.join(Hyde_folder, current_folder,
                                    f"2_{current_folder}{cu_part}{num_part}_rena.txt")
            map_path = "2_map.txt"
            prefix = os.path.join(Hyde_folder, current_folder,
                                f"2_{current_folder}{cu_part}{num_part}")
            command = f"/mnt/c/Users/linta/Temp/HyDe/scripts/run_hyde.py -i {file_path} -m {map_path} -o Outgroup -n 16 -t 4 -s {int(ts.sequence_length)} --prefix {prefix}"
            try:
                subprocess.run(command, shell=True, check=True)
                print("HyDe2-run_hyDe run successfully.\n")
            except subprocess.CalledProcessError as e:
                print(f"HyDe2-run_hyDe command failed with error: {e}")
            
            file2 = os.path.join(Hyde_folder, current_folder, f"2_{current_folder}{cu_part}{num_part}-out-filtered.txt")
            with open(file2, "r") as f:
                lines = f.readlines()
                if len(lines) > 1:
                    command = f"/mnt/c/Users/linta/Temp/HyDe/scripts/individual_hyde.py -tr {file2} -i {file_path} -m {map_path} -o Outgroup -n 16 -t 4 -s {int(ts.sequence_length)} --prefix {prefix}"
                    try:
                        subprocess.run(command, shell=True, check=True)
                        print("HyDe2-individual_hyde run successfully.\n")
                    except subprocess.CalledProcessError as e:
                        print(f"HyDe2-individual_hyde command failed with error: {e}")
                elif len(lines) == 1:
                    print("out-filtered file is empty, skip HyDe2-individual_hyde.\n")

            print("HyDe Done.\n")
        except Exception as e:
            print(f"Simulation {num+1} has an error: {e}")

    # save all the configurations in an excel file
    if not os.path.exists(os.path.join(original_fasta_folder,
                                       current_folder)):
        os.makedirs(os.path.join(original_fasta_folder,
                                 current_folder))
    df_config.to_excel(os.path.join(original_fasta_folder,
                                    current_folder, f"{current_folder}{cu_part}n{run_nums}_sims_config_list.xlsx"), index=False)

    print(f"The simulation of {run_nums} {cu_part} hybrid data is done.\n")
