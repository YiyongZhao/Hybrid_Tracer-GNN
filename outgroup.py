import itertools
import os
import numpy as np
import json
import subprocess

K=3
D=3
N=K*2
 
edge_1 = [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [0, 14], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9], [1, 10], [1, 11], [1, 12], [1, 13], [1, 14], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9], [2, 10], [2, 11], [2, 12], [2, 13], [2, 14], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9], [3, 10], [3, 11], [3, 12], [3, 13], [3, 14], [4, 5], [4, 6], [4, 7], [4, 8], [4, 9], [4, 10], [4, 11], [4, 12], [4, 13], [4, 14], [5, 6], [5, 7], [5, 8], [5, 9], [5, 10], [5, 11], [5, 12], [5, 13], [5, 14], [6, 7], [6, 8], [6, 9], [6, 10], [6, 11], [6, 12], [6, 13], [6, 14], [7, 8], [7, 9], [7, 10], [7, 11], [7, 12], [7, 13], [7, 14], [8, 9], [8, 10], [8, 11], [8, 12], [8, 13], [8, 14], [9, 10], [9, 11], [9, 12], [9, 13], [9, 14], [10, 11], [10, 12], [10, 13], [10, 14], [11, 12], [11, 13], [11, 14], [12, 13], [12, 14], [13, 14]]

motif1=str("A")



def chunk_gen_json_GNN(input_file, output_dir, logname, log_dir):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    num_chunks = len(lines[0]) // 100000
    
    for i in range(num_chunks):
        start = i * 100000
        end = start + 100000
        
        output_file = os.path.join(output_dir, f'chunk_{i+1}.txt')
        output_json_file = os.path.join(output_dir, f'{logname}_{i+1}.json')
        with open(output_file, 'w') as f:
            for line in lines:
                f.write(line[start:end] + '\n')


        lines_json=[]    
        with open(output_file, 'r') as file:

            for line in file:
                lines_json.append(line.strip())
                if len(lines_json) == 4:
                    spe1 = [0]
                    spe2 = [1]
                    spe3 = [2]
                    spe4 = [3]


                    aaaa, aaad, aaca, aacc, aabd, abaa, abab, abad, abba, abbb, abbd, abca, abcb, abcc, abcd=0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
               
                
                    for i in range(0, len(lines_json[0].strip())-(2*K+D)+1):

                        feature_spe1, feature_spe2, feature_spe3, feature_spe4 = [], [], [], []

                        for n in range(1,5):

                            spen = eval("spe"+ str(n))
                            feature_spen = eval("feature_spe"+ str(n))

                            for m in range(len(spen)):
                                kmer1d4 = (lines_json[spen[m]].strip()[i:i+K])
                                kmer2d4 = (lines_json[spen[m]].strip()[(i+K+D):(i+2*K+D)]) 
                                kmerNd4 = str(kmer1d4) + str(kmer2d4) 
                                kmerNd10 = int(kmerNd4, 4)
                                if kmerNd10 not in feature_spen:                            
                                    feature_spen.append(kmerNd10)


 
                    

                        set_1 = set(feature_spe1)
                        set_2 = set(feature_spe2)
                        set_3 = set(feature_spe3)
                        set_4 = set(feature_spe4)
                        if len(set_1.intersection(set_2)) > 0:
                            motif2 = str("A")
                        else:
                            motif2 = str("B")


                        if len(set_1.intersection(set_3)) > 0:
                            motif3 = str("A")
                        elif len(set_2.intersection(set_3)) > 0:
                            motif3 = str("B")
                        else:
                            motif3 = str("C")


                        if len(set_1.intersection(set_4)) > 0:
                            motif4 = str("A")
                        elif len(set_2.intersection(set_4)) > 0:
                            motif4 = str("B")
                        elif len(set_3.intersection(set_4)) > 0:
                            motif4 = str("C")
                        else:
                            motif4 = str("D")
                        motif = motif1 + motif2 + motif3 + motif4
                        if motif == "AAAA":
                            aaaa+=1
                
                        if motif == "AAAD":
                            aaad+=1
                
                        if motif == "AACA":
                            aaca+=1
                
                        if motif == "AACC":
                            aacc+=1
                
                        if motif == "AABD":
                            aabd+=1
                
                        if motif == "ABAA":
                            abaa+=1
                
                        if motif == "ABAB":
                            abab+=1
                
                        if motif == "ABAD":
                            abad+=1
                
                        if motif == "ABBA":
                            abba+=1
                
                        if motif == "ABBB":
                            abbb+=1
                
                        if motif == "ABBD":
                            abbd+=1
                
                        if motif == "ABCA":
                            abca+=1
                
                        if motif == "ABCB":
                            abcb+=1
                
                        if motif == "ABCC":
                            abcc+=1
                
                        if motif == "ABCD":
                            abcd+=1
                                 
                
                    feature_1=[aaaa, aaad, aaca, aacc, aabd, abaa, abab, abad, abba, abbb, abbd, abca, abcb, abcc, abcd]

                    data = {"graph_1": edge_1,"ged": 2, "labels_1": feature_1}
                
                    with open(output_json_file, 'w') as f:
                        json.dump(data, f)

    print("GNN ing")

    command = f"python3 ./src4types/main.py --load-path ./4typesreal-4000-initial-001-his16-e10-32-512abs.pt --learning-rate 0.001 --dropout 0.8 --epochs 10 --batch-size 512 --filters-1 128 --filters-2 64 --filters-3 32 --histogram --bins 16 --training-graphs ./dataset/train/ --testing-graphs {output_dir}/ >{log_dir}/{logname}.log"
    os.system(command)
    print(f'End of generating log {logname}')




phy_folder = 'phy'


phy_files = [f for f in os.listdir(phy_folder) if f.endswith('.phy')]


for phy_file in phy_files:

    phy_file_path = os.path.join(phy_folder, phy_file)


    with open(phy_file_path, 'r') as file:
        lines = file.readlines()
        outgroup = lines[0]
        lines = lines[1:]

    # Generate all possible permutations of four elements
    line_permutations = itertools.permutations(lines, 3)


    output_folder = 'output_files'
    os.makedirs(output_folder, exist_ok=True)


    for index, permutation in enumerate(line_permutations):
        file_name_parts = [outgroup.split()[0]] + ([line.split()[0] for line in permutation])  # Using the first part of each line as part of a filename

        file_name = '_'.join(file_name_parts)
        file_path = os.path.join(output_folder, f'{file_name}.txt')


        with open(file_path, 'w') as output_file:
            output_file.writelines(outgroup)
            output_file.writelines(permutation)

        print(f'Generated file: {file_name}.txt')




        with open(file_path, 'r') as file:
            lines = [list(line.split('\t')[1]) for line in file]
        # 处理文件中的每一列
        for i in range(len(lines[0])):
            if any(lines[n][i] not in "atcgATCG" for n in range(4)):
                for n in range(4):
                    lines[n][i] = "-"

        # 将列表转换回字符串
        lines = [''.join(line) for line in lines]

        # 写入到新文件中
        with open(file_path, 'w') as f:
            for line in lines:
                f.write(line + '\n')

 
        # Process each column in the file.
        processed_lines = []
        for line in lines:
            processed_line = []
            for char in line.strip():
                if char in "atcguATCGU":
                    processed_line.append(char.upper())
                else:
                    processed_line.append("-")
            processed_lines.append("".join(processed_line))
        
        # Write the processed content to a new file (overwrite the original file)
        with open(file_path, 'w') as f:
            f.writelines("\n".join(processed_lines))
        

        with open(file_path, 'r+') as f:
            text = f.read()
            text = text.replace("-", "")
            text = text.replace("a", "0")
            text = text.replace("t", "1")
            text = text.replace("c", "2")
            text = text.replace("g", "3")
            text = text.replace("A", "0")
            text = text.replace("T", "1")
            text = text.replace("C", "2")
            text = text.replace("G", "3")
            f.seek(0)
            f.write(text)
            f.truncate()

        print(f"file {file_name} completed.")
        output_json_dir = os.path.join(output_folder, f'{file_name}')

        if not os.path.exists(output_json_dir):
            os.makedirs(output_json_dir)
    
        chunk_gen_json_GNN(file_path, output_json_dir,file_name, output_folder)
        
        log_path = os.path.join(output_folder, f'{file_name}.log')


        with open(log_path, 'r') as f:
            for line in f:
                if line.startswith('ACC:'):
                    try:
                        # Retrieve the numbers following 'ACC:' and convert them to floating-point numbers
                        acc_value = float(line.split('ACC:')[1].strip())
                        print(f"{file_name} HYB Possibility: {acc_value}")
                        if acc_value >= 0.6:
                            print(f"""
  {file_name_parts[0]}  {file_name_parts[1]}  {file_name_parts[2]}  {file_name_parts[3]}
     |      |     |     |
     |      |     |     |
     |      |     |     |
     |      |----/ \----|
     |      |           |
     |      |-----------|
     |            |
     |            |
     |------------|
            |
                            """)
                            # Using GNN to get Gamma
                            command = f"sudo python3 srcgamma/main.py --load-path ./gamma-001-his16-e10-32-512abs.pt  --learning-rate 0.001 --dropout 0.8 --epochs 10 --batch-size 512  --filters-1 128 --filters-2 64 --filters-3 32 --histogram --bins 16 --training-graphs ./dataset/train/ --testing-graphs {output_json_dir}/ >{output_folder}/{file_name}_gamma.log"
                            os.system(command)
                            gamma_path = os.path.join(output_folder, f'{file_name}_gamma.log')
                            with open(gamma_path, 'r') as f:
                                for line in f:
                                    if line.startswith('Gamma:'):

                                        gamma_value = float(line.split('Gamma:')[1].strip())
                                print(f"Hyb group {file_name} Gamma : {gamma_value}.")

                    except Exception as e:
                        print(f"An error occurred: {e}") 
        #Remove tmp txt  
#        os.system(f"rm -f {file_path}")  
