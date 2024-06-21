```
###############################################################################################

██╗  ██╗██╗   ██╗██████╗ ██████╗ ██╗██████╗ ████████╗██████╗  █████╗  ██████╗███████╗██████╗        ██████╗ ███╗   ██╗███╗   ██╗
██║  ██║╚██╗ ██╔╝██╔══██╗██╔══██╗██║██╔══██╗╚══██╔══╝██╔══██╗██╔══██╗██╔════╝██╔════╝██╔══██╗      ██╔════╝ ████╗  ██║████╗  ██║
███████║ ╚████╔╝ ██████╔╝██████╔╝██║██║  ██║   ██║   ██████╔╝███████║██║     █████╗  ██████╔╝█████╗██║  ███╗██╔██╗ ██║██╔██╗ ██║
██╔══██║  ╚██╔╝  ██╔══██╗██╔══██╗██║██║  ██║   ██║   ██╔══██╗██╔══██║██║     ██╔══╝  ██╔══██╗╚════╝██║   ██║██║╚██╗██║██║╚██╗██║
██║  ██║   ██║   ██████╔╝██║  ██║██║██████╔╝   ██║   ██║  ██║██║  ██║╚██████╗███████╗██║  ██║      ╚██████╔╝██║ ╚████║██║ ╚████║
╚═╝  ╚═╝   ╚═╝   ╚═════╝ ╚═╝  ╚═╝╚═╝╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚══════╝╚═╝  ╚═╝       ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═══╝                                                                                                                                                                                                                     
HybridTracer-GNN enables inference of hybrid speciation and admixture with fast graph similarity computation neural network.                                                                   
Pypi: https://pypi.org/project/Hybrid_Tracer-GNN                                               
Github: https://github.com/YiyongZhao/Hybrid_Tracer-GNN                                        
Licence: MIT license                                                                     
Release Date: 2024-7                                                                     
Contacts: Xinzheng Du(XXX); Yiyong Zhao(yiyongzhao1991@gmail.com)
                                                                         
###############################################################################################
```
![Version](https://img.shields.io/badge/Version-1.0.0-blue)
[![Documentation Status](http://readthedocs.org/projects/hybridization-detection/badge/?version=latest)](http://hybridization-detection.readthedocs.io)
[![Hybrid_Tracer-GNN Issues](https://img.shields.io/badge/HybridTracer--CNN--Issues-blue)](https://github.com/YiyongZhao/Hybrid_Tracer-GNN/issues)
![Build Status](https://travis-ci.org/YiyongZhao/Hybrid_Tracer-GNN.svg?branch=master)
[![PyPI](https://img.shields.io/pypi/v/Hybrid_Tracer-GNN.svg)](https://pypi.python.org/pypi/Hybrid_Tracer-GNN)


### Introduction
Hybrid_Tracer-GNN enables inference of hybrid speciation and admixture with fast graph similarity computation neural network.
The reference Tensorflow implementation is accessible [[here]](https://github.com/yunshengb/GNN) and another implementation is [[here]](https://github.com/NightlyJourney/GNN).


### Clone and install environment:

```bash
#A convenient one-click installation by using conda (https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html) with the following commands:
git clone https://github.com/YiyongZhao/Hybrid_Tracer-GNN.git
cd Hybrid_Tracer-GNN
conda env create -f environment.yml
conda activate Hybrid_Tracer-GNN

#Alternatively, a convenient one-click installation by using pip (the package installer for Python) with the following commands:
chmod +x install_packages.sh
bash install_package.sh

#Required dependencies:
Python 3.0+
  Python modules:
  networkx          2.4
  tqdm              4.28.1
  numpy             1.15.4
  pandas            0.23.4
  texttable         1.5.0
  scipy             1.1.0
  argparse          1.1.0
  torch             1.1.0
  torch-scatter     1.4.0
  torch-sparse      0.4.3
  torch-cluster     1.4.5
  torch-geometric   1.3.2
  torchvision       0.3.0
  scikit-learn      0.20.0
```
### Install from PyPI with pip:

```bash
pip install HybridTracer-GNN
```

## Usage 
### Datasets for GNN trainning
<p align="justify">
The code takes pairs of graphs for training from an input folder where each pair of graph is stored as a JSON. Pairs of graphs used for testing are also stored as JSON files. Every node id and node label has to be indexed from 0. Keys of dictionaries are stored strings in order to make JSON serialization possible.</p>

Every JSON file has the following key-value structure:

```javascript
{"graph_1": [[0, 1], [1, 2], [2, 3], [3, 4]],
 "labels_1": [2, 2, 2, 2, 2],
 "ged": 1}
```
<p align="justify">
The **graph_1** keys have edge list values which descibe the connectivity structure. Similarly, the **labels_1** keys have labels for each node which are stored as list - positions in the list correspond to node identifiers. The **ged** key has an integer value which is the raw graph edit distance for the pair of graphs.</p>

### Example input format of multiple sequence alignment
```
-----------MSA.fas-----------------------------------------------------------------------------------
>sps1
GAAGTTAGTA-TGA-ACTGATTAGGTTCCTT
>sps2
GAC-TTAGTACTGA-ACTGA--AGGTTCCTT
>sps3
GAC-TTAGT-CTGATACTGATGAGGTTCCTT
>sps4
GAC-TTAGTACTGATAC-ATTAGGTTCCTC
>sps5
GAACTGAGTACTGATACTGATTAGGTTCCTT

```

### Options
<p align="justify">
Training a GNN model is handled by the `src/main.py` script which provides the following command line arguments.</p>

#### Input and output options
```
  --training-graphs   STR    Training graphs folder.      Default is `dataset/train/`.
  --testing-graphs    STR    Testing graphs folder.       Default is `dataset/test/`.
```
#### Model options
```
  --filters-1             INT         Number of filter in 1st GCN layer.       Default is 128.
  --filters-2             INT         Number of filter in 2nd GCN layer.       Default is 64. 
  --filters-3             INT         Number of filter in 3rd GCN layer.       Default is 32.
  --bottle-neck-neurons   INT         Bottle neck layer neurons.               Default is 16.
  --bins                  INT         Number of histogram bins.                Default is 16.
  --batch-size            INT         Number of pairs processed per batch.     Default is 128. 
  --epochs                INT         Number of GNN training epochs.           Default is 5.
  --dropout               FLOAT       Dropout rate.                            Default is 0.5.
  --learning-rate         FLOAT       Learning rate.                           Default is 0.001.
  --weight-decay          FLOAT       Weight decay.                            Default is 10^-5.
  --histogram             BOOL        Include histogram features.              Default is False.
```
### Examples
<p align="justify">
The following commands learn a neural network and score on the test set. Training a GNN model on the default dataset.</p>

```
python src/main.py
```
<p align="center">
<img style="float: center;" src="GNN_run.jpg">
</p>

Training a GNN model for a 100 epochs with a batch size of 512.
```
python src/main.py --epochs 100 --batch-size 512
```
Training a GNN with histogram features.
```
python src/main.py --histogram
```
Training a GNN with histogram features and a large bin number.
```
python src/main.py --histogram --bins 32
```
Increasing the learning rate and the dropout.
```
python src/main.py --learning-rate 0.01 --dropout 0.9
```
You can save the trained model by adding the `--save-path` parameter.
```
python src/main.py --save-path /path/to/model-name
```
Then you can load a pretrained model using the `--load-path` parameter; **note that the model will be used as-is, no training will be performed**.
```
python src/main.py --load-path /path/to/model-name
```


## Bug Reports
You can report bugs or request features through our [GitHub Issues page](https://github.com/YiyongZhao/Hybrid_Tracer-GNN/issues). If you have any questions, suggestions, or issues, please do not hesitate to contact us.

## Contributing
If you're interested in contributing code or reporting bugs, we welcome your ideas and contributions to improve HybridTracer-GNN! Please check out [Contribution Guidelines](https://docs.github.com/en/issues).

## Version History
Check the [Changelog](https://github.com/YiyongZhao/Hybrid_Tracer-GNN/commits/Hybrid_Tracer-GNN_v1.0.0) for details on different versions and updates.

## License
Hybrid_Tracer-GNN  is licensed under the [MIT LICENSE](LICENSE).



