<p style="text-align: center; font-size: 20px; font-weight: bold;">Hybrid Tracer Concatenated CNN</p>

**Hybrid_Tracer_Concatenated_CNN**: Hybridization Detection using a Concatenated CNN Model and site pattern frequencies.

## Introduction:
This repository contains a Python script that implements a concatenated Convolutional Neural Network (CNN) model to detect hybridization using frequencies of four groups of site patterns: `15_summary_site_patterns`, `256_one_base_site_patterns`, `75_summary_kmer_site_patterns`, and `256_seq_kmer_patterns`. The script trains and evaluates a multi-target CNN model using site pattern frequencies saved in `JSON` files, leveraging PyTorch for deep learning and Scikit-learn for evaluation metrics.

---
## Features
- **Multi-Target Prediction:** The model simultaneously predicts two classification targets: hybridization or not (`ged`) and the arrangement order of hybridization parents and child (`target3`).
- **Early Stopping:** Stops training when the average loss falls below a specified threshold.
- **Evaluation Metrics:** Provides accuracy, precision, recall, F1 score, and confusion matrices for predictions.

---
## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/hybrid_tracer_concatenated_cnn.git # May need to be updated
   cd hybrid_tracer_concatenated_cnn
   ```
2. **Install the required Python packages for Python 3.0+**
    ```bash
    pip install -r requirements.txt
    ```
---
## Usage
1. **Prepare training and testing data**  
    [JSON](https://www.json.org/json-en.html) (JavaScript Object Notation) is a lightweight data-interchange format that is both human-readable and machine-parsable, facilitating straightforward data exchange. The site pattern frequencies of each MSA (Multiple Sequence Alignment) file is stored in JSON format. An example `JSON` file for model training can be found in the `train_example` folder. You can calculate site pattern frequencies from MSA data (in `phylip` format) and save them in `JSON` files using the scripts in the `data_preprocessing` folder. Then place your training data and test data into two different folders.

2. **Run the script with desired configurations**  
    ```bash
    python hybrid_tracer_concatenated_CNN.py
    ```
    **Command-line Arguments**

    | Argument         | Type   | Default                    | Description                                           |
    |------------------|--------|----------------------------|-------------------------------------------------------|
    | `--train_folder` | `str`  | `./train`                  | Path to the training dataset folder.                  |
    | `--test_folder`  | `str`  | `./test`                   | Path to the testing dataset folder.                   |
    | `--epochs`       | `int`  | `3000`                     | Number of training epochs.                            |
    | `--lr`           | `float`| `0.001`                    | Learning rate for the optimizer.                      |
    | `--batch_size`   | `int`  | `512`                      | Batch size for training.                              |
    | `--maxloss`      | `float`| `0.001`                    | Early stopping threshold for average loss.            |
    | `--model_save_name` | `str` | `concatenated_CNN_model.pth` | File name for saving the trained model.             |
    | `--loss_save_name`  | `str` | `loss_history.json`         | File name for saving loss history.                   |

    ---


    **More examples for training models**  
    Training a GNN model for 10000 epochs with a batch size of 1024.
    ```bash
    python hybrid_tracer_concatenated_CNN.py --epochs 10000 --batch_size 1024
    ```

    Training a GNN model for 5000 epochs with a batch size of 1024 and saving the trained model with the name "0109_concat_CNN_1.pth".
    ```bash
    python hybrid_tracer_concatenated_CNN.py --epochs 5000 --batch_size 1024 --model_save_name 0109_concat_CNN_1.pth
    ```

    Training a GNN model for 5000 epochs with a learning rate of 0.00001 and testing the model using `JSON` files in folder `./butterfly`.
    ```bash
    python hybrid_tracer_concatenated_CNN.py --epochs 5000 --lr 0.00001 --test_folder ./butterfly
    ```
3. **Outputs**  
    The model training progress will be displayed in the terminal throughout the training process. After model training, two new folders will be created in the current working directory:
    * `model` folder: Contains the trained model, named as specified by the `--model_save_name` argument.
    * `loss_history` folder: Contains a `JSON` file with the average loss of each epoch, named as specified by the `--loss_save_name` argument.  
    
