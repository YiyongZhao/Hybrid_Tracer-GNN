import argparse
import numpy as np
import torch
import json
import os
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from tqdm import tqdm
import pandas as pd
import time

start_time = time.time()

# Using GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(
    f"#################### Your GPU usage is {torch.cuda.is_available()}! ########################\n")

def parse_args():
    parser = argparse.ArgumentParser(description="Train and test a concatenated CNN model for hybridization detection")
    parser.add_argument('--train_folder', type=str, default='./train', help="Folder containing training data files")
    parser.add_argument('--test_folder', type=str, default='./test', help="Folder containing test data files")
    parser.add_argument('--epochs', type=int, default=3000, help="Number of epochs during model training.")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for model training")
    parser.add_argument('--batch_size', type=int, default=512, help="Batch size during model training")
    parser.add_argument('--maxloss', type=float, default=0.001, help="Maximum average loss for early stopping")
    parser.add_argument('--model_save_name', type=str, default='concatenated_CNN_model.pth', help = "File name of saved trained model")
    parser.add_argument('--loss_save_name', type=str, default='loss_history.json', help = "File name of saved loss history")
    return parser.parse_args()

args = parse_args()
os.makedirs("./model", exist_ok= True)
os.makedirs("./loss_history", exist_ok= True)

train_folder = args.train_folder
test_folder = args.test_folder
epochs = args.epochs
batch_size = args.batch_size
maxloss = args.maxloss
lr = args.lr
model_save_path = f"./model/{args.model_save_name}"
loss_save_path = f"./loss_history/{args.loss_save_name}"

# Parameters
input_dim = 15+256+75+256
num_classes_1 = 2
num_classes_3 = 27

# Print parameters
print("=" * 40)
print("MODEL PARAMETERS".center(40))
print("=" * 40)
print(f"{'Train Folder:':<20} {train_folder}")
print(f"{'Test Folder:':<20} {test_folder}")
print(f"{'Epochs:':<20} {epochs}")
print(f"{'Batch Size:':<20} {batch_size}")
print(f"{'Maximum Loss:':<20} {maxloss}")
print(f"{'Learning Rate:':<20} {lr}")
print(f"{'Model Save Path:':<20} {model_save_path}")
print(f"{'Loss Save Path:':<20} {loss_save_path}")
print("=" * 40)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(64 * (input_dim // 8), 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3_1 = nn.Linear(64, num_classes_1)
        self.fc3_3 = nn.Linear(64, num_classes_3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool1d(x, 2)
        
        x = x.view(-1, 64 * (input_dim // 8))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x1 = self.fc3_1(x)
        x2 = self.fc3_3(x)
        
        return x1, x2
        # CrossEntropyLoss expect raw logits as input
        # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss

def load_data_from_json(folder_path):
    X = []
    X_labels_1 = []
    X_labels_2 = []
    X_labels_3 = []
    X_labels_8 = []

    y = []
    target3 = []
    file_names = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    onedim = np.concatenate([
                    np.array(data['labels_1']), # 15 summary site patterns
                    np.array(data['labels_2']), # 256 one base site patterns
                    np.array(data['labels_3']).flatten(), # 75 summary kmer site patterns
                    np.array(data['labels_8']).flatten(), # 256 seq kmer patterns
                ])

                    if len(data['labels_1']) == 15:
                        X.append(onedim)
                        X_labels_1.append(data['labels_1'])  # 1x15
                        X_labels_2.append(data['labels_2'])  # 1x256
                        X_labels_3.append(data['labels_3'])  
                        X_labels_8.append(data['labels_8']) 

                        ged_mapping = {2: 1, 8: 0}
                        y.append(ged_mapping.get(data['ged'], -1))  # Default to -1 for unexpected values

                        target3.append(data['target3']) 

                        file_names.append(filename.replace('.json', ''))
                    else:
                        print(f"Warning: Inconsistent shape in file {filename}, skipped.")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file {filename}: {e}")
            except KeyError as e:
                print(f"Key error in file {filename}: {e}")
            except Exception as e:
                print(f"An error occurred in file {filename}: {e}")

    return (np.array(X_labels_1, dtype=np.float32), #1
            np.array(X_labels_2, dtype=np.float32), #2
            np.array(X_labels_3, dtype=np.float32), #3
            np.array(X_labels_8, dtype=np.float32), #4
            np.array(y, dtype=np.int32), #5
            np.array(target3, dtype=np.int32), #6
            file_names, #7
            np.array(X, dtype=np.float32)) #8


# load data
_, _, _, _, y_train_a, target3_train_a, _, X_train = load_data_from_json(train_folder)
_, _, _, _, y_test_a, target3_test, file_names_b, X_test = load_data_from_json(test_folder)


X_train_tensor = torch.tensor(X_train).unsqueeze(1)
y_train_tensor = torch.tensor(y_train_a, dtype=torch.long)
target3_train_tensor = torch.tensor(target3_train_a, dtype=torch.long)

X_test_tensor = torch.tensor(X_test).unsqueeze(1)
y_test_tensor = torch.tensor(y_test_a, dtype=torch.long)
target3_test_tensor = torch.tensor(target3_test, dtype=torch.long)

# Load training data
train_dataset = TensorDataset(X_train_tensor, y_train_tensor, target3_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    
class MultiTargetLoss(nn.Module):
    def __init__(self):
        super(MultiTargetLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, ged_pred, target3_pred, ged_true, target3_true):
        # calculate loss for each instance
        loss_ged = self.cross_entropy(ged_pred, ged_true)
        loss_target3 = self.cross_entropy(target3_pred, target3_true)
        total_loss = loss_ged + loss_target3
        return total_loss

# initialize model, loss function, and optimizer
model = CNNModel().to(device)
criterion = MultiTargetLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# save training loss history
loss_history = []

# train the model
for epoch in range(epochs):
    model.train()
    total_loss = 0
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
        for X_batch, y_batch, target3_batch in train_loader:
            optimizer.zero_grad()

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            target3_batch = target3_batch.to(device)
            ged_pred, target3_pred= model(X_batch)
            
            loss = criterion(ged_pred, target3_pred, y_batch, target3_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            pbar.update(1)

    average_loss = total_loss / len(train_loader)
    loss_history.append(average_loss)
    print(f"Epoch {epoch+1} finished with average loss: {average_loss}, total loss: {total_loss}")

    # check average loss for early stopping
    if average_loss < maxloss:
        print(f"Average loss is below {maxloss}. Stopping training at epoch {epoch+1}.")
        break  # early stopping
    torch.save(model.state_dict(), model_save_path)

# save loss history
with open(loss_save_path, 'w') as f:
    json.dump(loss_history, f)

def evaluate(model, y_true, target3_true, file_names, X):
    model.eval()
    predictions = []
    target3_predictions = []

    with torch.no_grad():
        X_tensor = torch.tensor(X).unsqueeze(1).to(device)
        ged_pred, target3_pred= model(X_tensor)

        ged_prob = F.softmax(ged_pred, dim = 1)
        _, ged_prediction = torch.max(ged_prob, 1)

        target3_prob = F.softmax(target3_pred, dim = 1)
        _, target3_prediction = torch.max(target3_prob, 1)

        predictions.extend(ged_prediction.cpu().numpy())
        target3_predictions.extend(target3_prediction.cpu().numpy())

    # calculate evaluation metrics
    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions, average='weighted', zero_division=0)
    recall = recall_score(y_true, predictions, average='weighted', zero_division=0)
    f1 = f1_score(y_true, predictions, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, predictions)

    target3_accuracy = accuracy_score(target3_true, target3_predictions)
    cm_3 = confusion_matrix(target3_true, target3_predictions)

    return (predictions, y_true, accuracy, precision,
            recall, f1, cm, cm_3, target3_predictions,
            target3_true, target3_accuracy)

predictions, actuals, accuracy, precision, recall, f1, cm, cm_3, target3_predictions, target3_actuals, target3_accuracy= evaluate(model, y_test_a, target3_test, file_names_b, X_test)
unique_target3 = np.unique(list(target3_actuals) + list(target3_predictions))

# print file name, predicted value, actual value, target3 predicted value, target3 actual value
for file_name, prediction, actual, target3_predictions, target3_actuals in zip(file_names_b, predictions, actuals, target3_predictions, target3_actuals):
    print(f'File: {file_name}, hyb Predicted: {prediction}, hyb Actual: {actual}, target3_predictions:{target3_predictions}, target3_actuals:{target3_actuals}')

cm1_df = pd.DataFrame(cm, index=["no-Hybrid", "Hybrid"], columns=["Predicted no-Hybrid", "Predicted Hybrid"])
cm3_df = pd.DataFrame(cm_3, index = unique_target3, columns= unique_target3)

# print evaluation metrics
print(f'\nAccuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print('\nGed Confusion Matrix:')
print(cm1_df)
print(f'\nTarget 3 Accuracy: {target3_accuracy}')
print("\nTarget3 Confusion Matrix")
print(cm3_df)

end_time = time.time()
duration = end_time - start_time

print(f"\nModel Training and Testing time: {duration} seconds")