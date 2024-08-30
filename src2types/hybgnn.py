
import math
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv
from layers import AttentionModule, TenorNetworkModule
from utils import process_pair, calculate_loss, calculate_normalized_ged
# from utils import prec_at_ks
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score

# Using GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(
    f"#################### Your GPU usage is {torch.cuda.is_available()}! ########################\n")

LABELS = {2:0,8:1}

class HybGNN(nn.Module):
    def __init__(self, args):
        super(HybGNN, self).__init__()
        self.args = args

        # Node feature embedding layers
        self.embed_layers = nn.Sequential(
            nn.Linear(15, 15*32),
            nn.ReLU(),
            nn.Linear(15*32, 15*128),
            nn.ReLU(),
            nn.Linear(15*128, 15*self.args.embed_dim)
        )
        
        # GCN layers
        self.conv1 = GCNConv(self.args.embed_dim, self.args.filters_1)
        self.conv2 = GCNConv(self.args.filters_1, self.args.filters_2)

        # Attention module
        self.attention = AttentionModule(args)

        # Final classification layer
        self.fc = nn.Linear(self.args.filters_2, 3)  # 3-class classification
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, data):
        x, edge_index, target = data['features_1'], data['edge_index_1'], data['target']

        # Embed node features
        x = self.embed_layers(x)
        x = x.view(15, -1)
        # print(f"Node feature embedding shape: {x.shape}")

        # Apply GCN layers
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        # print(f"GCN output shape: {x.shape}")
        
        # Apply attention
        x = self.attention(x).squeeze(1)
        
        # print(f"Attention output shape: {x.shape}")

        # Classification
        logits = self.fc(x)
        
        # Calculate loss
        loss = self.criterion(logits, target.argmax(dim=0))
        
        # Calculate prediction probabilities
        predictions = F.softmax(logits, dim=0)

        return loss, predictions

class HybGNNTrainer(object):

    def __init__(self, args):

        self.args = args
        self.initial_label_enumeration()
        self.setup_model()

    def setup_model(self):

        self.model = HybGNN(self.args).to(device)

    def initial_label_enumeration(self):
        print("\nEnumerating unique labels.\n")
        self.training_graphs = glob.glob(self.args.training_graphs + "*.json")
        self.testing_graphs = glob.glob(self.args.testing_graphs + "*.json")
        
        
        print("Training graphs:", len(self.training_graphs))
        print("Testing graphs:", len(self.testing_graphs))
        
        # graph_pairs = self.training_graphs + self.testing_graphs

        # self.beforesorted_train_labels = set()
        # self.aftersorted_train_labels = set()

        # for training_graph in tqdm(self.training_graphs):
        #     traindata = process_pair(training_graph)
        #     self.beforesorted_train_labels = self.beforesorted_train_labels.union(
        #         set(traindata["labels_1"]))

        # self.aftersorted_train_labels = sorted(self.beforesorted_train_labels)
        # self.aftersorted_train_labels = {
        #     val: index for index, val in enumerate(self.aftersorted_train_labels)}
        # self.train_number_of_labels = len(self.aftersorted_train_labels)

        # self.number_of_labels = self.train_number_of_labels + 15

        # print("number_of_labels", self.number_of_labels)

    def create_batches(self, train=True):
        
        if train:
            graphs = self.training_graphs
        else:
            graphs = self.testing_graphs

        random.shuffle(graphs)
        batches = []
        for graph in range(0, len(graphs), self.args.batch_size):
            batches.append(
                graphs[graph:graph+self.args.batch_size])
        return batches

    def transfer_testset_to_torch(self, testdata):
        
        edges_1 = testdata["graph_1"] + [[y, x] for x, y in testdata["graph_1"]]
        edges_1 = torch.from_numpy(np.array(edges_1, dtype=np.int64).T).type(torch.long).to(device)
        
        features_1 = testdata["labels_1"]
        features_1 = torch.tensor(np.array(features_1), dtype=torch.float32).to(device)
        features_1 = torch.nn.functional.normalize(features_1, p=2, dim=0).to(device)
        
        labels_1 = LABELS[testdata["ged"]]
        labels_1 = torch.tensor(labels_1).to(device)
        labels_1 = torch.nn.functional.one_hot(labels_1, num_classes=3).float().to(device)
        
        # print(f"input dim: edges_1: {edges_1.shape}, features_1: {features_1.shape}, labels_1: {labels_1.shape}")
        
        new_testdata = dict()
        new_testdata["edge_index_1"] = edges_1
        new_testdata["features_1"] = features_1
        new_testdata["target"] = labels_1

        return new_testdata

    def transfer_trainset_to_torch(self, traindata):
        edges_1 = traindata["graph_1"] + [[y, x] for x, y in traindata["graph_1"]]
        edges_1 = np.array(edges_1, dtype=np.int64).T
        edges_1 = torch.tensor(edges_1, dtype=torch.long).to(device)

        features_1 = traindata["labels_1"]
        features_1 = torch.tensor(np.array(features_1), dtype=torch.float32).to(device)
        features_1 = torch.nn.functional.normalize(features_1, p=2, dim=0).to(device)
        
        labels_1 = LABELS[traindata["ged"]]
        labels_1 = torch.tensor(labels_1).to(device)
        labels_1 = torch.nn.functional.one_hot(labels_1, num_classes=3).float().to(device)
        
        # print(f"input dim: edges_1: {edges_1.shape}, features_1: {features_1.shape}, labels_1: {labels_1.shape}")
        
        new_traindata = dict()
        new_traindata["edge_index_1"] = edges_1
        new_traindata["features_1"] = features_1
        new_traindata["target"] = labels_1
        
        return new_traindata

    def process_batch(self, batch):
        self.optimizer.zero_grad()
        losses = 0
        for training_graph in batch:
            traindata = process_pair(training_graph)
            traindata = self.transfer_trainset_to_torch(traindata)
            target = traindata["target"]

            loss, score = self.model(traindata)
            
            # print("Prediction:", prediction)
            # print("Ground truth:", target)

            losses = losses + loss
        
        losses.backward(retain_graph=True)
        self.optimizer.step()
        loss = losses.item()
        return loss

    def fit(self):

        print("\nModel training.\n")

        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)

        self.model.train().to(device)
        epochs = trange(self.args.epochs, leave=True, desc="Epoch")
        for epoch in epochs:
            batches = self.create_batches(train=True)
            self.loss_sum = 0
            main_index = 0
            for index, batch in tqdm(enumerate(batches), total=len(batches), desc="Batches"):
                loss_score = self.process_batch(batch)
                main_index = main_index + len(batch)
                self.loss_sum = self.loss_sum + loss_score * len(batch)
                loss = self.loss_sum/main_index
                epochs.set_description("Epoch (Loss=%g)" % round(loss, 5))
              
    def score(self):  
        
        print("\nModel testing.\n")
        
        self.model.eval().to(device)
        
        batches = self.create_batches(train=False)
        predictions = []
        true_labels = []
        
        for index, batch in tqdm(enumerate(batches), total=len(batches), desc="Batches"):
            for testing_graph in batch:
                testdata = process_pair(testing_graph)
                testdata = self.transfer_testset_to_torch(testdata)
                _,prediction = self.model(testdata)
                predictions.append(prediction.cpu().detach().numpy())
                true_labels.append(testdata['target'].cpu().detach().numpy())
        
        # Convert predictions and true_labels to numpy arrays
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        
        print(f"predictions shape: {predictions.shape}")
        print(f"true labels shape: {true_labels.shape}")
        
        true_labels_class = np.argmax(true_labels, axis=1)
        predictions_class = np.argmax(predictions, axis=1)
        
        # Calculate accuracy
        acc = accuracy_score(true_labels_class, predictions_class)
        print(f"Accuracy: {acc:.4f}")

        # Calculate F1 score, precision, and recall
        f1 = f1_score(true_labels_class, predictions_class, average='weighted', zero_division=0)
        precision = precision_score(true_labels_class, predictions_class, average='weighted', zero_division=0)
        recall = recall_score(true_labels_class, predictions_class, average='weighted', zero_division=0)
        
        # Print F1 score, precision, and recall
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")

        # Check if there is more than one unique class in true_labels
        if len(np.unique(true_labels_class)) > 1:
            roc_auc = roc_auc_score(true_labels, predictions, multi_class='ovr')
            print(f"ROC AUC: {roc_auc:.4f}")
        else:
            print("Only one class present in true labels. Skipping ROC AUC calculation.")

    

    # def score(self):

    #     print("\n\nModel evaluation.\n")
    #     self.model.to(device)
    #     self.model.eval()
    #     self.scores = []
    #     self.ground_truth = []
    #     self.predictions = []
    #     self.precision = []
    #     self.hyb = []
    #     self.nohyb = []
    #     self.trueTargets = []
    #     self.predTargets = []

    #     for graph_pair in tqdm(self.testing_graphs):
    #         testdata = process_pair(graph_pair)
    #         testdata = self.transfer_testset_to_torch(testdata)
    #         prediction = self.model(testdata).to(device).squeeze(0)
    #         target = testdata['target']
            
    #         self.ground_truth.append(testdata['target'])
    #         self.predictions.append(prediction)
    #         self.scores.append(torch.nn.functional.mse_loss(prediction, target).item())
    #         self.trueTargets.append(torch.argmax(target).item())
    #         self.predTargets.append(torch.argmax(prediction).item())

    #         print("Prediction:", torch.argmax(prediction).item())
    #         print("Ground truth:", torch.argmax(target).item())
            
    #         print("Pred prob:", prediction)

    #         if torch.argmax(prediction).item() == torch.argmax(target).item():
    #             self.precision.append(1)
    #         else:
    #             self.precision.append(0)
    #         if torch.argmax(prediction).item() == 0:
    #             self.hyb.append(1)
    #         if torch.argmax(prediction).item() == 1:
    #             self.nohyb.append(1)

    #     print("\nAccuracy:", np.mean(self.precision))
    #     print("Hybrid_count:", len(self.hyb))
    #     print("Non-hybrid_count:", len(self.nohyb), "\n")

    #     # TODO: calculate rho, tau
    #     # self.predictions_array = np.array(self.predictions)
    #     # self.ground_truth_array = np.array(self.ground_truth)

    #     # self.coef_rho, self.p_rho = spearmanr(self.predictions_array, self.ground_truth_array)
    #     # self.coef_tau, self.p_tau = kendalltau(self.predictions_array, self.ground_truth_array)

    #     # print(f"Spearman's rho: {round(self.coef_rho,5)}, p-value: {round(self.p_rho,5)}")
    #     # print(f"Kendall's tau: {round(self.coef_tau,5)}, p-value: {round(self.p_tau,5)}")
    #     # self.print_evaluation()
    #     # self.print_common_metrics()

    def print_evaluation(self):
        norm_ged_mean = np.mean(self.ground_truth)
        base_error = np.mean([(n-norm_ged_mean)**2 for n in self.ground_truth])
        model_error = np.mean(self.scores)
        print("\nBaseline error: " + str(round(base_error, 5))+".")
        print("\nModel test error: " + str(round(model_error, 5))+".")

    def print_common_metrics(self):
        self.true_targets = ["no-Hybrid" if x ==
                             8.0 else "Hybrid" for x in self.trueTargets]
        self.pred_targets = ["no-Hybrid" if x ==
                             8.0 else "Hybrid" for x in self.predTargets]
        self.true_targets_num = [
            0 if x == 8.0 else 1 for x in self.trueTargets]
        self.pred_targets_num = [
            0 if x == 8.0 else 1 for x in self.predTargets]

        self.accuracy = accuracy_score(self.true_targets, self.pred_targets)
        self.precision = precision_score(
            self.true_targets, self.pred_targets, pos_label="Hybrid", average='binary')
        self.recall = recall_score(
            self.true_targets, self.pred_targets, pos_label="Hybrid", average='binary')
        self.f1 = f1_score(self.true_targets, self.pred_targets,
                           pos_label="Hybrid", average='binary')
        self.macro_f1 = f1_score(
            self.true_targets, self.pred_targets, average='macro')
        self.weighted_f1 = f1_score(
            self.true_targets, self.pred_targets, average='weighted')
        self.roc_auc = roc_auc_score(
            self.true_targets_num, self.pred_targets_num, average='weighted')
        self.conf_matrix = confusion_matrix(
            self.true_targets, self.pred_targets, labels=["Hybrid", "no-Hybrid"])
        conf_matrix_df = pd.DataFrame(self.conf_matrix, index=[
                                      "Hybrid", "no-Hybrid"], columns=["Predicted Hybrid", "Predicted no-Hybrid"])
        report = classification_report(
            self.true_targets, self.pred_targets, target_names=["Hybrid", "no-Hybrid"])

        print(f"\nAccuracy: {round(self.accuracy,5)}.")
        print(f"Precision: {round(self.precision,5)}.")
        print(f"Recall: {round(self.recall,5)}.")
        print(f"F1 score: {round(self.f1,5)}.")
        print(f"Macro F1 score: {round(self.macro_f1,5)}.")
        print(f"Weighted F1 score: {round(self.weighted_f1,5)}.")
        print(f"ROC-AUC score: {round(self.roc_auc,5)}.\n")
        print("Confusion Matrix:")
        print(conf_matrix_df)
        print("\nClassification Report:")
        print(report)

    def save(self):
        torch.save(self.model.state_dict(), self.args.save_path)

    def load(self):
        self.model.load_state_dict(torch.load(
            self.args.load_path, weights_only=True))
