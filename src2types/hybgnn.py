
import math
import glob
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
from tqdm import tqdm, trange
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

class HybGNN(torch.nn.Module):

    def __init__(self, args, embed_dim):

        super(HybGNN, self).__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.setup_layers()

    def calculate_bottleneck_features(self):

        if self.args.histogram == True:
            self.feature_count = self.args.filters_3 + self.args.bins
        else:
            self.feature_count = self.args.filters_3

    def setup_layers(self):

        self.calculate_bottleneck_features()
        self.embedding = torch.nn.Sequential(
            torch.nn.Linear(1, self.embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embed_dim, self.embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embed_dim, self.embed_dim),
            torch.nn.ReLU(),
        )
        self.convolution_1 = GCNConv(
            self.embed_dim, self.args.filters_1).to(device)
        self.convolution_2 = GCNConv(
            self.args.filters_1, self.args.filters_2).to(device)
        self.convolution_3 = GCNConv(
            self.args.filters_2, self.args.filters_3).to(device)
        self.attention = AttentionModule(self.args).to(device)
        self.fully_connected_first = torch.nn.Linear(self.feature_count,
                                                     self.args.bottle_neck_neurons).to(device)
        self.scoring_layer = torch.nn.Linear(
            self.args.bottle_neck_neurons, 3).to(device)

    def calculate_histogram(self, abstract_features_1, abstract_features_2):

        scores = torch.mm(abstract_features_1,
                          abstract_features_2).detach().to(device)
        scores = scores.view(-1, 1).to(device)
        hist = torch.histc(scores, bins=self.args.bins).to(device)
        hist = hist/torch.sum(hist).to(device)
        hist = hist.view(1, -1).to(device)
        return hist

    def convolutional_pass(self, edge_index, features):
        features = self.embedding(features).to(device)
        features = self.convolution_1(features, edge_index).to(device)
        features = torch.nn.functional.relu(features).to(device)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training).to(device)

        features = self.convolution_2(features, edge_index).to(device)
        features = torch.nn.functional.relu(features).to(device)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training).to(device)

        features = self.convolution_3(features, edge_index).to(device)
        return features

    def forward(self, data):

        edge_index_1 = data["edge_index_1"]

        features_1 = data["features_1"].unsqueeze(1)

        abstract_features_1 = self.convolutional_pass(
            edge_index_1, features_1).to(device)

        if self.args.histogram == True:

            hist = self.calculate_histogram(abstract_features_1,
                                            torch.abs(torch.t(abstract_features_1))).to(device)
        pooled_features_1 = self.attention(abstract_features_1).to(device)

        scores = pooled_features_1
        scores = torch.t(scores).to(device)

        if self.args.histogram == True:
            scores = torch.cat((scores, hist), dim=1).view(1, -1).to(device)

        scores = torch.nn.functional.relu(
            self.fully_connected_first(scores)).to(device)
        
        score = torch.softmax(self.scoring_layer(scores),dim=1).to(device)

        return score


class HybGNNTrainer(object):

    def __init__(self, args):

        self.args = args
        self.initial_label_enumeration()
        self.setup_model()

    def setup_model(self):

        self.model = HybGNN(self.args, self.args.embed_dim).to(device)

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

    def create_batches(self):

        random.shuffle(self.training_graphs)
        batches = []
        for graph in range(0, len(self.training_graphs), self.args.batch_size):
            batches.append(
                self.training_graphs[graph:graph+self.args.batch_size])
        return batches

    def transfer_testset_to_torch(self, testdata):
        
        edges_1 = testdata["graph_1"] + [[y, x] for x, y in testdata["graph_1"]]
        edges_1 = torch.from_numpy(np.array(edges_1, dtype=np.int64).T).type(torch.long).to(device)
        
        features_1 = testdata["labels_1"]
        features_1 = torch.FloatTensor(np.array(features_1)).to(device)
        features_1 = features_1/features_1.sum(dim=0, keepdim=True) # normalize features
        
        labels_1 = LABELS[testdata["ged"]]
        labels_1 = torch.tensor(labels_1).to(device)
        labels_1 = torch.nn.functional.one_hot(labels_1, num_classes=3).float().to(device)
        
        print(f"input dim: edges_1: {edges_1.shape}, features_1: {features_1.shape}, labels_1: {labels_1.shape}")
        
        new_testdata = dict()
        new_testdata["edge_index_1"] = edges_1
        new_testdata["features_1"] = features_1
        new_testdata["target"] = labels_1

        return new_testdata

    def transfer_trainset_to_torch(self, traindata):
        edges_1 = traindata["graph_1"] + [[y, x] for x, y in traindata["graph_1"]]
        edges_1 = torch.from_numpy(np.array(edges_1, dtype=np.int64).T).type(torch.long).to(device)

        features_1 = traindata["labels_1"]
        features_1 = torch.FloatTensor(np.array(features_1)).to(device)
        features_1 = features_1/features_1.sum(dim=0, keepdim=True) # normalize features
        
        labels_1 = LABELS[traindata["ged"]]
        labels_1 = torch.tensor(labels_1).to(device)
        labels_1 = torch.nn.functional.one_hot(labels_1, num_classes=3).float().to(device)
        
        print(f"input dim: edges_1: {edges_1.shape}, features_1: {features_1.shape}, labels_1: {labels_1.shape}")
        
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

            prediction = self.model(traindata).to(device)

            losses = losses + \
                torch.nn.functional.mse_loss(
                    traindata["target"], prediction[0]).to(device)  # prediction -> prediction[0] to fix the warning: Using a target size (torch.Size([1, 1])) that is different to the input size (torch.Size([1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.

        losses.backward(retain_graph=True)
        self.optimizer.step()
        loss = losses.item()
        return loss

    def fit(self):

        print("\nModel training.\n")

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)

        self.model.train().to(device)
        epochs = trange(self.args.epochs, leave=True, desc="Epoch")
        for epoch in epochs:
            batches = self.create_batches()
            self.loss_sum = 0
            main_index = 0
            for index, batch in tqdm(enumerate(batches), total=len(batches), desc="Batches"):
                loss_score = self.process_batch(batch)
                main_index = main_index + len(batch)
                self.loss_sum = self.loss_sum + loss_score * len(batch)
                loss = self.loss_sum/main_index
                epochs.set_description("Epoch (Loss=%g)" % round(loss, 5))

    def score(self):

        print("\n\nModel evaluation.\n")
        self.model.to(device)
        self.model.eval()
        self.scores = []
        self.ground_truth = []
        self.predictions = []
        self.precision = []
        self.hyb = []
        self.nohyb = []
        self.trueTargets = []
        self.predTargets = []

        for graph_pair in tqdm(self.testing_graphs):
            testdata = process_pair(graph_pair)
            testdata = self.transfer_testset_to_torch(testdata)
            prediction = self.model(testdata).to(device).squeeze(0)
            target = testdata['target']
            
            self.ground_truth.append(testdata['target'])
            self.predictions.append(prediction)
            self.scores.append(torch.nn.functional.mse_loss(prediction, target).item())
            self.trueTargets.append(torch.argmax(target).item())
            self.predTargets.append(torch.argmax(prediction).item())

            print("Prediction:", torch.argmax(prediction).item())
            print("Ground truth:", torch.argmax(target).item())
            
            print("Pred prob:", prediction)

            if torch.argmax(prediction).item() == torch.argmax(target).item():
                self.precision.append(1)
            else:
                self.precision.append(0)
            if torch.argmax(prediction).item() == 0:
                self.hyb.append(1)
            if torch.argmax(prediction).item() == 1:
                self.nohyb.append(1)

        print("\nAccuracy:", np.mean(self.precision))
        print("Hybrid_count:", len(self.hyb))
        print("Non-hybrid_count:", len(self.nohyb), "\n")

        # TODO: calculate rho, tau
        # self.predictions_array = np.array(self.predictions)
        # self.ground_truth_array = np.array(self.ground_truth)

        # self.coef_rho, self.p_rho = spearmanr(self.predictions_array, self.ground_truth_array)
        # self.coef_tau, self.p_tau = kendalltau(self.predictions_array, self.ground_truth_array)

        # print(f"Spearman's rho: {round(self.coef_rho,5)}, p-value: {round(self.p_rho,5)}")
        # print(f"Kendall's tau: {round(self.coef_tau,5)}, p-value: {round(self.p_tau,5)}")
        # self.print_evaluation()
        # self.print_common_metrics()

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
