
import math
import glob
import torch
import random
import numpy as np
from tqdm import tqdm, trange
from torch_geometric.nn import GCNConv
from layers import AttentionModule, TenorNetworkModule
from utils import process_pair, calculate_loss, calculate_normalized_ged

# Using GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class HybGNN(torch.nn.Module):

    def __init__(self, args, number_of_labels):

        super(HybGNN, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()

    def calculate_bottleneck_features(self):

        if self.args.histogram == True:
            self.feature_count = self.args.filters_3 + self.args.bins
        else:
            self.feature_count = self.args.filters_3

    def setup_layers(self):

        self.calculate_bottleneck_features()
        self.convolution_1 = GCNConv(self.number_labels, self.args.filters_1).to(device)
        self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2).to(device)
        self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3).to(device)
        self.attention = AttentionModule(self.args).to(device)
        self.fully_connected_first = torch.nn.Linear(self.feature_count,
                                                     self.args.bottle_neck_neurons).to(device)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1).to(device)

    def calculate_histogram(self, abstract_features_1,abstract_features_2):

        scores = torch.mm(abstract_features_1, abstract_features_2).detach().to(device)
        scores = scores.view(-1, 1).to(device)
        hist = torch.histc(scores, bins=self.args.bins).to(device)
        hist = hist/torch.sum(hist).to(device)
        hist = hist.view(1, -1).to(device)
        return hist

    def convolutional_pass(self, edge_index, features):

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

        features_1 = data["features_1"]


        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1).to(device)


        if self.args.histogram == True:

            hist = self.calculate_histogram(abstract_features_1,
                                            torch.abs(torch.t(abstract_features_1))).to(device)
        pooled_features_1 = self.attention(abstract_features_1).to(device)

        scores = pooled_features_1
        scores = torch.t(scores).to(device)

        if self.args.histogram == True:
            scores = torch.cat((scores, hist), dim=1).view(1, -1).to(device)

        scores = torch.nn.functional.relu(self.fully_connected_first(scores)).to(device)
        score = torch.sigmoid(self.scoring_layer(scores)).to(device)
        return score

class HybGNNTrainer(object):
 
    def __init__(self, args):

        self.args = args
        self.initial_label_enumeration()
        self.setup_model()

    def setup_model(self):
 
        self.model = HybGNN(self.args, self.number_of_labels).to(device)


    def initial_label_enumeration(self):

        print("\nEnumerating unique labels.\n")
        self.training_graphs = glob.glob(self.args.training_graphs + "*.json")
        self.testing_graphs = glob.glob(self.args.testing_graphs + "*.json")
        graph_pairs = self.training_graphs + self.testing_graphs

        self.beforesorted_train_labels = set()
        self.aftersorted_train_labels = set()

        for training_graph in tqdm(self.training_graphs):
            traindata = process_pair(training_graph)
            self.beforesorted_train_labels = self.beforesorted_train_labels.union(set(traindata["labels_1"]))


        self.aftersorted_train_labels = sorted(self.beforesorted_train_labels)
        self.aftersorted_train_labels = {val:index  for index, val in enumerate(self.aftersorted_train_labels)}
        self.train_number_of_labels = len(self.aftersorted_train_labels)
        

        

        self.number_of_labels =  self.train_number_of_labels + 15

        print ("number_of_labels", self.number_of_labels)
        
    def create_batches(self):

        random.shuffle(self.training_graphs)
        batches = []
        for graph in range(0, len(self.training_graphs), self.args.batch_size):
            batches.append(self.training_graphs[graph:graph+self.args.batch_size])
        return batches

    def transfer_testset_to_torch(self, testdata):


        self.beforesorted_global_labels = set()
        self.aftersorted_global_labels = set()          
        self.beforesorted_test_labels = set()
        self.aftersorted_test_labels = set()
                
        self.beforesorted_test_labels = set(testdata["labels_1"])
        self.beforesorted_global_labels = self.beforesorted_train_labels.union(self.beforesorted_test_labels)
        
        self.aftersorted_global_labels = sorted(self.beforesorted_global_labels)
        self.aftersorted_global_labels = {val:index  for index, val in enumerate(self.aftersorted_global_labels)}
        self.realnumber_of_labels = len(self.aftersorted_global_labels)
        
                
        new_testdata = dict()
        edges_1 = testdata["graph_1"] + [[y, x] for x, y in testdata["graph_1"]]



        edges_1 = torch.from_numpy(np.array(edges_1, dtype=np.int64).T).type(torch.long).to(device)


        features_1 = []

        for n in testdata["labels_1"]:
            features_1.append([1.0 if self.aftersorted_global_labels[n] == i else 0.0 for i in self.aftersorted_global_labels.values()] + [0] * (self.number_of_labels - self.realnumber_of_labels))

        features_1 = torch.FloatTensor(np.array(features_1)).to(device)


        new_testdata["edge_index_1"] = edges_1


        new_testdata["features_1"] = features_1


        norm_ged = testdata["ged"]/(len(testdata["labels_1"]))

        new_testdata["target"] = torch.from_numpy(np.exp(-norm_ged).reshape(1, 1)).view(-1).float().to(device)
        return new_testdata
        
    def transfer_trainset_to_torch(self, traindata):

        new_traindata = dict()
        edges_1 = traindata["graph_1"] + [[y, x] for x, y in traindata["graph_1"]]



        edges_1 = torch.from_numpy(np.array(edges_1, dtype=np.int64).T).type(torch.long).to(device)


        features_1 = []

        for n in traindata["labels_1"]:
            features_1.append([1.0 if self.aftersorted_train_labels[n] == i else 0.0 for i in self.aftersorted_train_labels.values()] + [0] * (self.number_of_labels - self.train_number_of_labels))

        features_1 = torch.FloatTensor(np.array(features_1)).to(device)


        new_traindata["edge_index_1"] = edges_1


        new_traindata["features_1"] = features_1


        norm_ged = traindata["ged"]/(len(traindata["labels_1"]))

        new_traindata["target"] = torch.from_numpy(np.exp(-norm_ged).reshape(1, 1)).view(-1).float().to(device)
        return new_traindata        


    def process_batch(self, batch):

        self.optimizer.zero_grad()
        losses = 0
        for training_graph in batch:
            traindata = process_pair(training_graph)           
            traindata = self.transfer_trainset_to_torch(traindata)          
            target = traindata["target"]

            prediction = self.model(traindata).to(device)
            losses = losses + torch.nn.functional.mse_loss(traindata["target"], prediction).to(device)
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

        self.model.eval()
        self.scores = []
        self.ground_truth = []
        self.precision = []
        self.hyb = []
        self.admix = []
        self.nohyb = []
        self.agf = []
        self.gamma = []

        for graph_pair in tqdm(self.testing_graphs):      
            testdata = process_pair(graph_pair)          
            gedfactor = (len(testdata["labels_1"]))
            self.ground_truth.append(calculate_normalized_ged(testdata))
            testdata = self.transfer_testset_to_torch(testdata)            
            target = testdata["target"]
            prediction = self.model(testdata).to(device)
            self.scores.append(calculate_loss(prediction, target))
            print("Ged:", -math.log(prediction) * gedfactor)
            print("Target:", -math.log(target) * gedfactor)

            if (math.log(target) * gedfactor-math.log(prediction) * gedfactor)**2 < 0.25:
                self.precision.append(1)

            else:
                self.precision.append(0)
            self.gamma.append(round(-math.log(prediction) * gedfactor, 1))
# 使用Counter来计算每个GED值的频数
#        freq = Counter(self.ged)

# 找到频数最高的GED值（即众数）
#        mode_ged = freq.most_common(1)[0][0]
       

        print("ACC:", np.mean(self.precision))
        print("Gamma:", np.mean(self.gamma))
            
        self.print_evaluation()

    def print_evaluation(self):

        norm_ged_mean = np.mean(self.ground_truth)
        base_error = np.mean([(n-norm_ged_mean)**2 for n in self.ground_truth])
        model_error = np.mean(self.scores)
        print("\nBaseline error: " +str(round(base_error, 5))+".")
        print("\nModel test error: " +str(round(model_error, 5))+".")

    def save(self):

        torch.save(self.model.state_dict(), self.args.save_path)

    def load(self):

        self.model.load_state_dict(torch.load(self.args.load_path))
