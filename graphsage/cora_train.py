## Script for training 

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
from sklearn.metrics import f1_score
import time

from graphsage.aggregators import MeanAggregator
from graphsage.encoders import Encoder
from graphsage.models import SupervisedGraphSage
from graphsage.datasets import load_cora

## Prepares the dataset
def prepare_cora_data():
    from graphsage.datasets import load_cora
    feat_data, labels, adj_lists = load_cora()
    features = nn.Embedding(2708, 1433)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    return features, labels, adj_lists


## Builds the model
def build_graphsage_model(features, adj_lists):
    from graphsage.aggregators import MeanAggregator
    from graphsage.encoders import Encoder
    from graphsage.models import SupervisedGraphSage

    agg1 = MeanAggregator(features, cuda=False)
    enc1 = Encoder(features, 1433, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
                   base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 5
    enc2.num_samples = 5

    model = SupervisedGraphSage(7, enc2)
    return model

# Trains the model
def train(graphsage, labels, train_nodes, optimizer, num_epochs=100):
    times = []
    for batch in range(num_epochs):
        batch_nodes = train_nodes[:256]
        random.shuffle(train_nodes)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes,
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print(batch, loss.item())
    return times


# evaluates the model and calculates f1 score 
def validate(graphsage, labels, val_nodes):
    val_output = graphsage.forward(val_nodes)
    f1 = f1_score(labels[val_nodes], val_output.data.numpy().argmax(axis=1), average="micro")
    print("Validation F1:", f1)
    return f1


# puts everything together and runs 
def run_cora():
    np.random.seed(1)
    random.seed(1)
    features, labels, adj_lists = prepare_cora_data()

    graphsage = build_graphsage_model(features, adj_lists)

    num_nodes = 2708
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train_nodes = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.7)

    times = train(graphsage, labels, train_nodes, optimizer, num_epochs=100)
    f1 = validate(graphsage, labels, val)

    print("Average batch time:", np.mean(times))

if __name__ == "__main__":
    run_cora()