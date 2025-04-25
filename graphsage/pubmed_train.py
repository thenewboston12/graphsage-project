# train_pubmed.py
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
from graphsage.datasets import load_pubmed

def prepare_pubmed_data():
    feat_data, labels, adj_lists = load_pubmed()
    features = nn.Embedding(19717, 500)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    return features, labels, adj_lists

def build_graphsage_model_pubmed(features, adj_lists):
    agg1 = MeanAggregator(features, cuda=False)
    enc1 = Encoder(features, 500, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
                   base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 10
    enc2.num_samples = 25

    model = SupervisedGraphSage(3, enc2)
    return model

def train(graphsage, labels, train_nodes, optimizer, num_epochs=200):
    times = []
    for batch in range(num_epochs):
        batch_nodes = train_nodes[:1024]
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

def validate(graphsage, labels, val_nodes):
    val_output = graphsage.forward(val_nodes)
    f1 = f1_score(labels[val_nodes], val_output.data.numpy().argmax(axis=1), average="micro")
    print("Validation F1:", f1)
    return f1

def run_pubmed():
    np.random.seed(1)
    random.seed(1)
    features, labels, adj_lists = prepare_pubmed_data()

    graphsage = build_graphsage_model_pubmed(features, adj_lists)

    num_nodes = 19717
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train_nodes = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.7)

    times = train(graphsage, labels, train_nodes, optimizer, num_epochs=200)
    f1 = validate(graphsage, labels, val)

    print("Average batch time:", np.mean(times))

if __name__ == "__main__":
    run_pubmed()
