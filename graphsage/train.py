# train.py
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
from sklearn.metrics import f1_score
import time
import argparse

from graphsage.aggregators import MeanAggregator, AttentionAggregator
from graphsage.encoders import Encoder, EncoderSC
from graphsage.models import SupervisedGraphSage
from graphsage.datasets import load_cora, load_pubmed

# -------------------
# Dataset Preparation
# -------------------

def prepare_cora_data():
    feat_data, labels, adj_lists = load_cora()
    features = nn.Embedding(2708, 1433)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    return features, labels, adj_lists

def prepare_pubmed_data():
    feat_data, labels, adj_lists = load_pubmed()
    features = nn.Embedding(19717, 500)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    return features, labels, adj_lists

# -------------------
# Model Building
# -------------------

def build_graphsage_model(features, adj_lists, dataset, model_type):
    if dataset == 'cora':
        input_dim = 1433
        num_classes = 7
        num_samples_1 = 5
        num_samples_2 = 5
    elif dataset == 'pubmed':
        input_dim = 500
        num_classes = 3
        num_samples_1 = 10
        num_samples_2 = 25
    else:
        raise ValueError("Dataset must be 'cora' or 'pubmed'.")

    # First aggregator
    if model_type in ['baseline', 'skip']:
        agg1 = MeanAggregator(features, cuda=False)
    elif model_type in ['attention', 'skip_and_attention']:
        agg1 = AttentionAggregator(features, embed_dim=input_dim, cuda=False)
    else:
        raise ValueError("Unknown model type")

    # First encoder
    if model_type in ['skip', 'skip_and_attention']:
        enc1 = EncoderSC(features, input_dim, 128, adj_lists, agg1, gcn=True, cuda=False)
    else:
        enc1 = Encoder(features, input_dim, 128, adj_lists, agg1, gcn=True, cuda=False)

    # Second aggregator
    if model_type in ['baseline', 'skip']:
        agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)
    elif model_type in ['attention', 'skip_and_attention']:
        agg2 = AttentionAggregator(lambda nodes: enc1(nodes).t(), embed_dim=128, cuda=False)

    # Second encoder
    if model_type in ['skip', 'skip_and_attention']:
        enc2 = EncoderSC(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
                         base_model=enc1, gcn=True, cuda=False)
    else:
        enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
                       base_model=enc1, gcn=True, cuda=False)

    # Sampling settings
    enc1.num_samples = num_samples_1
    enc2.num_samples = num_samples_2

    model = SupervisedGraphSage(num_classes, enc2)

    return model


# -------------------
# Training and Validation
# -------------------

def train(graphsage, labels, train_nodes, optimizer, batch_size, num_epochs):
    times = []
    for batch in range(num_epochs):
        batch_nodes = train_nodes[:batch_size]
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

# -------------------
# Runner
# -------------------

def run(dataset, model_type):
    np.random.seed(1)
    random.seed(1)

    if dataset == 'cora':
        features, labels, adj_lists = prepare_cora_data()
        num_nodes = 2708
        batch_size = 256
        num_epochs = 100
    elif dataset == 'pubmed':
        features, labels, adj_lists = prepare_pubmed_data()
        num_nodes = 19717
        batch_size = 1024
        num_epochs = 200
    else:
        raise ValueError("Unknown dataset.")

    graphsage = build_graphsage_model(features, adj_lists, dataset, model_type)

    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train_nodes = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.7)

    times = train(graphsage, labels, train_nodes, optimizer, batch_size, num_epochs)
    f1 = validate(graphsage, labels, val)

    print("Average batch time:", np.mean(times))

# -------------------
# Main Entry
# -------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GraphSAGE on Cora or Pubmed.")

    ## Choose dataset
    parser.add_argument('--dataset', type=str, choices=['cora', 'pubmed'], default='cora',
                        help="Dataset to use: 'cora' or 'pubmed'")
    
    ## Choose model architecture
    parser.add_argument('--model', type=str, choices=['baseline', 'attention', 'skip', 'skip_and_attention'], default='baseline',
                    help="Model type: 'baseline', 'attention', 'skip', or 'skip_and_attention'")

    


    args = parser.parse_args()
    run(args.dataset, args.model)


    