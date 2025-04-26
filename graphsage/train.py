import torch
import torch.nn as nn
import numpy as np
import random
import time
import argparse
import matplotlib.pyplot as plt


from graphsage.aggregators import MeanAggregator, AttentionAggregator
from graphsage.encoders import Encoder, EncoderSC
from graphsage.models import SupervisedGraphSage
from graphsage.datasets import load_cora, load_pubmed
from graphsage.trainer import train_epoch, validate_epoch, test_epoch

### Dataset Preparation


def prepare_cora_data():
    feat_data, labels, adj_lists = load_cora()

    ### randomly initialized at first 
    features = nn.Embedding(2708, 1433)

    ## then set to features of the dataset
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    return features, labels, adj_lists

def prepare_pubmed_data():
    feat_data, labels, adj_lists = load_pubmed()
    features = nn.Embedding(19717, 500)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    return features, labels, adj_lists

### Model Building


def build_graphsage_model(features, adj_lists, dataset, model_type):
    ## building a 2 layer graphsage model
    ### Node IDs -> enc2 -> enc1 -> agg1 -> features -> enc1 returns -> enc2 aggregates again -> final prediction
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

    if model_type in ['baseline', 'skip']:
        agg1 = MeanAggregator(features, cuda=False)
    elif model_type in ['attention', 'skip_and_attention']:
        agg1 = AttentionAggregator(features, embed_dim=input_dim, cuda=False)
    else:
        raise ValueError("Unknown model type")

    if model_type in ['skip', 'skip_and_attention']:
        enc1 = EncoderSC(features, input_dim, 128, adj_lists, agg1, gcn=True, cuda=False)
    else:
        enc1 = Encoder(features, input_dim, 128, adj_lists, agg1, gcn=True, cuda=False)

    if model_type in ['baseline', 'skip']:
        agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)
    elif model_type in ['attention', 'skip_and_attention']:
        agg2 = AttentionAggregator(lambda nodes: enc1(nodes).t(), embed_dim=128, cuda=False)

    if model_type in ['skip', 'skip_and_attention']:
        enc2 = EncoderSC(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
                         base_model=enc1, gcn=True, cuda=False)
    else:
        enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
                       base_model=enc1, gcn=True, cuda=False)

    enc1.num_samples = num_samples_1
    enc2.num_samples = num_samples_2

    model = SupervisedGraphSage(num_classes, enc2)

    return model

#### Runner

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
        num_epochs = 100
    else:
        raise ValueError("Unknown dataset.")

    graphsage = build_graphsage_model(features, adj_lists, dataset, model_type)

    rand_indices = np.random.permutation(num_nodes)
    test_nodes = rand_indices[:1000]
    val_nodes = rand_indices[1000:1500]
    train_nodes = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.7)

    train_losses = []
    train_f1s = []
    val_f1s = []

    print("Starting training...")
    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_f1 = train_epoch(graphsage, labels, train_nodes, optimizer, batch_size)
        val_f1 = validate_epoch(graphsage, labels, val_nodes)

        train_losses.append(train_loss)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)

        elapsed = time.time() - start_time
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Train F1 = {train_f1:.4f}, Val F1 = {val_f1:.4f}, Time = {elapsed:.2f}s")

    print("Training completed!")

    test_f1 = test_epoch(graphsage, labels, test_nodes)
    print(f"Test F1 Score: {test_f1:.4f}")

    ## plot curves
    print("Plotting curves... ")

    plt.figure(figsize=(8,6))
    plt.plot(range(num_epochs), train_f1s, label='Train F1')
    plt.plot(range(num_epochs), val_f1s, label='Validation F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title(f'Train vs Validation F1 Curve ({dataset.upper()} - {model_type})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"f1_curve_{dataset}_{model_type}.png")
    plt.show()

    ## Plot Training Loss Curve
    plt.figure(figsize=(8,6))
    plt.plot(range(num_epochs), train_losses, label='Train Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Curve ({dataset.upper()} - {model_type})')
    plt.grid(True)
    plt.savefig(f"loss_curve_{dataset}_{model_type}.png")
    plt.show()

### parsing args here!

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GraphSAGE on Cora or Pubmed.")

    parser.add_argument('--dataset', type=str, choices=['cora', 'pubmed'], default='cora',
                        help="Dataset to use: 'cora' or 'pubmed'")
    parser.add_argument('--model', type=str, choices=['baseline', 'attention', 'skip', 'skip_and_attention'], default='baseline',
                        help="Model type: 'baseline', 'attention', 'skip', or 'skip_and_attention'")

    args = parser.parse_args()
    run(args.dataset, args.model)
