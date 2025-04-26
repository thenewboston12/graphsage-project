import torch
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import f1_score

def train_epoch(model, labels, train_nodes, optimizer, batch_size):
    model.train()
    losses = []
    preds = []
    trues = []
    np.random.shuffle(train_nodes)

    for i in range(0, len(train_nodes), batch_size):
        batch_nodes = train_nodes[i:i+batch_size]

        optimizer.zero_grad()
        loss = model.loss(batch_nodes, 
            Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        ### predict on a BATCH
        output = model.forward(batch_nodes)
        preds.append(output.data.numpy().argmax(axis=1))
        trues.append(labels[np.array(batch_nodes)])

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    train_f1 = f1_score(trues, preds, average="micro")

    return np.mean(losses), train_f1

def validate_epoch(model, labels, val_nodes, batch_size=256):
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for i in range(0, len(val_nodes), batch_size):
            batch_nodes = val_nodes[i:i+batch_size]
            output = model.forward(batch_nodes)
            preds.append(output.data.numpy().argmax(axis=1))
            trues.append(labels[np.array(batch_nodes)])

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    val_f1 = f1_score(trues, preds, average="micro")

    return val_f1

def test_epoch(model, labels, test_nodes, batch_size=256):
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for i in range(0, len(test_nodes), batch_size):
            batch_nodes = test_nodes[i:i+batch_size]
            output = model.forward(batch_nodes)
            preds.append(output.data.numpy().argmax(axis=1))
            trues.append(labels[np.array(batch_nodes)])

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    test_f1 = f1_score(trues, preds, average="micro")

    return test_f1
