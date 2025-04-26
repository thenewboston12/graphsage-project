import torch
import torch.nn as nn
import random

"""
Set of modules for aggregating embeddings of neighbors.
"""

class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, features, cuda=False, gcn=False): 
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        self.features = features
        self.cuda = cuda
        self.gcn = gcn
        
    def forward(self, nodes, to_neighs, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        _set = set
        if num_sample is not None:
            _sample = random.sample
            samp_neighs = [_set(_sample(list(to_neigh), 
                            num_sample,
                            )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        if self.gcn:
            samp_neighs = [samp_neigh.union({nodes[i]}) for i, samp_neigh in enumerate(samp_neighs)]
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        
        mask = torch.zeros(len(samp_neighs), len(unique_nodes))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]   
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        
        if self.cuda:
            mask = mask.cuda()
        
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
        
        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        
        to_feats = mask.mm(embed_matrix)
        return to_feats


## Attention Aggregator!!!

class AttentionAggregator(nn.Module):
    """
    Aggregates a node's embeddings using attention-weighted mean of neighbors' embeddings.
    """
    def __init__(self, features, embed_dim=128, cuda=False, gcn=False):
        super(AttentionAggregator, self).__init__()

        self.features = features
        self.cuda = cuda
        self.gcn = gcn
        
        self.embed_dim = embed_dim

        # Linear layer to compute attention scores
        self.attention_fc = nn.Linear(embed_dim, 1, bias=False)

    def forward(self, nodes, to_neighs, num_sample=10):
        _set = set
        if num_sample is not None:
            _sample = random.sample
            samp_neighs = [_set(_sample(list(to_neigh), 
                            num_sample,
                            )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        if self.gcn:
            samp_neighs = [samp_neigh.union({nodes[i]}) for i, samp_neigh in enumerate(samp_neighs)]

        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))

        # Build neighbor feature matrices for each batch node
        to_feats = []
        for i in range(len(samp_neighs)):
            neighbor_feats = embed_matrix[torch.LongTensor([unique_nodes[n] for n in samp_neighs[i]])]
            attn_scores = self.attention_fc(neighbor_feats)  # [num_neighbors, 1]
            attn_weights = torch.softmax(attn_scores, dim=0)  # Normalize attention scores

            agg_feat = (attn_weights * neighbor_feats).sum(0)  # Weighted sum of neighbor features
            to_feats.append(agg_feat)

        to_feats = torch.stack(to_feats)

        return to_feats
