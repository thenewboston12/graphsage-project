import torch
import torch.nn as nn
import random

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

        ## sample neighbors of a node. If there are less than 10 neighbors, use all of its neighbors
        if num_sample is not None:
            _sample = random.sample
            samp_neighs = [_set(_sample(list(to_neigh), 
                            num_sample,
                            )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs


        ## use self loops if self.gcn set to true 
        ## in GCNs nodes aggregate from themselves and their neighbors
        if self.gcn:
            samp_neighs = [samp_neigh.union({nodes[i]}) for i, samp_neigh in enumerate(samp_neighs)]


        ## makes a list of all the unique nodes from all samples neighbor sets
        ## i.e. map node id -> index
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        
        ## create a mask that tells which neighbor is associated with which node
        ## dimensionality is [batch_size x unique_nodes_size] 
        mask = torch.zeros(len(samp_neighs), len(unique_nodes))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]   
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        
        ## use GPU if available 
        if self.cuda:
            mask = mask.cuda()
        
        ## Normalize rows by total amount of neighbors
        ## this allows to take averages (mean)
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
        
        ## embed_matrix dimensionality: [unique_nodes_size x feature_dimension]
        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        
        ## matrix multiplication to average neighbor embeddings per node 
        ## [batch_size x feature_dimension]
        to_feats = mask.mm(embed_matrix)
        return to_feats


## Attention Aggregator!!!

class AttentionAggregator(nn.Module):
    """
    Aggregates a node's embeddings using attention-weighted mean of neighbors' embeddings.
    """
    def __init__(self, features, embed_dim=128, cuda=False, gcn=False):
        ## mostly reusing the params from meanaggregator
        super(AttentionAggregator, self).__init__()

        self.features = features
        self.cuda = cuda
        self.gcn = gcn
        
        ## node's feature embeddings dimension (1433 for CORA and 500 for pubmed) but defaults to 128 if those not provided
        self.embed_dim = embed_dim

        ## torch.nn linearn layer function to calculate attention between neigbors.
        ## embed_dim vector  -> scalar 
        self.attention_fc = nn.Linear(embed_dim, 1, bias=False)

    def forward(self, nodes, to_neighs, num_sample=10):
        ### REUSING code from MeanAggregator until #############################
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

        #########################################################################

        ## For each node in the batch, build neighbor feature matrix 
        ## samp_neighbors's length is the number of nodes in a batch:
        ## samp_neighs = [{1,2}, {0,2,3}] --- 2 nodes in a batch and their neighbors
        to_feats = []
        for i in range(len(samp_neighs)):
            ### Grab each node's neighbors features
            neighbor_feats = embed_matrix[torch.LongTensor([unique_nodes[n] for n in samp_neighs[i]])]

            ## calculate attention scores for each neigbors: #[neighbors x 1] 1 scalar number for each neighbor of 
            ## this particular node samp_neghs[i] node
            attn_scores = self.attention_fc(neighbor_feats)  

            ## using softmax to turn scalar scores into probabilities and use them as weights
            attn_weights = torch.softmax(attn_scores, dim=0)  

            ## Now weigh each neighbor's features according to those attention weights and sum 
            ## agg_feat has (embed_dim, 1) dimensionality
            agg_feat = (attn_weights * neighbor_feats).sum(0)  
            to_feats.append(agg_feat)

        ## stack resulting features into 1 output matrix   
        ## resultiing shape is : (batch_size, embed_dim) or (batch_size, feature_dim)  
        to_feats = torch.stack(to_feats)

        return to_feats
