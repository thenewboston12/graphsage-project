import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, features, feature_dim, 
            embed_dim, adj_lists, aggregator,
            num_sample=10,
            base_model=None, gcn=False, cuda=False, 
            feature_transform=False): 
        super(Encoder, self).__init__()

        self.features = features # function to look up features given node 
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample ## how many neighbors we are ssampling

        ## For stacking models
        if base_model is not None:
            self.base_model = base_model

        self.gcn = gcn
        self.embed_dim = embed_dim # diemension of the embeddings ( we use 128)
        self.cuda = cuda
        self.aggregator.cuda = cuda

        ## learnable weight matrix of the encoder
        ## when gcn=True: [embed_dim, feat_dim]
        ##  if gcn=False: [embed_dim, 2 * feat_dim]  because we concatenate self too
        self.weight = nn.Parameter(
            torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim)
        )

        ## initalization with xavier uniform
        init.xavier_uniform_(self.weight)  

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.

        nodes -- list of nodes
        """

        ## Use aggregator(from aggregators.py) to compute neighbor features
        ## resulting shape = [batch_size x feat_dim]
        neigh_feats = self.aggregator.forward(
            nodes, [self.adj_lists[int(node)] for node in nodes], self.num_sample
        )

        ## when not in GCN mode, we also use the node's own features
        if not self.gcn:
            if self.cuda:
                self_feats = self.features(torch.LongTensor(nodes).cuda())
            else:
                self_feats = self.features(torch.LongTensor(nodes))
            ## concat self featurs and neighbors!
            combined = torch.cat([self_feats, neigh_feats], dim=1)
        else:
            ## do nothinig
            combined = neigh_feats

        
        ## Multiply by learnable WEIGHT matrix and pass thru ReLU 
        ## W x combined^T
        ## (embed_dim x input_dim )(input_dim x batch_size)
        ## output [embed_dim x batch_size]
        combined = F.relu(self.weight.mm(combined.t()))
        return combined



class EncoderSC(nn.Module):
    """
    Encoder with Skip Connection (Residual Connection) for GraphSAGE.
    """

    def __init__(self, features, feature_dim, 
                 embed_dim, adj_lists, aggregator,
                 num_sample=10,
                 base_model=None, gcn=False, cuda=False, 
                 feature_transform=False):
        super(EncoderSC, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample
        if base_model is not None:
            self.base_model = base_model

        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator.cuda = cuda
        self.weight = nn.Parameter(
            torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim)
        )
        init.xavier_uniform_(self.weight)

        ## Extra projection layer for skip connection so that dimensions match when adding raw to transformed
        if (self.feat_dim if self.gcn else 2 * self.feat_dim) != embed_dim:
            ## we insert a LINEAR Layer that fixes the dimensionality issue
            self.skip_proj = nn.Linear(
                self.feat_dim if self.gcn else 2 * self.feat_dim, embed_dim, bias=False
            )
        else:
            self.skip_proj = None

    def forward(self, nodes):
        ### -------------------- Identical to the original architecture ---------------- ### 
        neigh_feats = self.aggregator.forward(
            nodes, [self.adj_lists[int(node)] for node in nodes], self.num_sample
        )
        if not self.gcn:
            if self.cuda:
                self_feats = self.features(torch.LongTensor(nodes).cuda())
            else:
                self_feats = self.features(torch.LongTensor(nodes))
            combined = torch.cat([self_feats, neigh_feats], dim=1)
        else:
            combined = neigh_feats

        ### -----------------------------------------------------------------------------

        ## BEFORE multiplying by weight matrix and going through ReLU, do skip Connection
        ## save combined features before transforming themn 
        raw = combined  # Save original

        ## Apply weight transfomration as before
        ## output = [embed_dim x batch_size]
        combined = self.weight.mm(combined.t())  

        ## if diimensions dont match then project raw features
        if self.skip_proj is not None:
            ## [batch_size x input_dim] -> [batch_size x embed_dim]
            raw = self.skip_proj(raw)

        ## main skip connection logic, adding to transformed the raw projected weights
        combined = combined + raw.t()  

        ## then going thru nonlinearity
        combined = F.relu(combined)

        ## final shape is still [embed_dim x batch_size]
        return combined