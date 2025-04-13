import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv #Need to look at this one
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import scatter
import sys

from NGG.autoencoders.components.deepsets import DeepSets
from NGG.autoencoders.components.encoders.GIN_base import GIN
from NGG.autoencoders.components.encoders.GIN_concat import GIN_concat   
from NGG.autoencoders.components.decoders.decoder_base import Decoder
from NGG.autoencoders.components.decoders.decoder_norm import Decoder_normalized


# Variational Autoencoder
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, n_max_nodes, deepsets: DeepSets = None,norm=False,attention=True):
            super(VariationalAutoEncoder, self).__init__()
            self.n_max_nodes = n_max_nodes
            self.input_dim = input_dim
            self.latent_dim = latent_dim
            # self.encoder = GIN(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc)
            self.encoder = GIN_concat(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc,attention=attention)
            self.fc_mu = nn.Linear(hidden_dim_enc, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim_enc, latent_dim)
            # self.decoder = Decoder(latent_dim, hidden_dim_dec, n_layers_dec, n_max_nodes)
            self.decoder = Decoder(latent_dim, hidden_dim_dec, n_layers_dec, n_max_nodes)
            # self.decoder = Decoder_normalized(latent_dim+7, hidden_dim_dec, n_layers_dec, n_max_nodes)
            self.deepsets = deepsets

    def forward(self, data):
        x_g = self.encoder(data, self.deepsets)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g)
        return adj

    def encode(self, data):
        x_g = self.encoder(data, self.deepsets)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        return x_g

    def reparameterize(self, mu, logvar, eps_scale=1.):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, mu, logvar):
       x_g = self.reparameterize(mu, logvar)
       adj = self.decoder(x_g)
       return adj

    def decode_mu(self, mu):
       adj = self.decoder(mu)
       return adj
   
   
    def decode_and_recon_loss(self, x_g,data):
        adj= self.decoder(x_g)
        recon = F.l1_loss(adj, data.A, reduction='mean')
        return recon

    def edge_node_coherence_loss(self,adj_matrices,alpha1=1,alpha2=1,alpha3=1):
        """
        Computes the edge-node coherence loss for a given adjacency matrix.
        Every node with an edge must have a self-cycle.
        
        Parameters:
            adj_matrix (torch.Tensor): A square adjacency matrix of shape (N, N).
                                    Entries should be differentiable (e.g., predictions).
        
        Returns:
            torch.Tensor: The computed edge-node coherence loss.
        """
        row_sum = torch.sum(adj_matrices, dim=2)  # Shape: (B, N)
        # Extract diagonal entries (self-cycles) for each matrix in the batch
        diag = torch.diagonal(adj_matrices, dim1=1, dim2=2)  # Shape: (B, N)
        # print(f" verify there is 1 in diag {torch.sum(diag)}")
        row_penalty = torch.relu(row_sum - row_sum.sum()*diag)  # Shape: (B, N)
        return row_penalty.sum()


    def loss(self, data, 
            beta=0.05,
            contrastive_hyperparameters: list = None, 
            penalization_hyperparameters: float = None,): 
        
        x_g  = self.encoder(data, self.deepsets) # This encodes the input graph into a latent space but without any information about the prompt and stats...
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar) 
        adj = self.decoder(x_g) # BS*max_nodes*max_nodes This basically randomly sampes a graph from the distribution with no information whatsoever about the input prompt and stats
        
        
        recon = F.l1_loss(adj, data.A, reduction='mean')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + beta*kld

        results = {
            'loss': loss,
            'kl': kld,
            'recon': recon,
        }
        ###############################################
        #data is a torch_geometric.data.Data object that contains a batch of graphs
        # print(data.x.shape)# N_nodes_in_all_batch*(Spectral_embedding+1) these are the node features tensors, they are all aggregated in a single tensor!
        # print(data.edge_index.shape) # these are the edge indices, they are also aggregated in a single tensor, and they are indexed by the new node indices that can go up to 7000
        # print(data.A.shape) #BS*max_nodes*max_nodes this is not aggregated
        # print(data.stats.shape) # BS*7
        # print(len(data.filename)) #list of BS strings containing the filenames
        # print(data.ptr.shape) #BS+1 I don't understand this one ??
        # print(data.batch.shape) #N_nodes_in_all_batch this keeps track of "to which graph does the node belong to"
        # sys.exit()
        return results

    


    def create_mask_data(self, data):
        mask = torch.zeros(data.A.shape)
        for i in range(data.A.shape[0]): # iterate through the batch of graphs
            n_nodes=data.stats[i,0].item()
            mask[i,:n_nodes,:n_nodes]=1
        return mask
    
    def create_mask_latent(self, x_g,edge_index=False):
        batch_size = x_g.shape[0] if len(x_g.shape) > 1 else 1
        
        mask=torch.zeros((batch_size,self.n_max_nodes,self.n_max_nodes)).to(x_g.device)
        for i in range(batch_size):
            n_nodes=int(x_g[i,self.latent_dim])
            # print(f"the entire x_g row is {x_g[i]}")
            # print(f"n_nodes: {n_nodes}")
            # sys.exit()
            mask[i,:n_nodes,:n_nodes]=1
        
        return mask

