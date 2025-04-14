import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv #Need to look at this one
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import scatter
import sys

from NGG.autoencoders.autoencoder_base import VariationalAutoEncoder
from NGG.autoencoders.components.deepsets import DeepSets
from NGG.autoencoders.components.encoders.GIN_base import GIN
from NGG.autoencoders.components.encoders.GIN_concat import GIN_concat   
from NGG.autoencoders.components.decoders.decoder_base import Decoder
from NGG.autoencoders.components.decoders.decoder_norm import Decoder_normalized



class VariationalAutoEncoder_concat(VariationalAutoEncoder):
    def __init__(
        self, 
        input_dim, 
        hidden_dim_enc, 
        hidden_dim_dec, 
        latent_dim, 
        n_layers_enc, 
        n_layers_dec, 
        n_max_nodes,
        deepsets: DeepSets = None,
        normalize: bool = False,
        attention: bool = True,
    ):
        super().__init__(input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, n_max_nodes, deepsets)
        self.encoder = GIN_concat(input_dim, hidden_dim_enc, hidden_dim_enc, n_layers_enc, attention=attention)
        additional_dim = 7
        self.additional_dim = additional_dim
        self.norm = normalize   
        if normalize:
            self.decoder = Decoder_normalized(latent_dim+additional_dim, hidden_dim_dec, n_layers_dec, n_max_nodes)
            print("Using normalized decoder")
        else:
            self.decoder = Decoder(latent_dim+additional_dim, hidden_dim_dec, n_layers_dec, n_max_nodes)


    def forward(self, data):
        x_g = self.encoder(data, self.deepsets)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        stats = data.stats  # Shape: (batch_size, num_stats)
        x_g = torch.cat((x_g, stats), dim=1) 

        mask = self.create_mask_latent(x_g)
        adj = self.decoder(x_g,mask)

        return adj


    def decode(self, mu, logvar):
        
       x_g = self.reparameterize(mu, logvar)

       mask = self.create_mask_latent(x_g)  
       adj = self.decoder(x_g,mask)

       
       if self.norm:
           #take off self cycles by setting the diagonal to 0
            adj = adj - torch.diag_embed(torch.diagonal(adj, dim1=1, dim2=2))
       return adj

    def decode_mu(self, mu):

        mask = self.create_mask_latent(mu)
        adj = self.decoder(mu,mask)

        if self.norm:
            adj = adj - torch.diag_embed(torch.diagonal(adj, dim1=1, dim2=2))
        return adj
    
    def loss(
        self, 
        data, 
        beta=0.05, 
        contrastive_hyperparameters: list = None, 
        penalization_hyperparameters: float = None,
    ): #this loss variant concatenates the stats to the latent space before decoding

        x_g  = self.encoder(data, self.deepsets) # This encodes the input graph into a latent space but without any information about the prompt and stats...
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar) 
        # Concatenate stats and one-hot-encoded labels
        stats = data.stats  # Shape: (batch_size, num_stats)
        x_g = torch.cat((x_g, stats), dim=1) 

        mask = self.create_mask_latent(x_g)
        adj = self.decoder(x_g,mask) # BS*max_nodes*max_nodes This basically randomly sampes a graph from the distribution with no information whatsoever about the input prompt and stats

        
        recon = F.l1_loss(adj, data.A, reduction='mean')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + beta*kld
        results = {
            "recon": recon,
            "kld": kld,
        }
        if contrastive_hyperparameters is not None:
            contrastive_loss = self.get_weighted_contrastive_loss(mu, logvar, data, contrastive_hyperparameters[0], contrastive_hyperparameters[1])
            loss += contrastive_loss
            results["contrastive_loss"] = contrastive_loss
        if penalization_hyperparameters is not None:
            # print("penalization_hyperparameters", penalization_hyperparameters)
            penalization = self.get_weighted_penalization(adj, data, penalization_hyperparameters)
            loss += penalization
            results["penalization"] = penalization

        results["loss"] = loss
        
        return results
            
    
    def loss_function_concat_stats_pn(self, data, beta=0.05,alpha=0.05): # This loss variants concatenates the stats to the latent space before decoding and tries to punish n_nodes and n_edges
        
        
        x_g  = self.encoder(data, self.deepsets) # This encodes the input graph into a latent space but with info about the prompt and stats...
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar) 
        x_g = torch.cat((x_g, data.stats), dim=1)
        mask = self.create_mask_latent(x_g)
        adj = self.decoder(x_g,mask) # BS*max_nodes*max_nodes This basically randomly sampes a graph from the distribution with information about the input prompt and stats

        
        
        n_edges= adj.sum(dim=(1,2))/2
        num_nodes = torch.sum(torch.diagonal(adj, dim1=1, dim2=2),dim=1)
        coherence_loss = self.edge_node_coherence_loss(adj)
        
        MSE_n_nodes= torch.square(num_nodes - data.stats[:,0]).mean()
        MSE_n_edges= torch.square(n_edges - data.stats[:,1]).mean()

        feature_losses= (MSE_n_nodes+MSE_n_edges+coherence_loss)
        
        recon = F.l1_loss(adj, data.A, reduction='mean')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + beta*kld + alpha*feature_losses
        
        
        # print(loss)
        # print(feature_losses)
        return loss, recon, kld,feature_losses
    
    def get_weighted_contrastive_loss(self, mu, logvar, data, lambda_contrastive, lambda_entropy):
        """
        Computes the contrastive loss of the batch using cosine similarity. 
        The contrastive loss is multiplied by lambda_contrastive.
        Additionally, a regularization term is added to maximize the entropy between clusters.
        
        The dynamic margin is computed based on the mean pairwise cosine similarity between the samples.
        Cosine similarity is computed for both mu and logvar.
        """
        batch_size = mu.size(0)
        
        # Normalize the mu (mean vectors) and logvar (log variance vectors) for cosine similarity
        mu_norm = F.normalize(mu, p=2, dim=1)  # Normalize along the feature dimension for mu
        logvar_norm = F.normalize(logvar, p=2, dim=1)  # Normalize along the feature dimension for logvar

        # Compute cosine similarity using dot product (cosine similarity = mu[i] . mu[j])
        cosine_sim_matrix_mu = torch.matmul(mu_norm, mu_norm.t())  # Shape: (batch_size, batch_size)
        cosine_sim_matrix_logvar = torch.matmul(logvar_norm, logvar_norm.t())  # Shape: (batch_size, batch_size)

        # Compute the dynamic margin based on the average cosine similarity
        avg_cosine_sim_mu = cosine_sim_matrix_mu.sum() / (batch_size * (batch_size - 1))  # Average cosine similarity for mu
        avg_cosine_sim_logvar = cosine_sim_matrix_logvar.sum() / (batch_size * (batch_size - 1))  # Average cosine similarity for logvar
        
        # Combine the cosine similarity matrices for mu and logvar
        cosine_sim_matrix = (cosine_sim_matrix_mu + cosine_sim_matrix_logvar) / 2  # Averaging cosine similarities

        # Calculate dynamic margin based on the mean cosine similarity
        dynamic_margin = avg_cosine_sim_mu + avg_cosine_sim_logvar

        # Determine whether pairs belong to the same cluster
        labels = torch.tensor([data[i].label for i in range(len(data))]).to("cuda")
        same_cluster = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()  # Shape: (batch_size, batch_size)

        # Apply contrastive loss formula based on cosine similarity
        loss_same_cluster = same_cluster * (1 - cosine_sim_matrix).pow(2)  # Loss for same cluster: minimize cosine distance (maximize similarity)
        loss_diff_cluster = (1 - same_cluster) * F.relu(cosine_sim_matrix - dynamic_margin).pow(2)  # Loss for different clusters: enforce dynamic margin

        # Total contrastive loss
        contrastive_loss = (loss_same_cluster + loss_diff_cluster).sum() / (batch_size * (batch_size - 1))

        # Compute the entropy of the cluster assignments for each sample
        # Using the softmax over mu (assuming mu can be used to assign clusters)
        cluster_probs = F.softmax(mu, dim=1)  # Assuming softmax over the latent variables for clustering
        entropy_loss = -torch.sum(cluster_probs * torch.log(cluster_probs + 1e-8)) / batch_size  # Regularization term

        # Total loss includes the contrastive loss and the entropy regularization
        total_loss = lambda_contrastive * contrastive_loss + lambda_entropy * entropy_loss

        return total_loss
    
    
    def get_weighted_penalization(self, adj, data, lambda_penalization):
        """
        Compute a penalization representing the MSE with the expected number of 
        nodes and edges and multiply it by lambda_penalization.
        """
        n_edges= adj.sum(dim=(1,2))/2
        num_nodes = torch.sum(torch.diagonal(adj, dim1=1, dim2=2),dim=1)

        adj_cubed = torch.matmul(adj, torch.matmul(adj, adj))
        # print(f"adj_cubed {adj_cubed.shape}")

        num_triangles = torch.diagonal(adj_cubed, dim1=1, dim2=2).sum(dim=1) / 6.0

        coherence_loss = self.edge_node_coherence_loss(adj)
        
        MSE_n_nodes = self.compute_mse(num_nodes, data.stats[:,0])
        MSE_n_edges = self.compute_mse(n_edges, data.stats[:,1]) 
        MSE_n_triangles = self.compute_mse(num_triangles, data.stats[:,2])   


        # feature_losses = (MSE_n_nodes + MSE_n_edges + 0.0001*coherence_loss)
        feature_losses = (MSE_n_nodes + MSE_n_edges + 0.001*MSE_n_triangles+0.0001*coherence_loss)
        # print(f"node loss {MSE_n_nodes}, grad_fn {MSE_n_nodes.grad_fn}")
        # print(f"edge loss {MSE_n_edges}, grad_fn {MSE_n_edges.grad_fn}")
        # print(f"triangle loss {0.001*MSE_n_triangles}, grad_fn {MSE_n_triangles.grad_fn}")
        # print(f"coherence loss {0.0001*coherence_loss}, grad_fn {coherence_loss.grad_fn}")
        # print(f"feature loss {feature_losses}, grad_fn {feature_losses.grad_fn}")
        # sys.exit()

        return lambda_penalization * feature_losses
    
    def compute_mse(self, pred, gt):
        """
        Compute the mean squared error (MSE) between the predicted adjacency matrix and the ground truth.
        """
        mean_gt_per_column = torch.mean(gt, dim=0)
        std_gt_per_column = torch.std(gt, dim=0)
        
        norm_gt = (gt - mean_gt_per_column) / std_gt_per_column
        norm_pred = (pred - mean_gt_per_column) / std_gt_per_column
        
        mse = F.mse_loss(norm_pred, norm_gt, reduction='mean')
    
        return mse




    
