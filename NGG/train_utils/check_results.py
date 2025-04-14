
import torch
import csv
from tqdm import tqdm
from NGG.denoiser.denoise_model import sample
from NGG.utils.utils import construct_nx_from_adj
import os 
from NGG.utils.verify_graph_features import load_graphs_into_dataframe, compute_features_vectorized, compare_reconstructed_and_prompted_graphs_v2
import json

import networkx as nx


def check_results(args, device, autoencoder, denoise_model, test_loader,testset,betas):
    folder_directory = os.path.join("progression_archive",args.name)
    if os.path.exists(folder_directory):
        #find the iteration of this path with the biggest number at the end
        paths= [f for f in os.listdir("progression_archive") if f.startswith(args.name)]
        # print(paths)
        if len(paths)==1:
            folder_directory = folder_directory + "_1"
        else:
            numbers = [int(f.split("_")[-1]) if f.split("_")[-1].isdigit() else 0 for f in paths ]
            # print(numbers)
            max_number = max(numbers)
            folder_directory = folder_directory + "_" + str(max_number+1)
    os.makedirs(folder_directory)
    
    csv_file = os.path.join(folder_directory, "output.csv")
    # Save to a CSV file
    with open(csv_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write the header
        writer.writerow(["graph_id", "edge_list"])
        for k, data in enumerate(tqdm(test_loader, desc='Processing test set',)):
            data = data.to(device)
            
            stat = data.stats
            bs = stat.size(0)

            graph_ids = data.filename

            samples = sample(denoise_model, data.stats, latent_dim=args.latent_dim, timesteps=args.timesteps, betas=betas, batch_size=bs)
            x_sample = samples[-1]
            # print(x_sample.shape)
            # sys.exit()

            if args.AE != 'GMVAE':
                x_sample = torch.cat((x_sample, stat), dim=1) 
            else:
                # Assume labels from the test dataset
                labels = torch.tensor([data[i].label for i in range(len(data))], device=x_sample.device)  # Shape: (batch_size,)

                # Create one-hot encodings for the labels
                # y_onehot = torch.zeros(labels.size(0), 3, device=labels.device)
                # y_onehot.scatter_(1, labels.unsqueeze(1).long(), 1)

                # Concatenate z and y_onehot
                x_sample = torch.cat([x_sample, stat, labels.unsqueeze(1)], dim=1)

            adj = autoencoder.decode_mu(x_sample)
            stat_d = torch.reshape(stat, (-1, args.n_condition))


            for i in range(stat.size(0)):
                stat_x = stat_d[i]

                Gs_generated = construct_nx_from_adj(adj[i,:,:].detach().cpu().numpy())
                stat_x = stat_x.detach().cpu().numpy()

                # n_nodes, n_edges, n_triangles, avg_degree, global_clustering_coeff, max_k_core, n_communities = compute_graph_properties(Gs_generated)
                # print(f"Number of nodes: {n_nodes}, Number of edges: {n_edges}, Number of triangles: {n_triangles}, Average degree: {avg_degree}, Global clustering coefficient: {global_clustering_coeff}, Max k-core: {max_k_core}, Number of communities: {n_communities}")
                # print(f"Stats: {stat_x}")
                # print(f"------------------------------------------")

                # Define a graph ID
                graph_id = graph_ids[i]

                # Convert the edge list to a single string
                edge_list_text = ", ".join([f"({u}, {v})" for u, v in Gs_generated.edges()])           
                # Write the graph ID and the full edge list as a single row
                writer.writerow([graph_id, edge_list_text])
    
    graphs_df = load_graphs_into_dataframe(csv_file)
    result_df = compute_features_vectorized(graphs_df)
    result_df.to_csv(csv_file.replace(".csv", "_with_features.csv"), index=False)
    
    metrics = compare_reconstructed_and_prompted_graphs_v2(result_df, testset,return_log=csv_file.replace(".csv", "_log.txt"))
    
    # save json args
    with open(os.path.join(folder_directory, "args.json"), "w") as f:
        json.dump(vars(args), f)
    
    return metrics

def compute_graph_properties(graph):
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    n_triangles = sum(nx.triangles(graph).values()) // 3
    avg_degree = sum(dict(graph.degree()).values()) / n_nodes
    global_clustering_coeff = nx.average_clustering(graph)
    max_k_core = max(nx.core_number(graph).values())
    n_communities = nx.algorithms.community.modularity_max.greedy_modularity_communities(graph)
    return n_nodes, n_edges, n_triangles, avg_degree, global_clustering_coeff, max_k_core, len(n_communities)

