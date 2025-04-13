import os
import math
import networkx as nx
import numpy as np
import scipy as sp
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
import argparse

import multiprocessing as mp

from tqdm import tqdm
import scipy.sparse as sparse
from sklearn.cluster import KMeans
from torch_geometric.data import Data
from torch_geometric.utils import scatter

from NGG.utils.extract_feats import extract_feats, extract_numbers

import sys

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import requests
import tarfile
import shutil
from itertools import product
from argparse import Namespace


def download_data_from_s3(url, target_folder):
    """
    Downloads and extracts the dataset from the given S3 URL if the target folder does not exist.

    Args:
    - url (str): The S3 URL to download the dataset from.
    - target_folder (str): The local folder where the dataset should be stored.
    """
    if not os.path.exists(target_folder):
        print(f"Downloading dataset from {url}...")
        gz_path = "./data.gz"
        with requests.get(url, stream=True) as r:
            with open(gz_path, "wb") as f:
                shutil.copyfileobj(r.raw, f)
        print("Extracting dataset...")
        with tarfile.open(gz_path, "r:gz") as tar_ref:
            tar_ref.extractall("./")
        os.remove(gz_path)
        print(f"Dataset downloaded and extracted to {target_folder}.")
    else:
        print(f"Dataset already exists at {target_folder}.")


s3_url = "https://minio.lab.sspcloud.fr/lstepien/NGG/data.tar.gz"
target_folder = "./data"


def preprocess_dataset(
    dataset,
    n_max_nodes,
    spectral_emb_dim,
    normalize=False,
    labelize=False,
    additional_features_bool=False,
    generate=False,
    kmeans=None,
    n_clusters=None,
):
    if not os.path.exists(target_folder) or not os.listdir(target_folder):
        download_data_from_s3(s3_url, target_folder)

    data_lst = []
    if dataset == "test":
        filename = f"./data/dataset_{dataset}_nodes_{n_max_nodes}_embed_dim{spectral_emb_dim}_with_labels_{labelize}_gen{generate}.pt"

        desc_file = "./data/" + dataset + "/test.txt"

        if os.path.isfile(filename):
            data_lst = torch.load(filename)
            if labelize:

                data_lst, kmeans = assign_labels(data_lst, kmeans, n_clusters)

            print(f"Dataset {filename} loaded from file")

        else:
            fr = open(desc_file, "r")
            for line in fr:
                line = line.strip()
                tokens = line.split(",")
                graph_id = tokens[0]
                desc = tokens[1:]
                desc = "".join(desc)

                feats_stats = extract_numbers(desc)
                feats_stats = torch.FloatTensor(feats_stats).unsqueeze(0)
                data_lst.append(
                    Data(stats=feats_stats, filename=graph_id)
                )  # prompt=desc for testing
            if labelize:
                data_lst, kmeans = assign_labels(data_lst, kmeans, n_clusters)

            fr.close()
            torch.save(data_lst, filename)
            print(f"Dataset {filename} saved")

    else:

        filename = f"./data/dataset_{dataset}_nodes_{n_max_nodes}_embed_dim{spectral_emb_dim}norm_{normalize}_with_labels_{labelize}_additional_{additional_features_bool}_gen{generate}.pt"

        graph_path = "./data/" + dataset + "/graph"
        desc_path = "./data/" + dataset + "/description"

        if os.path.isfile(filename):
            data_lst = torch.load(filename)
            if labelize:
                data_lst, kmeans = assign_labels(data_lst, kmeans, n_clusters)

            print(f"Dataset {filename} loaded from file")

        else:
            # traverse through all the graphs of the folder
            files = [f for f in os.listdir(graph_path) if f != ".keep"]
            adjs = []
            eigvals = []

            # eigvecs = []
            # n_nodes = []

            max_eigval = 0
            min_eigval = 0
            for fileread in tqdm(files):
                tokens = fileread.split("/")
                idx = tokens[-1].find(".")
                filen = tokens[-1][:idx]
                extension = tokens[-1][idx + 1 :]
                fread = os.path.join(graph_path, fileread)
                fstats = os.path.join(desc_path, filen + ".txt")
                # load dataset to networkx
                if extension == "graphml":
                    G = nx.read_graphml(fread)
                    # Convert node labels back to tuples since GraphML stores them as strings
                    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
                else:
                    G = nx.read_edgelist(fread)
                # use canonical order (BFS) to create adjacency matrix
                ### BFS & DFS from largest-degree node

                CGs = [G.subgraph(c) for c in nx.connected_components(G)]

                # rank connected componets from large to small size
                CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)

                node_list_bfs = []
                for ii in range(len(CGs)):
                    node_degree_list = [(n, d) for n, d in CGs[ii].degree()]
                    degree_sequence = sorted(
                        node_degree_list, key=lambda tt: tt[1], reverse=True
                    )

                    bfs_tree = nx.bfs_tree(CGs[ii], source=degree_sequence[0][0])
                    node_list_bfs += list(bfs_tree.nodes())

                adj_bfs = nx.to_numpy_array(G, nodelist=node_list_bfs)

                adj = torch.from_numpy(adj_bfs).float()
                diags = np.sum(adj_bfs, axis=0)
                diags = np.squeeze(np.asarray(diags))
                D = sparse.diags(diags).toarray()
                L = D - adj_bfs
                with sp.special.errstate(singular="ignore"):
                    diags_sqrt = 1.0 / np.sqrt(diags)
                diags_sqrt[np.isinf(diags_sqrt)] = 0
                DH = sparse.diags(diags).toarray()
                L = np.linalg.multi_dot((DH, L, DH))
                L = torch.from_numpy(L).float()
                eigval, eigvecs = torch.linalg.eigh(L)
                eigval = torch.real(eigval)
                eigvecs = torch.real(eigvecs)
                idx = torch.argsort(eigval)
                eigvecs = eigvecs[:, idx]

                edge_index = torch.nonzero(adj).t()

                size_diff = n_max_nodes - G.number_of_nodes()
                x = torch.zeros(G.number_of_nodes(), spectral_emb_dim + 1)
                x[:, 0] = torch.mm(adj, torch.ones(G.number_of_nodes(), 1))[:, 0] / (
                    n_max_nodes - 1
                )
                mn = min(G.number_of_nodes(), spectral_emb_dim)
                mn += 1
                x[:, 1:mn] = eigvecs[:, :spectral_emb_dim]
                # additional_features_dim: number of additional features
                # additional_features: degree of node, sum of degrees of neighbourhood, number of nodes in connected component,number of edges in connected component, number of triangles where node is involved, is there a path back to this node ? (0,1), shortest path back to node if 1 else 0, longest_path back to node if 1 else 0
                if additional_features_bool:
                    additional_features = calculate_additional_features(G)
                    x = torch.cat((x, additional_features), dim=1)
                    # print(f"additional features added, New shape of x: {x.shape}")
                if normalize:
                    adj = adj + torch.eye(G.number_of_nodes())
                    # print(adj)
                    # sys.exit()

                adj = F.pad(adj, [0, size_diff, 0, size_diff])
                adj = adj.unsqueeze(0)

                feats_stats = extract_feats(fstats)
                feats_stats = torch.FloatTensor(feats_stats).unsqueeze(0)

                edge_features = torch.ones(edge_index.size(1), x.size(1) * 2)
                for i in range(edge_index.size(1)):
                    edge_features[i] = torch.cat(
                        (x[edge_index[0, i]], x[edge_index[1, i]])
                    )

                data_lst.append(
                    Data(
                        x=x,
                        edge_index=edge_index,
                        edge_features=edge_features,
                        A=adj,
                        stats=feats_stats,
                        filename=filen,
                    )
                )

            if generate and dataset == "train":
                print("Generating graphs")
                # Generate 50 000 graphs
                for i in tqdm(range(100000), desc="Generating graphs"):
                    n_nodes = np.random.randint(5, 50)
                    p_edges = np.random.uniform(0, 1)
                    G = nx.fast_gnp_random_graph(n_nodes, p_edges)
                    x, edge_index, adj = find_spectral_embedding(G, spectral_emb_dim)
                    if normalize:
                        adj = adj + torch.eye(n_nodes)
                    if additional_features_bool:
                        additional_features = calculate_additional_features(G)
                        x = torch.cat((x, additional_features), dim=1)
                    size_diff = n_max_nodes - G.number_of_nodes()
                    adj = F.pad(adj, [0, size_diff, 0, size_diff])
                    adj = adj.unsqueeze(0)
                    feats_stats = compute_graph_properties(G)
                    feats_stats = torch.FloatTensor(feats_stats).unsqueeze(0)

                    # #check everything is fine
                    # print(f"Graph {i} :")
                    # print(f"x and x.shape: {x} and {x.shape}")
                    # print(f"edge_index and edge_index.shape: {edge_index} and {edge_index.shape}")
                    # print(f"adj and adj.shape: {adj} and {adj.shape}")
                    # print(f"feats_stats and feats_stats.shape: {feats_stats} and {feats_stats.shape}")

                    data_lst.append(
                        Data(
                            x=x,
                            edge_index=edge_index,
                            A=adj,
                            stats=feats_stats,
                            filename=f"gen_{i}",
                        )
                    )

            # if generate:

            #     print("Generating graphs in parallel...")
            #     n_graphs = 100
            #     data_lst += generate_graphs_parallel(n_graphs, spectral_emb_dim, normalize, additional_features_bool, n_max_nodes)
            #     print("Graph generation completed.")

            if labelize:
                data_lst, kmeans = assign_labels(data_lst, kmeans, n_clusters)

            data_lst = normalize_last_n_columns(data_lst, 11)
            torch.save(data_lst, filename)
            print(f"Dataset {filename} saved")

    if labelize:
        return data_lst, kmeans
    return data_lst


def generate_single_graph(args):
    i, spectral_emb_dim, normalize, additional_features_bool, n_max_nodes = args
    n_nodes = np.random.randint(5, 50)
    p_edges = np.random.uniform(0, 1)
    G = nx.fast_gnp_random_graph(n_nodes, p_edges)
    x, edge_index, adj = find_spectral_embedding(G, spectral_emb_dim)

    if normalize:
        adj = adj + torch.eye(n_nodes)
    if additional_features_bool:
        additional_features = calculate_additional_features(G)
        x = torch.cat((x, additional_features), dim=1)

    size_diff = n_max_nodes - G.number_of_nodes()
    adj = F.pad(adj, [0, size_diff, 0, size_diff])
    adj = adj.unsqueeze(0)

    feats_stats = compute_graph_properties(G)
    feats_stats = torch.FloatTensor(feats_stats).unsqueeze(0)

    return Data(
        x=x, edge_index=edge_index, A=adj, stats=feats_stats, filename=f"gen_{i}"
    )


def generate_graphs_parallel(
    n_graphs, spectral_emb_dim, normalize, additional_features_bool, n_max_nodes
):
    args = [
        (i, spectral_emb_dim, normalize, additional_features_bool, n_max_nodes)
        for i in range(n_graphs)
    ]
    with mp.Pool(mp.cpu_count()) as pool:
        for _ in tqdm(
            pool.imap(generate_single_graph, args),
            total=n_graphs,
            desc="Generating graphs",
        ):
            pass

    with mp.Pool(mp.cpu_count()) as pool:
        data_lst = pool.map(generate_single_graph, args)
    return data_lst


def compute_graph_properties(graph):
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    n_triangles = sum(nx.triangles(graph).values()) // 3
    avg_degree = sum(dict(graph.degree()).values()) / n_nodes
    global_clustering_coeff = nx.average_clustering(graph)
    max_k_core = max(nx.core_number(graph).values())
    n_communities = (
        nx.algorithms.community.modularity_max.greedy_modularity_communities(graph)
        if n_edges > 0
        else [0] * n_nodes
    )

    return [
        n_nodes,
        n_edges,
        n_triangles,
        avg_degree,
        global_clustering_coeff,
        max_k_core,
        len(n_communities),
    ]


def find_spectral_embedding(G, spectral_emb_dim):
    CGs = [G.subgraph(c) for c in nx.connected_components(G)]
    # rank connected componets from large to small size
    CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
    node_list_bfs = []
    for ii in range(len(CGs)):
        node_degree_list = [(n, d) for n, d in CGs[ii].degree()]
        degree_sequence = sorted(node_degree_list, key=lambda tt: tt[1], reverse=True)

        bfs_tree = nx.bfs_tree(CGs[ii], source=degree_sequence[0][0])
        node_list_bfs += list(bfs_tree.nodes())
    adj_bfs = nx.to_numpy_array(G, nodelist=node_list_bfs)
    adj = torch.from_numpy(adj_bfs).float()
    diags = np.sum(adj_bfs, axis=0)
    diags = np.squeeze(np.asarray(diags))
    D = sparse.diags(diags).toarray()
    L = D - adj_bfs
    with sp.special.errstate(singular="ignore"):
        diags_sqrt = np.sqrt(diags)
        diags_sqrt[diags_sqrt == 0] = 1e-10
        diags_sqrt = 1.0 / diags_sqrt
    diags_sqrt[np.isinf(diags_sqrt)] = 0
    DH = sparse.diags(diags).toarray()
    L = np.linalg.multi_dot((DH, L, DH))
    L = torch.from_numpy(L).float()
    eigval, eigvecs = torch.linalg.eigh(L)
    eigval = torch.real(eigval)
    eigvecs = torch.real(eigvecs)
    idx = torch.argsort(eigval)
    eigvecs = eigvecs[:, idx]

    edge_index = torch.nonzero(adj).t()

    x = torch.zeros(G.number_of_nodes(), spectral_emb_dim + 1)
    x[:, 0] = torch.mm(adj, torch.ones(G.number_of_nodes(), 1))[:, 0] / (
        n_max_nodes - 1
    )
    mn = min(G.number_of_nodes(), spectral_emb_dim)
    mn += 1
    x[:, 1:mn] = eigvecs[:, :spectral_emb_dim]

    return x, edge_index, adj


def normalize_last_n_columns(data_lst, n):
    # Step 1: Extract the last n columns from each x in data_lst
    last_n_features = [data.x[:, -n:] for data in data_lst]

    # Step 2: Stack these columns vertically to form a matrix
    stacked_features = torch.vstack(last_n_features)

    # Step 3: Compute the maximum for each column
    max_values = torch.max(stacked_features, dim=0)[0]

    # Step 4: Normalize each column by dividing by its maximum value
    normalized_features = stacked_features / max_values

    # Step 5: Replace the last n columns of each x with the normalized values
    start_idx = 0
    for data in data_lst:
        num_nodes = data.x.shape[0]
        normalized_data = normalized_features[start_idx : start_idx + num_nodes]
        data.x[:, -n:] = normalized_data
        start_idx += num_nodes

    return data_lst


def calculate_additional_features(G):
    """
    Calculate additional features for each node in the graph.

    Parameters:
    G (networkx.Graph): The graph for which features are calculated.

    Returns:
    torch.Tensor: A tensor containing the additional features for each node.
    """

    # Initialize the additional_features tensor
    additional_features_dim = 11  # Updated to include more features
    additional_features = torch.zeros(G.number_of_nodes(), additional_features_dim)

    # Degree of node
    degrees = torch.tensor([d for _, d in G.degree()]).float()
    additional_features[:, 0] = degrees

    # Sum of degrees of the neighborhood
    neighborhood_degrees = torch.tensor(
        [sum(degrees[list(map(int, G.neighbors(node)))]) for node in G.nodes()]
    ).float()
    additional_features[:, 1] = neighborhood_degrees

    # Number of nodes in the connected component
    components = list(nx.connected_components(G))
    component_sizes = torch.tensor([len(c) for c in components])
    component_map = {
        node: size
        for component, size in zip(components, component_sizes)
        for node in component
    }
    n_node_component = torch.tensor([component_map[node] for node in G.nodes()]).float()
    additional_features[:, 2] = n_node_component

    # Number of edges in the connected component
    component_edges = torch.tensor(
        [G.subgraph(c).number_of_edges() for c in components]
    )
    edge_map = {
        node: edges
        for component, edges in zip(components, component_edges)
        for node in component
    }
    n_edges_component = torch.tensor([edge_map[node] for node in G.nodes()]).float()
    additional_features[:, 3] = n_edges_component

    # Number of triangles where the node is involved
    triangles = nx.triangles(G)
    n_triangles = torch.tensor([triangles[node] for node in G.nodes()]).float()
    additional_features[:, 4] = n_triangles

    # Is there a path back to this node? (Self-loop existence)
    cycle_existence = torch.tensor(
        [1 if nx.has_path(G, node, node) else 0 for node in G.nodes()]
    )
    additional_features[:, 5] = cycle_existence

    # Local clustering coefficient
    local_clustering = nx.clustering(G)
    clustering_coeffs = torch.tensor(
        [local_clustering[node] for node in G.nodes()]
    ).float()
    additional_features[:, 6] = clustering_coeffs

    # Betweenness centrality
    betweenness_centrality = nx.betweenness_centrality(G)
    betweenness = torch.tensor(
        [betweenness_centrality[node] for node in G.nodes()]
    ).float()
    additional_features[:, 7] = betweenness

    # Core number
    core_number = nx.core_number(G)
    core_nums = torch.tensor([core_number[node] for node in G.nodes()]).float()
    additional_features[:, 8] = core_nums

    # Closeness centrality
    closeness_centrality = nx.closeness_centrality(G)
    closeness = torch.tensor([closeness_centrality[node] for node in G.nodes()]).float()
    additional_features[:, 9] = closeness

    # PageRank
    pagerank = nx.pagerank(G)
    pagerank_values = torch.tensor([pagerank[node] for node in G.nodes()]).float()
    additional_features[:, 10] = pagerank_values

    # Check the vector of the first node
    # print(f"Additional features for the first node: {additional_features[0,6]}")
    if torch.isnan(additional_features).any():
        print("NaN in additional features")
        for i in range(len(additional_features)):
            if torch.isnan(additional_features[i]).any():
                print(f"NaN found in additional features at index {i}")
                print(f"The row is {additional_features[i]}")
        sys.exit()

    return additional_features


def construct_nx_from_adj(adj):
    G = nx.from_numpy_array(adj, create_using=nx.Graph)
    to_remove = []
    for node in G.nodes():
        if G.degree(node) == 0:
            to_remove.append(node)
    G.remove_nodes_from(to_remove)
    return G


def handle_nan(x):
    if math.isnan(x):
        return float(-100)
    return x


def masked_instance_norm2D(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-5):
    """
    x: [batch_size (N), num_objects (L), num_objects (L), features(C)]
    mask: [batch_size (N), num_objects (L), num_objects (L), 1]
    """
    mask = mask.view(x.size(0), x.size(1), x.size(2), 1).expand_as(x)
    mean = torch.sum(x * mask, dim=[1, 2]) / torch.sum(mask, dim=[1, 2])  # (N,C)
    var_term = (
        (x - mean.unsqueeze(1).unsqueeze(1).expand_as(x)) * mask
    ) ** 2  # (N,L,L,C)
    var = torch.sum(var_term, dim=[1, 2]) / torch.sum(mask, dim=[1, 2])  # (N,C)
    mean = mean.unsqueeze(1).unsqueeze(1).expand_as(x)  # (N, L, L, C)
    var = var.unsqueeze(1).unsqueeze(1).expand_as(x)  # (N, L, L, C)
    instance_norm = (x - mean) / torch.sqrt(var + eps)  # (N, L, L, C)
    instance_norm = instance_norm * mask
    return instance_norm


def masked_layer_norm2D(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-5):
    """
    x: [batch_size (N), num_objects (L), num_objects (L), features(C)]
    mask: [batch_size (N), num_objects (L), num_objects (L), 1]
    """
    mask = mask.view(x.size(0), x.size(1), x.size(2), 1).expand_as(x)
    mean = torch.sum(x * mask, dim=[3, 2, 1]) / torch.sum(mask, dim=[3, 2, 1])  # (N)
    var_term = ((x - mean.view(-1, 1, 1, 1).expand_as(x)) * mask) ** 2  # (N,L,L,C)
    var = torch.sum(var_term, dim=[3, 2, 1]) / torch.sum(mask, dim=[3, 2, 1])  # (N)
    mean = mean.view(-1, 1, 1, 1).expand_as(x)  # (N, L, L, C)
    var = var.view(-1, 1, 1, 1).expand_as(x)  # (N, L, L, C)
    layer_norm = (x - mean) / torch.sqrt(var + eps)  # (N, L, L, C)
    layer_norm = layer_norm * mask
    return layer_norm


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def assign_labels(data, kmeans=None, n_clusters=3):
    """
    Assigns cluster labels to each graph in the data list.

    Args:
    - data: List of Data objects containing graph information.
    - n_clusters: Number of clusters for K-means. Set to 3 after WCSS visualization.

    Returns:
    - data: Updated list with cluster labels assigned.
    """
    properties = torch.cat([graph.stats for graph in data], axis=0)

    properties_np = properties.numpy()

    # Perform K-means clustering
    if kmeans is None:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(properties_np)
    else:
        labels = kmeans.predict(properties_np)

    # Assign labels back to each graph
    for graph, label in zip(data, labels):
        graph.label = torch.tensor([label])  # Add the label as a tensor

    return data, kmeans


def create_deepsets_train_dataset(hidden_dim, batch_size, device):
    n_train = 100000

    X_train, batch = [], []
    for i in range(n_train):
        sample = np.random.normal(0, 5, hidden_dim)
        X_train.append(sample)
        batch.append(i // batch_size)

    X_train = torch.tensor(X_train).to(device).to(torch.float)
    batch = torch.tensor(batch).to(device)
    y_train = scatter(X_train, batch, dim=0, reduce="sum").to(device)
    return X_train, y_train, batch


# Function to compute graph features
def compute_graph_features_from_adj(adj_matrix):
    graph = nx.from_numpy_array(adj_matrix)

    # Compute features
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    avg_degree = sum(dict(graph.degree()).values()) / n_nodes if n_nodes > 0 else 0
    n_triangles = sum(nx.triangles(graph).values()) // 3
    clustering_coeff = nx.average_clustering(graph)
    max_core = max(nx.core_number(graph).values())
    communities = nx.community.greedy_modularity_communities(graph)
    n_communities = len(communities)

    return [
        n_nodes,
        n_edges,
        avg_degree,
        n_triangles,
        clustering_coeff,
        max_core,
        n_communities,
    ]


def to_labels(adj, kmeans=None):
    """
    Computes graph features and clustering labels using torch_geometric utilities.
    """

    arr_adj = adj.detach().cpu().numpy()
    # Use joblib for parallel computation
    all_properties = Parallel(n_jobs=-1)(
        delayed(compute_graph_features_from_adj)(adj_matrix) for adj_matrix in arr_adj
    )

    # Convert features to numpy float64
    all_properties = np.array(all_properties, dtype=np.float64)

    if kmeans is not None:
        kmeans.cluster_centers_ = kmeans.cluster_centers_.astype(
            float
        )  # Handles type errors
        # Cluster label prediction
        labels = kmeans.predict(
            all_properties
        )  # Predict labels for all graphs in the batch

        # Add the labels to the properties
        all_properties = np.hstack((all_properties, labels.reshape(-1, 1)))

    # Transform back to a tensor of size (batch_size, nfeatures + 1)
    return torch.tensor(all_properties, dtype=torch.float32).to(adj.device)


def generate_args_from_config(config):
    """
    Generate argparse.Namespace-like objects for each combination of hyperparameters
    defined in the config.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        List[Namespace]: A list of Namespace objects, one for each combination of hyperparameters.
    """
    # TODO: change the name of the experiment for each set of configs
    # Extract model configuration (fixed values)
    model_config = config["model_config"]

    # Extract hyperparameters (grid search values)
    hyperparameters = config["hyperparameters"]

    # Generate all combinations of hyperparameters
    keys, values = zip(*hyperparameters.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]

    # Create Namespace objects for each combination
    args_list = []
    for combination in combinations:
        # Merge model_config and the current hyperparameter combination
        merged_config = {**model_config, **combination}
        # Convert to Namespace
        args = Namespace(**merged_config)
        args_list.append(args)

    return args_list


## testing script
if __name__ == "__main__":
    # create parser for normalize, labelize, additional_features_bool
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--normalize", action="store_true", help="normalize the adjacency matrix"
    )
    parser.add_argument("--labelize", action="store_true", help="labelize the dataset")
    parser.add_argument(
        "--additional_features",
        action="store_true",
        help="add additional features to the dataset",
    )

    parser.add_argument("--gen", action="store_true", help="generate 50 000 graphs")

    args = parser.parse_args()
    # print(f"Visualizing the Test dataset")
    # dataset = 'test'
    # n_max_nodes = 50
    # spectral_emb_dim = 10
    # data_lst = preprocess_dataset(dataset, n_max_nodes, spectral_emb_dim,args.normalize, args.labelize, args.additional_features)
    # print(len(data_lst))
    # print(data_lst[0])
    # print(data_lst[0].x.shape)
    # print(data_lst[0].stats)# tensor of size 1x7
    # # print(data_lst[0].prompt)
    # print(data_lst[0].filename) #file name or index of the graph
    # print("-------------------")
    print(f"Visualizing the Train dataset")
    dataset = "train"
    n_max_nodes = 50
    spectral_emb_dim = 10
    if args.labelize:

        data_lst, kmeans = preprocess_dataset(
            dataset,
            n_max_nodes,
            spectral_emb_dim,
            args.normalize,
            args.labelize,
            args.additional_features,
            args.gen,
        )
    else:
        data_lst = preprocess_dataset(
            dataset,
            n_max_nodes,
            spectral_emb_dim,
            args.normalize,
            args.labelize,
            args.additional_features,
            args.gen,
        )

    print(len(data_lst))
    print(data_lst[0])
    print("\n")
    print(data_lst[0].x)  # tensor of shape num_nodes, spectral_emb_dim+1
    print("\n")
    print(data_lst[0].edge_index)  # tensor of shape 2 x num_edges
    print("\n")

    print(
        data_lst[0].edge_features
    )  # tensor of shape num_edges x 2*(spectral_emb_dim+1)
    print("\n")

    print(data_lst[0].A)  # tensor of shape 1 , max nodes, max nodes
    print("\n")
    print(data_lst[0].stats)  # tensor of size 1x7
    print("\n")
    print(data_lst[0].filename)  # file name or index of the graph
