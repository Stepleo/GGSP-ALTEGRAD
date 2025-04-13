import pandas as pd
import networkx as nx
import ast
from typing import List

from NGG.utils.utils import preprocess_dataset, compute_graph_features_from_adj, Data

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import math

# Load graphs into a DataFrame
def load_graphs_into_dataframe(csv_path):
    df = pd.read_csv(csv_path)

    # Convert edge list to adjacency matrix and store in a DataFrame column
    def edge_list_to_adj_matrix(edge_list_str): 
        edge_list = ast.literal_eval(edge_list_str)
        graph = nx.Graph()
        graph.add_edges_from(edge_list)
        adj_matrix = nx.to_numpy_array(graph, nodelist=sorted(graph.nodes()))
        return adj_matrix

    df['adj_matrix'] = df['edge_list'].apply(edge_list_to_adj_matrix)
    return df

# Vectorized feature computation
def compute_features_vectorized(df):
    # Convert adjacency matrices back to graphs and compute features
    df['features'] = df['adj_matrix'].apply(compute_graph_features_from_adj)
    feature_names = [
        "n_nodes", "n_edges", "avg_degree", 
        "n_triangles", "clustering_coeff", "max_k_core", "n_communities"
    ]

    # Split features into separate columns
    features_df = pd.DataFrame(df['features'].tolist(), columns=feature_names)
    return pd.concat([df.drop(columns=['features']), features_df], axis=1)

def compare_reconstructed_and_prompted_graphs(result_df: pd.DataFrame, data_lst: List[Data],csv_path: str):
    diff_nodes, diff_edges, diff_avg_degree, diff_triangles, diff_clustering_coeff, diff_max_core, diff_communities = [], [], [], [], [], [], []
    MSE_vectors=[]
    
    for i, data in enumerate(data_lst):
        filename=data.filename
        stats=data.stats
        #get the row of the dataframe where graph_id == filename
        row = result_df[result_df['graph_id'] == filename]
        if row.empty:
            print(f"Graph {filename} not found in the DataFrame")
            continue
        
        diff_nodes.append(stats[0,0] - row['n_nodes'].values[0])
        diff_edges.append(stats[0,1] - row['n_edges'].values[0])
        diff_avg_degree.append(stats[0,2] - row['avg_degree'].values[0])
        diff_triangles.append(stats[0,3] - row['n_triangles'].values[0])
        diff_clustering_coeff.append(stats[0,4] - row['clustering_coeff'].values[0])
        diff_max_core.append(stats[0,5] - row['max_k_core'].values[0])
        diff_communities.append(stats[0,6] - row['n_communities'].values[0])
        
        #compute MSe for the entire vector
        MSE_vector=np.square(stats - row[['n_nodes', 'n_edges', 'avg_degree', 'n_triangles', 'clustering_coeff', 'max_k_core', 'n_communities']].values[0])
        
    MSE_vectors.append(MSE_vector)
        
    print(f"Mean difference in number of nodes: {sum(diff_nodes) / len(diff_nodes)}")
    print(f"Mean difference in number of edges: {sum(diff_edges) / len(diff_edges)}")
    print(f"Mean difference in average degree: {sum(diff_avg_degree) / len(diff_avg_degree)}")
    print(f"Mean difference in number of triangles: {sum(diff_triangles) / len(diff_triangles)}")
    print(f"Mean difference in clustering coefficient: {sum(diff_clustering_coeff) / len(diff_clustering_coeff)}")
    print(f"Mean difference in max k-core: {sum(diff_max_core) / len(diff_max_core)}")
    print(f"Mean difference in number of communities: {sum(diff_communities) / len(diff_communities)}")
    
    #plot the differences in one major plot using subplots
    
    # Create subplots
    fig, axs = plt.subplots(4, 2, figsize=(15, 15))

    # Data and titles
    data_and_titles = [
        (diff_nodes, 'Difference in number of nodes', axs[0, 0]),
        (diff_edges, 'Difference in number of edges', axs[0, 1]),
        (diff_avg_degree, 'Difference in average degree', axs[1, 0]),
        (diff_triangles, 'Difference in number of triangles', axs[1, 1]),
        (diff_clustering_coeff, 'Difference in clustering coefficient', axs[2, 0]),
        (diff_max_core, 'Difference in max k-core', axs[2, 1]),
        (diff_communities, 'Difference in number of communities', axs[3, 0]),
    ]

    # Plot each histogram with mean line and improved aesthetics
    for diff, title, ax in data_and_titles:
        ax.hist(diff, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        mean_value = np.mean(diff)
        ax.axvline(mean_value, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_value:.2f}')
        ax.set_title(title, fontsize=14, weight='bold')
        ax.set_xlabel('Difference', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)

    # Turn off the unused subplot
    axs[3, 1].axis('off')

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(csv_path.replace(".csv", "_differences_with_prompted_features_plot.png"))
    print(f"Plot saved as ",csv_path.replace(".csv", "_differences_with_prompted_features_plot.png"))
    
    
    #Average MSE for the entire dataset
    MSE_vectors=np.array(MSE_vectors)
    MSE_vectors=MSE_vectors.squeeze()
    MSE_vectors=MSE_vectors.mean(axis=0)
    print(f"Average MSE for the entire dataset: {MSE_vectors}") # Clearly this is wrong caus eour entry on kagggle sits at 0.89 whereas this is 1889.88....
    
def precompute_missing(y, y_pred):
    y = np.array(y)
    y_pred = np.array(y_pred)
    y = np.nan_to_num(y, nan=-100.0)
    y_pred = np.nan_to_num(y_pred, nan=-100.0)
    # Find indices where y is -100
    indices_to_change = np.where(y == -100.0)

    # Set corresponding elements in y and y_pred to 0
    y[indices_to_change] = 0.0
    y_pred[indices_to_change] = 0.0
    zeros_per_column = np.count_nonzero(y, axis=0)

    list_from_array = zeros_per_column.tolist()
    dc = {}
    for i in range(len(list_from_array)):
        dc[i] = list_from_array[i]
    return dc, y, y_pred



def sum_elements_per_column(matrix, dc):
    num_rows = len(matrix)
    num_cols = len(matrix[0])

    column_sums = [0] * num_cols

    for col in range(num_cols):
        for row in range(num_rows):
            column_sums[col] += matrix[row][col]

    res = []
    for col in range(num_cols):
        x = column_sums[col]/dc[col]
        res.append(x)

    return res   
    
def z_score_norm(y, y_pred, mean, std, eps=1e-10):

    y = np.array(y)
    y_pred = np.array(y_pred)

    normalized_true = (y - mean) / std

    normalized_gen = (y_pred - mean) / std

    dc, normalized_true, normalized_gen = precompute_missing(normalized_true, normalized_gen)

    #print(np.isnan(normalized_true).any())
    #print(np.isnan(normalized_gen).any())

    # Calculate MSE using normalized tensors
    mse_st = (normalized_true - normalized_gen) ** 2
    mae_st = np.absolute(normalized_true - normalized_gen)

    mse = sum_elements_per_column(mse_st, dc)
    mae = sum_elements_per_column(mae_st, dc)

    mse = np.sum(mse)/7
    mae = np.sum(mae)/7

    a = np.absolute(normalized_true - normalized_gen)
    b = np.absolute(normalized_true) + np.absolute(normalized_gen) + eps
    norm_error_st = (a/b)
    norm_error = sum_elements_per_column(norm_error_st, dc)
    norm_error = np.sum(norm_error)/7


    return mse, mae, norm_error   

def calculate_mean_std(x):
    print(x.shape)
    sm = [0 for i in range(7)]
    samples = [0 for i in range(7)]

    for el in x:
        for i, it in enumerate(el):
            if not math.isnan(it):
                sm[i] += it
                samples[i] += 1

    mean = [k / y for k,y in zip(sm, samples)]


    sm2 = [0 for i in range(8)]

    std = []

    for el in x:
        for i, it in enumerate(el):
            if not math.isnan(it):
                k = (it - mean[i])**2
                sm2[i] += k

    std = [(k / y)**0.5 for k,y in zip(sm2, samples)]
    return mean, std

def evaluation_metrics(y, y_pred, eps=1e-10):
    dc, y, y_pred = precompute_missing(y, y_pred)

    mse_st = (y - y_pred) ** 2
    mae_st = np.absolute(y - y_pred)

    mse = sum_elements_per_column(mse_st, dc)
    mae = sum_elements_per_column(mae_st, dc)

    #mse = [sum(x)/len(mse_st) for x in zip(*mse_st)]
    #mae = [sum(x)/len(mae_st) for x in zip(*mae_st)]

    a = np.absolute(y - y_pred)
    b = np.absolute(y) + np.absolute(y_pred)+ eps
    norm_error_st = (a/b)

    norm_error = sum_elements_per_column(norm_error_st, dc)
    #[sum(x)*100/len(norm_error_st) for x in zip(*norm_error_st)]

    return mse, mae, norm_error


def compare_reconstructed_and_prompted_graphs_v2(result_df: pd.DataFrame, data_lst: List[Data],return_log=None):
    # prep data for z-score normalization
    y_pred=result_df[['n_nodes', 'n_edges', 'avg_degree', 'n_triangles', 'clustering_coeff', 'max_k_core', 'n_communities']].values
    y=[]
    for data in data_lst:
        stats=data.stats.squeeze()
        y.append(stats)
    y=np.array(y)
    
    mean, std = calculate_mean_std(y)
    mse_all, mae_all, norm_error_all = z_score_norm(y, y_pred, mean, std)
    print(f"MSE for all features: {mse_all}")
    print(f"MAE for all features: {mae_all}")
    print(f"Normalized error for all features: {norm_error_all*100}")
    
    mses, maes, norm_errors = evaluation_metrics(y, y_pred)
    feats_lst = ["number of nodes", "number of edges","avg degree","triangles", "global clustering coeff", "max k-core", "communities"]
    id2feats = {i:feats_lst[i] for i in range(len(mses))}
    for i in range(len(mses)):
        print("MSE for the samples for the feature \""+str(id2feats[i])+"\" is equal to: "+str(mses[i]))
        print("MAE for the samples for the feature \""+str(id2feats[i])+"\" is equal to: "+str(maes[i]))
        print("Symmetric Mean absolute Percentage Error for the samples for the feature \""+str(id2feats[i])+"\" is equal to: "+str(norm_errors[i]*100))
        print("=" * 100)
    if return_log is not None:
        with open(return_log, 'w') as f:
            f.write(f"MSE for all features: {mse_all}\n")
            f.write(f"MAE for all features: {mae_all}\n")
            f.write(f"Normalized error for all features: {norm_error_all*100}\n")
            for i in range(len(mses)):
                f.write("MSE for the samples for the feature \""+str(id2feats[i])+"\" is equal to: "+str(mses[i])+"\n")
                f.write("MAE for the samples for the feature \""+str(id2feats[i])+"\" is equal to: "+str(maes[i])+"\n")
                f.write("Symmetric Mean absolute Percentage Error for the samples for the feature \""+str(id2feats[i])+"\" is equal to: "+str(norm_errors[i]*100)+"\n")
                f.write("=" * 100 + "\n")
    return {
        "mse_all_features": mse_all,
        "mae_all_features": mae_all,
        "normalized_error": norm_error_all,
        "nodes_mse": mses[0],
        "nodes_mae": maes[0],
        "nodes_smape": norm_errors[0]*100,
        "edges_mse": mses[1],
        "edges_mae": maes[1], 
        "edges_smape": norm_errors[1]*100,
        "degree_mse": mses[2],
        "degree_mae": maes[2],
        "degree_smape": norm_errors[2]*100,
        "triangles_mse": mses[3],
        "triangles_mae": maes[3],
        "triangles_smape": norm_errors[3]*100,
        "clustering_mse": mses[4],
        "clustering_mae": maes[4],
        "clustering_smape": norm_errors[4]*100,
        "kcore_mse": mses[5],
        "kcore_mae": maes[5],
        "kcore_smape": norm_errors[5]*100,
        "communities_mse": mses[6],
        "communities_mae": maes[6],
        "communities_smape": norm_errors[6]*100
    }


def main():
    parser = argparse.ArgumentParser(description='Process a CSV file for graph features.')
    parser.add_argument('--csv', type=str, help='Path to CSV file')
    parser.add_argument('--v2', action='store_true', help='Use the second version of the comparison function')
    args = parser.parse_args()

    csv_path = args.csv if args.csv else "progression_archive/attempt_1/output.csv"  # Default example path
    if not os.path.isfile(csv_path.replace(".csv", "_with_features.csv")):
        graphs_df = load_graphs_into_dataframe(csv_path)
        result_df = compute_features_vectorized(graphs_df)
        result_df.to_csv(csv_path.replace(".csv", "_with_features.csv"), index=False)
    else:
        result_df = pd.read_csv(csv_path.replace(".csv", "_with_features.csv"))
        

    data_lst = preprocess_dataset("test", 50, 10)

    
    if args.v2:
        compare_reconstructed_and_prompted_graphs_v2(result_df, data_lst, csv_path)
    else:
        compare_reconstructed_and_prompted_graphs(result_df, data_lst, csv_path)


if __name__ == "__main__":
    main()


