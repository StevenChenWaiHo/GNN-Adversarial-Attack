import networkx as nx
from torch_geometric.utils import from_networkx
import pandas as pd
import os
import torch
# from Datasets.get_dataframe import csv_to_dataframe

def convert_df_to_PyG(dataframe, source, destination, edge_attrs, create_using=nx.MultiDiGraph(), **kwargs):
    """
    Convert a DataFrame to a PyG Data object.
    
    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        
    Returns:
        torch_geometric.data.Data: The converted PyG Data object.
    """

    # Extract node features and labels
    G_nx = nx.from_pandas_edgelist(dataframe, source, destination, edge_attrs, create_using=create_using, **kwargs)
    G_pyg = from_networkx(G_nx)

    return G_nx, G_pyg

# def get_graph(csv_path, source, destination, edge_attrs, create_using=nx.MultiDiGraph(), **kwargs):
#     """
#     Get a PyG Data object from a saved file or convert a CSV file and save the graph.
    
#     Args:
#         csv_path (str): Path to the CSV file.
        
#     Returns:
#         torch_geometric.data.Data: The PyG Data object.
#     """
#     # Define the path to save the graph
#     save_path = os.path.join("../logs", os.path.relpath(os.path.dirname(csv_path), "../Datasets"), "/graph/graph.pt")

#     # Check if the graph is already saved
#     if os.path.exists(save_path):
#         # Load the saved graph
#         G_nx, G_pyg = torch.load(save_path)
#         print(f"Graph loaded from {save_path}")
#     else:
#         # Read the CSV file
#         dataframe = csv_to_dataframe(csv_path)

#         feature_cols = dataframe.columns.tolist()
#         feature_cols.remove(source)
#         feature_cols.remove(destination)
#         for attr in edge_attrs:
#             if attr in feature_cols:
#                 feature_cols.remove(attr)
#             elif attr != "h":
#                 print(f"Warning: {attr} not found in feature columns.")

#         dataframe['h'] = dataframe[ feature_cols ].values.tolist()
        
#         # Convert to PyG Data object
#         G_nx, G_pyg = convert_csv_to_PyG(dataframe, source, destination, edge_attrs, create_using=create_using, **kwargs)
        
#         # Save the graph
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         torch.save([G_nx, G_pyg], save_path)
#         print(f"Graph saved to {save_path}")
    
#     return G_nx, G_pyg