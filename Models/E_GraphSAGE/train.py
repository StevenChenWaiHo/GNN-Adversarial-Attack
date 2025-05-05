import os
import sys

import joblib
from graph_generator import convert_df_to_PyG
import torch as th
from sklearn.preprocessing import LabelEncoder
from model import EGraphSAGE
from sklearn.utils import class_weight
import torch.nn as nn
import numpy as np
from torch_geometric.utils import subgraph
from torch_geometric.data import Data

project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
sys.path.append(project_root)

from Models.UNSW_NB15_config import UNSW_NB15_Config
from Datasets.get_dataframe import csv_to_dataframe

config_dict = {
    'UNSW_NB15': UNSW_NB15_Config
}

def train(dataset_name, epochs = 20, batch_size = 64):

    config = config_dict.get(dataset_name)
    if config is None:
        raise ValueError(f"Dataset '{dataset_name}' is not supported.")
    
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    print(device)

    # Load the dataset
    train_df = csv_to_dataframe(config.TRAIN_PATH)

    # Split the dataset into features and labels
    X_train = train_df.drop(columns=[config.ATTACK_CLASS_COL_NAME, config.IS_ATTACK_COL_NAME])

    le = LabelEncoder()
    attack_labels = le.fit_transform(train_df[config.ATTACK_CLASS_COL_NAME])
    class_map = le.classes_
    num_classes = len(class_map)
    print(class_map)
    print("Attack label mapping:", dict(zip(class_map, range(len(class_map)))))
    joblib.dump(le, f'./logs/label_encoder/label_encoder_{dataset_name}.pkl')

    G_nx, G_pyg = convert_df_to_PyG(train_df, config.SOURCE_NODE, config.DESTINATION_NODE, ['h', config.ATTACK_CLASS_COL_NAME])

    num_nodes = G_pyg.num_nodes
    num_edges = G_pyg.num_edges

    G_pyg.x = th.ones(num_nodes, len(X_train['h'].iloc[0])) 

    edge_attr_list = []
    edge_label_list = []

    for u, v, key, data in G_nx.edges(keys=True, data=True):
        edge_attr_list.append(data['h']) 
        edge_label_list.append(data[config.IS_ATTACK_COL_NAME]) 

    G_pyg.edge_attr = th.tensor(edge_attr_list, dtype=th.float32)
    G_pyg.edge_class = th.tensor(attack_labels, dtype=th.long)

    print("Number of edges in G_pyg:", G_pyg.num_edges)
    print("Number of node in G_pyg:", G_pyg.num_nodes)
    print("Shape of node in G_pyg:", G_pyg.x.shape)
    print("Shape of edge attr in G_pyg:", G_pyg.edge_attr.shape)
    print("Shape of edge label in G_pyg:", G_pyg.edge_label.shape)
    print("Shape of edge class in G_pyg:", G_pyg.edge_class.shape)

    model = EGraphSAGE(node_in_channels=G_pyg.num_node_features, 
                   edge_in_channels=G_pyg.num_edge_features,
                   hidden_channels=128, 
                   out_channels=num_classes).to(device)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    model.apply(init_weights)

    labels = G_pyg.edge_class.cpu().numpy()
    class_weights = class_weight.compute_class_weight('balanced',
                                                    classes=np.unique(labels),
                                                    y=labels)

    class_weights = th.FloatTensor(class_weights).cuda()
    criterion = nn.CrossEntropyLoss(weight = class_weights)
    optimizer = th.optim.Adam(model.parameters(), lr=0.001)



    def compute_accuracy(pred, labels):
        return (pred.argmax(1) == labels).float().mean().item()

    G_pyg.edge_label = G_pyg.edge_label.to(device)
    G_pyg.edge_attr = G_pyg.edge_attr.to(device)

    def generate_edge_based_batches_with_node_expansion(graph, batch_size, min_nodes):
        num_edges = graph.edge_index.size(1)  # Get total number of edges
        edge_indices = th.arange(num_edges)   # Create list of edge indices
        num_edges_processed = 0
        
        while num_edges_processed < num_edges:
            # Select a batch of edges
            batch_edge_indices = edge_indices[num_edges_processed : min(num_edges_processed + batch_size, num_edges)]
            edge_index = graph.edge_index[:, batch_edge_indices]
            
            # Update the number of edges processed
            num_edges_processed += batch_size
            
            # Get the unique nodes associated with these edges
            batch_nodes = th.cat([edge_index[0], edge_index[1]]).unique()

            # Check if the batch has enough unique nodes
            while batch_nodes.size(0) < min_nodes:
                # Sample additional neighboring nodes to ensure diversity
                additional_edges = int(batch_size / 8)  # Ensure additional_edges is an integer
                batch_edge_indices = th.cat([batch_edge_indices, edge_indices[num_edges_processed : min(num_edges_processed + additional_edges, num_edges)]])
                edge_index = graph.edge_index[:, batch_edge_indices]
                batch_nodes = th.cat([edge_index[0], edge_index[1]]).unique()
                num_edges_processed += additional_edges

                # Avoid potential infinite loops by breaking if no more edges can be added
                if num_edges_processed >= num_edges:
                    break

            # Create subgraph from the selected nodes and edges
            edge_index, _, edge_mask = subgraph(batch_nodes, graph.edge_index, relabel_nodes=True, return_edge_mask=True)

            # Use edge_mask to select edge attributes and labels
            edge_attr = graph.edge_attr[edge_mask]
            edge_label = graph.edge_label[edge_mask]

            yield batch_nodes, edge_index, edge_attr, edge_label
    
    for epoch in range(epochs):
        print(f'epoch : {epoch}')
        all_preds = []
        all_labels = []
        
        try:
            for batch_idx, (batch_nodes, edge_index, edge_attr, edge_label) in enumerate(generate_edge_based_batches_with_node_expansion(G_pyg, batch_size, 20)):
                # print(f"Processing epoch {epoch}, batch {batch_idx} with {batch_nodes.size(0)} nodes and {edge_index.size(1)} edges")
                batch = Data(x=G_pyg.x[batch_nodes], edge_index=edge_index, edge_attr=edge_attr, edge_label=edge_label)
                
                if batch.edge_index.size(1) == 0 or batch.edge_label.size(0) == 0:
                    print(f"Warning: Empty batch at batch {batch_idx}")
                    continue
                    
                if batch is None or batch.num_nodes == 0:
                    print(f"Warning: Empty batch at Batch {batch_idx}")
                    continue 
        
                if th.isnan(batch.x).any() or th.isinf(batch.x).any() or th.isnan(batch.edge_attr).any() or th.isinf(batch.edge_attr).any():
                    print(f"Warning: batch x and edge_attr contains NaN or Inf at Batch {batch_idx}")
                    continue 
                    
                try:
                    batch = batch.to(device)
                except Exception as batch_error:
                    print(f"Error moving batch to device at Batch {batch_idx}: {batch_error}")
                    continue
                
                try:
                    out = model(batch)
        
                    if th.isnan(out).any() or th.isinf(out).any():
                        print(f"Warning: out contains NaN or Inf at Batch {batch_idx}")
                        continue 
                    all_preds.append(out)
                    all_labels.append(batch.edge_label)
        
                    loss = criterion(out, batch.edge_label)
                    if th.isnan(loss):
                        print(f"loss: {loss}")
                        print(f"out: {out}")
                        print(f"edge_labels: {batch.edge_label}")
                        
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                except Exception as forward_error:
                    print(f"Error during forward/backward pass at Epoch {epoch}, Batch {batch_idx}: {forward_error}")
                    continue
            
            all_preds = th.cat(all_preds)
            all_labels = th.cat(all_labels)
            
            epoch_accuracy = compute_accuracy(all_preds, all_labels)
            print(f'Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
            print(all_labels.shape)

        except Exception as e:
            print(f"An error occurred at epoch {epoch}, batch {batch_idx}: {str(e)}")
    print("Training is over")

    th.save(model.state_dict(), f"./logs/GNN_model_weights_{dataset_name}.pth")


if __name__ == "__main__":
    dataset_name = 'UNSW_NB15'
    epochs = 20
    batch_size = 64

    train(dataset_name, epochs, batch_size)
    print("Training completed.")