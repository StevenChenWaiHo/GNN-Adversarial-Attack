import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

class EGraphSAGEConv(MessagePassing):
    def __init__(self, node_in_channels, edge_in_channels, out_channels):
        super(EGraphSAGEConv, self).__init__(aggr='mean')  # mean aggregation
        self.lin_node = nn.Linear(node_in_channels, out_channels)
        self.lin_edge = nn.Linear(edge_in_channels, out_channels)
        self.lin_update = nn.Linear(node_in_channels + out_channels, out_channels) # out_channels * 2

    def forward(self, x, edge_index, edge_attr):
        # x: Node features, edge_attr: Edge features, edge_index: Connectivity
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        if edge_attr is not None:
            if edge_attr.size(0) != edge_index.size(1):
                loop_attr = th.zeros((edge_index.size(1) - edge_attr.size(0), edge_attr.size(1))).to(edge_attr.device)
                edge_attr = th.cat([edge_attr, loop_attr], dim=0)
        else:
            print("edge_attr is unexist")
        
        # Propagate and aggregate neighbor information
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # x_j represents the adjacent nodes of x
        # Compute messages by combining node and edge features
        return self.lin_node(x_j) + self.lin_edge(edge_attr)

    def update(self, aggr_out, x):
        # Update node features after message passing
        return self.lin_update(th.cat([x, aggr_out], dim=1))

class MLPPredictor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLPPredictor, self).__init__()
        self.lin = nn.Linear(in_channels * 2, out_channels)

    def forward(self, data, z):
        row, col = data.edge_index
        # Concatenate the features of source and target nodes for each edge
        edge_feat = th.cat([z[row], z[col]], dim=1)
        return self.lin(edge_feat)

class EGraphSAGE(nn.Module):
    def __init__(self, node_in_channels, edge_in_channels, hidden_channels, out_channels):
        super(EGraphSAGE, self).__init__()
        self.conv1 = EGraphSAGEConv(node_in_channels, edge_in_channels, hidden_channels)
        self.conv2 = EGraphSAGEConv(hidden_channels, edge_in_channels, hidden_channels)
        self.mlp_predictor = MLPPredictor(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return self.mlp_predictor(data, x)