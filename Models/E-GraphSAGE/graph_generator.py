import networkx as nx
from torch_geometric.utils import from_networkx

def convert_csv_to_PyG(dataframe, source, destination, edge_attrs, create_using=nx.MultiDiGraph(), **kwargs):
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

    return G_pyg