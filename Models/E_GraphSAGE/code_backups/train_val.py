from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import os

def train(dataset_name, epochs=20, batch_size=64, checkpoint_path="./checkpoint.pth"):

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
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = th.optim.Adam(model.parameters(), lr=0.001)

    def compute_accuracy(pred, labels):
        return (pred.argmax(1) == labels).float().mean().item()

    G_pyg.edge_label = G_pyg.edge_label.to(device)
    G_pyg.edge_attr = G_pyg.edge_attr.to(device)

    # Split edges into train and validation sets
    edge_indices = th.arange(G_pyg.edge_label.size(0))
    train_indices, val_indices = train_test_split(edge_indices.cpu().numpy(), test_size=0.2, random_state=42)
    train_indices = th.tensor(train_indices, dtype=th.long).to(device)
    val_indices = th.tensor(val_indices, dtype=th.long).to(device)

    best_f1 = 0.0
    best_model_state = None

    # Load checkpoint if exists
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = th.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint['best_f1']
        print(f"Resumed training from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        print(f'Epoch: {epoch}')
        model.train()
        all_preds = []
        all_labels = []

        for batch_idx, (batch_nodes, edge_index, edge_attr, edge_label) in enumerate(generate_edge_based_batches_with_node_expansion(G_pyg, batch_size, 20)):
            if batch_idx not in train_indices:
                continue

            batch = Data(x=G_pyg.x[batch_nodes], edge_index=edge_index, edge_attr=edge_attr, edge_label=edge_label)
            batch = batch.to(device)

            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.edge_label)
            loss.backward()
            optimizer.step()

            all_preds.append(out)
            all_labels.append(batch.edge_label)

        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        with th.no_grad():
            for batch_idx, (batch_nodes, edge_index, edge_attr, edge_label) in enumerate(generate_edge_based_batches_with_node_expansion(G_pyg, batch_size, 20)):
                if batch_idx not in val_indices:
                    continue

                batch = Data(x=G_pyg.x[batch_nodes], edge_index=edge_index, edge_attr=edge_attr, edge_label=edge_label)
                batch = batch.to(device)

                out = model(batch)
                val_preds.append(out)
                val_labels.append(batch.edge_label)

        val_preds = th.cat(val_preds)
        val_labels = th.cat(val_labels)
        val_f1 = f1_score(val_labels.cpu(), val_preds.argmax(1).cpu(), average='weighted')

        print(f'Epoch {epoch}, Loss: {loss:.4f}, Validation F1: {val_f1:.4f}')

        # Save the best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = model.state_dict()
            th.save(best_model_state, f"./logs/best_model_{dataset_name}.pth")
            print("Saved best model.")

        # Save checkpoint
        th.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_f1': best_f1
        }, checkpoint_path)

    print("Training is over")
