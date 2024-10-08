from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, SAGPooling, global_mean_pool, global_max_pool, GAE
from torch_geometric.data import Data, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn

print("Loading dataset...")
dataset = load_dataset('AI-MO/NuminaMath-CoT')

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-math-7b-base")

def extract_pairs(data):
    """
    Extracts problem-solution pairs from the dataset.

    Args:
        data (list): A list of dataset items containing problems and solutions.

    Returns:
        tuple: A tuple containing two lists - problems and solutions.
    """
    problems = []
    solutions = []
    for item in tqdm(data, desc="Extracting problem-solution pairs"):
        problems.append(item['problem'])
        solutions.append(item['solution'])
    return problems, solutions

print("Extracting train data...")
train_problems, train_solutions = extract_pairs(dataset['train'])

print("Extracting test data...")
test_problems, test_solutions = extract_pairs(dataset['test'])

def text_to_graph(text):
    """
    Converts a text input into a graph representation.

    Args:
        text (str): The input text to be converted into a graph.

    Returns:
        tuple: A tuple containing node features and edge indices.
    """
    tokens = tokenizer.tokenize(text)
    node_features = []
    edge_index = [[], []]
    for i, token in enumerate(tokens):
        node_feature = [0] * 3
        token_id = tokenizer.convert_tokens_to_ids(token)
        node_feature[0] = token_id
        node_features.append(node_feature)
        if i > 0:
            edge_index[0].append(i-1)
            edge_index[1].append(i)
            edge_index[0].append(i)
            edge_index[1].append(i-1)
    return node_features, edge_index

print("Building train graphs...")
train_graphs = [text_to_graph(problem) for problem in tqdm(train_problems, desc="Building train graphs")]

print("Building test graphs...")
test_graphs = [text_to_graph(problem) for problem in tqdm(test_problems, desc="Building test graphs")]

def create_data_objects(graphs, solutions):
    """
    Creates data objects for the graphs and their corresponding solutions.

    Args:
        graphs (list): A list of graph representations.
        solutions (list): A list of corresponding solutions.

    Returns:
        list: A list of Data objects containing node features, edge indices, and labels.
    """
    data_list = []
    for (node_features, edge_index), solution in tqdm(zip(graphs, solutions), total=len(graphs), desc="Creating data objects"):
        if node_features and edge_index[0]:
            x = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            try:
                y = torch.tensor([[float(len(solution))]], dtype=torch.float)
                data_list.append(Data(x=x, edge_index=edge_index, y=y))
            except Exception as e:
                print(f"Skipping invalid solution: {solution}")
                print(f"Error: {str(e)}")
        else:
            print("Skipping invalid graph")
    return data_list

print("Creating train data objects...")
train_data_objects = create_data_objects(train_graphs, train_solutions)
print(f"Number of valid train samples: {len(train_data_objects)}")

print("Creating test data objects...")
test_data_objects = create_data_objects(test_graphs, test_solutions)
print(f"Number of valid test samples: {len(test_data_objects)}")

if len(train_data_objects) > 0 and len(test_data_objects) > 0:
    train_loader = DataLoader(train_data_objects, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data_objects, batch_size=16, shuffle=False)

class MathGAE(torch.nn.Module):
    """
    Graph Autoencoder for mathematical problem-solving.

    Attributes:
        encoder (HierarchicalEncoder): The encoder module.
        decoder (HierarchicalDecoder): The decoder module.
        regressor (ImprovedRegressor): The regression module.
    """

    def __init__(self, num_node_features, hidden_channels=4096, num_layers=12):
        """
        Initializes the MathGAE model.

        Args:
            num_node_features (int): The number of features per node.
            hidden_channels (int): The number of hidden channels for the layers.
            num_layers (int): The number of layers in the encoder and decoder.
        """
        super(MathGAE, self).__init__()
        self.encoder = HierarchicalEncoder(num_node_features, hidden_channels, num_layers)
        self.decoder = HierarchicalDecoder(hidden_channels, num_layers)
        self.regressor = ImprovedRegressor(hidden_channels)

    def encode(self, x, edge_index, batch):
        """
        Encodes the input graph data.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices.
            batch (torch.Tensor): Batch indices.

        Returns:
            torch.Tensor: Encoded node representations.
        """
        return self.encoder(x, edge_index, batch)

    def decode(self, z, edge_index):
        """
        Decodes the encoded representations back to graph structure.

        Args:
            z (torch.Tensor): Encoded node representations.
            edge_index (torch.Tensor): Edge indices.

        Returns:
            torch.Tensor: Reconstructed graph representation.
        """
        return self.decoder(z, edge_index)

    def regress(self, z, batch):
        """
        Performs regression on the encoded representations.

        Args:
            z (torch.Tensor): Encoded node representations.
            batch (torch.Tensor): Batch indices.

        Returns:
            torch.Tensor: Regression outputs.
        """
        return self.regressor(z, batch)

class HierarchicalEncoder(torch.nn.Module):
    """
    Hierarchical encoder for processing graph data.

    Attributes:
        num_layers (int): Number of layers in the encoder.
        convs (torch.nn.ModuleList): List of convolutional layers.
        bns (torch.nn.ModuleList): List of batch normalization layers.
        pools (torch.nn.ModuleList): List of pooling layers.
        gat (GATv2Conv): Graph attention layer.
        bn_gat (torch.nn.BatchNorm1d): Batch normalization for GAT.
        final_linear (torch.nn.Linear): Final linear layer.
    """

    def __init__(self, in_channels, hidden_channels, num_layers):
        """
        Initializes the HierarchicalEncoder.

        Args:
            in_channels (int): Number of input channels.
            hidden_channels (int): Number of hidden channels.
            num_layers (int): Number of layers in the encoder.
        """
        super(HierarchicalEncoder, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.convs.append(GCNConv(in_channels, hidden_channels))
            else:
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            self.pools.append(SAGPooling(hidden_channels, ratio=0.8))
        self.gat = GATv2Conv(hidden_channels, hidden_channels, heads=16, concat=True)
        self.bn_gat = torch.nn.BatchNorm1d(hidden_channels * 16)
        self.final_linear = torch.nn.Linear(hidden_channels * 16, hidden_channels)

    def forward(self, x, edge_index, batch):
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices.
            batch (torch.Tensor): Batch indices.

        Returns:
            tuple: Encoded node representations and batch indices.
        """
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(self.bns[i](x))
            x, edge_index, _, batch, _, _ = self.pools[i](x, edge_index, None, batch)
        x = self.gat(x, edge_index)
        x = F.relu(self.bn_gat(x))
        x = self.final_linear(x)
        return x, batch

class HierarchicalDecoder(torch.nn.Module):
    """
    Hierarchical decoder for reconstructing graph data from encoded representations.

    Attributes:
        num_layers (int): Number of layers in the decoder.
        lins (torch.nn.ModuleList): List of linear layers.
        final_lin (torch.nn.Linear): Final linear layer.
    """

    def __init__(self, hidden_channels, num_layers):
        """
        Initializes the HierarchicalDecoder.

        Args:
            hidden_channels (int): Number of hidden channels.
            num_layers (int): Number of layers in the decoder.
        """
        super(HierarchicalDecoder, self).__init__()
        self.num_layers = num_layers
        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.final_lin = torch.nn.Linear(hidden_channels, hidden_channels)

    def forward(self, z, edge_index):
        """
        Forward pass through the decoder.

        Args:
            z (torch.Tensor): Encoded node representations.
            edge_index (torch.Tensor): Edge indices.

        Returns:
            torch.Tensor: Reconstructed graph representation.
        """
        for i in range(self.num_layers):
            z = F.relu(self.lins[i](z))
        z = self.final_lin(z)
        return torch.sigmoid((z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1))

class ImprovedRegressor(torch.nn.Module):
    """
    Improved regressor for predicting outputs based on encoded representations.

    Attributes:
        lins (torch.nn.ModuleList): List of linear layers.
        bns (torch.nn.ModuleList): List of batch normalization layers.
        final_lin (torch.nn.Linear): Final linear layer.
    """

    def __init__(self, hidden_channels):
        """
        Initializes the ImprovedRegressor.

        Args:
            hidden_channels (int): Number of hidden channels.
        """
        super(ImprovedRegressor, self).__init__()
        self.lins = torch.nn.ModuleList([
            torch.nn.Linear(hidden_channels * 2, hidden_channels * 2) for _ in range(8)
        ])
        self.bns = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(hidden_channels * 2) for _ in range(8)
        ])
        self.final_lin = torch.nn.Linear(hidden_channels * 2, 1)

    def forward(self, z, batch):
        """
        Forward pass through the regressor.

        Args:
            z (torch.Tensor): Encoded node representations.
            batch (torch.Tensor): Batch indices.

        Returns:
            torch.Tensor: Final regression outputs.
        """
        z_mean = global_mean_pool(z, batch)
        z_max = global_max_pool(z, batch)
        z = torch.cat([z_mean, z_max], dim=1)
        for lin, bn in zip(self.lins, self.bns):
            z_new = F.relu(bn(lin(z)))
            z = z + z_new  # Residual connection
        z = F.dropout(z, p=0.5, training=self.training)
        return self.final_lin(z)

hidden_channels = 2048
model = MathGAE(num_node_features=3, hidden_channels=hidden_channels, num_layers=12)
device = torch.device("cuda")
model = model.to(device)

# model = torch.nn.DataParallel(model)
# torchinfo.summary(model)

gae = GAE(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-5)

def train():
    """
    Trains the model for one epoch.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    total_loss = 0
    for data in tqdm(train_loader, desc="Training", leave=False):
        data = data.to(device)
        optimizer.zero_grad()
        z, batch = model.encode(data.x, data.edge_index, data.batch)

        # Filter out invalid edges
        edge_index = data.edge_index
        valid_mask = (edge_index[0] < z.size(0)) & (edge_index[1] < z.size(0))
        edge_index = edge_index[:, valid_mask]

        loss = gae.recon_loss(z, edge_index)
        out = model.regress(z, batch)
        loss += F.mse_loss(out, data.y.view(-1, 1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test(loader):
    """
    Evaluates the model on the given data loader.

    Args:
        loader (DataLoader): The data loader for evaluation.

    Returns:
        tuple: Average error and accuracy of the model on the dataset.
    """
    model.eval()
    total_error = 0
    total_correct = 0
    for data in tqdm(loader, desc="Evaluating", leave=False):
        data = data.to(device)
        z, batch = model.encode(data.x, data.edge_index, data.batch)
        out = model.regress(z, batch)
        total_error += F.l1_loss(out, data.y.view(-1, 1), reduction='sum').item()
        total_correct += ((out.round() == data.y.view(-1, 1)).sum().item())

    accuracy = total_correct / len(loader.dataset)
    return total_error / len(loader.dataset), accuracy

def train_and_evaluate():
    """
    Trains and evaluates the model over multiple epochs.

    Returns:
        tuple: Lists of training losses, training accuracies, test losses, and test accuracies.
    """
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    print("Starting training...")
    for epoch in tqdm(range(1, 4), desc="Epochs"):
        loss = train()
        train_mae, train_accuracy = test(train_loader)
        test_mae, test_accuracy = test(test_loader)

        train_losses.append(loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_mae)
        test_accuracies.append(test_accuracy)

        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train MAE: {train_mae:.4f}, '
              f'Train Acc: {train_accuracy:.4f}, Test MAE: {test_mae:.4f}, '
              f'Test Acc: {test_accuracy:.4f}')

    return train_losses, train_accuracies, test_losses, test_accuracies

# Run the training
train_losses, train_accuracies, test_losses, test_accuracies = train_and_evaluate()

def plot_training_curves(train_losses, train_accuracies, test_losses, test_accuracies):
    """
    Plots the training and test losses and accuracies over epochs.

    Args:
        train_losses (list): List of training losses.
        train_accuracies (list): List of training accuracies.
        test_losses (list): List of test losses.
        test_accuracies (list): List of test accuracies.
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, test_losses, 'r-', label='Test MAE')
    plt.title('Training and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, test_accuracies, 'r-', label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

plot_training_curves(train_losses, train_accuracies, test_losses, test_accuracies)
print("Training curves have been saved as 'training_curves.png'")

else:
    print("Error: No valid samples found in the dataset.")
    print("Please check the original dataset and the tokenization process.")
