import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as F

class GATModel(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_heads, dropout_rate, output_size):
        super(GATModel, self).__init__()
        # Primera capa de GAT con múltiples cabezas de atención
        self.conv1 = GATConv(num_features, hidden_dim, heads=num_heads, dropout=dropout_rate)
        # Segunda capa de GAT para reducir la dimensión a la salida deseada; aquí usamos solo una cabeza para la salida
        self.conv2 = GATConv(hidden_dim * num_heads, output_size, heads=1, concat=False, dropout=dropout_rate)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=0.6, training=self.training)  # Aplicar dropout a las entradas
        x = self.conv1(x, edge_index)  # Pasar por la primera capa GAT
        x = F.elu(x)  # Aplicar función de activación ELU

        x = F.dropout(x, p=0.6, training=self.training)  # Aplicar dropout después de la primera capa GAT
        x = self.conv2(x, edge_index)  # Pasar por la segunda capa GAT

        return x  # No aplicamos softmax al final ya que es una tarea de regresión



