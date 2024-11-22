import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_size, initial_hidden_size, drop_prob=0.1, output_size=1, num_layers=1):
        super(Model, self).__init__()
        layers = []
        current_size = input_size
        hidden_size = initial_hidden_size

        for _ in range(num_layers):
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=drop_prob))
            layers.append(nn.BatchNorm1d(hidden_size))
            current_size = hidden_size
            hidden_size //= 2  # Reduce el tamaño de la siguiente capa oculta a la mitad

        # Añadir la capa de salida
        self.layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(current_size, output_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x
