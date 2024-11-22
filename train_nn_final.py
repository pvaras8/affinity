import random
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from configs.config_nn import HYPERPARAMETERS
from models.model_nn import Model
from utils.data_preprocessing import prepare_data
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Xtr, Ytr, Xdev, Ydev, Xte, Yte = prepare_data('data/espacio_latenteBuche.csv')

model = Model(input_size=HYPERPARAMETERS['input_size'], initial_hidden_size=HYPERPARAMETERS['initial_hidden_size'], drop_prob=HYPERPARAMETERS['dropout_rate'], num_layers=HYPERPARAMETERS['num_layers'], output_size=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=HYPERPARAMETERS['learning_rate'])

for i in range(HYPERPARAMETERS['epochs']):
    model.train()
    optimizer.zero_grad()
    predictions = model(Xtr)
    loss = F.mse_loss(predictions, Ytr)
    loss.backward()
    optimizer.step()
    
    if (i + 1) % 100 == 0:
        # Evaluar la pérdida de validación
        model.eval() # Cambiar el modelo a modo de evaluación
        with torch.no_grad(): # No calcular gradientes
            predictions_dev = model(Xdev)
            loss_dev = F.mse_loss(predictions_dev, Ydev)
        print(f'Epoch {i + 1}: Training Loss = {loss.item()}, Validation Loss = {loss_dev.item()}')
        model.train() # Volver al modo de entrenamiento

# Mueve el modelo a CPU para poder guardarlo con pickle
model.to('cpu')

# Guardar el modelo
model_filename = 'models/nn_latente_buche_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

print(f"Modelo guardado con éxito en {model_filename}")