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

# Detectar el dispositivo automáticamente
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Cargar los datos (usar PyTorch tensores)
Xtr, Ytr, Xdev, Ydev, Xte, Yte = prepare_data('data/baseDyrk1a.csv', use_torch=True)

# Instanciar el modelo
model = Model(
    input_size=HYPERPARAMETERS['input_size'],
    initial_hidden_size=HYPERPARAMETERS['initial_hidden_size'],
    drop_prob=HYPERPARAMETERS['dropout_rate'],
    num_layers=HYPERPARAMETERS['num_layers'],
    output_size=1
).to(device)

# Configurar el optimizador
optimizer = optim.Adam(model.parameters(), lr=HYPERPARAMETERS['learning_rate'])

# Entrenamiento del modelo
for epoch in range(HYPERPARAMETERS['epochs']):
    model.train()  # Modo entrenamiento
    optimizer.zero_grad()  # Reiniciar gradientes
    
    # Predicciones y cálculo de la pérdida
    predictions = model(Xtr)
    loss = F.mse_loss(predictions, Ytr)
    
    # Backpropagation y actualización de los parámetros
    loss.backward()
    optimizer.step()
    
    # Evaluar cada 100 épocas
    if (epoch + 1) % 100 == 0:
        model.eval()  # Cambiar a modo evaluación
        with torch.no_grad():  # Desactivar gradientes para validación
            predictions_dev = model(Xdev)
            loss_dev = F.mse_loss(predictions_dev, Ydev)
        print(f'Epoch {epoch + 1}: Training Loss = {loss.item()}, Validation Loss = {loss_dev.item()}')
        model.train()  # Volver al modo entrenamiento

# Guardar el modelo entrenado
model.to('cpu')  # Mover a CPU para guardarlo
model_filename = 'models/dyrk1a/nn_model_dyrk1a.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

print(f"Modelo guardado con éxito en {model_filename}")
