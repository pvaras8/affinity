# Definición de hiperparámetros
HYPERPARAMETERS = {
    'input_size': 2048,
    'initial_hidden_size': 512,  # Tamaño reducido de la capa oculta para evaluar el impacto en la capacidad del modelo
    'dropout_rate': 0.1,  # Tasa de dropout ligeramente más baja para reducir la regularización
    'epochs': 3000,  # Menor número de épocas para implementar una detención temprana manual
    'learning_rate': 1e-04,  # Ajuste fino del learning rate
    'num_layers': 3  # Incrementar el número de capas para explorar mayor complejidad del modelo
}
