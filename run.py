import os
import pandas as pd
import pickle
import torch
from utils.data_preprocessing import build_dataset

def load_model(model_path):
    """
    Cargar el modelo desde un archivo pickle. Puede ser un modelo de Scikit-learn o PyTorch.
    """
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def make_prediction(model, X, is_torch_model=False):
    """
    Generar predicciones usando el modelo.
    
    Args:
        model: Modelo cargado (Scikit-learn o PyTorch).
        X: Datos de entrada (NumPy array o PyTorch tensor).
        is_torch_model: Indica si el modelo es de PyTorch.
    
    Returns:
        Predicciones generadas por el modelo.
    """
    if is_torch_model:
        model.eval()  # Cambiar a modo evaluación
        with torch.no_grad():  # No calcular gradientes
            predictions = model(X).cpu().numpy()  # Convertir a NumPy para guardar resultados
    else:
        predictions = model.predict(X)
    return predictions

def load_input_file(file_path):
    """
    Cargar un archivo de entrada, que puede ser .txt o .csv.
    - Si es .txt, cada línea se asume como una molécula SMILES.
    - Si es .csv, se asume que tiene una columna llamada 'SMILES'.
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.txt':
        with open(file_path, 'r') as file:
            smiles_list = file.read().splitlines()
        df = pd.DataFrame(smiles_list, columns=['SMILES'])
    elif file_extension == '.csv':
        df = pd.read_csv(file_path)
        if 'SMILES' not in df.columns:
            raise ValueError("El archivo CSV debe contener una columna llamada 'SMILES'.")
    else:
        raise ValueError("El archivo debe ser .txt o .csv.")
    return df

if __name__ == "__main__":
    # Pedir al usuario que introduzca la ruta del archivo de entrada
    input_file_path = input("Por favor, introduce la ruta del archivo (.txt o .csv): ").strip()

    # Verificar que el archivo existe
    if not os.path.exists(input_file_path):
        print(f"El archivo {input_file_path} no existe. Por favor, revisa la ruta.")
        exit()

    # Ruta al archivo del modelo
    model_path = input("Introduce la ruta del modelo (p. ej., 'models/dyrk1a_model.pkl'): ").strip()
    if not os.path.exists(model_path):
        print(f"El archivo del modelo {model_path} no existe. Por favor, revisa la ruta.")
        exit()

    # Cargar modelo
    model = load_model(model_path)

    # Identificar si el modelo es de PyTorch
    is_torch_model = isinstance(model, torch.nn.Module)

    # Cargar los datos desde el archivo proporcionado por el usuario
    try:
        df = load_input_file(input_file_path)
    except ValueError as e:
        print(f"Error al cargar el archivo: {e}")
        exit()

    # Preparar los datos
    if is_torch_model:
        # Si el modelo es de PyTorch, usamos tensores
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X = build_dataset(df, include_y=False, use_torch=True).to(device)
    else:
        # Si el modelo es de Scikit-learn, usamos NumPy
        X = build_dataset(df, include_y=False, use_torch=False)

    # Hacer predicciones
    predictions = make_prediction(model, X, is_torch_model=is_torch_model)

    # Preparar DataFrame de salida incluyendo SMILES
    output_df = pd.DataFrame({'SMILES': df['SMILES'], 'pChEMBL Predicted': predictions})

    # Definir el nombre del archivo de salida basado en el nombre del archivo de entrada
    base_name = os.path.splitext(os.path.basename(input_file_path))[0]
    output_filename = os.path.join(os.path.dirname(input_file_path), f'{base_name}_predictions.csv')

    # Guardar las predicciones en un archivo CSV
    output_df.to_csv(output_filename, index=False)
    print("Predicciones guardadas con éxito en", output_filename)
