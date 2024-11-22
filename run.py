import os
import pandas as pd
import pickle
import torch
from utils.data_preprocessing import build_dataset

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def make_prediction(model, X):
    """
    Generar predicciones usando el modelo.
    """
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
    model_path = 'models/dyrk1a/rf_dyrk1a_model.pkl'

    # Cargar modelo
    model = load_model(model_path)

    # Cargar los datos desde el archivo proporcionado por el usuario
    try:
        df = load_input_file(input_file_path)
    except ValueError as e:
        print(f"Error al cargar el archivo: {e}")
        exit()

    # Definir el dispositivo (CPU o GPU si está disponible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preparar los datos
    X = build_dataset(df, include_y=False)  # Asume que devuelve un array de NumPy

    # Hacer predicciones
    predictions = make_prediction(model, X)

    # Preparar DataFrame de salida incluyendo SMILES
    output_df = pd.DataFrame({'SMILES': df['SMILES'], 'pChEMBL Predicted': predictions})

    # Definir el nombre del archivo de salida basado en el nombre del archivo de entrada
    base_name = os.path.splitext(os.path.basename(input_file_path))[0]
    output_filename = os.path.join(os.path.dirname(input_file_path), f'{base_name}_affinity.csv')

    # Guardar las predicciones en un archivo CSV
    output_df.to_csv(output_filename, index=False)
    print("Predicciones guardadas con éxito en", output_filename)
