import os
import pandas as pd
import pickle
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

# Cargar el generador de Morgan Fingerprints
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# Función para calcular Morgan Fingerprints
def calculate_morgan_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return list(mfpgen.GetFingerprint(mol))
    else:
        return None

# Función para construir el dataset con Standard Type
def build_dataset(df, include_y=False):
    """
    Convierte SMILES a Morgan Fingerprints y añade el Standard Type (IC50/EC50).
    
    Args:
        df (pd.DataFrame): DataFrame con columnas 'SMILES' y 'Standard Type'.
        include_y (bool): Si True, incluye la columna 'pChEMBL Value'.

    Returns:
        X (np.array): Matriz de Morgan Fingerprints con Standard Type.
        Y (np.array | None): Valores pChEMBL si include_y=True.
    """
    X, y = [], []
    
    for _, row in df.iterrows():
        fingerprint = calculate_morgan_fingerprint(row["SMILES"])
        if fingerprint:
            standard_type = row["Standard Type"]
            X.append(fingerprint + [standard_type])  # Concatenar Standard Type
        
            if include_y:
                y.append(row["pChEMBL Value"])

    X = np.array(X, dtype=np.float32)
    if include_y:
        y = np.array(y, dtype=np.float32)
        return X, y
    return X

# Función para cargar el modelo guardado
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Función para hacer predicciones con el modelo
def make_prediction(model, X):
    return model.predict(X)

# Función para cargar archivo de entrada
def load_input_file(file_path):
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
    # Pedir la ruta del archivo de entrada
    input_file_path = input("Introduce la ruta del archivo (.txt o .csv): ").strip()
    if not os.path.exists(input_file_path):
        print(f"Error: el archivo {input_file_path} no existe.")
        exit()

    # Pedir la ruta del modelo
    model_path = input("Introduce la ruta del modelo (ej: 'models/dyrk1a_model.pkl'): ").strip()
    if not os.path.exists(model_path):
        print(f"Error: el modelo {model_path} no existe.")
        exit()

    # Cargar modelo
    model = load_model(model_path)

    # Cargar datos de entrada
    df = load_input_file(input_file_path)

    # Preguntar por el Standard Type si no está en el archivo
    if 'Standard Type' not in df.columns:
        std_type_input = input("El dataset no tiene Standard Type. ¿Son IC50 o EC50? (IC50/EC50): ").strip().upper()
        std_type = 0 if std_type_input == "IC50" else 1
        df["Standard Type"] = std_type  # Agregar la columna a todo el dataset

    # Construir dataset con Morgan Fingerprints y Standard Type
    X = build_dataset(df, include_y=False)

    # Hacer predicciones
    predictions = make_prediction(model, X)

    # Guardar resultados en CSV
    df["pChEMBL Predicted"] = predictions
    output_filename = os.path.join(os.path.dirname(input_file_path), "predictions.csv")
    df.to_csv(output_filename, index=False)
    
    print(f"✅ Predicciones guardadas en {output_filename}")
