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
    Convierte SMILES a Morgan Fingerprints y añade la columna 'Standard Type' (IC50=0, EC50=1).
    
    Args:
        df (pd.DataFrame): DataFrame con columnas 'SMILES' y 'Standard Type' (numérica).
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
            # Concatenar la huella molecular (2048) con el Standard Type (1)
            X.append(fingerprint + [standard_type])
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
    model_path = input("Introduce la ruta del modelo (ej: 'models/xg_model_dyrk1a.pkl'): ").strip()
    if not os.path.exists(model_path):
        print(f"Error: el modelo {model_path} no existe.")
        exit()

    # Cargar modelo
    model = load_model(model_path)

    # Cargar datos de entrada
    df = load_input_file(input_file_path)

    # Caso 1: Si ya existe la columna 'Standard Type',
    #         convertir los valores "IC50"/"EC50" a 0/1
    if 'Standard Type' in df.columns:
        # Convertir los valores de IC50 y EC50 a 0 y 1
        df["Standard Type"] = df["Standard Type"].replace({"IC50": 0, "EC50": 1})

        # Si hay valores distintos de 0 o 1, los asignamos a 0 (IC50 por defecto)
        df.loc[~df["Standard Type"].isin([0, 1]), "Standard Type"] = 0

    else:
        # Si no existe la columna, asumir siempre IC50 (0) por defecto
        df["Standard Type"] = 0

    # Construir dataset con Morgan Fingerprints y Standard Type
    X = build_dataset(df, include_y=False)

    # Hacer predicciones
    predictions = make_prediction(model, X)

    # Guardar resultados en CSV
    df["pChEMBL Predicted"] = predictions
    output_filename = os.path.join(os.path.dirname(input_file_path), "predictions.csv")
    df.to_csv(output_filename, index=False)
    
    print(f"✅ Predicciones guardadas en {output_filename}")
