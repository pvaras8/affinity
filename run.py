import os
import pandas as pd
import pickle
import numpy as np
from utils.data_preprocessing import calculate_morgan_fingerprint, build_dataset  # 👈 Importa tus funciones

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
        print(f"❌ Error: el archivo {input_file_path} no existe.")
        exit()

    # Pedir la ruta del modelo
    model_path = input("Introduce la ruta del modelo (ej: 'models/xg_model_dyrk1a.pkl'): ").strip()
    if not os.path.exists(model_path):
        print(f"❌ Error: el modelo {model_path} no existe.")
        exit()

    # Cargar modelo
    model = load_model(model_path)

    # Cargar datos de entrada
    df = load_input_file(input_file_path)

    # Procesar Standard Type
    if 'Standard Type' in df.columns:
        df["Standard Type"] = df["Standard Type"].replace({"IC50": 0, "EC50": 1})
        df.loc[~df["Standard Type"].isin([0, 1]), "Standard Type"] = 0
    else:
        df["Standard Type"] = 0  # IC50 por defecto

    # Construir dataset
    X = build_dataset(df, include_y=False)

    # Hacer predicciones
    predictions = make_prediction(model, X)

    # Guardar resultados
    df["pChEMBL Predicted"] = predictions
    output_filename = os.path.join(os.path.dirname(input_file_path), "predictions.csv")
    df.to_csv(output_filename, index=False)

    print(f"✅ Predicciones guardadas en {output_filename}")
