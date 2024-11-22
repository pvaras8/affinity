import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import train_test_split
import numpy as np

# Crear el generador de Morgan Fingerprints
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# Función para calcular Morgan Fingerprints
def calculate_morgan_fingerprint(molecule):
    mol = Chem.MolFromSmiles(molecule)
    if mol is not None:
        return list(mfpgen.GetFingerprint(mol))
    else:
        return None

# Construir dataset para modelos tabulares
def build_dataset(df, include_y=True):
    # Determinar automáticamente el dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X = []
    for smile in df['SMILES']:
        fp = calculate_morgan_fingerprint(smile)
        if fp is not None:
            X.append(fp)

    X = torch.tensor(X).float().to(device)

    if include_y and 'pChEMBL Value' in df.columns:
        Y = torch.tensor(df['pChEMBL Value'].values, dtype=torch.float).view(-1, 1).to(device)
        return X, Y
    else:
        return X  # Solo devolvemos X si no incluimos Y o si 'pChEMBL Value' no está en las columnas

# Preparar datos tabulares con división aleatoria
def prepare_data(filepath):
    df = pd.read_csv(filepath, encoding='latin-1', sep=';')
    df = df[['SMILES', 'Standard Type', 'pChEMBL Value']]
    df = df[df['Standard Type'] == 'IC50']
    df = df.dropna()
    df = df[~df['SMILES'].str.contains('\.')]
    df = df.reset_index(drop=True)

    # Generar las huellas moleculares
    df['MorganFingerprint'] = df['SMILES'].apply(calculate_morgan_fingerprint)
    df = df.dropna(subset=['MorganFingerprint'])

    # Preparar los datos para entrenamiento
    X = np.array(df['MorganFingerprint'].tolist())
    y = df['pChEMBL Value']

    # División aleatoria en entrenamiento, validación y prueba (80%, 10%, 10%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2)
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

    # Convertir a tensores y mover a GPU si está disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = torch.tensor(X_train).float().to(device)
    y_train = torch.tensor(y_train.values).float().view(-1, 1).to(device)
    X_dev = torch.tensor(X_dev).float().to(device)
    y_dev = torch.tensor(y_dev.values).float().view(-1, 1).to(device)
    X_test = torch.tensor(X_test).float().to(device)
    y_test = torch.tensor(y_test.values).float().view(-1, 1).to(device)

    return X_train, y_train, X_dev, y_dev, X_test, y_test

