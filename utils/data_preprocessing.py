import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import train_test_split
import numpy as np

# Crear el generador de Morgan Fingerprints
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# Detectar el dispositivo adecuado: GPU (CUDA), MPS (macOS), o CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Metal Performance Shaders en macOS
elif torch.cuda.is_available():
    device = torch.device("cuda")  # CUDA en GPUs
else:
    device = torch.device("cpu")  # CPU como fallback

# Función para calcular Morgan Fingerprints
def calculate_morgan_fingerprint(molecule):
    """Calcula las huellas moleculares (Morgan Fingerprints) para un SMILES dado."""
    mol = Chem.MolFromSmiles(molecule)
    if mol is not None:
        return list(mfpgen.GetFingerprint(mol))
    else:
        return None

# Construir dataset para modelos tabulares y redes neuronales
def build_dataset(df, include_y=True, use_torch=False):
    """
    Construye el dataset basado en las huellas moleculares.
    
    Args:
        df (pd.DataFrame): DataFrame con al menos una columna 'SMILES'.
        include_y (bool): Incluir la variable de salida (Y) si está disponible.
        use_torch (bool): Devuelve tensores de PyTorch si es True; NumPy si es False.
    
    Returns:
        X (np.array | torch.Tensor): Huellas moleculares como matriz NumPy o tensor.
        Y (np.array | torch.Tensor | None): Valores de la columna 'pChEMBL Value' si include_y es True.
    """
    X = []
    for smile in df['SMILES']:
        fp = calculate_morgan_fingerprint(smile)
        if fp is not None:
            X.append(fp)

    if use_torch:
        X = torch.tensor(X, dtype=torch.float32).to(device)
    else:
        X = np.array(X, dtype=np.float32)

    if include_y and 'pChEMBL Value' in df.columns:
        Y = df['pChEMBL Value'].values.astype(np.float32)
        if use_torch:
            Y = torch.tensor(Y, dtype=torch.float32).view(-1, 1).to(device)
        return X, Y
    else:
        return X

# Preparar datos tabulares con división aleatoria
def prepare_data(filepath, use_torch=False):
    """
    Lee el archivo CSV y divide los datos en conjuntos de entrenamiento, validación y prueba.
    
    Args:
        filepath (str): Ruta al archivo CSV.
        use_torch (bool): Si True, devuelve tensores de PyTorch.
    
    Returns:
        X_train, y_train, X_dev, y_dev, X_test, y_test (np.array | torch.Tensor): Conjuntos divididos.
    """
    # Leer el archivo CSV
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
    X = np.array(df['MorganFingerprint'].tolist(), dtype=np.float32)
    y = df['pChEMBL Value'].values.astype(np.float32)

    # División aleatoria en entrenamiento, validación y prueba (80%, 10%, 10%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    if use_torch:
        # Convertir a tensores de PyTorch
        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
        X_dev = torch.tensor(X_dev, dtype=torch.float32).to(device)
        y_dev = torch.tensor(y_dev, dtype=torch.float32).view(-1, 1).to(device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

    return X_train, y_train, X_dev, y_dev, X_test, y_test
