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
    Construye el dataset basado en las huellas moleculares y añade Standard Type.
    
    Args:
        df (pd.DataFrame): DataFrame con al menos una columna 'SMILES'.
        include_y (bool): Incluir la variable de salida (Y) si está disponible.
        use_torch (bool): Devuelve tensores de PyTorch si es True; NumPy si es False.
    
    Returns:
        X (np.array | torch.Tensor): Huellas moleculares + Standard Type.
        Y (np.array | torch.Tensor | None): Valores de la columna 'pChEMBL Value' si include_y es True.
    """
    X = []
    for _, row in df.iterrows():
        fp = calculate_morgan_fingerprint(row['SMILES'])
        if fp is not None:
            # Agregar Standard Type como una característica adicional
            fp.append(row['Standard Type'])  # Se agrega como última columna
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


def prepare_data(filepath, activity_cliff_path=None, use_torch=False, threshold_dup=0.5):
    """
    Prepara los datos tabulares eliminando duplicados inconsistentes,
    promediando duplicados válidos, y reservando activity cliffs para test.

    Args:
        filepath (str): Ruta al archivo CSV con datos base.
        activity_cliff_path (str): Ruta al CSV de activity cliffs (formato de pares).
        use_torch (bool): Si True, devuelve tensores de PyTorch.
        threshold_dup (float): Diferencia máxima aceptable para promediar duplicados.

    Returns:
        X_train, y_train, X_dev, y_dev, X_test, y_test
    """
    # Leer CSV principal
    df = pd.read_csv(filepath, encoding='latin-1', sep=';')
    df = df[df['Standard Type'].isin(['IC50', 'EC50'])]
    df['Standard Type'] = df['Standard Type'].map({'IC50': 0, 'EC50': 1})
    df = df[['SMILES', 'Standard Type', 'pChEMBL Value']].dropna().reset_index(drop=True)
    df = df[~df['SMILES'].str.contains('\.')]

    # Eliminar duplicados inconsistentes y promediar los válidos
    to_remove = set()
    merged = []
    duplicated_total = 0
    consistent_total = 0
    inconsistent_total = 0

    for (smiles, stype), group in df.groupby(['SMILES', 'Standard Type']):
        values = group['pChEMBL Value'].values
        if len(values) > 1:
            duplicated_total += 1
            diffs = [abs(a - b) for i, a in enumerate(values) for b in values[i+1:]]
            if any(diff > threshold_dup for diff in diffs):
                to_remove.add((smiles, stype))
                inconsistent_total += 1
            else:
                merged.append({"SMILES": smiles, "Standard Type": stype, "pChEMBL Value": np.mean(values)})
                consistent_total += 1

    print(f"Número total de SMILES duplicados: {duplicated_total}")
    print(f"→ Eliminados por inconsistentes: {inconsistent_total}")
    print(f"→ Conservados (promediados): {consistent_total}")

    df = df[~df.set_index(['SMILES', 'Standard Type']).index.isin(to_remove)]
    df = df.drop_duplicates(subset=['SMILES', 'Standard Type'])

    if merged:
        df_merged = pd.DataFrame(merged)
        df = pd.concat([df, df_merged], ignore_index=True).drop_duplicates(subset=['SMILES', 'Standard Type'])

    # Calcular fingerprints
    df['MorganFingerprint'] = df['SMILES'].apply(calculate_morgan_fingerprint)
    df = df.dropna(subset=['MorganFingerprint']).reset_index(drop=True)

    # Reservar cliffs en test si se proporciona
    if activity_cliff_path:
        df_cliffs = pd.read_csv(activity_cliff_path)
        cliff_smiles = set(df_cliffs['SMILES']) if 'SMILES' in df_cliffs.columns else \
                        set(df_cliffs['SMILES_1']).union(df_cliffs['SMILES_2'])
        df_cliffs = df[df['SMILES'].isin(cliff_smiles)]
        df_noncliffs = df[~df['SMILES'].isin(cliff_smiles)]
        print(f"El número de cliffs detectados es de {len(df_cliffs)}")
    else:
        print(f"No hay cliffs detectados")
        df_cliffs = pd.DataFrame(columns=df.columns)
        df_noncliffs = df

    # Preparar arrays
    X = np.array([fp + [stype] for fp, stype in zip(df_noncliffs['MorganFingerprint'], df_noncliffs['Standard Type'])], dtype=np.float32)
    y = df_noncliffs['pChEMBL Value'].values.astype(np.float32)

    # Dividir en train/dev/test (cliffs van aparte)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2)
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

    # Añadir cliffs al test set
    if not df_cliffs.empty:
        X_cliff = np.array([fp + [stype] for fp, stype in zip(df_cliffs['MorganFingerprint'], df_cliffs['Standard Type'])], dtype=np.float32)
        y_cliff = df_cliffs['pChEMBL Value'].values.astype(np.float32)
        X_test = np.concatenate([X_test, X_cliff], axis=0)
        y_test = np.concatenate([y_test, y_cliff], axis=0)

    if use_torch:
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_dev = torch.tensor(X_dev, dtype=torch.float32)
        y_dev = torch.tensor(y_dev, dtype=torch.float32).view(-1, 1)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    return X_train, y_train, X_dev, y_dev, X_test, y_test