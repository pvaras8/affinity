import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import train_test_split
import numpy as np
from rdkit.DataStructs import TanimotoSimilarity
import itertools

# Crear el generador de Morgan Fingerprints
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# Detectar el dispositivo adecuado: GPU (CUDA), MPS (macOS), o CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Función para calcular Morgan Fingerprints como objeto RDKit
def calculate_morgan_fingerprint(molecule):
    mol = Chem.MolFromSmiles(molecule)
    if mol is not None:
        return mfpgen.GetFingerprint(mol)  # 🔁 Devuelve objeto RDKit (no lista)
    else:
        return None

# Construir dataset para redes/tabulares
def build_dataset(df, include_y=True, use_torch=False):
    X = []
    for _, row in df.iterrows():
        fp = calculate_morgan_fingerprint(row['SMILES'])
        if fp is not None:
            vector = list(fp)
            vector.append(row['Standard Type'])  # Añadir como feature
            X.append(vector)

    X = torch.tensor(X, dtype=torch.float32).to(device) if use_torch else np.array(X, dtype=np.float32)

    if include_y and 'pChEMBL Value' in df.columns:
        Y = df['pChEMBL Value'].values.astype(np.float32)
        return (X, torch.tensor(Y, dtype=torch.float32).view(-1, 1).to(device)) if use_torch else (X, Y)
    return X

# Preparar y dividir datos, detectando cliffs automáticamente
def prepare_data(filepath, use_torch=False, threshold_dup=0.5):
    df = pd.read_csv(filepath, encoding='latin-1', sep=';')
    df = df[df['Standard Type'].isin(['IC50', 'EC50'])]
    df['Standard Type'] = df['Standard Type'].map({'IC50': 0, 'EC50': 1})
    df = df[['SMILES', 'Standard Type', 'pChEMBL Value']].dropna().reset_index(drop=True)
    df = df[~df['SMILES'].str.contains('\\.')]

    # Filtrar duplicados inconsistentes
    to_remove, merged = set(), []
    duplicated_total = consistent_total = inconsistent_total = 0

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

    print(f"🔍 SMILES duplicados: {duplicated_total}")
    print(f"⛔ Eliminados: {inconsistent_total}")
    print(f"✅ Promediados: {consistent_total}")

    df = df[~df.set_index(['SMILES', 'Standard Type']).index.isin(to_remove)]
    df = df.drop_duplicates(subset=['SMILES', 'Standard Type'])

    if merged:
        df = pd.concat([df, pd.DataFrame(merged)], ignore_index=True).drop_duplicates(subset=['SMILES', 'Standard Type'])

    # Fingerprints
    df['MorganFingerprint'] = df['SMILES'].apply(lambda s: mfpgen.GetFingerprint(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else None)
    df = df.dropna(subset=['MorganFingerprint']).reset_index(drop=True)

    # Detectar cliffs automáticamente
    cliffs = set()
    for i, j in itertools.combinations(range(len(df)), 2):
        if df.loc[i, 'Standard Type'] != df.loc[j, 'Standard Type']:
            continue
        sim = TanimotoSimilarity(df.loc[i, 'MorganFingerprint'], df.loc[j, 'MorganFingerprint'])
        if sim >= 0.9:
            delta = abs(df.loc[i, 'pChEMBL Value'] - df.loc[j, 'pChEMBL Value'])
            if delta >= 1.0:
                cliffs.update([df.loc[i, 'SMILES'], df.loc[j, 'SMILES']])

    df_cliffs = df[df['SMILES'].isin(cliffs)]
    df_noncliffs = df[~df['SMILES'].isin(cliffs)]

    print(f"🧗 Activity cliffs detectados: {len(df_cliffs)}")

    # Arrays no cliffs
    X = np.array([list(fp) + [stype] for fp, stype in zip(df_noncliffs['MorganFingerprint'], df_noncliffs['Standard Type'])], dtype=np.float32)
    y = df_noncliffs['pChEMBL Value'].values.astype(np.float32)

    # Dividir dataset
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2)
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

    # Añadir cliffs al test
    if not df_cliffs.empty:
        X_cliff = np.array([list(fp) + [stype] for fp, stype in zip(df_cliffs['MorganFingerprint'], df_cliffs['Standard Type'])], dtype=np.float32)
        y_cliff = df_cliffs['pChEMBL Value'].values.astype(np.float32)
        X_test = np.concatenate([X_test, X_cliff])
        y_test = np.concatenate([y_test, y_cliff])

    if use_torch:
        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
        X_dev = torch.tensor(X_dev, dtype=torch.float32).to(device)
        y_dev = torch.tensor(y_dev, dtype=torch.float32).view(-1, 1).to(device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

    return X_train, y_train, X_dev, y_dev, X_test, y_test
