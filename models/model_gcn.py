import torch
import sys
import urllib.request
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from tqdm.notebook import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import IPythonConsole

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, balanced_accuracy_score, roc_auc_score
from torch_geometric.nn.models import AttentiveFP
from torch_geometric.nn import GINEConv, GATv2Conv, global_add_pool
# -------------------------------
# Leemos los datos 
# -------------------------------

df = pd.read_csv('espacio_latente.csv', encoding='latin-1')
smiles = df.loc[(df['Standard Type'] == 'IC50')]
df = smiles[['Smiles', 'pChEMBL Value']]
smi = df['Smiles'][1]
mol = Chem.MolFromSmiles(smi)


# -------------------------------
# Coger los enlaces
# -------------------------------

edges = []
for bond in mol.GetBonds():
  i = bond.GetBeginAtomIdx()
  j = bond.GetEndAtomIdx()
  edges.extend([(i,j), (j,i)])
  
edge_index = list(zip(*edges))

def atom_feature(atom):
  return [atom.GetAtomicNum(), 
          atom.GetDegree(),
          atom.GetNumImplicitHs(),
          atom.GetIsAromatic()]

def bond_feature(bond):
  return [bond.GetBondType(), 
          bond.GetStereo()]

node_features = [atom_feature(a) for a in mol.GetAtoms()]
edge_features = [bond_feature(b) for b in mol.GetBonds()]

g = Data(edge_index=torch.LongTensor(edge_index),
         x=torch.FloatTensor(node_features),
         edge_attr=torch.FloatTensor(edge_features),
         smiles=smi,
         mol=mol)

def smi_to_pyg(smi, y):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
      return None

    id_pairs = ((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds())
    atom_pairs = [z for (i, j) in id_pairs for z in ((i, j), (j, i))]

    bonds = (mol.GetBondBetweenAtoms(i, j) for (i, j) in atom_pairs)
    atom_features = [atom_feature(a) for a in mol.GetAtoms()]
    bond_features = [bond_feature(b) for b in bonds]

    return Data(edge_index=torch.LongTensor(list(zip(*atom_pairs))),
                x=torch.FloatTensor(atom_features),
                edge_attr=torch.FloatTensor(bond_features),
                y=torch.FloatTensor([y]),
                mol=mol,
                smiles=smi)

class MyDataset(Dataset):
  def __init__(self, smiles, response):
    mols = [smi_to_pyg(smi, y) for smi, y in \
            tqdm(zip(smiles, response), total=len(smiles))]
    self.X = [m for m in mols if m]

  def __getitem__(self, idx):
    return self.X[idx]

  def __len__(self):
    return len(self.X)

base_dataset = MyDataset(df['Smiles'], df['pChEMBL Value'])

# -------------------------------
# Dividimos en train test y val
# -------------------------------

N = len(base_dataset)
M = N // 10

indices = np.random.permutation(range(N))

idx = {'train': indices[:8*M],
      'valid': indices[8*M:9*M],
      'test': indices[9*M:]}

modes = ['train', 'valid', 'test']

dataset = {m: Subset(base_dataset, idx[m]) for m in modes}
loader = {m: DataLoader(dataset[m], batch_size=200, shuffle=True) if m == 'train' \
          else DataLoader(dataset[m], batch_size=200) for m in modes}
    
# -------------------------------
# Creamos el modelo de random forest junto con los descriptores moleculares
# -------------------------------
    
def ECFP4(mol):
  return np.asarray(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))

X = {m: np.vstack([ECFP4(data.mol) for data in dataset[m]]) for m in modes}
y = {m: np.asarray([data.y.numpy() for data in dataset[m]]).flatten() for m in modes}

model = RandomForestRegressor()
model.fit(X['train'], y['train'])

for m in ['valid', 'test']:
  y_pred = model.predict(X[m])
  for metric in [mean_absolute_error, r2_score]:
    print("{} {} {:.3f}".format(m, metric.__name__, metric(y[m], y_pred)))
    
# -------------------------------
# Creamos el modelo attentive FP
# -------------------------------

node_dim = base_dataset[0].num_node_features
edge_dim = base_dataset[0].num_edge_features
node_dim, edge_dim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AttentiveFP(out_channels=1, # active or inactive
                    in_channels=node_dim, edge_dim=edge_dim,
                    hidden_channels=200, num_layers=6, num_timesteps=2,
                    dropout=0.2)
model = model.to(device)

train_epochs = 100
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, \
                                                steps_per_epoch=len(loader['train']),
                                                epochs=train_epochs)
criterion = nn.L1Loss()

def train(loader):
    total_loss = total_examples = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        total_examples += data.num_graphs
    return total_loss / total_examples

@torch.no_grad()
def test(loader):
    total_loss = total_examples = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = criterion(out, data.y)
        total_loss += loss.item()
        total_examples += data.num_graphs
    return total_loss / total_examples

@torch.no_grad()
def predict(loader):
    y_true = []
    y_pred = []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        y_true.extend(data.y.cpu().numpy())
        y_pred.extend(out.cpu().numpy())
    return y_true, y_pred

best_val = float("inf")

learn_curve = defaultdict(list)
func = {'train': train, 'valid': test, 'test': test}

for epoch in tqdm(range(1, train_epochs+1)):
    loss = {}
    for mode in ['train', 'valid', 'test']: 
      loss[mode] = func[mode](loader[mode])
      learn_curve[mode].append(loss[mode])
    if loss['valid'] < best_val:
      torch.save(model.state_dict(), 'best_val.model')
    if epoch % 20 == 0:
      print(f'Epoch: {epoch:03d} Loss: ' + ' '.join(
          ['{} {:.6f}'.format(m, loss[m]) for m in modes]
      ))

fig, ax = plt.subplots()
for m in modes:
  ax.plot(learn_curve[m], label=m)
ax.legend()
ax.set_xlabel('epochs')
ax.set_ylabel('loss')
ax.set_yscale('log')
plt.show()

model.load_state_dict(torch.load('best_val.model'))

for m in ['valid', 'test']:
  y_true, y_pred = predict(loader[m])
  for metric in [mean_absolute_error, r2_score]:
    print("{} {} {:.3f}".format(m, metric.__name__, metric(y_true, y_pred)))

model.eval()

# Lista para almacenar las etiquetas verdaderas y las predicciones
y_true_train = []
y_pred_train = []

for data in loader['train']:  # Supongo que 'loader' es tu dataloader de entrenamiento
    data = data.to(device)  # Asegúrate de que los datos estén en el mismo dispositivo que el modelo
    out = model(data.x, data.edge_index, data.edge_attr, data.batch)
    y_true_train.extend(data.y.cpu().detach().numpy())
    y_pred_train.extend(out.cpu().detach().numpy())

# Calcula el MAE en el conjunto de entrenamiento
mae_train = mean_absolute_error(y_true_train, y_pred_train)
r2score = r2_score(y_true_train, y_pred_train)
print("MAE en el conjunto de entrenamiento:", mae_train)
print("R2 score en el conjunto de entrenamiento:", r2_score)




    
    
# -------------------------------
# Creamos el modelo de gintw
# -------------------------------

def MyConv(node_dim, edge_dim, arch='GIN'):
  conv = None
  if arch == 'GIN':
    h = nn.Sequential(nn.Linear(node_dim, node_dim, bias=True))
    conv = GINEConv(h, edge_dim=edge_dim)
  elif arch == 'GAT':
    conv = GATv2Conv(node_dim, node_dim, edge_dim=edge_dim)
  return conv

class MyGNN(nn.Module):
  def __init__(self, node_dim, edge_dim, arch, num_layers=3):
    super().__init__()
    layers = [MyConv(node_dim, edge_dim, arch) for _ in range(num_layers)]
    self.convs = nn.ModuleList(layers)

  def forward(self, x, edge_index, edge_attr):
    for conv in self.convs:
      x = conv(x, edge_index, edge_attr)
      x = F.leaky_relu(x)
    return x

[int(x) for x in Chem.rdchem.BondType.names.values()]

ptable = Chem.GetPeriodicTable()
for i in range(200):
  try:
    s = ptable.GetElementSymbol(i)
  except:
    print(f'max id {i-1} for {s}')
    break
ptable.GetElementSymbol(i-1)

class MyFinalNetwork(nn.Module):
  def __init__(self, node_dim, edge_dim, arch, num_layers=3, 
               encoding='onehot'):
    super().__init__()

    self.encoding = encoding
    if encoding != 'onehot':
      self.atom_encoder = nn.Embedding(num_embeddings=118+1, embedding_dim=64)
      self.bond_encoder = nn.Embedding(num_embeddings=21+1, embedding_dim=8)
      node_dim = (node_dim-1) + 64
      edge_dim = (edge_dim-1) + 8
    else:
      node_dim = (node_dim-1) + 118+1
      edge_dim = (edge_dim-1) + 21+1

    self.gnn = MyGNN(node_dim, edge_dim, arch, num_layers=num_layers)
    embed_dim = int(node_dim / 2)
    self.head = nn.Sequential(
        nn.BatchNorm1d(node_dim),
        nn.Dropout(p=0.5),
        nn.Linear(node_dim, embed_dim, bias=True),
        nn.ReLU(),
        nn.BatchNorm1d(embed_dim),
        nn.Dropout(p=0.5),
        nn.Linear(embed_dim, 1)
    )
  def forward(self, x, edge_index, edge_attr, batch):
    if self.encoding == 'onehot':
      x0 = F.one_hot(x[:, 0].to(torch.int64), num_classes=118+1)
      edge_attr0 = F.one_hot(edge_attr[:, 0].to(torch.int64), num_classes=21+1)
    else:
      x0 = self.atom_encoder(x[:, 0].int())
      edge_attr0 = self.bond_encoder(edge_attr[:, 0].int())
    
    x = torch.cat([x0, x[:, 1:]], dim=1)
    edge_attr = torch.cat([edge_attr0, edge_attr[:, 1:]], dim=1)

    node_out = self.gnn(x, edge_index, edge_attr)
    graph_out = global_add_pool(node_out, batch)
    return self.head(graph_out)

model = MyFinalNetwork(node_dim, edge_dim, arch='GAT', num_layers=3, encoding='embedding')
model = model.to(device)

train_epochs = 100
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, \
                                                steps_per_epoch=len(loader['train']),
                                                epochs=train_epochs)
criterion = nn.MSELoss()
best_val = float("inf")

learn_curve = defaultdict(list)
func = {'train': train, 'valid': test, 'test': test}

for epoch in tqdm(range(1, train_epochs+1)):
    loss = {}
    for mode in ['train', 'valid', 'test']: 
      loss[mode] = func[mode](loader[mode])
      learn_curve[mode].append(loss[mode])
    if loss['valid'] < best_val:
      torch.save(model.state_dict(), 'best_val.model')
    if epoch % 20 == 0:
      print(f'Epoch: {epoch:03d} Loss: ' + ' '.join(
          ['{} {:.6f}'.format(m, loss[m]) for m in modes]
      ))
fig, ax = plt.subplots()
for m in modes:
  ax.plot(learn_curve[m], label=m)
ax.legend()
ax.set_xlabel('epochs')
ax.set_ylabel('loss')
ax.set_yscale('log')
plt.show()

model.load_state_dict(torch.load('best_val.model'))

for m in ['valid', 'test']:
  y_true, y_pred = predict(loader[m])
  for metric in [mean_absolute_error, r2_score]:
    print("{} {} {:.3f}".format(m, metric.__name__, metric(y_true, y_pred)))
    