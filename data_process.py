import os
import json, pickle
import requests

import pandas as pd
import networkx as nx
import numpy as np

from rdkit import Chem
from rdkit.Chem import MACCSkeys
from collections import OrderedDict


def protein_feature(sequence):
    seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
    seq_dict = {v: i for i, v in enumerate(seq_voc)}
    max_seq_len = 1000

    def seq_cat(prot):
        x = np.zeros(max_seq_len)
        for i, ch in enumerate(prot[:max_seq_len]):
            x[i] = seq_dict[ch]
        return x

    return [seq_cat(sequence)]


def molecule_feature(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fingerprints = MACCSkeys.GenMACCSKeys(mol)  # 167 bit
    return [float(s) for s in fingerprints.ToBitString()]


def data_prepare(dataset, fold):
    dataset_path = os.path.join("data", dataset)
    train_fold_origin = json.load(open(dataset_path + "/folds/train_fold_setting1.txt"))
    train_fold_origin = [e for e in train_fold_origin]  # for 5 folds
    train_fold = []
    valid_fold = train_fold_origin[fold]
    for i in range(len(train_fold_origin)):
        if i != fold:
            train_fold += train_fold_origin[i]
    affinity = pickle.load(open(dataset_path + "/Y", "rb"), encoding='latin1')
    test_fold = json.load(open(dataset_path + "/folds/test_fold_setting1.txt"))
    ligands = json.load(open(dataset_path + "/ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(dataset_path + "/proteins.txt"), object_pairs_hook=OrderedDict)

    pro_features = []
    drug_features = []
    for d in ligands.keys():
        # lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        drug_features.append(molecule_feature(ligands[d]))

    for t in proteins.keys():
        pro_features.append(protein_feature(proteins[t]))

    data = {}
    if dataset == 'davis':
        affinity = [-np.log10(y / 1e9) for y in affinity]
    affinity = np.asarray(affinity)
    opts = ['train', 'valid', 'test']
    for opt in opts:
        if opt == 'train':
            rows, cols = np.where(np.isnan(affinity) == False)
            rows, cols = rows[train_fold], cols[train_fold]
            train_entries = [rows, cols]
            data["train"] = train_entries
        elif opt == 'valid':
            rows, cols = np.where(np.isnan(affinity) == False)
            rows, cols = rows[valid_fold], cols[valid_fold]
            valid_entries = [rows, cols]
            data["valid"] = valid_entries
        elif opt == 'test':
            rows, cols = np.where(np.isnan(affinity) == False)
            rows, cols = rows[test_fold], cols[test_fold]
            test_entries = [rows, cols]
            data["test"] = test_entries

    return data, drug_features, pro_features, affinity
