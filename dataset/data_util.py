import math
import json
import os
import numpy as np
from torch.utils.data import Dataset, sampler, DataLoader
from typing import Dict, List, Set, Tuple, Union
import torch
import random
from collections import defaultdict
import dataset.load_dataset as preprocess
import numpy
import math, os
from tqdm import tqdm
import random
import csv
from collections import OrderedDict
import json, pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import SaltRemover

# absolute_path = os.path.abspath(__file__)
# DATA_PATH = "/".join(absolute_path.split("/")[:-2]+["data"])

def check_smiles_validity(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 'smiles_unvaild'
        HA_num = mol.GetNumHeavyAtoms()
        if HA_num <= 2:
            return 'smiles_unvaild'
        return smiles
    except:
        return 'smiles_unvaild'

def gen_standardize_smiles(smiles, kekule=False, random=False):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 'smiles_unvaild'
        desalt = SaltRemover.SaltRemover() ## defnData="[Cl,Br,I,Fe,Na,K,Ca,Mg,Ni,Zn]"
        mol = desalt.StripMol(mol)
        if mol is None:
            return 'smiles_unvaild'
        HA_num = mol.GetNumHeavyAtoms()
        if HA_num <= 2:
            return 'smiles_unvaild'
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol, kekuleSmiles=kekule, doRandom=random, isomericSmiles=True)
        return smiles
    except:
        smiles = 'smiles_unvaild'


def get_chembl_targets():

    # E3FP
    save_path = "D:/YQ/code/Act_GeminiMol/data/chembl/float_fps_501156.pkl"
    # save_path = "D:/YQ/code/Act_GeminiMol/data/chembl/chembl_data.pkl"
    smiles_e3fps = pickle.load(open(save_path, "rb"))

    datas = csv.reader(open("D:/YQ/code/Act_GeminiMol/data/chembl/processed_chembl_target_assay.csv", "r"),
                       delimiter=',')
    header = next(datas)
    print(f"header:{header}")
    target_id_dicts = {}

    for line in datas:
        unit = line[7]   # nM
        if unit=="%":
            continue
        assay_id = line[11]
        target_chembl_id = line[15]
        # DHODH Target ID
        if target_chembl_id == 'CHEMBL1966':
            continue
        # target_id = "{}_{}_{}".format(line[15], line[7], line[8]).replace("/", "_")
        target_id = "{}_{}".format(line[15], line[11]).replace("/", "_")
        # target_id = "{}_{}_{}_{}".format(line[15], line[11], line[7], line[8]).replace("/", "_")
        if target_id not in target_id_dicts:
            target_id_dicts[target_id] = []

        smiles = line[13]
        # Get E3FP smiles
        e3fp = smiles_e3fps.get(smiles, None)
        # if the smiles is not found then skip
        if e3fp is None:
            continue

        assay_type = line[9]
        bao_endpoint = line[4]
        bao_format = line[10]
        std_type = line[8]
        # if std_type.lower() != "kd":
        #     continue
        unit = line[7]
        std_rel = line[5]

        if std_rel != "=" and std_rel != "'='":
            continue
        is_does = unit in ['ug.mL-1', 'ug ml-1', 'mg.kg-1', 'mg kg-1',
                           'mg/L', 'ng/ml', 'mg/ml', 'ug kg-1', 'mg/kg/day', 'mg kg-1 day-1',
                           "10'-4 ug/ml", 'M kg-1', "10'-6 ug/ml", 'ng/L', 'pmg kg-1', "10'-8mg/ml",
                           'ng ml-1', "10'-3 ug/ml", "10'-1 ug/ml", ]
        pic50_exp = -math.log10(float(line[6]))
        affi_prefix = line[5]
        ligand_info = {
            "assay_type": std_type,
            "smiles": smiles,
            "e3fp": e3fp,
            "pic50_exp": pic50_exp,
            "affi_prefix": affi_prefix,
            "is_does": is_does,
            "chembl_assay_type": assay_type,
            "bao_endpoint": bao_endpoint,
            "bao_format": bao_format,
            "unit": unit,
            "domain": "chembl"
        }
        target_id_dicts[target_id].append(ligand_info)

    split_path = f'D:/YQ/code/Act_GeminiMol/data/chembl/chembl_valid_test_split.json'
    split_chembl_val_test = json.load(open(split_path, "r"))
    train_targets = split_chembl_val_test['train']
    # train_targets 10111
    # print('train_targets', len(train_targets))

    target_id_dicts_new = {}
    for target_id, ligands in target_id_dicts.items():
        # pic50_exp_list = [x["pic50_exp"] for x in ligands]
        # pic50_std = np.std(pic50_exp_list)
        # if pic50_std <= 0.2:
        #     continue
        if len(ligands) < 20 or len(ligands) > 30000:
            continue
        # if len(ligands) > 10000:
            # print(f'len(ligands)={len(ligands)}', target_id)
        # target_id_dicts_new[target_id] = ligands
        ''' 12 = 6 + 4 + 1 + 1
        len(ligands)=21977 CHEMBL1293294_nM_Potency
        len(ligands)=13873 CHEMBL4040_nM_Potency
        len(ligands)=12659 CHEMBL203_nM_IC50
        len(ligands)=20460 CHEMBL4159_nM_Potency
        len(ligands)=10832 CHEMBL205_nM_Ki
        len(ligands)=23043 CHEMBL2392_nM_Potency
        len(ligands)=13590 CHEMBL340_nM_Potency
        len(ligands)=10937 CHEMBL279_nM_IC50
        len(ligands)=75813 CHEMBL3577_nM_Potency
        len(ligands)=13280 CHEMBL1075189_nM_Potency
        len(ligands)=34993 CHEMBL2608_nM_Potency
        len(ligands)=24247 CHEMBL1293255_nM_Potency
        '''
        if target_id in train_targets:
            if len(ligands) > 64:
                target_set = len(ligands) // 64
                target_ligand = len(ligands) % 64
                for i in range(target_set):
                    first_idx = i * 64
                    end_idx = first_idx + 64
                    ligands_set = ligands[first_idx:end_idx]
                    target_id_dicts_new[f"{target_id}_{i + 1}"] = ligands_set
                if target_ligand > 0:
                    if target_ligand < 20:
                        last_target_id = f"{target_id}_{target_set}"
                        target_id_dicts_new[last_target_id].extend(ligands[-target_ligand:])
                    else:
                        new_target_id = f"{target_id}_{target_set + 1}"
                        target_id_dicts_new[new_target_id] = ligands[-target_ligand:]
            else:
                target_id_dicts_new[target_id] = ligands
        else:
            target_id_dicts_new[target_id] = ligands
    return {"ligand_sets": target_id_dicts_new, "targets": list(target_id_dicts_new.keys())}


def get_bdb_targets():
    data_dir = f"D:/YQ/code/Act_GeminiMol/data/bdb/polymer_bdb"
    ligand_sets = {}
    means = []

    # E3FP
    save_path = "D:/YQ/code/Act_GeminiMol/data/bdb/bdb_fps_558599.pkl"
    # save_path = "D:/YQ/code/Act_GeminiMol/data/bdb/bdb_data.pkl"
    smiles_e3fps = pickle.load(open(save_path, "rb"))

    for target_name in tqdm(list(os.listdir(data_dir))):
        # DHODH Target Name
        if target_name == 'Dihydroorotate dehydrogenase':
            continue
        for assay_file in os.listdir(os.path.join(data_dir, target_name)):
            target_assay_name = target_name + "/" + assay_file
            # target_assay_name = target_name + "/" + assay_file[-5]
            entry_assay = "_".join(assay_file.split("_")[:2])
            affi_idx = int(assay_file[-5])
            ligands = []
            affix = []
            file_lines = list(open(os.path.join(data_dir, target_name, assay_file), "r", encoding="utf-8").readlines())
            for i, line in enumerate(file_lines):
                line = line.strip().split("\t")
                affi_prefix = ""
                pic50_exp = line[8+affi_idx].strip()
                if pic50_exp.startswith(">") or pic50_exp.startswith("<"):
                    continue
                    # affi_prefix = pic50_exp[0]
                    # pic50_exp = pic50_exp[1:]
                try:
                    pic50_exp = -math.log10(float(pic50_exp))
                    # print('pic50_exp', pic50_exp)
                except:
                    print("error ic50s:", pic50_exp)
                    continue
                smiles = line[1]
                # Get E3FP smiles
                e3fp = smiles_e3fps.get(smiles, None)
                if e3fp is None:
                    continue
                affix.append(pic50_exp)
                ligand_info = {
                    "affi_idx": affi_idx,
                    "affi_prefix": affi_prefix,
                    "smiles": smiles,
                    "e3fp": e3fp,
                    "pic50_exp": pic50_exp,
                    "domain": "bdb"
                }
                ligands.append(ligand_info)
            # pic50_exp_list = [x["pic50_exp"] for x in ligands]
            # pic50_std = np.std(pic50_exp_list)
            # if pic50_std <= 0.2:
            #     continue
            if len(ligands) <= 20:
                continue
            means.append(np.mean([x["pic50_exp"] for x in ligands]))
            ligand_sets[target_assay_name] = ligands

    # target_id_dicts = {}
    # for target_assay_name, ligands in ligand_sets.items():
    #     target_name, assay_file = target_assay_name.split("/")
    #     max_idx = assay_file.split("_")[-1].replace(".tsv", "")
    #     target_id = f"{target_name}_{max_idx}"
    #     if target_id not in target_id_dicts:
    #         target_id_dicts[target_id] = []
    #     target_id_dicts[target_id].extend(ligands)
    #
    # target_id_dicts_new = {}
    # for target_id, ligands in target_id_dicts.items():
    #     # pic50_exp_list = [x["pic50_exp"] for x in ligands]
    #     # pic50_std = np.std(pic50_exp_list)
    #     # if pic50_std <= 0.2:
    #     #     continue
    #     if len(ligands) < 10:
    #         # print(f'len(ligands)={len(ligands)}', target_id)
    #         continue
    #     means.append(np.mean([x["pic50_exp"] for x in ligands]))
    #     target_id_dicts_new[target_id] = ligands
    # means - 2.254907135561684
    print('means', np.mean(means))
    # target_id_dicts_new = ligand_sets

    split_path = f'D:/YQ/code/Act_GeminiMol/data/bdb/bdb_valid_test_split.json'
    split_bdb_val_test = json.load(open(split_path, "r"))
    train_targets = split_bdb_val_test['train']
    # train_targets 14555
    print('train_targets', len(train_targets))

    target_id_dicts_new = {}
    for target_id, ligands in ligand_sets.items():
        # pic50_exp_list = [x["pic50_exp"] for x in ligands]
        # pic50_std = np.std(pic50_exp_list)
        # if pic50_std <= 0.2:
        #     continue
        if len(ligands) < 20 or len(ligands) > 30000:
            continue
        if target_id in train_targets:
            if len(ligands) > 64:
                target_set = len(ligands) // 64
                target_ligand = len(ligands) % 64
                for i in range(target_set):
                    first_idx = i * 64
                    end_idx = first_idx + 64
                    ligands_set = ligands[first_idx:end_idx]
                    target_id_dicts_new[f"{target_id}_{i + 1}"] = ligands_set
                if target_ligand > 0:
                    if target_ligand < 20:
                        last_target_id = f"{target_id}_{target_set}"
                        target_id_dicts_new[last_target_id].extend(ligands[-target_ligand:])
                    else:
                        new_target_id = f"{target_id}_{target_set + 1}"
                        target_id_dicts_new[new_target_id] = ligands[-target_ligand:]
            else:
                target_id_dicts_new[target_id] = ligands
        else:
            target_id_dicts_new[target_id] = ligands
    return {"ligand_sets": target_id_dicts_new,
            "targets": list(target_id_dicts_new.keys())}


def get_activity_cliff_targets():
    # E3FP
    save_path = "D:/YQ/code/Act_GeminiMol/data/activity_cliff/activity_cliff_fps.pkl"
    smiles_e3fps = pickle.load(open(save_path, "rb"))

    smiles_as_target = csv.reader(open("D:/YQ/code/Act_GeminiMol/data/activity_cliff/processed_activity_cliff.csv", "r"), delimiter=',')
    header = next(smiles_as_target)
    print(f"header of activity_cliff:{header}")

    target_dicts = {}
    for line in smiles_as_target:
        smiles = line[0]
        exp_value = line[2]

        if exp_value.strip() == '':
            continue

        # Get E3FP smiles
        e3fp = smiles_e3fps.get(smiles, None)
        # if the smiles is not found then skip
        if e3fp is None:
            continue

        assay_type = line[4]
        target_chembl_id = line[5]
        target_id = "{}_{}".format(assay_type, target_chembl_id).replace("/", "_")
        if target_id not in target_dicts:
            target_dicts[target_id] = []

        pic50_exp = -math.log10(float(exp_value))
        ligand_info = {
            "smiles": smiles,
            "e3fp": e3fp,
            "pic50_exp": pic50_exp,
            "domain": "activity_cliff"
        }
        target_dicts[target_id].append(ligand_info)

    target_dicts_new = {}
    for target_id, ligands in target_dicts.items():
        # pic50_exp_list = [x["pic50_exp"] for x in ligands]
        # pic50_std = np.std(pic50_exp_list)
        # if pic50_std <= 0.2:
        #     continue
        if len(ligands) < 20 or len(ligands) > 10000:
            continue
        target_set = len(ligands) // 32
        target_ligand = len(ligands) % 32
        for i in range(target_set):
            first_idx = i * 32
            end_idx = first_idx + 32
            ligands_set = ligands[first_idx:end_idx]
            target_dicts_new[f"{target_id}_{i+1}"] = ligands_set
        if target_ligand > 0:
            if target_ligand < 20:
                    last_target_id = f"{target_id}_{target_set}"
                    target_dicts_new[last_target_id].extend(ligands[-target_ligand:])
            else:
                new_target_id = f"{target_id}_{target_set + 1}"
                target_dicts_new[new_target_id] = ligands[-target_ligand:]

    return {"ligand_sets": target_dicts_new, "targets": list(target_dicts_new.keys())}


def preprocess_targets(in_data):
    lines = in_data
    x_tmp = []
    smiles_list = []
    activity_list = []

    if lines is None:
        return None

    if len(lines) > 30000:
        return None

    for line in lines:
        smiles = line["smiles"]

        # Generate ECFP fingerprint
        # mol = Chem.MolFromSmiles(smiles)
        # if mol is None:
        #     continue
        # fingerprints_vect = rdFingerprintGenerator.GetCountFPs(
        #     [mol], fpType=rdFingerprintGenerator.MorganFP
        # )[0]
        # fp_numpy = np.zeros((0,), np.int8)
        # DataStructs.ConvertToNumpyArray(fingerprints_vect, fp_numpy)

        # Get e3fp fingerprint
        fp_numpy = np.array(line["e3fp"])
        # print(f"type(fp_numpy):{type(fp_numpy)},{fp_numpy}")

        pic50_exp = line["pic50_exp"]
        activity_list.append(pic50_exp)
        x_tmp.append(fp_numpy)
        smiles_list.append(smiles)

    x_tmp = np.array(x_tmp).astype(np.float32)
    affis = np.array(activity_list).astype(np.float32)
    if len(x_tmp) < 20 and lines[0].get("domain", "none") in ['chembl', 'bdb']:
        return None
    return x_tmp, affis, smiles_list


if __name__ == "__main__":
    # res = get_chembl_targets()
    res = get_bdb_targets()
    # res = get_activity_cliff_targets()
    '''
    chembl:(no e3fp)-targets_num:2259(no assay_id)10573/10516(assay_id);
    bdb:(no e3fp)targets_num:2606/3210(no assay_id)15510
    chembl:targets_num:2238-20,2719-10,2717(no assay_id)
           10511/(assay_id)
    bdb:targets_num:2606/3132(no assay_id)
        (assay_id)targets_num:15155
    chembl_split_64:targets_num:15978
    bdb_split_64:targets_num:20436
    '''
    # "Aurora-kinase-B_2", {res.get("ligand_sets")["Aurora-kinase-B_2"]}
    print(f'targets_num:{len(res.get("targets"))}')
    lengths = [len(lst) for lst in res.get("ligand_sets").values()]
    '''
    chembl:(no e3fp)9949 20 353.1974324922532(no assay_id)
                    9819 20 58.30871086730351(assay_id)9819 20 58.33339672879422
    BDB:(no e3fp) 9572 10 302.5753894080997(no assay_id)
                  2225 20 61.00322372662798(assay_id)
    chembl:9932 20 354.142091152815(no assay_id)
           75813 10 395.29716807649874(>30000)
           24247 10 354.8056680161943;
           assay_id 24165 20 69.31091237750928
    bdb:9544 10 306.9406130268199
        assay_id:2217 21 61.503134279115805
    chembl_split_64:length 11884 20 45.595631493303294
    bdb_split_64:length 1101 20 45.60970835779996
    
    activity_cliff
    [849, 765, 817, 659, 2923, 1802, 3796, 1073, 2057, 1962,
     1069, 3506, 4113, 2595, 2955, 1087, 3037, 1255, 1936, 3272, 
     860, 3180, 704, 1528, 1048, 1209, 1022, 731, 754, 1603]
    '''
    print("length", max(lengths), min(lengths), sum(lengths) / len(lengths))

    ligand_sets = res.get("ligand_sets")
    smiles_list = []
    for assay in ligand_sets.values():
        for ligand in assay:
            smiles_list.append(ligand["smiles"])

    # split_path = f'D:/YQ/code/Act_GeminiMol/data/chembl/chembl_valid_test_split.json'
    # split_chembl_val_test = json.load(open(split_path, "r"))
    split_path = f'D:/YQ/code/Act_GeminiMol/data/bdb/bdb_valid_test_split.json'
    split_bdb_val_test = json.load(open(split_path, "r"))
    split_val_test = split_bdb_val_test
    train_dict = {'train': []}
    for target_id, ligands in ligand_sets.items():
        if target_id not in split_val_test['valid'] and target_id not in split_val_test['test']:
            train_dict['train'].append(target_id)
    # chembl_train_dict 15578;bdb_train_dict 19836
    print('train_dict', len(train_dict['train']))
    # chembl_file_path = 'D:/YQ/code/Act_GeminiMol/data/chembl/chembl_train_split.json'
    bdb_file_path = 'D:/YQ/code/Act_GeminiMol/data/bdb/bdb_train_split.json'
    # with open(file_path, 'w', encoding='utf-8') as f:
    #     json.dump(train_dict, f, ensure_ascii=False, indent=4)
    '''
    bdb:926474 549211; 968001 563856;
    chembl:792570 377122;1074813 491175-70000;999000 458757-50000;
           964007 447339;(assay_id) 728527 370692           
    bdb:961338 558582;(assay_id)932080 550333(no e3fp)556203
    '''
    print('smiles_list', len(smiles_list), len(set(smiles_list)))
    # df = pd.DataFrame(set(smiles_list), columns=['smiles']).dropna()
    # df.to_csv("D:/YQ/code/Act_GeminiMol/data/BDB_unique_smiles.csv", index=False, encoding='utf-8')
    # # bdb_smiles_data = pd.read_csv('D:/YQ/code/Act_GeminiMol/data/unique_BDB_smiles(valid).csv')
    # bdb_smiles = set(list(bdb_smiles_data['smiles']))
    # print('bdb_smiles', len(bdb_smiles))   # 549207
    # unique_to_smiles = set(smiles_list) - bdb_smiles
    # print('unique', len(unique_to_smiles)) # 14649
    # df_unique = pd.DataFrame(unique_to_smiles, columns=['smiles']).dropna()
    # df_unique.to_csv("D:/YQ/code/Act_GeminiMol/data/BDB_unique_smiles_14649.csv", index=False, encoding='utf-8')

    # data = res.get("targets")
    # random.shuffle(data)
    # valid_size = 200
    # test_size = 200
    # train_size = len(data) - test_size - valid_size
    # test_data = data[:test_size]
    # valid_data = data[test_size:test_size + valid_size]
    # train_data = data[test_size + valid_size:]
    # # no assay_id chembl:2567 50 100;bdb:2962 70 100
    # # assay_id chembl:10111 200 200;bdb:14555 300 300
    # print(len(train_data), len(valid_data), len(test_data))
    # data_split = {
    #     'train': train_data,
    #     'test': test_data,
    #     'valid': valid_data}
    # print(len(data_split['train']), len(data_split['valid']), len(data_split['test']))
    # file_path = 'D:/YQ/code/Act_GeminiMol/data/chembl/chembl_target_assay_split.json'
    # # file_path = 'D:/YQ/code/Act_GeminiMol/data/bdb/bdb_target_assay_split.json'
    # with open(file_path, 'w', encoding='utf-8') as f:
    #     json.dump(data_split, f, ensure_ascii=False, indent=4)


class BaseMetaDataset(Dataset):
    def __init__(self, args, exp_string):
        self.args = args
        self.current_set_name = "train"
        self.exp_string = exp_string

        self.init_seed = {"train": args.train_seed, "val": args.val_seed, 'test': args.test_seed,
                          'train_weight': args.train_seed}
        self.batch_size = args.meta_batch_size

        self.train_index = 0
        self.val_index = 0
        self.test_index = 0
        self.split_all = []

        self.current_epoch = 0
        self.load_dataset()

    def load_dataset(self):
        raise NotImplementedError

    # y_opls4=None, scaffold_split=None, smiles=None
    def get_split(self, X_in, y_in, is_test=False, sup_num=None,  rand_seed=None, smiles=None):
        def data_split(data_len, sup_num_, rng_):
            if not is_test:
                # 0.3
                min_num = math.log10(max(10, int(0.3 * data_len)))
                max_num = math.log10(int(0.85 * data_len))
                # todo:for few-shot setting
                sup_num_ = random.uniform(min_num, max_num)
                sup_num_ = math.floor(10 ** sup_num_)
            split = [1] * sup_num_ + [0] * (data_len - sup_num_)
            rng_.shuffle(split)
            return np.array(split)

        def data_split_byvalue(y, sup_num_):
            sup_index = np.argpartition(y, sup_num_)[:sup_num_].tolist()
            split = []
            for i in range(len(y)):
                if i in sup_index:
                    split.append(1)
                else:
                    split.append(0)
            return np.array(split)

        def data_split_activity(y, sup_num_):
            sup_index = np.argpartition(y, -sup_num_)[-sup_num_:].tolist()
            split = []
            for i in range(len(y)):
                split.append(1 if i in sup_index else 0)
            return np.array(split)

        # def data_split_bysaffold(smiles, sup_num_):
        #     scaffold_dict = scaffold_to_smiles(smiles, use_indices=True)
        #     scaffold_id_list = [(k, v) for k, v in scaffold_dict.items()]
        #     scaffold_id_list = sorted(scaffold_id_list, key=lambda x: len(x[1]))
        #     idx_list_all = []
        #     for scaffold, idx_list in scaffold_id_list:
        #         idx_list_all += idx_list
        #
        #     sup_index = idx_list_all[:sup_num_]
        #     split = []
        #     for i in range(len(y)):
        #         if i in sup_index:
        #             split.append(1)
        #         else:
        #             split.append(0)
        #     return np.array(split)

        def data_split_bysim(Xs, sup_num_, rng_, sim_cut_):
            def get_sim_matrix(a, b):
                a_bool = (a > 0.).float()
                b_bool = (b > 0.).float()
                and_res = torch.mm(a_bool, b_bool.transpose(0, 1))
                or_res = a.shape[-1] - torch.mm((1. - a_bool), (1. - b_bool).transpose(0, 1))
                sim = and_res / or_res
                return sim

            Xs_torch = torch.tensor(Xs).cuda()
            sim_matrix = get_sim_matrix(Xs_torch, Xs_torch).cpu().numpy() - np.eye(len(Xs))
            split = [1] * sup_num_ + [0] * (len(Xs) - sup_num_)
            rng_.shuffle(split)
            sup_index = [i for i, t in enumerate(split) if t == 1]

            split = []
            for i in range(len(y)):
                if i in sup_index:
                    split.append(1)
                else:
                    max_sim = np.max(sim_matrix[i][sup_index])
                    if max_sim >= sim_cut_:
                        split.append(-1)
                    else:
                        split.append(0)
            return np.array(split)

        # if np.std(y_in) > 0:
        #     y_in = (np.array(y_in) - np.mean(y_in)) / np.std(y_in)
        # else:
        #     y_in = np.array(y_in) - np.mean(y_in)
        rng = np.random.RandomState(seed=rand_seed)
        # 64
        if len(X_in) > 64 and not is_test:
            subset_num = 64
            raw_data_len = len(X_in)
            select_idx = [1] * subset_num + [0] * (raw_data_len - subset_num)
            rng.shuffle(select_idx)
            select_idx = np.nonzero(np.array(select_idx))
            X, y = X_in[select_idx], y_in[select_idx]
        else:
            X, y = X_in, y_in

        sup_num = self.args.test_sup_num
        if sup_num <= 1:
            sup_num = sup_num * len(X)
        sup_num = int(sup_num)
        if self.args.similarity_cut < 1.:
            assert is_test
            split = data_split_bysim(X, sup_num, rng, self.args.similarity_cut)
            X = np.array([t for i, t in enumerate(X) if split[i] != -1])
            y = [t for i, t in enumerate(y) if split[i] != -1]
            split = np.array([t for i, t in enumerate(split) if t != -1], dtype=np.int)
        else:
            ''' split random'''
            split = data_split(len(X), sup_num, rng)
            ''' split by activity'''
            # split = data_split_activity(y, sup_num)
        # if y_opls4 is not None:
        #     assert len(y_opls4) == len(y)
        #     y = (1 - split) * y + split * np.array(y_opls4)

        return [X, y, split]

    def get_set(self, current_set_name, idx):
        datas = []
        targets = []
        si_list = []
        ligand_nums = []
        ligands_all = []

        if current_set_name == 'train':
            si_list = self.train_indices[idx * self.batch_size: (idx + 1) * self.batch_size]
            ret_weight = [1. for _ in si_list]
        elif current_set_name == 'val':
            si_list = [self.val_indices[idx]]
            ret_weight = [1.]
        elif current_set_name == 'test':
            si_list = [self.test_indices[idx]]
            ret_weight = [1.]
        elif current_set_name == 'train_weight':
            if self.idxes is not None:
                si_list = self.idxes[idx * self.weighted_batch: (idx + 1) * self.weighted_batch]
                ret_weight = self.train_weight[idx * self.weighted_batch: (idx + 1) * self.weighted_batch]
            else:
                si_list = [self.train_indices[idx]]
                ret_weight = [1.]

        for si in si_list:
            ligand_nums.append(len(self.Xs[si]))
            # if self.args.expert_test == "fep_opls4":
            #     y_opls4 = self.assayid2opls4_dict.get(self.assaes[si], None)
            # else:
            #     y_opls4 = None
            target_name = self.targets[si]
            # if len(self.split_all) > 0:
            #     scaffold_split = self.split_all[si]
            # else:
            #     scaffold_split = None
            datas.append(self.get_split(self.Xs[si], self.ys[si],
                                        is_test=current_set_name in ['test', 'val', 'train_weight'],
                                        # scaffold_split=scaffold_split,
                                        # y_opls4=y_opls4,
                                        rand_seed=self.init_seed[current_set_name] + si + self.current_epoch,
                                        smiles=self.smiles_all[si]
                                        ))
            targets.append(target_name)
            ligands_all.append(self.smiles_all[si])

        return tuple([[torch.tensor(x[i]) for x in datas] for i in range(0, 3)] +
                     [targets, ret_weight, ligands_all])

    def __len__(self):
        if self.current_set_name == "train":
            total_samples = self.data_length[self.current_set_name] // self.args.meta_batch_size
        elif self.current_set_name == "train_weight":
            if self.idxes is not None:
                total_samples = len(self.idxes) // self.weighted_batch
            else:
                total_samples = self.data_length["train_weight"] // self.weighted_batch
        else:
            total_samples = self.data_length[self.current_set_name]
        return total_samples

    def length(self, set_name):
        self.switch_set(set_name=set_name)
        return len(self)

    def set_train_weight(self, train_weight=None, idxes=None, weighted_batch=1):
        self.train_weight = train_weight
        self.idxes = idxes
        self.weighted_batch = weighted_batch

    def switch_set(self, set_name, current_epoch=0):
        self.current_set_name = set_name
        self.current_epoch = current_epoch
        if set_name == "train":
            rng = np.random.RandomState(seed=self.init_seed["train"] + current_epoch)
            rng.shuffle(self.train_indices)

    def __getitem__(self, idx):
        return self.get_set(self.current_set_name, idx=idx)


def my_collate_fn(batch):
    batch = batch[0]
    return batch


class SystemDataLoader(object):
    def __init__(self, args, MetaDataset, current_epoch=0, exp_string=None):
        """
        Initializes a meta learning system dataloader. The data loader uses the Pytorch DataLoader class to parallelize
        batch sampling and preprocessing.
        :param args: An arguments NamedTuple containing all the required arguments.
        :param current_epoch: Current iter of experiment. Is used to make sure the data loader continues where it left
        of previously.
        """
        self.args = args
        self.batch_size = args.meta_batch_size
        self.total_train_epochs = 0
        self.dataset = MetaDataset(args, exp_string=exp_string)
        self.full_data_length = self.dataset.data_length
        self.continue_from_epoch(current_epoch=current_epoch)

    def get_train_dataloader(self):
        """
        Returns a data loader with the correct set (train, val or test), continuing from the current iter.
        :return:
        """
        return DataLoader(self.dataset, batch_size=1, num_workers=2, shuffle=False, drop_last=True,
                          collate_fn=my_collate_fn)

    def get_dataloader(self):
        """
        Returns a data loader with the correct set (train, val or test), continuing from the current iter.
        :return:
        """
        return DataLoader(self.dataset, batch_size=1, shuffle=False, drop_last=True, collate_fn=my_collate_fn)

    def continue_from_epoch(self, current_epoch):
        """
        Makes sure the data provider is aware of where we are in terms of training iterations in the experiment.
        :param current_epoch:
        """
        self.total_train_epochs += current_epoch

    def get_train_batches_weighted(self, weights=None, idxes=None, weighted_batch=1):
        """
        Returns a training batches data_loader
        :param total_batches: The number of batches we want the data loader to sample
        :param augment_images: Whether we want the images to be augmented.
        """
        self.dataset.switch_set(set_name="train_weight", current_epoch=self.total_train_epochs)
        self.dataset.set_train_weight(weights, idxes, weighted_batch=weighted_batch)
        self.total_train_epochs += 1
        return self.get_dataloader()

    def get_train_batches(self, total_batches=-1):
        """
        Returns a training batches data_loader
        :param total_batches: The number of batches we want the data loader to sample
        :param augment_images: Whether we want the images to be augmented.
        """
        if total_batches == -1:
            self.dataset.data_length = self.full_data_length
        else:
            self.dataset.data_length["train"] = total_batches  # * self.dataset.batch_size
        self.dataset.switch_set(set_name="train", current_epoch=self.total_train_epochs)
        self.total_train_epochs += self.batch_size
        return self.get_train_dataloader()

    def get_val_batches(self, total_batches=-1, repeat_cnt=0):
        """
        Returns a validation batches data_loader
        :param total_batches: The number of batches we want the data loader to sample
        :param repeat_cnt:
        """
        if total_batches == -1:
            self.dataset.data_length = self.full_data_length
        else:
            self.dataset.data_length['val'] = total_batches  # * self.dataset.batch_size
        self.dataset.switch_set(set_name="val", current_epoch=repeat_cnt)
        return self.get_dataloader()

    def get_test_batches(self, total_batches=-1, repeat_cnt=0):
        """
        Returns a testing batches data_loader
        :param total_batches: The number of batches we want the data loader to sample
        :param repeat_cnt:
        """
        if total_batches == -1:
            self.dataset.data_length = self.full_data_length
        else:
            self.dataset.data_length['test'] = total_batches  # * self.dataset.batch_size
        self.dataset.switch_set(set_name='test', current_epoch=repeat_cnt)
        return self.get_dataloader()
