import json
import pickle
import os
import numpy as np
import torch
from torch.utils.data import Dataset, sampler, DataLoader
import tqdm
from multiprocessing import Pool
from dataset.data_util import get_chembl_targets, get_bdb_targets, get_activity_cliff_targets, preprocess_targets, BaseMetaDataset

# absolute_path = os.path.abspath(__file__)
# DATA_PATH = "/".join(absolute_path.split("/")[:-2]+["data"])

class MetaDataset(BaseMetaDataset):
    def __init__(self, args, exp_string):
        super(MetaDataset, self).__init__(args, exp_string)

    def load_dataset(self):
        datasource = self.args.datasource

        if datasource == "chembl":
            experiment_train = get_chembl_targets()
        elif datasource == "bdb":
            experiment_train = get_bdb_targets()
        # elif datasource == "chembl_bdb":
        #     experiment_chembl = get_chembl_targets()
        #     experiment_bdb = get_bdb_targets()
        else:
            print("dataset not exist")
            exit()

        self.target_ids = experiment_train["targets"]
        ligand_set = experiment_train["ligand_sets"]
        print(f'len(ligand_set) of {datasource}:', len(ligand_set))

        if datasource == "chembl":
            # save_path = f'D:/YQ/code/Act_GeminiMol/data/chembl/chembl_targets_split.json'
            self.split_name_train_val_test = {}
            valid_test_path = f'D:/YQ/code/Act_GeminiMol/data/chembl/chembl_valid_test_split.json'
            valid_test_data = json.load(open(valid_test_path, "r"))
            self.split_name_train_val_test.update(valid_test_data)
            train_path = f'D:/YQ/code/Act_GeminiMol/data/chembl/chembl_train_split.json'
            train_data = json.load(open(train_path, "r"))
            self.split_name_train_val_test.update(train_data)
            print(f"number of {datasource} training set:", len(self.split_name_train_val_test['train']))
        elif datasource == "bdb":
            # save_path = f'D:/YQ/code/Act_GeminiMol/data/bdb/bdb_targets_split.json'
            self.split_name_train_val_test = {}
            valid_test_path = f'D:/YQ/code/Act_GeminiMol/data/bdb/bdb_valid_test_split.json'
            valid_test_data = json.load(open(valid_test_path, "r"))
            self.split_name_train_val_test.update(valid_test_data)
            train_path = f'D:/YQ/code/Act_GeminiMol/data/bdb/bdb_train_split.json'
            train_data = json.load(open(train_path, "r"))
            self.split_name_train_val_test.update(train_data)
            print(f"number of {datasource} training set:", len(self.split_name_train_val_test['train']))

        self.Xs = []
        self.smiles_all = []
        self.ys = []
        self.targets = []
        self.train_indices = []
        self.val_indices = []
        self.test_indices = []

        target_list = []
        # test set
        if self.args.expert_test != "":
            if self.args.expert_test == "act_cliff":
                experiment_test = get_activity_cliff_targets()
            else:
                raise ValueError(f"no expert_test {self.args.expert_test}")
            ligand_set = {**ligand_set, **experiment_test['ligand_sets']}
            # print(f'len(ligand_set)', len(ligand_set))
            self.split_name_train_val_test['test'] = experiment_test['targets']
        target_list += self.split_name_train_val_test['test']
        # valid set
        target_list += self.split_name_train_val_test['valid']
        # train set
        if self.args.train == 1:
            target_list += self.split_name_train_val_test['train']
        # print('target_list', len(target_list))

        data_cnt = 0
        with Pool(8) as p:
            res_all = p.map(preprocess_targets, tqdm.tqdm([ligand_set.get(x, None) for x in target_list]))
            print(f"len(res_all):{len(res_all)}")
            for res, target_id in zip(res_all, target_list):
                if res is None:
                    continue
                x_tmp, y_tmp, smiles_list = res

                self.Xs.append(x_tmp)
                self.ys.append(y_tmp)
                self.smiles_all.append(smiles_list)
                self.targets.append(target_id)
                if target_id in self.split_name_train_val_test['train']:
                    self.train_indices.append(data_cnt)
                    data_cnt += 1
                    # # chembl--512;
                    # repeat_cnt = len(smiles_list) // 512
                    # for i in range(repeat_cnt):
                    #     self.Xs.append(x_tmp)
                    #     self.ys.append(y_tmp)
                    #     self.smiles_all.append(smiles_list)
                    #     self.targets.append(f"{target_id}_{i}")
                    #     self.train_indices.append(data_cnt)
                    #     data_cnt += 1
                elif target_id in self.split_name_train_val_test['valid']:
                    self.val_indices.append(data_cnt)
                    data_cnt += 1
                elif target_id in self.split_name_train_val_test['test']:
                    self.test_indices.append(data_cnt)
                    data_cnt += 1
                else:
                    print(target_id)
                    data_cnt += 1

        train_cnt = len(self.train_indices)
        val_cnt = len(self.val_indices)
        test_cnt = len(self.test_indices)

        self.data_length = {}
        self.data_length['train'] = train_cnt
        self.data_length['val'] = val_cnt
        self.data_length['test'] = test_cnt
        self.data_length['train_weight'] = train_cnt

        print(f'{datasource}_train/valid/test:',train_cnt, val_cnt, test_cnt)
        print(f'{datasource}:',np.max([len(x) for x in self.Xs]), np.mean([len(x) for x in self.Xs]))

if __name__ == "__main__":

    # experiment_train = get_chembl_targets()
    experiment_train = get_bdb_targets()
    target_ids = experiment_train["targets"]
    ligand_set = experiment_train["ligand_sets"]
    '''
    bdb:(no assay_id)len(ligand_set) 3132
    chembl(assay_id):len(ligand_set) 10511
    bdb(assay_id):len(ligand_set) 15155
    chembl_split_64:len(ligand_set) 15978
    bdb_split_64:len(ligand_set) 20436
    '''
    print('len(ligand_set)', len(ligand_set))

    # save_path = f'D:/YQ/code/Act_GeminiMol/data/chembl/chembl_targets_split.json'
    # save_path = f'D:/YQ/code/Act_GeminiMol/data/bdb/bdb_targets_split.json'
    # save_path = f'D:/YQ/code/Act_GeminiMol/data/chembl/chembl_target_assay_split.json'
    # save_path = f'D:/YQ/code/Act_GeminiMol/data/bdb/bdb_target_assay_split.json'
    split_name_train_val_test = {}
    # valid_test_path = f'D:/YQ/code/Act_GeminiMol/data/chembl/chembl_valid_test_split.json'
    valid_test_path = f'D:/YQ/code/Act_GeminiMol/data/bdb/bdb_valid_test_split.json'
    valid_test_data = json.load(open(valid_test_path, "r"))
    split_name_train_val_test.update(valid_test_data)
    # train_path = f'D:/YQ/code/Act_GeminiMol/data/chembl/chembl_train_split.json'
    train_path = f'D:/YQ/code/Act_GeminiMol/data/bdb/bdb_train_split.json'
    train_data = json.load(open(train_path, "r"))
    split_name_train_val_test.update(train_data)
    '''
    bdb(no assay_id):number of training set: 2962
    chembl(assay_id):number of training set: 10111
    bdb(assay_id):number of training set: 14555
    chembl_split_64:number of training set: 15578
    bdb_split_64:number of training set: 19836
    '''
    print("number of training set:", len(split_name_train_val_test['train']))

    Xs = []
    smiles_all = []
    ys = []
    targets = []
    train_indices = []
    val_indices = []
    test_indices = []

    target_list = []
    # test set
    expert_test = "act_cliff"
    expert_test = ""
    if expert_test != "":
        if expert_test == "act_cliff":
            experiment_test = get_activity_cliff_targets()
        else:
            raise ValueError(f"no expert_test {expert_test}")
        ligand_set = {**ligand_set, **experiment_test['ligand_sets']}
        print(f'len(ligand_set)', len(ligand_set))
        split_name_train_val_test['test'] = experiment_test['targets']
    target_list += split_name_train_val_test['test']
    # valid set
    target_list += split_name_train_val_test['valid']
    # train set
    # if self.args.train == 1:
    target_list += split_name_train_val_test['train']
    print('target_list', len(target_list))

    data_cnt = 0
    with Pool(4) as p:
        res_all = p.map(preprocess_targets,
                        tqdm.tqdm([ligand_set.get(x, None) for x in target_list]))
        print(f"len(res_all):{len(res_all)}")

        for res, target_id in zip(res_all, target_list):
            if res is None:
                continue
            x_tmp, y_tmp, smiles_list = res
            # print(f"x_tmp:{x_tmp.shape}")            # (n_ligands, 2048)
            # print(f"y_tmp:{y_tmp.shape}")            # (n_ligands,)
            # print(f"smiles_list:{len(smiles_list)}") # n_ligands

            Xs.append(x_tmp)
            ys.append(y_tmp)
            smiles_all.append(smiles_list)
            targets.append(target_id)

            if target_id in split_name_train_val_test['train']:
                train_indices.append(data_cnt)
                data_cnt += 1
                # repeat_cnt = len(smiles_list) // 512
                # for i in range(repeat_cnt):
                #     Xs.append(x_tmp)
                #     ys.append(y_tmp)
                #     smiles_all.append(smiles_list)
                #     targets.append(f"{target_id}_{i}")
                #     train_indices.append(data_cnt)
                #     data_cnt += 1
            elif target_id in split_name_train_val_test['valid']:
                val_indices.append(data_cnt)
                data_cnt += 1
            elif target_id in split_name_train_val_test['test']:
                test_indices.append(data_cnt)
                data_cnt += 1
            else:
                print(target_id)
                data_cnt += 1

    train_cnt = len(train_indices)
    val_cnt = len(val_indices)
    test_cnt = len(test_indices)

    data_length = {}
    data_length['train'] = train_cnt
    data_length['val'] = val_cnt
    data_length['test'] = test_cnt
    data_length['train_weight'] = train_cnt

    '''    
    bdb(no assay_id):2962 70 100
    chembl(assay_id):10111 200 200
    bdb(assay_id):14555 300 300
    chembl_split_64:15578 200 200
    bdb_split_64:19836 300 300
    '''
    print(train_cnt, val_cnt, test_cnt)
    print(len(Xs), len(ys), len(smiles_all), len(targets))
    ''' 
    bdb:9544 306.9406130268199
    chembl(assay_id):24165 69.31091237750928
    bdb(assay_id):2217 61.503134279115805
    chembl_split_64:11884 45.595631493303294
    bdb_split_64:1101 45.60970835779996
    '''
    print(np.max([len(x) for x in Xs]), np.mean([len(x) for x in Xs]))
