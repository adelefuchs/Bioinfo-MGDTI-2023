import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold
import scipy.sparse as sp
import torch as th
from prepare_colddata import Prepare
from config import Config


class Dataset():
    def __init__(self):
        config = Config()
        self.fold_nums = config.fold_nums
        self.dg_dg_sim_path = config.dg_dg_sim_path
        self.pt_pt_sim_path = config.pt_pt_sim_path

        #self.cold_drug_path = config.cold_drug_path
        self.cold_protein_path = config.cold_protein_path
        self.read_data()
        # self.pre_process()

    def read_data(self):
        # sequence data
        self.dg_dg_sim = pd.read_table(
            self.dg_dg_sim_path, sep=' ', header=None).values
        self.pt_pt_sim = pd.read_table(
            self.pt_pt_sim_path, sep=' ', header=None).values

        self.num_dg = self.dg_dg_sim.shape[0]
        self.num_pt = self.pt_pt_sim.shape[0]

    def get_one_indexs(self, num_row, num_col, matrix, rows, cols, vals):
        index_row, index_col = np.where(matrix == 1)
        row = list(index_row + num_row)
        col = list(index_col + num_col)
        data = [1 for i in range(len(index_row))]
        rows.extend(row)
        cols.extend(col)
        vals.extend(data)

    def get_adjM(self, dg_dg, pt_pt, dg_pt):
        num_total = self.num_dg + self.num_pt
        rows = []
        cols = []
        vals = []
        self.get_one_indexs(0, 0, dg_dg, rows, cols, vals)
        self.get_one_indexs(0, self.num_dg, dg_pt, rows, cols, vals)
        self.get_one_indexs(self.num_dg, self.num_dg, pt_pt, rows, cols, vals)
        features = []
        dg_eye = sp.eye(self.num_dg)
        pt_eye = sp.eye(self.num_pt)
        features.append(dg_eye)
        features.append(pt_eye)

        adj = sp.coo_matrix((np.array(vals), (np.array(rows), np.array(cols))), shape=(
            num_total, num_total))
        return features, adj

    def shuffle(self, data):
        data_index = [i for i in range(data.shape[0])]
        np.random.shuffle(data_index)
        data = data[data_index]
        return data

    def get_sim_topk(self, a, k=5):
        print(f"Type of 'a': {type(a)}")
        
        # Check if 'a' contains only numeric values
        print(f"Data types in 'a': {set(type(x) for x in a.flatten())}")

        # Check the shape and values of 'a' for unexpected types
        print(f"Shape of 'a': {a.shape}")
        print(f"Sample values of 'a': {a[:5, :5]}")  # Print a small sample for inspection

        for i in range(len(a)):
            a[i][i] = 0
            
        print(f"Modified 'a' with diagonal set to 0: {a[:5, :5]}")
        try:
            max_indexes = np.argpartition(a, -k)[:, -k:]
        except Exception as e:
            print(f"Error during np.argpartition: {e}")
            return None
        
        print(f"max_indexes:\n{max_indexes[:5]}")
        result = np.take_along_axis(a, max_indexes, axis=1)
        print(f"Result of np.take_along_axis:\n{result[:5]}")
        return max_indexes
    
    def add_edges(self, dg_dg, pt_pt):
        pt_pt_sim = self.pt_pt_sim
        dg_dg_sim = self.dg_dg_sim
        dg_dg_sim = dg_dg_sim[:708, :708]
        dg_dg_topk = self.get_sim_topk(dg_dg_sim)
        pt_pt_topk = self.get_sim_topk(pt_pt_sim)
        for i, sim in enumerate(dg_dg_topk):
            for j in sim:
                dg_dg[i][j] = 1
        for i, sim in enumerate(pt_pt_topk):
            for j in sim:
                pt_pt[i][j] = 1

    def count_num_one(self, a):
        index = np.where(a == 1)
        print(len(index[0]))  
    
    def get_fold_data(self, fold_num, delete_rate):
        print(f"load data fold_{fold_num}")
        delete_rate = f"delete_rate_{delete_rate}"
        #cold_path = self.cold_drug_path
        cold_path = self.cold_protein_path
        dg_dg = np.load(os.path.join(cold_path, delete_rate, f"fold_{fold_num}_dg_dg.npy"))
        pt_pt = np.load(os.path.join(cold_path, delete_rate, f"fold_{fold_num}_pt_pt.npy"))
        dg_pt = np.load(os.path.join(cold_path, delete_rate, f"fold_{fold_num}_dg_pt.npy"))
        test_index = np.load(os.path.join(cold_path, delete_rate, f"fold_{fold_num}_test_index.npy"))
        train_index = np.load(os.path.join(cold_path, delete_rate, f"fold_{fold_num}_train_index.npy"))
        train_index[:,1] += self.num_dg
        test_index[:,1] += self.num_dg
        self.add_edges(dg_dg, pt_pt)
        features, adjM = self.get_adjM(dg_dg, pt_pt, dg_pt)
        features2, _, _, _, _= self.load_final_test_data()
        features += features2
        return features, adjM, train_index, test_index


    def load_final_test_data(self):
        """
        Loads the final testing data for drug-protein interactions.
        
        """
        dg_dg = np.load("pharos4DAVIS/dg_dg.npy")
        pt_pt = np.load("pharos4DAVIS/pt_pt.npy")
        dg_pt = np.load("pharos4DAVIS/dg_pt.npy")
        
        pt_pt_sim = pd.read_table("pharos4DAVIS/Similarity_Matrix_Proteins.txt", sep=' ', header=None).values
        dg_dg_sim = pd.read_table("pharos4DAVIS/Similarity_Matrix_Drugs.txt", sep=' ', header=None).values

        num_dg = dg_dg_sim.shape[0]
        num_pt = pt_pt_sim.shape[0]
        
        
        # Step 4: Add edges based on similarity
        self.add_edges2(dg_dg, pt_pt,dg_dg_sim,pt_pt_sim)
        
        # Step 5: Get features and adjacency matrix
        features, adjM = self.get_adjM2(dg_dg, pt_pt, dg_pt, num_dg, num_pt)
        
        drug_ids, protein_ids = np.where(dg_pt == 1)
        interaction_values = dg_pt[drug_ids, protein_ids]
        # Step 6: Create test index
        test_index = np.array([[int(drug_id), int(protein_id), value] for drug_id, protein_id, value in zip(drug_ids, protein_ids, interaction_values)])
        
        test_index[:, 1] += num_dg  # Adjust protein indices
                
        return features, adjM, test_index, num_dg, num_pt
    
    def get_adjM2(self, dg_dg, pt_pt, dg_pt, num_dg, num_pt):
        num_total = num_dg + num_pt
        print(num_total, " = ", num_dg, " + ", num_pt)
        rows = []
        cols = []
        vals = []
        self.get_one_indexs(0, 0, dg_dg, rows, cols, vals)
        self.get_one_indexs(0, num_dg, dg_pt, rows, cols, vals)
        self.get_one_indexs(num_dg, num_dg, pt_pt, rows, cols, vals)
        features = []
        dg_eye = sp.eye(num_dg)
        pt_eye = sp.eye(num_pt)
        features.append(dg_eye)
        features.append(pt_eye)

        print("After actions",num_total)
        print(f"Max row index: {max(rows)}")
        print(f"Max col index: {max(cols)}")
        adj = sp.coo_matrix((np.array(vals), (np.array(rows), np.array(cols))), shape=(
            num_total, num_total))
        return features, adj
        
    def add_edges2(self, dg_dg, pt_pt, dg_dg_sim, pt_pt_sim):
        # Truncate the drug-drug similarity matrix to size 708x708
        dg_dg_sim = dg_dg_sim[:708, :708]
        
        # Get the top-k similar nodes for drug-drug and protein-protein similarity matrices
        dg_dg_topk = self.get_sim_topk(dg_dg_sim)
        pt_pt_topk = self.get_sim_topk(pt_pt_sim)
        
        # Add edges based on the similarity information
        for i, sim in enumerate(dg_dg_topk):
            for j in sim:
                dg_dg[i][j] = 1
        
        for i, sim in enumerate(pt_pt_topk):
            for j in sim:
                pt_pt[i][j] = 1


if __name__ == '__main__':
    delete_rate_list = [1.0, 0.9, 0.7, 0.5]
    cold_data_prepare = Prepare()
    for delete_rate in delete_rate_list:
        cold_data_prepare.build_cold_protein(delete_rate)
    # for delete_rate in delete_rate_list:
    #     cold_data_prepare.build_cold_drug(delete_rate)
