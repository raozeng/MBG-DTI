import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import re

# Atom-level Tokenizer Regex (DeepChem/DeepDTA style)
SMILES_REGEX = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
regex = re.compile(SMILES_REGEX)

# Updated Vocabulary including multi-char atoms (Br, Cl)
# Maps Token -> Integer ID
CHARISOSMISET = {
    "#": 29, "%": 30, ")": 31, "(": 32, "+": 33, "-": 34, ".": 35, "0": 36, "1": 37, "2": 38, "3": 39, "4": 40, "5": 41, "6": 42, "7": 43, "8": 44, "9": 45, "=": 46, 
    "A": 47, "B": 48, "C": 49, "D": 50, "E": 51, "F": 52, "G": 53, "H": 54, "I": 55, "K": 56, "L": 57, "M": 58, "N": 59, "O": 60, "P": 61, "R": 62, "S": 63, "T": 64, "V": 65, "X": 66, "Y": 67, "Z": 68, 
    "[": 69, "\\": 70, "]": 71, "a": 72, "b": 73, "c": 74, "d": 75, "e": 76, "f": 77, "g": 78, "h": 79, "i": 80, "l": 81, "m": 82, "n": 83, "o": 84, "r": 85, "s": 86, "u": 87, "y": 88,
    "Br": 89, "Cl": 90, "Si": 91, "Se": 92, "Na": 93, "Mg": 94, "Ca": 95, "Fe": 96, "Al": 97, "Zn": 98, "Cu": 99, "Ag": 100, "Au": 101, "Hg": 102, "Mn": 103, "As": 104, "Unknown": 105
}

# CHARPROTSET for Protein Sequences (Amino Acids)
CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

def label_smiles(line, max_len, char_dict):
    X = np.zeros(max_len, dtype=np.int64)
    # Use Regex to find all tokens
    tokens = [token for token in regex.findall(line)]
    
    for i, token in enumerate(tokens[:max_len]): 
        X[i] = char_dict.get(token, char_dict.get("Unknown", 0))
    return X

def label_sequence(line, max_len, char_dict):
    X = np.zeros(max_len, dtype=np.int64)
    for i, ch in enumerate(line[:max_len]):
        X[i] = char_dict.get(ch, 0)
    return X

class SeqDTIDataset(Dataset):
    def __init__(self, file_path, max_len_drug=100, max_len_prot=1000, 
                 drug_char_dict=CHARISOSMISET, prot_char_dict=CHARPROTSET):
        """
        Sequence-based DTI Dataset for DeepDTA, MCANet, TransformerCPI
        Uses Label Encoding (Integer IDs) instead of Tokenizers.
        """
        self.max_len_drug = max_len_drug
        self.max_len_prot = max_len_prot
        self.drug_char_dict = drug_char_dict
        self.prot_char_dict = prot_char_dict
        
        # 1. Read Raw Data
        raw_data = []
        try:
            with open(file_path, 'r') as f:
                print(f"Loading data from {file_path}")
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        try:
                            label = float(parts[-1])
                            seq = parts[-2]
                            smiles = parts[-3]
                            
                            # Simple validation
                            if len(smiles) < 1 or len(seq) < 1:
                                continue
                                
                            raw_data.append({'smiles': smiles, 'sequence': seq, 'label': label})
                        except ValueError:
                            continue
        except FileNotFoundError:
            print(f"Warning: File {file_path} not found.")
            
        print(f"Pre-processing {len(raw_data)} samples (Sequence Mode)...")
        
        # 2. Convert to Integers
        self.drug_ids_list = []
        self.prot_ids_list = []
        self.labels_list = []
        
        for d in tqdm(raw_data, desc="Encoding"):
            self.drug_ids_list.append(label_smiles(d['smiles'], self.max_len_drug, self.drug_char_dict))
            self.prot_ids_list.append(label_sequence(d['sequence'], self.max_len_prot, self.prot_char_dict))
            self.labels_list.append(d['label'])
            
        self.drug_ids = np.array(self.drug_ids_list)
        self.prot_ids = np.array(self.prot_ids_list)
        self.labels = np.array(self.labels_list, dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'drug_ids': torch.tensor(self.drug_ids[idx], dtype=torch.long),
            'prot_ids': torch.tensor(self.prot_ids[idx], dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.float32)
        }
        
    def slice_subset(self, n):
        n = min(n, len(self.labels))
        self.labels = self.labels[:n]
        self.drug_ids = self.drug_ids[:n]
        self.prot_ids = self.prot_ids[:n]


def collate_seq(batch):
    """
    Simple stacking for sequence data
    Return format matches what train.py expects (tuple inputs)
    """
    drug_ids = torch.stack([item['drug_ids'] for item in batch]) # (B, max_len_d)
    prot_ids = torch.stack([item['prot_ids'] for item in batch]) # (B, max_len_p)
    labels = torch.stack([item['label'] for item in batch])
    
    # Create simple masks (1 for valid, 0 for pad/0)
    drug_mask = (drug_ids != 0).long()
    prot_mask = (prot_ids != 0).long()
    
    # Dummy elements for the 3rd tuple element (Graph or EdgeIndex)
    # The models expecting sequence input (DeepDTA etc.) only use the first element anyway
    # But just in case, we return None or empty
    
    return {
        'drug_input': (drug_ids, drug_mask, None), 
        'prot_input': (prot_ids, prot_mask, None),
        'labels': labels
    }
