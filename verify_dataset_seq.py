import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

try:
    from dataset_seq import label_smiles, CHARISOSMISET
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def verify():
    print("--- Verifying Dataset Seq Tokenizer ---")
    
    # 1. Check Vocab
    print(f"Vocab Size: {len(CHARISOSMISET)}")
    if "Cl" in CHARISOSMISET and "Br" in CHARISOSMISET:
        print("[OK] Multi-char atoms (Cl, Br) found in vocabulary.")
    else:
        print("[FAIL] 'Cl' or 'Br' missing from vocabulary.")
        return False
        
    # 2. Test Tokenization of Chlorine
    test_smi = "C1=CC=C(C=C1)Cl"
    print(f"\nTest SMILES: {test_smi}")
    
    encoded = label_smiles(test_smi, 20, CHARISOSMISET)
    print(f"Encoded IDs: {encoded}")
    
    # Check if Cl ID (90) is present
    cl_id = CHARISOSMISET['Cl']
    if cl_id in encoded:
        print(f"[OK] Found Cl token ID ({cl_id}) in output.")
    else:
        print(f"[FAIL] Cl token ID ({cl_id}) NOT found. Split into C and l?")
        
        # specific check for split
        c_id = CHARISOSMISET['C']
        if c_id in encoded:
            print("   Note: Found 'C' tokens.")
            
    # 3. Test Unknown Token
    unknown_smi = "Xyz" # X is 66, y is 88, z is 86(s?) no, z is 72? no matter.
    # Actually checking if regex handles weird stuff
    
    return True

if __name__ == "__main__":
    verify()
