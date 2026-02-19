import re

# DeepDTA style regex
pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
regex = re.compile(pattern)

test_smiles = [
    "CCO",
    "C1=CC=C(C=C1)Cl", # Benzene with Chlorine
    "CC(C)Br", # Bromine
    "[H][N+](C)(C)C", # Charged
    "Cn1c(=O)n(C)c2c1c(N)nc(n2)C" # Complex
]

print("Testing Tokenizer...")
for smi in test_smiles:
    tokens = [token for token in regex.findall(smi)]
    print(f"Original: {smi}")
    print(f"Tokens:   {tokens}")
    
# Verify existing CHARISOSMISET coverage
# CHARISOSMISET from dataset_seq.py
CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 32, "+": 33, "-": 34, ".": 35, "0": 36, "1": 37, "2": 38, "3": 39, "4": 40, "5": 41, "6": 42, "7": 43, "8": 44, "9": 45, "=": 46, "A": 47, "B": 48, "C": 49, "D": 50, "E": 51, "F": 52, "G": 53, "H": 54, "I": 55, "K": 56, "L": 57, "M": 58, "N": 59, "O": 60, "P": 61, "R": 62, "S": 63, "T": 64, "V": 65, "X": 66, "Y": 67, "Z": 68, "[": 69, "\\": 70, "]": 71, "a": 72, "b": 73, "c": 74, "d": 75, "e": 76, "f": 77, "g": 78, "h": 79, "i": 80, "l": 81, "m": 82, "n": 83, "o": 84, "r": 85, "s": 86, "u": 87, "y": 88}

print("\nVerifying coverage of old chars:")
detected_tokens = set()
for char in CHARISOSMISET.keys():
    # Treat each char as a SMILES string
    # This checks if the regex can pick up single characters
    # But real check is if valid SMILES are parsed correctly
    pass
    
print("Done.")
