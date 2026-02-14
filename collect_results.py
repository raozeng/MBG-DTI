import os
import csv
import pandas as pd

# Define datasets and models to look for
datasets = ['Davis', 'KIBA', 'GPCRs', 'DrugBank', 'ion_channel', 'Enzyme']
models = ['mamba_bilstm', 'deepdta', 'transformercpi', 'mcanet', 'mambabilstmseqonly', 'transformerbilstm']
model_display_names = {
    'mamba_bilstm': 'Mamba-BiLSTM (Ours)',
    'deepdta': 'DeepDTA',
    'transformercpi': 'TransformerCPI',
    'mcanet': 'MCANet',
    'mambabilstmseqonly': 'Mamba-BiLSTM (SeqOnly)',
    'transformerbilstm': 'Transformer-BiLSTM'
}

base_dir = os.getcwd()

def get_average_metrics(file_path):
    if not os.path.exists(file_path):
        return None
    
    try:
        # Read specifically looking for the 'Average' row we wrote
        # Format: Fold,ACC,Sn,Sp,Pre,F1,MCC,AUC
        # Last row should be Average
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if not lines:
                return None
            
            # Find the line starting with "Average"
            avg_line = None
            for line in reversed(lines):
                if line.startswith("Average"):
                    avg_line = line
                    break
            
            if avg_line:
                parts = avg_line.strip().split(',')
                # parts[0] is "Average"
                # parts[1] is ACC, etc.
                if len(parts) >= 8:
                    return {
                        'ACC': float(parts[1]),
                        'Sn': float(parts[2]),
                        'Sp': float(parts[3]),
                        'Pre': float(parts[4]),
                        'F1': float(parts[5]),
                        'MCC': float(parts[6]),
                        'AUC': float(parts[7])
                    }
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return None

def main():
    print("Collecting Results and Generating Tables...\n")
    
    # Store all results first to process best values
    all_data = {} # {dataset: [row_dict, ...]}
    
    for data_name in datasets:
        all_data[data_name] = []
        for model_name in models:
            log_path = os.path.join(base_dir, data_name, model_name, 'train_result', 'train_log.csv')
            metrics = get_average_metrics(log_path)
            
            if metrics:
                row = {
                    'Model': model_display_names.get(model_name, model_name),
                    'ACC': metrics['ACC'],
                    'Sn': metrics['Sn'],
                    'Sp': metrics['Sp'],
                    'Pre': metrics['Pre'],
                    'F1': metrics['F1'],
                    'MCC': metrics['MCC'],
                    'ROC': metrics['AUC']
                }
                all_data[data_name].append(row)

    # --- Generate LaTeX Table ---
    print("="*20 + " LaTeX Code (Copy to your paper) " + "="*20)
    print(r"\begin{table*}[ht]")
    print(r"\centering")
    print(r"\caption{Performance comparison on benchmark datasets.}")
    print(r"\label{tab:benchmark}")
    print(r"\begin{tabular}{llccccccc}")
    print(r"\toprule")
    print(r"Benchmark & Model & ACC & Sn & Sp & Pre & F1 & MCC & ROC \\")
    print(r"\midrule")

    for data_name in datasets:
        rows = all_data[data_name]
        if not rows:
            continue
            
        # Find max values for marking bold
        best_vals = {}
        for metric in ['ACC', 'Sn', 'Sp', 'Pre', 'F1', 'MCC', 'ROC']:
            best_vals[metric] = max([r[metric] for r in rows]) if rows else 0

        first_row = True
        for r in rows:
            # Prepare columns
            cols = []
            
            # 1. Benchmark Name (Multirow logic)
            if first_row:
                # Escape underscore for latex
                safe_data_name = data_name.replace('_', r'\_') 
                cols.append(f"\\multirow{{{len(rows)}}}{{*}}{{{safe_data_name}}}")
            else:
                cols.append("")
            
            # 2. Model Name
            cols.append(r['Model'])
            
            # 3. Metrics
            for metric in ['ACC', 'Sn', 'Sp', 'Pre', 'F1', 'MCC', 'ROC']:
                val = r[metric]
                val_str = f"{val:.4f}"
                if val >= best_vals[metric] and len(rows) > 1:
                    cols.append(f"\\textbf{{{val_str}}}")
                else:
                    cols.append(val_str)
            
            # Print Line
            print(" & ".join(cols) + r" \\")
            first_row = False
        
        # Add separator between datasets (except last one)
        if data_name != datasets[-1]:
             print(r"\midrule")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table*}")
    print("="*60 + "\n")

    # --- Clean Text Output (Optional, for quick view) ---
    print("--- Text Summary ---")
    for data_name, rows in all_data.items():
        if rows:
            print(f"Dataset: {data_name}")
            for r in rows:
                print(f"  {r['Model']}: ACC={r['ACC']:.4f}, ROC={r['ROC']:.4f}")

if __name__ == "__main__":
    main()
