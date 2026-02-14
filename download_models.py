import os
from transformers import AutoModel, AutoTokenizer

def download_model(model_name, local_path):
    if not os.path.exists(local_path):
        print(f"Downloading {model_name} to {local_path}...")
        os.makedirs(local_path, exist_ok=True)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            tokenizer.save_pretrained(local_path)
            model.save_pretrained(local_path)
            print(f"Successfully saved {model_name} to {local_path}")
        except Exception as e:
            print(f"Failed to download {model_name}: {e}")
            # Clean up empty directory if failed
            try:
                os.rmdir(local_path)
            except:
                pass
    else:
        print(f"{local_path} already exists. Skipping download.")

if __name__ == "__main__":
    # Ensure models directory exists
    if not os.path.exists('./models'):
        os.makedirs('./models')
        
    download_model('seyonec/ChemBERTa-zinc-base-v1', './models/ChemBERTa')
    download_model('facebook/esm2_t6_8M_UR50D', './models/ESM2')
