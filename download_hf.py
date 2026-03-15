import os
from huggingface_hub import snapshot_download

def download_model(repo_id, local_dir):
    print(f"Downloading {repo_id} to {local_dir}...")
    try:
        snapshot_download(repo_id=repo_id, local_dir=local_dir,
                          local_dir_use_symlinks=False, revision="main")
        print(f"Successfully downloaded {repo_id}")
    except Exception as e:
        print(f"Error downloading {repo_id}: {e}")

if __name__ == "__main__":
    base_dir = r"d:\Research\MBG-DTI\huggingface_models"
    models = [
        "seyonec/ChemBERTa-zinc-base-v1",
        "facebook/esm2_t6_8M_UR50D"
    ]
    
    for repo_id in models:
        safe_name = repo_id.replace("/", "_")
        local_dir = os.path.join(base_dir, safe_name)
        download_model(repo_id, local_dir)
        
    print(f"\nAll models downloaded to {base_dir}")
