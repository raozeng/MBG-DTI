import os
import sys
import subprocess
import time
import paramiko

def load_env(env_path='.env'):
    config = {}
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
    return config

def run_command_live(ssh, command):
    print(f"Running: {command}")
    stdin, stdout, stderr = ssh.exec_command(command, get_pty=True)
    
    while True:
        line = stdout.readline()
        if not line:
            break
        print(line, end="")
    
    exit_status = stdout.channel.recv_exit_status()
    if exit_status != 0:
        print(f"Command failed with exit status {exit_status}")
    return exit_status

def run_experiment(model='mamba_bilstm', dataset='Davis'):
    config = load_env()
    
    host = config.get('host_url')
    try:
        port = int(config.get('host_port', 22))
    except ValueError:
        port = 22
        
    user = config.get('host_user')
    password = config.get('host_pwd')
    
    if not all([host, user, password]):
        print("Missing configuration in .env")
        return

    print(f"Connecting to {user}@{host}:{port}...")
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(host, port=port, username=user, password=password)
        print("Connected successfully.")
        
        # Upload run_persistent.sh
        sftp = ssh.open_sftp()
        local_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_persistent.sh')
        remote_script = 'MBG-DTI/run_persistent.sh'
        
        if os.path.exists(local_script):
            print(f"Uploading {local_script} to {remote_script}...")
            # Ensure LF line endings before upload
            with open(local_script, 'rb') as f:
                content = f.read().replace(b'\r\n', b'\n')
            with open(local_script, 'wb') as f:
                f.write(content)
                
            sftp.put(local_script, remote_script)
            
            # Upload download_models.py
            local_download_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'download_models.py')
            remote_download_script = 'MBG-DTI/download_models.py'
            if os.path.exists(local_download_script):
                print(f"Uploading {local_download_script} to {remote_download_script}...")
                sftp.put(local_download_script, remote_download_script)
            
            sftp.close()
            
            # Run download_models.py first (if present)
            if os.path.exists(local_download_script):
                 print("Ensuring models are downloaded on remote...")
                 download_cmd = "cd MBG-DTI && python download_models.py"
                 run_command_live(ssh, download_cmd)
            
            # Run the script
            cmd = f"cd MBG-DTI && chmod +x run_persistent.sh && ./run_persistent.sh {model} {dataset}"
            run_command_live(ssh, cmd)
        else:
             print("Local run_persistent.sh not found.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ssh.close()

if __name__ == "__main__":
    run_experiment()
