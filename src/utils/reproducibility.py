import os
import time
import uuid
import json
import hashlib

def create_run_dir() -> tuple[str, str, str]:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = str(uuid.uuid4())[:8]
    dir_name = f"{timestamp}_{run_id}"
    path = os.path.join("runs", dir_name)
    
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "logs"), exist_ok=True)
    os.makedirs(os.path.join(path, "inputs"), exist_ok=True)
    os.makedirs(os.path.join(path, "results"), exist_ok=True)
    os.makedirs(os.path.join(path, "plots"), exist_ok=True)
    os.makedirs(os.path.join(path, "reports"), exist_ok=True)
    
    return path, run_id, timestamp

def save_manifest(run_dir: str, run_id: str, timestamp: str, command: str, args, seed: int = None):
    manifest = {
        "run_id": run_id,
        "timestamp": timestamp,
        "command": command,
        "seed": seed,
        "args": vars(args),
        "versions": {
            "numpy": __import__('numpy').__version__,
            "scipy": __import__('scipy').__version__,
            "pandas": __import__('pandas').__version__
        }
    }
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=4)
