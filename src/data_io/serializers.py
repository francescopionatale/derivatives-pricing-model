import json

def save_json(data: dict, path: str):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
