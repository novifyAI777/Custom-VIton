"""
Auto-generated template file
IO helpers
"""

import os, json

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_json(obj, path):
    with open(path,'w') as f:
        json.dump(obj,f,indent=2)
