import json, glob, os

def load_chunked(base_name, data_dir=None):
    if data_dir is None:
        data_dir = os.path.dirname(os.path.abspath(__file__))
    pattern = os.path.join(data_dir, f"{base_name}_part*.json")
    parts = sorted(glob.glob(pattern))
    if not parts:
        single = os.path.join(data_dir, f"{base_name}.json")
        if os.path.exists(single):
            with open(single) as f:
                return json.load(f)
        return []
    data = []
    for p in parts:
        with open(p) as f:
            data.extend(json.load(f))
    return data

def load_real_historical(data_dir=None):
    return load_chunked("real_historical", data_dir)

def load_polymarket_markets(data_dir=None):
    return load_chunked("polymarket_markets", data_dir)
