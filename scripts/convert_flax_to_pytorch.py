"""Download TimesFM Flax checkpoint from GCS and convert to PyTorch safetensors."""
import os
import sys
import json
import subprocess
import numpy as np

MODEL_DIR = os.environ.get("TIMESFM_MODEL_DIR", "/home/user/timesfm/poc/model_cache/pytorch")
FLAX_DIR = os.environ.get("TIMESFM_FLAX_DIR", "/home/user/timesfm/poc/model_cache/flax")
GCS_BASE = "https://storage.googleapis.com/vertex-model-garden-public-us/timesfm/timesfm-2.5-200m-flax"

SAFETENSORS_PATH = os.path.join(MODEL_DIR, "model.safetensors")


def main():
    if os.path.exists(SAFETENSORS_PATH):
        size_mb = os.path.getsize(SAFETENSORS_PATH) / 1024 / 1024
        if size_mb > 800:
            print(f"Model already exists at {SAFETENSORS_PATH} ({size_mb:.0f}MB). Skipping.")
            return 0
        print(f"Model file exists but too small ({size_mb:.0f}MB). Re-downloading.")

    print("Step 1/3: Downloading Flax checkpoint from GCS...")
    if not download_checkpoint():
        return 1

    print("Step 2/3: Loading and converting weights...")
    if not convert_weights():
        return 1

    print("Step 3/3: Verifying model loads...")
    if not verify_model():
        return 1

    print(f"\nDone! Model saved to {SAFETENSORS_PATH}")
    return 0


def download_checkpoint():
    """Download all checkpoint files from GCS using curl."""
    os.makedirs(FLAX_DIR, exist_ok=True)

    # Metadata files
    metadata_files = [
        "_CHECKPOINT_METADATA", "_METADATA", "_sharding",
        "manifest.ocdbt", "descriptor/descriptor.pbtxt",
        "array_metadatas/process_0", "d/99d9838f4ea666b0baf271caec0acb55",
        "ocdbt.process_0/manifest.ocdbt",
        "ocdbt.process_0/d/82e3580474fe958b6aca1f086b1801bb",
        "ocdbt.process_0/d/f0a794078cf86d67e5026770a0cb3aaa",
        "README.md",
    ]
    for f in metadata_files:
        outpath = os.path.join(FLAX_DIR, f)
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        _curl_download(f"{GCS_BASE}/{f}", outpath)

    # Large weight shards
    weight_files = [
        ("ocdbt.process_0/d/391be1dabf9d22a13dbd77a36b10b698", 122),
        ("ocdbt.process_0/d/f933434baa602db9432904cb5cfd5012", 695),
    ]
    for f, expected_mb in weight_files:
        outpath = os.path.join(FLAX_DIR, f)
        if os.path.exists(outpath) and os.path.getsize(outpath) > expected_mb * 1024 * 1024 * 0.9:
            print(f"  {f}: already downloaded")
            continue
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        print(f"  Downloading {f} ({expected_mb}MB)...")
        _curl_download(f"{GCS_BASE}/{f}", outpath)

    return True


def _curl_download(url, outpath):
    """Download a file using curl."""
    result = subprocess.run(
        ["curl", "-s", "-o", outpath, url],
        capture_output=True, text=True, timeout=600
    )
    return result.returncode == 0


def convert_weights():
    """Convert Flax OCDBT checkpoint to PyTorch safetensors."""
    import tensorstore as ts
    import torch
    from safetensors.torch import save_file

    ocdbt_path = os.path.join(FLAX_DIR, "ocdbt.process_0")
    metadata_path = os.path.join(FLAX_DIR, "_METADATA")

    with open(metadata_path) as f:
        meta = json.load(f)

    # Load all Flax parameters
    flax_params = {}
    for key_str, info in meta['tree_metadata'].items():
        key_tuple = eval(key_str)
        param_name = '.'.join(key_tuple[:-1])
        zarr_key = '.'.join(key_tuple)

        spec = {
            "driver": "zarr",
            "kvstore": {
                "driver": "ocdbt",
                "base": f"file://{ocdbt_path}",
                "path": zarr_key + "/",
            },
        }
        store = ts.open(spec).result()
        flax_params[param_name] = np.array(store.read().result())

    print(f"  Loaded {len(flax_params)} Flax parameters")

    # Convert to PyTorch state_dict
    new_state = {}

    # Tokenizer
    for layer in ['hidden_layer', 'output_layer', 'residual_layer']:
        new_state[f"tokenizer.{layer}.weight"] = torch.from_numpy(
            flax_params[f"tokenizer.{layer}.kernel"].T.copy())
        bias_key = f"tokenizer.{layer}.bias"
        if bias_key in flax_params:
            new_state[bias_key] = torch.from_numpy(flax_params[bias_key].copy())

    # Output projections
    for proj in ['output_projection_point', 'output_projection_quantiles']:
        for layer in ['hidden_layer', 'output_layer', 'residual_layer']:
            new_state[f"{proj}.{layer}.weight"] = torch.from_numpy(
                flax_params[f"{proj}.{layer}.kernel"].T.copy())

    # Transformer layers
    for i in range(20):
        for ln in ['pre_attn_ln', 'post_attn_ln', 'pre_ff_ln', 'post_ff_ln']:
            new_state[f"stacked_xf.{i}.{ln}.scale"] = torch.from_numpy(
                flax_params[f"stacked_xf.{ln}.scale"][i].copy())

        # Fuse Q, K, V into qkv_proj
        q = flax_params["stacked_xf.attn.query.kernel"][i].reshape(1280, -1)
        k = flax_params["stacked_xf.attn.key.kernel"][i].reshape(1280, -1)
        v = flax_params["stacked_xf.attn.value.kernel"][i].reshape(1280, -1)
        qkv = np.concatenate([q, k, v], axis=1)
        new_state[f"stacked_xf.{i}.attn.qkv_proj.weight"] = torch.from_numpy(qkv.T.copy())

        out = flax_params["stacked_xf.attn.out.kernel"][i].reshape(-1, 1280)
        new_state[f"stacked_xf.{i}.attn.out.weight"] = torch.from_numpy(out.T.copy())

        new_state[f"stacked_xf.{i}.attn.query_ln.scale"] = torch.from_numpy(
            flax_params["stacked_xf.attn.query_ln.scale"][i].copy())
        new_state[f"stacked_xf.{i}.attn.key_ln.scale"] = torch.from_numpy(
            flax_params["stacked_xf.attn.key_ln.scale"][i].copy())
        new_state[f"stacked_xf.{i}.attn.per_dim_scale.per_dim_scale"] = torch.from_numpy(
            flax_params["stacked_xf.attn.per_dim_scale.per_dim_scale"][i].copy())

        for ff in ['ff0', 'ff1']:
            new_state[f"stacked_xf.{i}.{ff}.weight"] = torch.from_numpy(
                flax_params[f"stacked_xf.{ff}.kernel"][i].T.copy())

    print(f"  Converted {len(new_state)} PyTorch parameters")

    os.makedirs(MODEL_DIR, exist_ok=True)
    save_file(new_state, SAFETENSORS_PATH)
    size_mb = os.path.getsize(SAFETENSORS_PATH) / 1024 / 1024
    print(f"  Saved: {SAFETENSORS_PATH} ({size_mb:.0f}MB)")
    return True


def verify_model():
    """Verify the converted model loads and produces output."""
    import timesfm
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        MODEL_DIR, torch_compile=False, local_files_only=True
    )
    model.compile(timesfm.ForecastConfig(
        max_context=512, max_horizon=128,
        normalize_inputs=True, use_continuous_quantile_head=True,
        fix_quantile_crossing=True,
    ))
    x = np.sin(np.linspace(0, 4 * np.pi, 128))
    point, _ = model.forecast(horizon=32, inputs=[x])
    if np.any(np.isnan(point)):
        print("  ERROR: NaN in output!")
        return False
    print(f"  Verification passed (forecast shape: {point.shape})")
    return True


if __name__ == "__main__":
    sys.exit(main())
