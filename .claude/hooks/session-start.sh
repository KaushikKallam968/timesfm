#!/bin/bash
set -euo pipefail

# Only run on Claude Code web (remote environment)
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

echo "=== TimesFM Session Start Hook ==="

# 0. Ensure MCP servers are auto-approved in global settings
GLOBAL_SETTINGS="$HOME/.claude/settings.json"
mkdir -p "$HOME/.claude"
if [ -f "$GLOBAL_SETTINGS" ]; then
  # Use python to safely merge MCP approval into existing settings
  python3 -c "
import json, sys
with open('$GLOBAL_SETTINGS', 'r') as f:
    s = json.load(f)
changed = False
if not s.get('enableAllProjectMcpServers'):
    s['enableAllProjectMcpServers'] = True
    changed = True
servers = ['sequential-thinking', 'sqlite', 'fetch']
if s.get('enabledMcpjsonServers') != servers:
    s['enabledMcpjsonServers'] = servers
    changed = True
if changed:
    with open('$GLOBAL_SETTINGS', 'w') as f:
        json.dump(s, f, indent=4)
    print('  Global settings updated with MCP approval.')
else:
    print('  MCP approval already configured.')
"
else
  cat > "$GLOBAL_SETTINGS" << 'SETTINGS'
{
    "enableAllProjectMcpServers": true,
    "enabledMcpjsonServers": ["sequential-thinking", "sqlite", "fetch"]
}
SETTINGS
  echo "  Global settings created with MCP approval."
fi

# 1. Install Python dependencies
echo "[1/3] Installing Python dependencies..."
pip install --quiet \
  numpy pandas matplotlib torch safetensors \
  tensorstore orbax-checkpoint gcsfs \
  yfinance 2>&1 | tail -3

# Install timesfm from GitHub (has the v2.5 API)
pip install --quiet git+https://github.com/google-research/timesfm.git 2>&1 | tail -3

# Fix setuptools compatibility for multitasking (yfinance dep)
pip install --quiet setuptools --upgrade 2>&1 | tail -1
pip install --quiet multitasking 2>&1 | tail -1

echo "  Dependencies installed."

# 2. Download and convert model weights (if not cached)
echo "[2/3] Checking model weights..."
MODEL_PATH="$CLAUDE_PROJECT_DIR/poc/model_cache/pytorch/model.safetensors"

if [ -f "$MODEL_PATH" ] && [ "$(stat -f%z "$MODEL_PATH" 2>/dev/null || stat -c%s "$MODEL_PATH" 2>/dev/null)" -gt 800000000 ]; then
  echo "  Model already cached. Skipping download."
else
  echo "  Downloading and converting Flax→PyTorch (this takes ~2-3 min)..."
  python "$CLAUDE_PROJECT_DIR/scripts/convert_flax_to_pytorch.py"
fi

# 3. Set environment variables
echo "[3/3] Setting environment..."
echo "export PYTHONPATH=\"$CLAUDE_PROJECT_DIR\"" >> "$CLAUDE_ENV_FILE"
echo "export TIMESFM_MODEL_DIR=\"$CLAUDE_PROJECT_DIR/poc/model_cache/pytorch\"" >> "$CLAUDE_ENV_FILE"

echo "=== Session Start Hook Complete ==="
