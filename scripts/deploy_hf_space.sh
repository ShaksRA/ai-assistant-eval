#!/usr/bin/env bash
# deploy_hf_space.sh — Deploy OSS assistant to HuggingFace Spaces
# Usage: HF_USERNAME=yourname HF_TOKEN=hf_xxx bash scripts/deploy_hf_space.sh

set -euo pipefail

HF_USERNAME="${HF_USERNAME:-your-hf-username}"
HF_TOKEN="${HF_TOKEN:-}"
SPACE_NAME="${SPACE_NAME:-qwen-personal-assistant}"
SPACE_URL="https://huggingface.co/spaces/${HF_USERNAME}/${SPACE_NAME}"

if [ -z "$HF_TOKEN" ]; then
  echo "ERROR: HF_TOKEN environment variable not set."
  echo "Get your token from https://huggingface.co/settings/tokens"
  exit 1
fi

echo "==> Deploying to HuggingFace Spaces: ${SPACE_URL}"

# 1. Create a temporary deployment directory
DEPLOY_DIR=$(mktemp -d)
trap "rm -rf $DEPLOY_DIR" EXIT

# 2. Copy OSS assistant files
cp -r oss_assistant/* "$DEPLOY_DIR/"

# 3. Create HF Spaces README (app.yaml)
cat > "$DEPLOY_DIR/README.md" << 'EOF'
---
title: Qwen Personal Assistant
emoji: 🤖
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: apache-2.0
---

# Qwen2.5 Personal Assistant

Multi-turn personal assistant powered by Qwen2.5-0.5B-Instruct.

Features:
- Multi-turn conversation with 10-turn memory
- Safety guardrails (input + output filtering)
- Streaming responses

## Model
- **Model**: `Qwen/Qwen2.5-0.5B-Instruct`
- **License**: Apache 2.0
- **Parameters**: 0.5B

EOF

# 4. Install huggingface_hub if needed
pip install huggingface_hub --quiet

# 5. Push to HF Spaces using Python
python3 << PYEOF
import os
from huggingface_hub import HfApi, create_repo

token = "${HF_TOKEN}"
username = "${HF_USERNAME}"
space_name = "${SPACE_NAME}"
deploy_dir = "${DEPLOY_DIR}"

api = HfApi(token=token)
repo_id = f"{username}/{space_name}"

print(f"Creating/updating Space: {repo_id}")
try:
    create_repo(
        repo_id=repo_id,
        token=token,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
        private=False,
    )
    print("Repo created/exists ✓")
except Exception as e:
    print(f"Note: {e}")

api.upload_folder(
    folder_path=deploy_dir,
    repo_id=repo_id,
    repo_type="space",
    token=token,
)
print(f"\n✅ Deployed successfully!")
print(f"🔗 Space URL: https://huggingface.co/spaces/{repo_id}")
print("⏳ Build may take 5-10 minutes for first model download")
PYEOF

echo ""
echo "==> Deployment complete!"
echo "==> Space URL: ${SPACE_URL}"
echo "==> Note: First cold start will download ~1GB model. Subsequent starts use cache."
