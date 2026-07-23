"""
upload_adapter.py — Automated Hugging Face repository upload script.

Verifies authentication, creates the target model repository if missing,
and uploads fine-tuned LoRA adapter files from `hg-face/json`.

Usage
-----
    python scripts/upload_adapter.py
    # Or override repository ID:
    python scripts/upload_adapter.py --repo-id MyUsername/my-adapter
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi
except ImportError:
    print("Error: huggingface_hub is required. Run: pip install huggingface_hub")
    sys.exit(1)


def upload_adapter(repo_id: str | None = None, local_folder: str = "hg-face/json") -> None:
    api = HfApi()

    # 1. Verify authentication
    try:
        user_info = api.whoami()
        username = user_info.get("name")
        print(f"✓ Authenticated with Hugging Face as user: '{username}'")
    except Exception as err:
        print(f"❌ Hugging Face authentication failed: {err}")
        print("Please run `hf auth login` or `huggingface-cli login` first.")
        sys.exit(1)

    # 2. Determine target repo ID
    if not repo_id:
        repo_id = f"{username}/prefilter-ai-json-0.8b"

    print(f"Target Repository: '{repo_id}'")

    # 3. Check local directory
    folder_path = Path(local_folder)
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"❌ Local folder '{local_folder}' not found.")
        sys.exit(1)

    # 4. Create repository if missing
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print(f"✓ Repository '{repo_id}' verified/created.")
    except Exception as err:
        print(f"⚠️ Could not create repo automatically: {err}")

    # 5. Upload files
    print(f"Uploading files from '{local_folder}' to 'https://huggingface.co/{repo_id}'...")
    try:
        api.upload_folder(
            folder_path=str(folder_path),
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"🎉 Success! Adapter published to https://huggingface.co/{repo_id}")
    except Exception as err:
        print(f"❌ Upload failed: {err}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload Prefilter AI fine-tuned adapter to Hugging Face Hub.")
    parser.add_argument("--repo-id", type=str, default=None, help="Hugging Face repo ID (default: <username>/prefilter-ai-json-0.8b)")
    parser.add_argument("--folder", type=str, default="hg-face/json", help="Local directory containing adapter files")
    args = parser.parse_args()

    upload_adapter(repo_id=args.repo_id, local_folder=args.folder)
