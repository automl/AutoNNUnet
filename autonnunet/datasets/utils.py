from __future__ import annotations

import os
from pathlib import Path
import tarfile


def download_file(url: str, target_path: Path) -> None:
    if os.path.exists(target_path):
        print(f"File already exists at {target_path}.")
        return

    import requests
    from tqdm import tqdm

    print(f"Downloading file from {url} to {target_path}...")

    response = requests.get(url, stream=True)

    if response.status_code == 200:
        total_size = int(response.headers.get("content-length", 0))

        with open(target_path, "wb") as f:
            for chunk in tqdm(response.iter_content(chunk_size=8192),
                              total=total_size//8192,
                              unit="KB",
                              unit_scale=True):
                if chunk:
                    f.write(chunk)
        print("File successfully downloaded.")
    else:
        raise Exception("Failed to download file.")


def untar_file(tar_path: Path, target_path: Path) -> None:
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=target_path)

    print(f"File successfully unpacked to {target_path}.")
    os.remove(tar_path)