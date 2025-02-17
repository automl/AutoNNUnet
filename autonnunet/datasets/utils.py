"""Dataset utilities for downloading and extracting files."""
from __future__ import annotations

import tarfile
from pathlib import Path


def download_file(url: str, target_path: Path) -> None:
    """Downloads a file from a URL to a target path.

    Parameters
    ----------
    url : str
        The URL of the file to download.

    target_path : Path
        The target path where the file will be saved.

    Raises:
    ------
    Exception
        If the download fails.
    """
    if Path(target_path).exists():
        print(f"File already exists at {target_path}.")
        return

    # We need to import requests
    import requests
    from tqdm import tqdm

    print(f"Downloading file from {url} to {target_path}...")

    response = requests.get(url, stream=True)       # noqa: S113

    if response.status_code == 200:     # noqa: PLR2004
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
    """Extracts a tar file to a target path.

    Parameters
    ----------
    tar_path : Path
        The path to the tar file.

    target_path : Path
        The target path where the tar file will be extracted.

    Raises:
    ------
    Exception
        If the extraction fails.
    """
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=target_path)    # noqa: S202

    print(f"File successfully unpacked to {target_path}.")
    Path(tar_path).unlink()