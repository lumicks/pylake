import os
import json
import hashlib
import urllib.error
from urllib.parse import urljoin
from urllib.request import urlopen

from tqdm.auto import tqdm

__all__ = ["download_from_doi"]


def get_url_from_doi(doi):
    """Obtains a Zenodo record from the DOI (e.g. 10.5281/zenodo.#)"""
    url = doi if doi.startswith("http") else urljoin("https://doi.org/", doi)
    try:
        with urlopen(url) as response:
            return response.url
    except urllib.error.HTTPError as exception:
        raise RuntimeError(f"DOI could not be resolved ({exception})") from exception


def download_record_metadata(record_number):
    """Download specific Zenodo record metadata"""
    zenodo_url = "https://zenodo.org/api/records/"
    with urlopen(urljoin(zenodo_url, str(record_number))) as response:
        if response.status == 200:
            return json.loads(response.read())
        else:
            raise ValueError(f"Failed to access Zenodo record {response}")


def download_file(url, target_path, download_path, show_progress=True, block_size=8192):
    """Stream a file from a URL while showing progress

    Parameters
    ----------
    url : str
        URL to pull from
    target_path : str
        Target path to download to
    download_path : str
        Path and filename to download
    show_progress : bool
        Show progress while downloading or not?
    block_size : int
        Block size to use when downloading
    """
    with urlopen(url) as response:
        size_bytes = int(response.headers.get("Content-Length", 0))

        full_path = os.path.join(target_path, download_path)  # Append download_path to target_path
        dir_name = os.path.dirname(full_path)  # Extract directory name
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with open(full_path, "wb") as file:
            with tqdm(
                total=size_bytes,
                unit="iB",
                unit_scale=True,
                desc=f"Download {download_path}",
                disable=not show_progress,
            ) as progress_bar:
                for data in iter(lambda: response.read(block_size), b""):
                    progress_bar.update(len(data))
                    file.write(data)


def verify_hash(file_name, algorithm, reference_hash, chunk_size=65536):
    """Verify the hash of a file"""
    m = hashlib.new(algorithm)
    with open(file_name, "rb") as f:
        b = f.read(chunk_size)
        while len(b) > 0:
            m.update(b)
            b = f.read(chunk_size)

    return m.hexdigest() == reference_hash


def download_from_doi(doi, target_path="", force_download=False, show_progress=True):
    """Download files from a Zenodo DOI (i.e. 10.5281/zenodo.#######)

    Note
    ----
    This function will not re-download files that have already been downloaded. You can therefore
    safely use it at the start of a notebook or script without worrying that it will download
    files unnecessarily.

    Parameters
    ----------
    doi : str
        DOI of the files to download.
    target_path : str
        Target path to download to. Default downloads to current folder.
    force_download : bool
        Force re-downloading the file even if it already exists and the hash checks out.
        When the hash of an existing file does not match, this will overwrite the existing file with
        a freshly downloaded copy.
    show_progress : bool
        Show a progress bar while downloading.

    Returns
    -------
    list of str
        List of downloaded file names.
    """
    url = get_url_from_doi(doi)

    if "zenodo" not in url:
        raise RuntimeError("Only Zenodo DOIs are supported.")

    record_number = url.split("/")[-1].strip()
    record_metadata = download_record_metadata(record_number)

    if show_progress:
        print(f"Fetching from record: '{record_metadata['metadata']['title']}'")

    file_names = []
    for file in record_metadata["files"]:
        file_name, url = file["key"], file["links"]["self"]
        full_path = os.path.join(target_path, file_name)

        # If the file doesn't exist, we can't skip it
        download = not os.path.exists(full_path)

        # If a file with the requested filename exists but does not match the data from Zenodo,
        # throw an error.
        hash_algorithm, checksum = file["checksum"].split(":")
        if not download and not verify_hash(full_path, hash_algorithm, checksum):
            if not force_download:
                raise RuntimeError(
                    f"File {file_name} does not match file from Zenodo. Set force_download=True "
                    f"if you wish to overwrite the existing file on disk with the version from "
                    f"Zenodo."
                )

        # Only download what we don't have yet.
        if download or force_download:
            download_file(url, target_path, file_name, show_progress)
            if not verify_hash(full_path, hash_algorithm, checksum):
                raise RuntimeError("Download failed. Invalid checksum after download.")
        else:
            print(f"Already downloaded {file_name}.")

        file_names.append(full_path)

    return file_names
