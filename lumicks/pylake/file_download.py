import os
import json
import hashlib
import urllib.error
from urllib.parse import quote, urljoin, urlparse
from urllib.request import urlopen

from tqdm.auto import tqdm

__all__ = ["download_from_doi"]


def strip_control_characters(url_str) -> str:
    """Strips control characters from a URL

    Parameters
    ----------
    url_str : str
        URL to encode

    Raises
    ------
    ValueError
        if the URL does not contain a scheme (e.g. https) or net location
    """
    url = urlparse(url_str)

    if not url.scheme or not url.netloc:
        raise ValueError(f"Invalid URL provided: {url}")

    base_url = url.scheme + "://" + url.netloc + quote(url.path)
    return base_url + "?" + quote(url.query) if url.query else base_url


def get_url_from_doi(doi):
    """Obtains a Zenodo record from the DOI (e.g. 10.5281/zenodo.#)"""
    url = doi if doi.startswith("http") else urljoin("https://doi.org/", doi)

    url = strip_control_characters(url)

    try:
        with urlopen(url) as response:
            return response.url
    except urllib.error.HTTPError as exception:
        raise RuntimeError(f"DOI could not be resolved ({exception})") from exception


def download_record_metadata(record_number):
    """Download specific Zenodo record metadata"""
    zenodo_url = "https://zenodo.org/api/records/"

    with urlopen(strip_control_characters(urljoin(zenodo_url, str(record_number)))) as response:
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
    url = strip_control_characters(url)

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


def download_from_doi(
    doi, target_path="", force_download=False, show_progress=True, allow_overwrite=False
):
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
    allow_overwrite : bool
        Re-download files for which the hash does not match the expected hash from Zenodo. Note that
        this will overwrite the existing file with a freshly downloaded copy.

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

        # If the file doesn't exist or we are forcing to download all, we can't skip it
        download = not os.path.exists(full_path) or force_download

        # Handle the case where a file with the requested filename exists, we are not forcing
        # all files to download, but the file we have does not match the checksum from Zenodo.
        hash_algorithm, checksum = file["checksum"].split(":")
        if not download and not verify_hash(full_path, hash_algorithm, checksum):
            if allow_overwrite:
                download = True
            else:
                raise RuntimeError(
                    f"File {file_name} does not match file from Zenodo. Set allow_overwrite=True "
                    f"if you wish to overwrite the existing file on disk with the version from "
                    f"Zenodo."
                )

        if download:
            download_file(url, target_path, file_name, show_progress)
            if not verify_hash(full_path, hash_algorithm, checksum):
                raise RuntimeError("Download failed. Invalid checksum after download.")
        else:
            print(f"Already downloaded {file_name}.")

        file_names.append(full_path)

    return file_names
