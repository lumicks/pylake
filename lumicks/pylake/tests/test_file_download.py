import pytest
from lumicks.pylake.file_download import (
    get_url_from_doi,
    download_record_metadata,
    verify_hash,
    download_from_doi,
)


@pytest.mark.slow
def test_grab_record():
    assert get_url_from_doi("10.5281/zenodo.4247279") == "https://zenodo.org/record/4247279"

    with pytest.raises(RuntimeError, match="DOI could not be resolved"):
        assert get_url_from_doi("10.55281/zenodo.4247279")


@pytest.mark.slow
def test_download_record_metadata():
    record = download_record_metadata("4280789")  # Older version of Pylake

    # Verify that the fields we rely on stay the same
    assert record["files"][0]["checksum"] == "md5:1a401193ab22f0983f87855e2581075b"
    assert record["files"][0]["key"] == "lumicks/pylake-v0.7.1.zip"
    assert record["files"][0]["links"]["self"].startswith("https://zenodo.org/")  # Link may change


@pytest.mark.slow
def test_non_zenodo_doi():
    with pytest.raises(RuntimeError, match="Only Zenodo DOIs are supported"):
        assert download_from_doi("https://doi.org/10.1109/5.771073")


@pytest.mark.slow
def test_download_from_doi(tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp("download_testing")
    record = download_record_metadata("4247279")
    files = download_from_doi("10.5281/zenodo.4247279", tmpdir, show_progress=False)

    # Validate checksum
    assert verify_hash(files[0], *record["files"][0]["checksum"].split(":"))

    # Add a random character such that the checksum fails
    with open(files[0], "ab") as f:
        f.write(b"\x21")

    # Validate that the hash is no longer correct
    assert not verify_hash(files[0], *record["files"][0]["checksum"].split(":"))

    with pytest.raises(
        RuntimeError,
        match="Set force_download=True if you wish to overwrite the existing file on disk with the "
        "version from Zenodo",
    ):
        download_from_doi("10.5281/zenodo.4247279", tmpdir, show_progress=False)

    download_from_doi("10.5281/zenodo.4247279", tmpdir, force_download=True, show_progress=False)

    # Validate checksum after forced re-download (should be OK again)
    assert verify_hash(files[0], *record["files"][0]["checksum"].split(":"))