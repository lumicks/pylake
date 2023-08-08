from pathlib import Path

import pytest

from lumicks.pylake.kymotracker.detail.sequence import read_genbank


def check_feature(index, feature, kind, location, label, note):
    assert feature[0] == kind, f"mismatch @ feature {index} - kind"
    assert feature[1] == location, f"mismatch @ feature {index} - location"
    assert feature[2][0] == label, f"mismatch @ feature {index} - label"
    assert feature[2][1] == note, f"mismatch @ feature {index} - note"


def test_read_genbank():
    filename = Path(__file__).parent / "./data/test_sequence.gb"
    reference_features = (
        ("source", "1..120", ("mol_type", '"genomic DNA"'), ("organism", '"unspecified"')),
        ("misc_feature", "1", ("label", "3x biotin"), ("note", '"attachment"')),
        ("misc_feature", "11", ("label", "ATTO 647N"), ("note", '"landmark:red"')),
        (
            "misc_feature",
            "13",
            ("label", "gibberish"),
            ("note", r'''"a`~!@# $%^&*()-_=+;:'"|,<.>/?\ "'''),
        ),
        ("misc_feature", "100", ("label", "ATTO 647N"), ("note", '"landmark:red"')),
        ("misc_feature", "120", ("label", "3x biotin"), ("note", '"attachment"')),
    )

    features = read_genbank(filename)
    assert len(features) == 6
    for j, (feature, reference) in enumerate(zip(features, reference_features)):
        check_feature(j, feature, *reference)


def test_read_genbank_no_features():
    filename = Path(__file__).parent / "./data/test_sequence_no_features.gb"
    features = read_genbank(filename)
    check_feature(
        0,
        features[0],
        "source",
        "1..120",
        ("mol_type", '"genomic DNA"'),
        ("organism", '"unspecified"'),
    )


def test_read_genbank_corrupted():
    filename = Path(__file__).parent / "./data/test_sequence_corrupted.gb"
    with pytest.raises(
        AttributeError, match="There is no feature table in this file; the format may be corrupted."
    ):
        read_genbank(filename)
