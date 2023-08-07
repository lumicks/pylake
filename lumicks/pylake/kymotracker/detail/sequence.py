import re


def read_genbank(filename):
    """Read a GenBank (.gb) sequence file and extract annotated features.

    The GenBank file format is a space delimited format for recording genomic sequences
    with custom annotations. A full sample record can be found at
    https://www.ncbi.nlm.nih.gov/Sitemap/samplerecord.html

    The feature table format is described at https://www.insdc.org/documents/feature_table.html#4
    The position of the data items within the feature descriptor line is as follows:
        column position    data item
        ---------------    ---------

        1-5                blank
        6-20               feature key
        21                 blank
        22-80              location

    Data on the qualifier and continuation lines (tags) begins in column position 22 (the
    first 21 columns contain blanks).

    A full example of expected landmark annotations is as follows:

         misc_feature    33786
                         /label=ATTO 647N
                         /note="landmark:red"
    ............................................
        |    |    |    |    |    |    |    |
        5    10   15   20   21   25   30   35

    where:
        'misc_feature' is the feature type
        '33786' is the feature location
        '/label=ATTO 647N' is a feature tag
        '/note="landmark:red"' is a feature tag

    Parameters
    ----------
    filename: str | os.PathLike
        Path to the file to be parsed.
    """

    with open(filename, "r") as gb_file:
        record = gb_file.read()

    # Extract the features table; located between the 'FEATURES' and 'ORIGIN' headers
    feature_table = re.search(r"FEATURES.+\n([\s\S]*)ORIGIN", record)
    if not feature_table:
        raise AttributeError("There is no feature table in this file; the format may be corrupted.")
    feature_table = feature_table.group(1)

    # Split the table into a list of repeating pattern: feature name, location, tags
    # Discard the first item in the list; from the `re` docs:
    #   If there are capturing groups in the separator and it matches at the start of the string,
    #   the result will start with an empty string.
    table_parts = re.split(r" {5}(\w+) +([\d\.]+)\n", feature_table)[1:]
    features = [table_parts[j : j + 3] for j in range(0, len(table_parts), 3)]

    # Parse tags string into individual items
    regex_tag = re.compile(r" {21}/([\w\"]+)=(.*)", re.MULTILINE)
    for feature in features:
        feature[-1] = re.findall(regex_tag, feature[-1])
    return features
