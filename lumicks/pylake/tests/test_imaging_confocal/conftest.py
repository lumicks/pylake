import pytest


@pytest.fixture
def grab_tiff_tags():
    def grab_tags(file):
        from ast import literal_eval

        import tifffile

        tiff_tags = []
        with tifffile.TiffFile(file) as tif:
            for page in tif.pages:
                page_tags = {}
                for tag in page.tags.values():
                    name, value = tag.name, tag.value
                    try:
                        page_tags[name] = literal_eval(value)
                    except (ValueError, SyntaxError):
                        page_tags[name] = value
                tiff_tags.append(page_tags)
        return tiff_tags

    return grab_tags
