def pre_mutation(context):
    skip = [
        "conftest.py",  # Don't mutate the test configuration
        "data",  # Don't mutate any of the data generation
        "tests",
        "benchmark",
    ]

    if any(s in context.filename for s in skip):
        context.skip = True
