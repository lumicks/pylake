# Contributing

Great that you're willing to contribute to Pylake!

This document is intended to provide you with some instructions on how to contribute to Pylake.

The first step in contributing to Pylake is always to open a GitHub issue. By first discussing the improvements you wish to make, we aim to prevent unnecessary work up front. We can also help by making suggestions on where said functionality might best fit in.

If you feel uncomfortable contributing code, you could also consider other ways to contribute.
Think of improvements to the documentation, additional tutorials or other tasks that generally improve Pylake and benefit its users. If you have an idea for a new feature, but do not know how to start implementing it, consider opening an issue for it. The same applies to bug fixes.

## Opening an issue

We use GitHub issues to track bug reports and feature requests.
Feel free to open an issue if you run into a bug or wish to see a feature implemented.

### Submitting a bug report

If you encounter a bug, please verify that the bug still exists on the latest version of Pylake.
You can find instructions on how to update Pylake [here](https://lumicks-pylake.readthedocs.io/en/stable/install.html#updating).
Please also check the [changelog](https://raw.githubusercontent.com/lumicks/pylake/main/changelog.md) to see whether it may have already been addressed in an unreleased version of Pylake.

If the bug has not been addressed, please open an issue describing the bug. This issue should at least contain the following information:

- A short summary of the bug. In most cases, this should only be a couple of sentences.
- A code snippet to reproduce the bug. Please do your best to reduce the code snippet to the minimum required.
- The result of the code snippet.
- The expected result of the code snippet (which is different from the obtained result because of the bug).

Please also make sure you mention the version of Python, the platform and the version of Pylake. If reproducing the bug relies on data, please make sure this data is accessible to us.

### Opening a feature request

Sometimes a feature may not be present.
If the feature is something that would benefit many users, it may be worthwhile to do a feature request.
When doing a feature request consider the following points:
- Make sure the motivation for including the new feature is clear by demonstrating its usefulness, ideally with some plots.
- Cite relevant literature that describes the method and/or evaluate its performance.
- If a reference implementation exists, please include it.

## Preparing a Pull Request (PR)

Code and documentation contributions will generally proceed through a Pull Request.

- Please provide a clear PR description. If the implemented method is based on a paper, please include a citation to the paper.
- If the PR is related to an open issue, please refer to this issue in the PR description.
- Keep a clean commit history.
- If a new feature is complex and requires multiple steps, consider breaking it up into multiple Pull Requests for the different sub-problems.
- Remember to update the [changelog](https://raw.githubusercontent.com/lumicks/pylake/main/changelog.md) and [documentation](https://github.com/lumicks/pylake/tree/main/docs). Please verify that the documentation builds. The `docs` directory contains instructions on how to locally build the docs.
- New features must include unit tests which ensure their correct function. Tests for new functionality should ideally have 100% coverage. To measure coverage, you can use the package [pytest-cov](https://pypi.org/project/pytest-cov/).
- Make sure all tests pass after you make your changes.
- You may merge the Pull Request once you have a sign-off of one core developer, or if you do not have permission to do that, you may request this developer to merge it for you.
- If you need to add any dependencies to be able to develop your feature, please discuss them in the issue with us first.
- We use [semantic versioning](https://semver.org/), meaning that if your code introduces a breaking API change this will require increasing the major version number.
- When preparing your PR, please follow the stylistic guidelines below.

## Stylistic Guidelines

- Please set up your editor to follow PEP8 (remove trailing white space, no tabs, etc.). Use [black](https://github.com/psf/black) to format your code.
- All public functions and classes require an informative docstring. Document these using the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) style.
- Features should ideally be grouped by category in their own folder, with implementational details in a sub-folder named `detail`. Tests for each specific category should be placed in a sub-folder `tests`.

### Installing pre-commit hooks

To help with formatting and code-style consider installing the pre-commit hooks.
These will check the formatting whenever you commit.
Install `pre-commit` with:

    pip install pre-commit

Next, set up the hook with::

    pre-commit install

That's all there is to it.
