# Pylake release procedure

We use a very simple branching scheme.
All the latest features are merged into the `main` branch.
We branch off `release/X.Y` as bugfix-only branches (all `.Z` patch versions go here).
We get bugfixes into the default branch by occasionally merging `release/X.Y` into `main`.

## Preflight Check

Within the Pylake repo dir:
- Check out the `release/X.Y` branch if it already exists (for `.Z > 0`), or create a new one by branching from `main` (for `.Z == 0`).
- Update the file `changelog.md`:
  - Update the date of the new release.
  - Review the changelog entries. Make sure everything is clear and informative for users.
  - Consider grouping some related entries if it makes sense.
- Bump the version number in `__about__.py`.
- Commit changes with message "release: release Pylake v<X.YY.Z>".
- Check whether any dependencies of Pylake have changed and check if required packages do exist on anaconda:
  - `git diff v<previous_release_version_number> pyproject.toml` to check changed dependencies.
  - Check availability of package versions on [anaconda](https://anaconda.org/) in the channel `anaconda` and `conda-forge`.
- Run `pytest` with `pytest --runpreflight --runslow ./lumicks/pylake` and verify that all tests pass (none may be skipped).
- Build the docs (see `docs/readme.md`) and verify that they build without warnings.
- Run the notebook testing script with `python docs/run_notebooks.py --reset`. The reset flag removes the cached results. If a notebook runs into an error, you fix it, rebuild the docs and then you can rerun the script without `--reset` to run only those that failed previously.
- Push to origin, create a PR targeting the `release/X.Y` branch, wait until all checks have passed and merge.

## Releasing

### PyPI

Perform the release procedure on GitHub: [Create a new release](https://github.com/lumicks/pylake/releases/new):
- Create a new tag `v<release_version_number>` with target `release/X.Y`.
- Set the release title to `v<release_version_number>`.
- Include the relevant changelog entries into the description.
- Wait until all [actions](https://github.com/lumicks/pylake/actions) of the release merge have passed.
- "Publish release".
- If it's a bugfix release, merge `release/X.Y` into `main` (see below).

### conda-forge

- Wait until the new release of [Pylake has been published on PyPI](https://pypi.org/project/lumicks.pylake/)
- Copy the sha256 hash of the tarball under [download files](https://pypi.org/project/lumicks.pylake/#files). You will need this for releasing on conda-forge.

- Update the Pylake conda package hosted on the `conda-forge` channel. For details see [Maintaining packages](https://conda-forge.org/docs/maintainer/updating_pkgs.html):
  - Use [our conda-forge feedstock fork](https://github.com/lumicks/lumicks.pylake-feedstock) and _not_ upstream
  - Rebase main on newest changes of upstream: `git rebase upstream/main` (see [2. Syncing your fork with conda-forges feedstock](https://conda-forge.org/docs/maintainer/updating_pkgs.html#example-workflow-for-updating-a-package)
  - Create a new branch `git checkout -b release_v<X_YY_Z>`
  - Update `recipe/meta.yaml`:
    - Update "version" if the Pylake version number changed.
    - Update the sha256 hash with the copy from PyPI.
    - Ensure the build number is zero, or if it is a re-release because some dependencies changed, increment the build number (but do not change the Pylake version).
    - Check if your github username is listed under `recipe-maintainers`
    - Commit changes with a message like "release Pylake <X.YY.Z>"
  - [Rerender the package with conda-smithy](https://conda-forge.org/docs/maintainer/updating_pkgs.html#rerendering-with-conda-smithy-locally).
  - Push the new branch to _our_ conda-forge feedstock fork: `git push -u origin release_v<X_YY_Z>` and _not_ upstream
  - Make a PR on the [upstream conda-forge feedstock repository](https://github.com/conda-forge/lumicks.pylake-feedstock).
  - Click `compare across forks` and choose the new branch from _our_ conda-forge feedstock fork as a source to pull from.
  - Double check the checklist of the PR.
  - Wait for the tests to complete and pull in the changes.

## Merging release branches into `main`

We merge bugfixes into `main` after every release (or sooner if critical fixes are needed).
Follow these steps:
- Create a merge branch: `git checkout main && git pull && git checkout -b merge/X.Y`.
- Merge the release branch and resolve any conflicts: `git merge release/X.Y`.
- Open a PR, but do not merge via the GitHub web UI (it would create an extra redundant merge commit).
- Instead, merge via git (the GitHub web UI will recognize this and mark the PR as merged): `git checkout main && git merge --ff-only merge/X.Y && git push main`.

You're all set! Pylake has been released.
