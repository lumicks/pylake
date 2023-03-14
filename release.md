# Pylake release procedure

## Preflight Check

Within the Pylake repo dir:
- Create and checkout a new branch for the new release from `main`.
- Update the file `changelog.md`:
  - Update the date of the new release.
  - Review the changelog entries. Make sure everything is clear and informative for users.
- Bump the version number in `./lumicks/pylake/__about__.py`.
- Commit changes with message "release: release Pylake v<X.YY.Z>".
- Check whether any dependencies of Pylake have changed and check if required packages do exist on anaconda:
  - `git diff v<previous_release_version_number> setup.py` to check changed dependencies.
  - Check availability of package versions on [anaconda](https://anaconda.org/) in the channel `anaconda` and `conda-forge`.
- Run `pytest` with `pytest --runpreflight --runslow ./lumicks/pylake` and verify that all tests pass (none may be skipped).
- Push to origin, create a PR, wait until all checks have passed and merge to `main`.

## Releasing

### PyPI

Perform the release procedure on GitHub: [Create a new release](https://github.com/lumicks/pylake/releases/new):
- Create a new tag `v<release_version_number>` with target `main`.
- Set the release title to `v<release_version_number>`.
- Include the relevant changelog entries into the description.
- Wait until all [actions](https://github.com/lumicks/pylake/actions) of the release merge have passed.
- "Publish release".

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

You're all set! Pylake has been released.
