# Pylake release procedure

## Preflight check

- Update the changelog with the date.
- Review the changelog. Make sure everything is clear and informative for users.
- Bump the version number in `__about__.py`.
- Check whether any dependencies have changed.

## Releasing

### Pypi

- Perform the release procedure on GitHub (just follow the instructions on "create a new release").
- Include the relevant changelog entries.
- Don't forget to check "This is a pre-release".

### Conda Forge

- Go to the PyPI page of the newly released version: https://pypi.org/project/lumicks.pylake/
- Under download files, check the sha256 hash of the tarball. You will need this for releasing on Conda Forge.
- Use our [conda-forge feedstock fork](https://github.com/lumicks/lumicks.pylake-feedstock) and create a new branch based on `upstream/master`.
- Create a new branch on our feedstock for your release.
- Update the version in the `recipe/meta.yaml` if the version number changed. If the build number is not zero, set it to zero. Also update the sha256 hash.
- If it's a re-release because some dependencies changed, increment the build number but do not change the version.
See https://conda-forge.org/docs/maintainer/updating_pkgs.html for more info.
- Make a PR on the [upstream conda-forge repository](https://github.com/conda-forge/lumicks.pylake-feedstock). Wait for the tests to complete and pull in the changes.

You're all set! Pylake has been released.
