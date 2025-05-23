# Releasing New WecOptTool Version
This section is for developers.

Before a release make sure to:

* change [version number](https://semver.org/) in `pyproject.toml` in the `dev` branch.
* Merge the `dev` branch into the `main` branch. **Note: the `dev` branch should only be merged into `main` when it is ready for a new release.**

## GitHub
In the GitHub repository, click on *Releases*, click on *Draft new release*.

* Title the release with the [version number](https://semver.org/) preceded by a `v`, e.g., `v1.0.0`. Nothing else should go in the title.
* Tag the release using the same name as the *Title*.
* Click on *Generate release notes*. This adds the PR messages and contributors. Ideally nothing more is needed, but might require minor editing/formatting.
* Select the *Create a discussion* checkmark and mark as *Announcement*.

This will trigger the PyPI, Conda, and GH-Pages build and deploy.

### Pre-releases:
For pre-releases make sure to use correct [semantic versioning](https://semver.org/) for the version number in `pyproject.toml`.
In the Github release, for the *title* and *tag* name append the pre-release version after the version name, e.g., `v1.2.0-alpha` or `v1.2.1-beta.3`, and select the *pre-release* checkmark. Do not select the *Create a discussion* checkmark.

## PyPI package
For details see the [Python packaging user guide](https://packaging.python.org/en/latest/) and in particular the [packaging tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/).

The PyPI package is created and uploaded automatically to [TestPyPI](https://test.pypi.org/) and [PyPI](https://pypi.org/) on every GitHub release.
This is done sequentially, so that if *TestPyPi* fails the package is not pushed to `PyPi`.
See the [GitHub release workflow](https://github.com/sandialabs/WecOptTool/blob/main/.github/workflows/release.yml).

**NOTE:** While GitHub lets you delete a release and then create a new one with the same name, PyPI does not. You can delete releases but you cannot upload a package with the same version as a previous one (even a deleted one).

## Conda package
The Conda package for WecOptTool is housed in the [wecopttool-feedstock](https://github.com/conda-forge/wecopttool-feedstock) repository.
When a new release is available on PyPI, Conda-forge has a [bot](https://github.com/regro/autotick-bot) that will automatically detect it, create a pull request in wecopttool-feedstock, and merge it.
In other words, the WecOptTool Conda package will usually update automatically within a few hours of a new release being published.

If any dependencies are added, removed, or changed in `pyproject.toml`[^1], the Conda recipe must be manually updated to reflect these changes.
To do this:

1. Fork `wecopttool-feedstock`
2. In a new branch, update [`wecopttool-feedstock/recipe/meta.yaml`](https://github.com/conda-forge/wecopttool-feedstock/blob/main/recipe/meta.yaml) (the file containing the Conda recipe) by editing the `run` list of `requirements` to mirror the dependency list in `pyproject.toml`.
3. Create a [pull request](https://github.com/conda-forge/wecopttool-feedstock/pulls) to the `main` branch of `wecopttool-feedstock` from your new branch.
4. Follow the instructions in the autogenerated checklist in the pull request description.
5. Merge the pull request once the linter comes back clean and the CI passes.

Other changes to the Conda configuration unrelated to a new version release, such as changing package metadata or the maintainers list, also require a manual update to the recipe through this same process.

[^1]: At time of writing, Conda-forge does not currently have full integration with `pyproject.toml` files, see https://github.com/conda-forge/wecopttool-feedstock/pull/8.
This may change in the future though: see https://github.com/conda/conda/issues/12462.
