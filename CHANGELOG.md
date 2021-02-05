# epytope: Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v3.0.0dev - [date]

Initial release of epytope. Epytope is the successor project of FRED-2, which was renamed to a more versioning friendly base name.

### `Added`

- [#1](https://github.com/KohlbacherLab/epytope/pull/1) - Switch CI/CD from Travis to GitHub Actions
- [#6](https://github.com/KohlbacherLab/epytope/pull/6) - Add CI for external tools
- [#9](https://github.com/KohlbacherLab/epytope/pull/9) - Initial Python 2 to Python 3 conversion based on 2to3conv
- [#11](https://github.com/KohlbacherLab/epytope/pull/11) - Fix python version matrix in CI, remove versions that fail
- [#16](https://github.com/KohlbacherLab/epytope/pull/16) - Fix epitope prediction 2to3 bugs and tests
- [#17](https://github.com/KohlbacherLab/epytope/pull/17) - Fix Invalid subprocess handling and mhcflurry polluting stdout
- [#18](https://github.com/KohlbacherLab/epytope/pull/18) - Fix external (NetMHC tool family) epitope prediction error and tempfile handling
- [#20](https://github.com/KohlbacherLab/epytope/pull/20) - Use logging module rather than print calls across the library
- [#23](https://github.com/KohlbacherLab/epytope/pull/23) - Rename the package from FRED-2 to epytope
- [#24](https://github.com/KohlbacherLab/epytope/pull/24) - Add keras dependency
- [#25](https://github.com/KohlbacherLab/epytope/pull/25) - Refactor CI to use more pip and less conda
- [#26](https://github.com/KohlbacherLab/epytope/pull/26) - Add a license file
- [#31](https://github.com/KohlbacherLab/epytope/pull/31) - Add deployment to PyPI
