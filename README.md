# MyScale Search-Index

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL_v3-yellow.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![Language](https://img.shields.io/badge/Language-C++20-blue.svg)](https://isocpp.org/)

The MyScale Search-Index library, incorporating vector indexing algorithms used in [MyScaleDB](https://github.com/myscale/myscaledb), offers a unified interface for tasks such as index creation, loading, serialization, and vector search. Although primarily developed for integration with MyScale, it also supports standalone compilation and unit test execution. The library supports various vector index algorithms, including Flat, IVF, HNSW (with optimized HNSWfast), and ScaNN (with automatic parameter tuning).

## Building from source code

### Install dependencies

Similar to MyScaleDB, Search-Index requires `clang-15` with `c++20` support to build from source:

```bash
sudo apt install clang-15 libc++abi-15-dev libc++-15-dev -y
sudo apt install libboost-all-dev libmkl-dev -y
```

### Build the project

After installing the dependencies, use `cmake` to build the project. The static library and unit test programs will be generated under the `build/` folder.

```bash
mkdir build && cd build
CC=clang-15 CXX=clang++-15 cmake .. && make -j
```

### Run Unit Tests

Execute the following commands under the `build` folder. The `run_tests.sh` script contains commands for testing vector indexes under various configurations:

```bash
cd build
bash ../scripts/run_tests.sh
```

## Contributing

Before submitting a pull request, please ensure that you run the pre-commit checks locally and execute the unit tests to verify that your code is ready for review.

Use the following commands to install and set up the pre-commit checks:

```bash
# install `clang-format` and `clang-tidy` if you haven't done so already
sudo apt install clang-format clang-tidy

# install pre-commit hooks
pip3 install pre-commit
pre-commit install
```

## Credits

This project utilizes the following open-source vector search libraries:

- [Faiss](https://github.com/facebookresearch/faiss) - A library for efficient similarity search and clustering of dense vectors, by Meta's Fundamental AI Research.
- [hnswlib](https://github.com/nmslib/hnswlib) - Header-only C++/python library for fast approximate nearest neighbors.
- [ScaNN](https://github.com/google-research/google-research/tree/master/scann) - Scalable Nearest Neighbors library by Google Research.
