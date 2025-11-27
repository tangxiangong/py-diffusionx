# Changelog

All notable changes to this project will be documented in this file.

## [0.1.3] - 2025-11-27

### 🚀 Features

- Add benchmarking for random number generation
- Add Langevin simulation classes and update visualization module
- Add test for Langevin simulation
- Add Langevin and GeneralizedLangevin simulation functions
- Add new Langevin simulation functions and optimize callback mechanism
- Add GitHub Actions workflow for publishing to PyPI
- Add Brownian motion simulation and benchmarking in Python and Rust
- Add new functions for Brownian motion, Lévy process, fractional Brownian motion, and continuous time random walk simulations
- Add new functions for raw and central moments of Brownian motion and occupation time
- Add functions for raw and central moments of CTRW and occupation time
- Add new functions for raw and central moments of fractional Brownian motion and occupation time
- Add new functions for raw and central moments of Lévy process and occupation time
- Add new functions for raw and central moments of generalized Langevin and Lévy processes
- Add new functions for subordinated Langevin and generalized Langevin processes
- Add new functions for first passage times and occupation times in Langevin processes
- Add new functions for first passage times and occupation times in generalized Langevin processes
- Add raw and central moment functions for Poisson processes
- Add raw and central moment functions for subordinated and inverse subordinated processes
- Add Poisson and subordinator functions for raw and central moments
- Add additional functions for raw and central moments in Poisson and subordinator processes
- Add functions for raw and central moments of first passage and occupation times in Brownian motion
- Add functions for raw and central moments of first passage and occupation times in CTRW
- Add functions for raw and central moments of first passage and occupation times in fractional Brownian motion
- Add functions for first passage times and occupation times in Langevin, generalized Langevin, and subordinated processes
- Add functions for raw and central moments of first passage and occupation times in Levy processes
- Add additional functions for raw and central moments of first passage and occupation times in Poisson processes
- Add Brownian bridge simulation functions for statistical analysis
- Add Brownian excursion simulation functions for statistical analysis
- Add Brownian meander simulation functions for statistical analysis
- Add Cauchy and Asymmetric Cauchy simulation functions for statistical analysis
- Add Gamma process simulation functions for statistical analysis
- Add Geometric Brownian motion simulation functions for statistical analysis
- Enhance simulation module with additional stochastic process simulations
- Add Lévy walk simulation functions for statistical analysis
- Add Ornstein-Uhlenbeck simulation functions for statistical analysis
- Add Asymmetric Lévy simulation functions for statistical analysis
- Enhance simulation module with additional stochastic process functions
- Add new stochastic process functions for simulation and analysis
- Add additional stochastic process classes to simulation module
- Add Asymmetric Cauchy process class to simulation module
- Add Asymmetric Lévy process class with simulation methods
- Add Brownian Bridge class with simulation and moment calculation methods
- Add Brownian Excursion class with simulation and moment calculation methods
- Add Cauchy process class with simulation and moment calculation methods
- Add Gamma process class with simulation and moment calculation methods
- Add Geometric Brownian Motion class with simulation and moment calculation methods
- Add Lévy Walk class with simulation and moment calculation methods
- Add Meander class with simulation and moment calculation methods
- Add Ornstein-Uhlenbeck process class with simulation and moment calculation methods
- Refactor stochastic process classes and update simulation methods
- Add Python wrapper for ContinuousProcess trait methods
- Add pyo3-stub-gen for automatic type stub generation and upgrade diffusionx to 0.8
- Make pyo3-stub-gen optional and configure Python environment

### 🐛 Bug Fixes

- Update manifest path in publish-python.yml
- Update manifest path and add debug steps in publish.yml
- Update uv pip install command in publish.yml workflow
- Update diffusionx version and adjust return types in simulation functions
- Update diffusionx version to 0.3 in Cargo.toml
- Update __init__.py to include distribution module in exports

### 💼 Other

- Enhance Brownian motion simulation with new MATLAB functions and update benchmarks

### 🚜 Refactor

- Update docstrings in CTRW and Fbm classes for consistency and clarity
- Update debug steps in publish.yml workflow
- Enhance publish.yml workflow with debugging and build steps
- Refactor simulation module by splitting into separate files
- Reorder random module declaration in lib.rs
- Improve error handling and type validation in simulation modules
- Update Python notebook to remove execution counts and outputs
- Update ctrw simulation function to use simulate_with_duration
- Clean up whitespace and improve code formatting in simulation scripts
- Enhance simulation module with additional functions for first passage and occupation times
- Rename Bb class to BrownianBridge and update methods for clarity and functionality
- Rename Be class to BrownianExcursion and update methods for clarity and functionality
- Update Bm class to inherit from ContinuousProcess and enhance moment calculation methods
- Update Cauchy and add AsymmetricCauchy classes to inherit from ContinuousProcess
- Change CTRW class to inherit from PointProcess and enhance simulation methods
- Update simulation methods and change class imports for CTRW and Lévy Walk
- Update simulation classes to inherit from ContinuousProcess and enhance moment calculation methods
- Remove obsolete benchmark files for random number generation and simulations
- Update imports and class names in simulation module
- Streamline publish workflow by consolidating build jobs and using maturin-action
- Reorganize simulation module into nested submodules
- Remove visualization module from simulation package
- Remove functional module from simulation package
- Rename step_size parameter to time_step across all simulation functions
- Reorganize _core.pyi type stubs by grouping related functions
- Add stub generation attribute to generalized_langevin_simulate and fix indentation
- Upgrade minimum Python version from 3.10 to 3.11 in pyo3 abi3 feature
- Upgrade minimum Python version to 3.11 and update dependencies
- Add generalized_langevin_simulate function to type stubs
- Standardize import order and parameter validation across simulation modules

### 📚 Documentation

- Streamline README by moving benchmark details to separate repository and adding badges
- Add docstrings to all process simulation functions

### ⚙️ Miscellaneous Tasks

- Initialize project structure for DiffusionX Python package
- Update dependencies and enhance type hints in diffusionx package
- Simplify dependency list format in pyproject.toml
- Update README files with new benchmark results and add MATLAB benchmark scripts
- Update dependencies and add new visualization module
- Bump version to 0.1.4 in pyproject.toml
- Add CHANGELOG.md file
- Add .gitattributes file to exclude benches from linguist detection
- Add benchmark configuration and update diffusionx version in Cargo.toml
- Update diffusionx dependency version to 0.3.13 in Cargo.toml
- Add .DS_Store to .gitignore to prevent tracking of macOS system files
- Remove .gitattributes file to clean up repository
- Update dependencies and remove obsolete benchmark configurations in Cargo.toml
- Update dependencies in pyproject.toml by removing obsolete dev dependencies
- Update dependencies and refactor Python type usage
- Upgrade dependencies and bump minimum Python version to 3.10
- Bump version to 0.1.4 and remove obsolete test files
- Downgrade version to 0.1.2 and add package metadata
- Add explicit Python 3.11 setup step to Linux and Windows build jobs
- Bump version from 0.1.2 to 0.1.3

<!-- generated by git-cliff -->
