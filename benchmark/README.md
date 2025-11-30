# Benchmarks

## Random Number Sampling (unit: ms, len: 10_000_000)

### Uniform distribution
| | mean | min | max |
|:---:|:---:|:---:|:---:|
| C++ | 12.848 | 11.963 | 25.771 |
| Rust | 1.641 | 1.628 | 1.655 |
| Julia | 4.198 | 4.030 | 33.727 |
| Python (Rust wrapper) | - | - | - |
| Python (NumPy / SciPy) | 26.22 | 25.84 | 26.61 |

### Normal distribution
| | mean | min | max |
|:---:|:---:|:---:|:---:|
| C++ | 50.348 | 45.629 | 134.988 |
| Rust | 6.701 | 6.631 | 6.810 |
| Julia | 12.831 | 12.393 | 43.719 |
| Python (Rust wrapper) | - | - | - |
| Python (NumPy / SciPy) | 108.02 | 106.00 | 132.79 |

### Exponential distribution
| | mean | min | max |
|:---:|:---:|:---:|:---:|
| C++ | 33.3436  | 28.1625 | 40.7075 |
| Rust | 7.010 | 6.978 | 7.048 |
| Julia | 13.545 | 13.312 | 26.987 |
| Python (Rust wrapper) | - | - | - |
| Python (NumPy / SciPy) | 86.37 | 86.07 | 87.04 |


### Stable distribution
| | mean | min | max |
|:---:|:---:|:---:|:---:|
| C++ | 145.7687 | 132.5238 | 187.3853 |
| Rust | 73.293  | 73.488 | 73.702 |
| Julia | 274.270 | 261.133 | 286.218 |
| Python (Rust wrapper) | - | - | - |
| Python (NumPy / SciPy) | 615.60 | 612.23 | 621.36 |
