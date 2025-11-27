# DiffusionX

English | [简体中文](README-zh.md)

![PyPI](https://img.shields.io/pypi/v/diffusionx)
![PyPI - License](https://img.shields.io/pypi/l/diffusionx)

> [!NOTE]
> Development is in progress. 
>
> The Python binding of the same-named Rust crate [diffusionx](https://crates.io/crates/diffusionx), a multi-threaded Rust crate for random number generation and stochastic process simulation, via [PyO3](https://github.com/PyO3/pyo3). 

## Usage

```python
from diffusionx.simulation import Bm

# Brownian motion simulation
bm = Bm()
traj = bm(10)
times, positions = traj.simulate(time_step=0.01)  # Simulate Brownian motion trajectory, returns ndarray

# Monte Carlo simulation of Brownian motion statistics
raw_moment = traj.raw_moment(order=1, particles=1000)  # First-order raw moment
central_moment = traj.central_moment(order=2, particles=1000)  # Second-order central moment

# First passage time of Brownian motion
fpt = bm.fpt((-1, 1))
```

## Progress
### Random Number Generation

- [x] Normal distribution
- [x] Uniform distribution
- [x] Exponential distribution
- [x] Poisson distribution
- [x] Alpha-stable distribution

### Stochastic Processes

- [x] Brownian motion
- [x] Alpha-stable Lévy process
- [x] Subordinator
- [x] Inverse Subordinator
- [x] Fractional Brownian motion
- [x] Poisson process
- [ ] Compound Poisson process
- [x] Langevin process
- [x] Generalized Langevin equation
- [x] Subordinated Langevin equation
  
### Functional

- [x] First passage time
- [x] Occupation time

## Benchmark

Performance benchmark tests compare the Rust, C++, Julia, and Python implementations, which can be found [here](https://github.com/tangxiangong/diffusionx-benches).

## License

This project is dual-licensed under:

* [MIT License](https://opensource.org/licenses/MIT)
* [Apache License Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)

You can choose to use either license. 

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
