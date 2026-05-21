# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`diffusionx` is a Python package for high-performance random number generation and
stochastic process simulation. It is a thin PyO3 binding layer: all numerical work is
done by the upstream Rust crate [`diffusionx`](https://crates.io/crates/diffusionx)
(pinned in `Cargo.toml`). This repo contains **bindings only** — no simulation
algorithms. Changing simulation behavior usually means bumping the upstream crate, not
editing this code.

## Build & develop

The package manager is `uv`; the Python toolchain is `maturin` (PyO3 extension module).

```sh
maturin develop            # compile the Rust _core extension into .venv and install the package
maturin develop --release  # optimized build (use when benchmarking)
cargo check                # fast type-check of the Rust side without producing the extension
cargo clippy               # Rust lint (the codebase compiles clippy-clean)
```

`.cargo/config.toml` pins `PYO3_PYTHON=.venv/bin/python`, so cargo always links against
the project venv — create/activate the venv before any cargo or maturin command.

There is **no test suite** in this repository (no `tests/` directory). Verify changes by
importing the package and exercising it manually, or against the benchmark repo linked in
the README.

## Regenerating type stubs

`python/diffusionx/_core/__init__.pyi` is **auto-generated** — never edit it by hand.
After adding, removing, or changing the signature of any `#[pyfunction]`, regenerate it:

```sh
./scripts/gen_stub.sh   # runs: cargo run --bin stub_gen --features stub_gen
```

Generation is driven by `#[gen_stub_pyfunction]` attributes (behind the `stub_gen`
feature). Every new pyfunction must carry `#[cfg_attr(feature = "stub_gen", gen_stub_pyfunction)]`.

## Architecture

Two layers, connected by a flat function ABI:

- **Rust `_core` extension** (`src/`) — exposes one flat `#[pyfunction]` per
  (process, operation) pair. The module is assembled in `src/lib.rs` via the
  `register_functions!` macro listing every exported function. `_core` is declared
  `#[pymodule(gil_used = false)]` (free-threaded / no-GIL capable) and initializes a
  global rayon thread pool sized to `num_cpus`.
- **Python wrapper package** (`python/diffusionx/`) — user-facing classes
  (`Bm`, `FBm`, `Levy`, ...) that validate arguments and dispatch to the matching
  `_core.<process>_<operation>` function.

### Naming & dispatch conventions

`_core` functions follow `<process>_<operation>`, e.g. `bm_simulate`, `bm_raw_moment`,
`bm_fpt`, `levy_occupation_time_central_moment`. The Python class for a process owns the
mapping from a clean OO API to these flat calls.

Two dispatch patterns recur in the Python wrappers — match them when adding a process:

- **central vs raw moments**: `central=True` → `<process>_central_moment`, else
  `<process>_raw_moment`.
- **integer vs fractional order**: `isinstance(order, int)` → `<process>_..._moment`;
  float order → `<process>_..._frac_..._moment`. See `bm.py`'s `moment` for the canonical
  nested-ternary form.

Arguments are passed **positionally** to `_core`; the Python keyword order and the Rust
signature order frequently differ (e.g. `particles`/`time_step` are swapped between some
Python signatures and their `_core` counterparts). Always check the Rust `#[pyfunction]`
signature, not the Python one, when wiring a call.

### Generic functionals

`moment`/`mean`/`msd`/`tamsd`/`eatamsd` (defined in `src/simulation/continuous.rs`,
exposed as bare `_core.moment` etc.) are process-agnostic: they receive a Python
`simulate` **callable**, then run `particles` simulations in parallel with rayon,
re-acquiring the GIL per call via `Python::attach`. `ContinuousProcess` in
`python/diffusionx/simulation/basic.py` provides these as mixin methods. Note: generic
`fpt` and `occupation_time` wrappers exist but are **commented out** in both
`src/lib.rs` and `basic.py` — per-process `<process>_fpt` functions are used instead.

### Validation & errors

All input validation lives in Python (`python/diffusionx/simulation/utils.py`:
`validate_positive_float`, `validate_domain`, `validate_order`, ...). The Rust side
assumes pre-validated inputs and surfaces upstream-crate errors as `XPyError`
(`src/error.rs`), which converts to a Python `ValueError`. Rust functions return
`XPyResult<T>`.

## Adding a new process function

1. Write the `#[pyfunction]` in `src/simulation/processes/<process>.rs` with the
   `#[cfg_attr(feature = "stub_gen", gen_stub_pyfunction)]` attribute.
2. Register it in the `register_functions!` block in `src/lib.rs`.
3. Add the calling code to the process's Python class in `python/diffusionx/simulation/`.
4. Run `maturin develop`, then `./scripts/gen_stub.sh` to refresh the stub.

## Releasing

`Cargo.toml` and `pyproject.toml` versions must stay in sync. Publishing is automated:
creating a GitHub Release triggers `.github/workflows/publish.yml`, which builds wheels
for Linux/macOS/Windows (including no-GIL `3.13t` variants) and an sdist, then publishes
to PyPI via trusted publishing. `CHANGELOG.md` is generated by `git-cliff` (`cliff.toml`).
