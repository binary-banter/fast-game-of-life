# Fast Game of Life Simulation

This repository provides code to perform extremely fast simulation of dense Game of Life instances. It is accompanied
by [this](https://binary-banter.github.io/game-of-life/) blogpost.

# How to Run Tests

You can choose to select what implementation to test by providing up to one feature.

```shell
# Test the `trivial` implementation.
cargo test --release

# Test the `simd`, `cuda` or `opencl` implementation.
cargo test --release --features simd
cargo test --release --features cuda
cargo test --release --features opencl
```

# How to Run Benchmarks

The basic benchmark provided steps a grid a given number of times.
Depending on the implementation, the grid-size and/or number of steps may need to be tweaked due to time constraints.

```shell
# Bench the `trivial` implementation.
cargo bench

# Bench the `simd`, `cuda` or `opencl` implementation.
cargo bench --features simd
cargo bench --features cuda
cargo bench --features opencl
```
