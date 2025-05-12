use criterion::{Criterion, criterion_group, criterion_main};
use diffusionx::simulation::{continuous::Bm, prelude::*};
use std::hint::black_box;

const _N: usize = 100_000;

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("Brownian motion simulation", |b| {
        b.iter(|| {
            let bm = Bm::default();
            bm.simulate(black_box(100), black_box(0.01))
                .unwrap();
        });
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
