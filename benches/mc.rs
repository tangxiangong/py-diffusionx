use criterion::{Criterion, criterion_group, criterion_main};
use diffusionx::simulation::{continuous::Bm, prelude::*};
use std::hint::black_box;

const N: usize = 1_000_000;

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("Brownian motion simulation", |b| {
        b.iter(|| {
            let bm = Bm::default();
            bm.simulate(black_box(5000), black_box(0.01))
                .unwrap();
        });
    });

    c.bench_function("Brownian motion MSD simulation", |b| {
        b.iter(|| {
            let bm = Bm::default();
            bm.msd(black_box(500), black_box(N), black_box(0.01))
                .unwrap();
        });
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
