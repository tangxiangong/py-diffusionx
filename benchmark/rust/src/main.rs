use diffusionx::random::normal;
use diffusionx::random::{stable, uniform};
use std::env;
use std::time;

fn timeit<F>(func: F, bench_size: usize) -> Vec<f64>
where
    F: Fn() -> i32,
{
    (0..bench_size)
        .map(|_| {
            let start_time = time::Instant::now();
            let _ = func();
            start_time.elapsed().as_secs_f64()
        })
        .collect()
}

fn show_timeit(result: Vec<f64>) {
    let mean = result.iter().sum::<f64>() / result.len() as f64;
    let stddev =
        (result.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / result.len() as f64).sqrt();
    let min = *result
        .iter()
        .reduce(|a, b| if *a < *b { a } else { b })
        .expect("Failed to find minimum value");
    let max = *result
        .iter()
        .reduce(|a, b| if *a > *b { a } else { b })
        .expect("Failed to find maximum value");
    println!("mean: {mean:.4}, stddev: {stddev:.4}, min: {min:.4}, max: {max:.4}");
    println!("unit: second");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let bench_size = args.get(1).map(|s| s.parse().unwrap_or(20)).unwrap_or(20);

    let len = 10_000_000;

    println!("=========================Rust==========================");
    println!();

    println!(
        "bench size: {}, length of random vectors: {}",
        bench_size, len
    );
    println!("unit: second");
    println!();

    let uniform = || {
        let _rnds = uniform::range_rands(0.0..1.0, len).unwrap();
        0
    };
    println!("------------uniform random number sampling------------");
    show_timeit(timeit(uniform, bench_size));

    let normal = || {
        let _rands = normal::standard_rands::<f64>(len);
        0
    };
    println!("------------normal random number sampling------------");
    show_timeit(timeit(normal, bench_size));

    let stable = || {
        let _rnds = stable::sym_standard_rands(0.7, len).unwrap();
        0
    };
    println!("------------stable random number sampling------------");
    show_timeit(timeit(stable, bench_size));

    println!("=======================================================");
    println!();
}
