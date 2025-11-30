#include "utils.hpp"

#include <format>
#include <print>

import diffusionx.random;

int main(int argc, char **argv) {
    size_t bench_size = 200;
    size_t len = 10000000;

    auto uniform = [len]() { auto result = rand(len).value(); };

    auto stable = [len]() { auto result = rand_stable(len, 0.7).value(); };

    auto normal = [len]() { auto result = randn(len).value(); };

    auto exponential = [len]() { auto result = randexp(len).value(); };

    std::println("==========================C++==========================");
    std::println("\n");

    std::println("bench size: {}, length of random vectors: {}", bench_size,
                 len);
    std::println("\n");

    bench("uniform random number sampling", uniform, bench_size);
    bench("normal random number sampling", normal, bench_size);
    bench("exponential random number sampling", exponential, bench_size);
    bench("stable random number sampling", stable, bench_size);

    return 0;
}
