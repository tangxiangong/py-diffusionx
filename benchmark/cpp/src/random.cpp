#include "utils.hpp"

#include <format>
#include <print>

import diffusionx.random;

int main(int argc, char **argv) {
    size_t bench_size = 20;
    if (argc > 1) {
        bench_size = std::atoi(argv[1]);
    }
    size_t len = 10000000;

    auto uniform = [len]() { auto result = rand(len).value(); };

    auto stable = [len]() { auto result = rand_stable(len, 0.7).value(); };

    auto normal = [len]() { auto result = randn(len).value(); };

    std::println("==========================C++==========================");
    std::println("\n");

    std::println("bench size: {}, length of random vectors: {}", bench_size,
                 len);
    std::println("\n");

    std::println("-----------uniform random number sampling--------------");
    bench("uniform random number sampling", uniform, bench_size);

    std::println("------------normal random number sampling--------------");
    bench("normal random number sampling", normal, bench_size);

    std::println("-----------stable random number sampling---------------");
    bench("stable random number sampling", stable, bench_size);

    std::println("=======================================================");
    std::println("\n");
    return 0;
}
