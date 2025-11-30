#ifndef DIFFUSIONX_BENCHES_UTILS_HPP
#define DIFFUSIONX_BENCHES_UTILS_HPP

#include <algorithm>
#include <chrono>
#include <concepts>
#include <numeric>
#include <print>
#include <vector>

using std::vector;

inline auto format_time(double nanoseconds) {
    struct {
        double value;
        const char *unit;
    } result;
    if (nanoseconds < 1e3) {
        result.value = nanoseconds;
        result.unit = "ns";
    } else if (nanoseconds < 1e6) {
        result.value = nanoseconds / 1e3;
        result.unit = "μs";
    } else if (nanoseconds < 1e9) {
        result.value = nanoseconds / 1e6;
        result.unit = "ms";
    } else {
        result.value = nanoseconds / 1e9;
        result.unit = "s";
    }
    return result;
}

template <typename F>
    requires std::invocable<F> && std::same_as<std::invoke_result_t<F>, void>
void bench(std::string_view &&name, F func, size_t bench_size) {
    func();
    vector<double> result(bench_size);
    for (auto &val : result) {
        auto start_time = std::chrono::high_resolution_clock::now();
        func();
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
            end_time - start_time);
        val = static_cast<double>(duration.count());
    }

    double mean_ns =
        std::accumulate(result.begin(), result.end(), 0.0) / result.size();
    auto [mean, mean_unit] = format_time(mean_ns);
    double variance = std::accumulate(result.begin(), result.end(), 0.0,
                                      [mean_ns](double acc, double x) {
                                          double diff = x - mean_ns;
                                          return acc + (diff * diff);
                                      }) /
                      result.size();
    auto [stddev, stddev_unit] = format_time(std::sqrt(variance));
    auto [min, min_unit] = format_time(*std::ranges::min_element(result));
    auto [max, max_unit] = format_time(*std::ranges::max_element(result));

    std::println("{}", name);
    std::println("mean: {:.3f} {}, stddev: {:.3f} {}, min: {:.3f} {}, max: "
                 "{:.3f} {}",
                 mean, mean_unit, stddev, stddev_unit, min, min_unit, max,
                 max_unit);
}

#endif // DIFFUSIONX_BENCHES_UTILS_HPP
