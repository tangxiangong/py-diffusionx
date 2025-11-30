#include "utils.hpp"
import diffusionx.simulation.continuous;

int main() {
    Bm bm{};

    double duration = 1000.0;
    double time_step = 0.01;
    size_t num_samples = 10000;

    auto func = [&]() {
        auto result = bm.msd(duration, num_samples, time_step).value();
    };

    bench("brownian motion", func, 10);
    return 0;
}
