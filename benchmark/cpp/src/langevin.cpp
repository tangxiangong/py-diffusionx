#include "utils.hpp"
import diffusionx.simulation.continuous;

int main() {
    auto drift = [](double x, double t) -> double { return -x; };
    auto diffusivity = [](double x, double t) -> double { return 1.0; };

    Langevin eq{drift, diffusivity, 0.0};

    double duration = 1000.0;
    double time_step = 0.01;
    size_t num_samples = 10000;

    auto func = [&]() {
        auto result = eq.simulate(duration, time_step).value();
    };

    bench("langevin", func, 10);
    return 0;
}
