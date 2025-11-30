#include "utils.hpp"
import diffusionx.simulation.continuous;

int main() {
    Levy levy{0.7, 0.0, 1.0, 0.0, 0.0};

    double duration = 1000.0;
    double time_step = 0.01;

    auto func = [&]() {
        auto result = levy.simulate(duration, time_step).value();
    };

    bench("levy", func, 10000);
    return 0;
}
