#include "alfa/tensor/Production.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace
{
using namespace alfa::tensor;
constexpr std::size_t ticks = 192000;
template <class S>
void measure(std::size_t history, bool rejectSome)
{
    PrimitiveT<S> primitive;
    std::string error;
    if (!production::createPrimitive("time.ring_delay.v1", history + 2, primitive, error))
        throw std::runtime_error(error);
    ConfigurationT<S> c;
    c.activation = {{"instance"}, {1}, {S(1)}};
    c.parameters = {{"parameter"}, {2}, {S(127), S(0.5)}};
    c.routing = {{"weight"}, {0}, {}};
    c.modulation = c.routing;
    c.instances = {{&primitive, 0, {0, 2}, {1, 0}, OutputBetweenUpdates::Hold}};
    c.initialState = {std::vector<S>(history + 2, S(0))};
    RuntimeT<S> runtime;
    if (!runtime.prepare(c, Budget{1, history + 2, 2, 0, 0, 1, 3, 0, 3}, 192000, error))
        throw std::runtime_error(error);
    std::array<S, 1> audio{S(0.25)};
    InputsT<S> inputs;
    inputs.audio.values = audio;
    // Verification is deliberately outside the timing interval.
    if (!runtime.tick(inputs))
        throw std::runtime_error("Initial valid tick rejected");
    const std::vector<S> before(runtime.state().begin(), runtime.state().end());
    const auto oldSample = runtime.sample();
    audio[0] = std::numeric_limits<S>::quiet_NaN();
    if (runtime.tick(inputs) || runtime.sample() != oldSample ||
        !std::equal(before.begin(), before.end(), runtime.state().begin()) || runtime.output()[0] != S(0))
        throw std::runtime_error("Rollback check failed");
    audio[0] = S(0.25);
    if (!runtime.tick(inputs))
        throw std::runtime_error("Post-rejection valid tick rejected");
    std::array<double, 7> ns{};
    double checksum = 0;
    for (std::size_t repeat = 0; repeat < ns.size(); ++repeat)
    {
        runtime.reset();
        audio[0] = S(0.25);
        for (std::size_t warm = 0; warm < 4096; ++warm)
            if (!runtime.tick(inputs))
                throw std::runtime_error("Warm-up rejected");
        const auto start = std::chrono::steady_clock::now();
        std::size_t successes = 0;
        for (std::size_t i = 0; i < ticks; ++i)
        {
            const bool invalid = rejectSome && i % 64 == 0;
            audio[0] = invalid ? std::numeric_limits<S>::quiet_NaN() : S(0.25);
            successes += runtime.tick(inputs) ? 1u : 0u;
            checksum += static_cast<double>(runtime.output()[0]);
        }
        const auto stop = std::chrono::steady_clock::now();
        if (successes != ticks - (rejectSome ? ticks / 64 : 0) || runtime.sample() != 4096 + successes)
            throw std::runtime_error("Unexpected commit count");
        ns[repeat] =
            std::chrono::duration<double, std::nano>(stop - start).count() / static_cast<double>(ticks);
    }
    std::sort(ns.begin(), ns.end());
    std::cout << (sizeof(S) == 4 ? "float32" : "float64") << ',' << history << ','
              << (rejectSome ? "reject_1_in_64" : "valid") << ',' << ticks << ','
              << runtime.numericStorageBytes() << ',' << std::fixed << std::setprecision(3) << ns[3] << ','
              << ns.front() << ',' << ns.back() << ',' << checksum << '\n';
}
} // namespace
int main()
{
    try
    {
        std::cout << "dtype,history,mode,ticks,numeric_bytes,median_ns_per_tick,min_ns,max_ns,checksum\n";
        for (const auto history : {std::size_t(128), std::size_t(65536), std::size_t(384000)})
            for (const bool rejectSome : {false, true})
            {
                measure<float>(history, rejectSome);
                measure<double>(history, rejectSome);
            }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
