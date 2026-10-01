#include "alfa/tensor/Runtime.h"
#include "alfa/tensor/Universe.h"

#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numbers>
#include <vector>
using namespace alfa::tensor;
template <class Scalar>
void profile()
{
    PrimitiveT<Scalar> primitive;
    std::string error;
    if (!universe::createPrimitive("physical.resonator", 2, primitive, error))
        throw std::runtime_error(error);
    ConfigurationT<Scalar> config;
    config.activation = {{"instance"}, {16}, std::vector<Scalar>(16, Scalar(1))};
    config.parameters = {{"parameter"}, {48}, std::vector<Scalar>(48)};
    config.routing = {{"weight"}, {0}, {}};
    config.modulation = config.routing;
    for (size_t i = 0; i < 16; ++i)
    {
        config.parameters.values[i * 3] = Scalar(40 + i * 13);
        config.parameters.values[i * 3 + 1] = Scalar(30);
        config.parameters.values[i * 3 + 2] = Scalar(1);
        config.instances.push_back({&primitive, i, {i * 3, 3}, {1, 0}, OutputBetweenUpdates::Zero});
        config.initialState.push_back({Scalar(0), Scalar(0)});
    }
    RuntimeT<Scalar> runtime;
    const Budget budget{16, 32, 48, 0, 0, 16, 64, 0};
    if (!runtime.prepare(config, budget, 192000, error))
        throw std::runtime_error(error);
    const std::array<std::string, 1> axes{"channel"};
    const std::array<size_t, 1> shape{1};
    std::array<Scalar, 1> audio{};
    InputsT<Scalar> inputs;
    inputs.audio = {axes, shape, audio, {}};
    constexpr size_t blocks = 4096, blockSize = 256;
    double checksum = 0;
    auto start = std::chrono::steady_clock::now();
    for (size_t block = 0; block < blocks; ++block)
        for (size_t sample = 0; sample < blockSize; ++sample)
        {
            audio[0] = block == 0 && sample == 0 ? Scalar(1) : Scalar(0);
            if (!runtime.tick(inputs))
                throw std::runtime_error("runtime rejected test input");
            checksum += runtime.output()[0];
        }
    const double elapsed =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
    std::cout << "scalar_bytes=" << sizeof(Scalar)
              << " numeric_backing_bytes=" << runtime.numericStorageBytes()
              << " us_per_256_sample_callback=" << elapsed / static_cast<double>(blocks)
              << " checksum=" << checksum << '\n';
}
int main()
{
    profile<float>();
    profile<double>();
}
