#include "alfa/tensor/Compiler.h"
#include "alfa/tensor/Model.h"
#include "alfa/tensor/Precision.h"

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>

#if defined(_WIN32)
#define AOM_EXPORT extern "C" __declspec(dllexport)
#else
#define AOM_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace
{
struct HandleBase
{
    virtual ~HandleBase() = default;
    virtual std::size_t outputCount() const noexcept = 0;
    virtual std::size_t scalarBytes() const noexcept = 0;
    virtual void reset() noexcept = 0;
    virtual int render(std::size_t frames, const float* const* sources, const std::size_t* widths,
                       float* output, std::size_t capacity) noexcept = 0;
};

template <typename Scalar>
struct Handle final : HandleBase
{
    alfa::tensor::RuntimeT<Scalar> runtime;
    std::unique_ptr<alfa::tensor::TensorModelT<Scalar>> model;
    alfa::tensor::Clock modelClock;
    std::array<std::string, 1> axes{"feature"};
    std::array<std::vector<Scalar>, 7> conversion;
    std::array<std::size_t, 7> inputCapacity{};
    bool boundedInputs = false;
    std::size_t outputCount() const noexcept override { return runtime.output().size(); }
    std::size_t scalarBytes() const noexcept override { return sizeof(Scalar); }
    void reset() noexcept override
    {
        runtime.reset();
        if (model)
            model->reset();
    }
    int render(std::size_t frames, const float* const* sources, const std::size_t* widths, float* output,
               std::size_t capacity) noexcept override
    {
        const auto outputs = outputCount();
        if ((outputs != 0 && frames > capacity / outputs) ||
            (outputs != 0 && frames != 0 && output == nullptr))
            return 0;
        for (std::size_t category = 0; category < 7; ++category)
        {
            if (widths[category] != 0 && sources[category] == nullptr)
                return 0;
            if (widths[category] != 0 && frames > std::numeric_limits<std::size_t>::max() / widths[category])
                return 0;
            if (boundedInputs && widths[category] > inputCapacity[category])
                return 0;
            if constexpr (!std::is_same_v<Scalar, float>)
                if (widths[category] > conversion[category].size())
                    return 0;
        }
        for (std::size_t sample = 0; sample < frames; ++sample)
        {
            std::array<alfa::tensor::TensorViewT<Scalar>, 7> views;
            std::array<std::array<std::size_t, 1>, 7> shapes;
            for (std::size_t category = 0; category < 7; ++category)
                if (widths[category] != 0)
                {
                    shapes[category] = {widths[category]};
                    std::span<const Scalar> values;
                    const auto* source = sources[category] + sample * widths[category];
                    if constexpr (std::is_same_v<Scalar, float>)
                        values = {source, widths[category]};
                    else
                    {
                        std::copy_n(source, widths[category], conversion[category].begin());
                        values = {conversion[category].data(), widths[category]};
                    }
                    views[category] = {axes, shapes[category], values, {}};
                }
            const alfa::tensor::InputsT<Scalar> inputs{views[0], views[1], views[2], views[3],
                                                       views[4], views[5], views[6]};
            if ((model && runtime.sample() % modelClock.periodSamples == modelClock.phaseSamples &&
                 !model->apply(inputs, runtime)) ||
                !runtime.tick(inputs))
                return 0;
            if (outputs != 0)
                std::transform(runtime.output().begin(), runtime.output().end(), output + sample * outputs,
                               [](Scalar value) { return static_cast<float>(value); });
        }
        return 1;
    }
};

template <typename Scalar>
std::unique_ptr<HandleBase> prepare(const char* text, double rate, const nlohmann::json& parsed,
                                    std::string& error)
{
    auto handle = std::make_unique<Handle<Scalar>>();
    if (!alfa::tensor::compileJSON(text, rate, handle->runtime, error))
        return {};
    if (parsed.contains("runtimeModel"))
    {
        alfa::tensor::ConfigurationT<Scalar> configuration;
        handle->model = std::make_unique<alfa::tensor::TensorModelT<Scalar>>();
        if (!alfa::tensor::parseJSON(text, configuration, error) ||
            !handle->model->prepare(parsed.at("runtimeModel").dump(), configuration, error))
            return {};
        const auto& clock = parsed.at("runtimeModel").at("clock");
        handle->modelClock = {clock.at("period_samples").get<uint64_t>(),
                              clock.at("phase_samples").get<uint64_t>()};
    }
    if (parsed.contains("input_capacity") || !std::is_same_v<Scalar, float>)
    {
        // The external ABI remains float32. Producers supply capacities for double boundary
        // conversion; storage is allocated only during prepare. A six-width list predates the
        // wave feature source, whose capacity is then zero.
        const auto& limits = parsed.at("input_capacity");
        if (!limits.is_array() || (limits.size() != 6 && limits.size() != 7))
            throw std::invalid_argument(
                "float64 C ABI requires explicit input_capacity for six or seven sources");
        handle->boundedInputs = true;
        std::size_t total = 0;
        const auto& requestedAllowance = parsed.at("budget").at("boundary_conversion_bytes");
        if (!requestedAllowance.is_number_unsigned() &&
            (!requestedAllowance.is_number_integer() || requestedAllowance.get<std::int64_t>() < 0))
            throw std::invalid_argument("Invalid boundary conversion memory allowance");
        const auto allowance = requestedAllowance.get<std::size_t>();
        for (std::size_t category = 0; category < limits.size(); ++category)
        {
            if (!limits[category].is_number_unsigned() &&
                (!limits[category].is_number_integer() || limits[category].get<std::int64_t>() < 0))
                throw std::invalid_argument("Invalid boundary source capacity");
            const auto width = limits[category].get<std::size_t>();
            handle->inputCapacity[category] = width;
            if constexpr (!std::is_same_v<Scalar, float>)
            {
                if (width > (allowance / sizeof(Scalar)) - total)
                    throw std::invalid_argument("Boundary conversion exceeds explicit memory allowance");
                total += width;
                handle->conversion[category].resize(width);
            }
        }
    }
    return handle;
}

void writeError(char* destination, std::size_t capacity, std::string_view error) noexcept
{
    if (destination == nullptr || capacity == 0)
        return;
    const auto count = std::min(capacity - 1, error.size());
    std::copy_n(error.data(), count, destination);
    destination[count] = '\0';
}
} // namespace

AOM_EXPORT void* aom_create(const char* document, double sampleRate, char* error,
                            std::size_t errorCapacity) noexcept
{
    try
    {
        if (document == nullptr)
        {
            writeError(error, errorCapacity, "No tensor computation document supplied");
            return nullptr;
        }
        std::string failure;
        alfa::tensor::PrecisionProfile precision;
        if (!alfa::tensor::resolvePrecisionJSON(document, precision, failure))
        {
            writeError(error, errorCapacity, failure);
            return nullptr;
        }
        const auto parsed = nlohmann::json::parse(document);
        auto handle = precision.scalarType == alfa::tensor::ScalarType::Float64
                          ? prepare<double>(document, sampleRate, parsed, failure)
                          : prepare<float>(document, sampleRate, parsed, failure);
        if (!handle)
        {
            writeError(error, errorCapacity, failure);
            return nullptr;
        }
        writeError(error, errorCapacity, "");
        return handle.release();
    }
    catch (const std::exception& failure)
    {
        writeError(error, errorCapacity, failure.what());
        return nullptr;
    }
}

AOM_EXPORT void aom_destroy(void* opaque) noexcept
{
    delete static_cast<HandleBase*>(opaque);
}
AOM_EXPORT std::size_t aom_output_count(void* opaque) noexcept
{
    return opaque == nullptr ? 0 : static_cast<HandleBase*>(opaque)->outputCount();
}
AOM_EXPORT std::size_t aom_scalar_bytes(void* opaque) noexcept
{
    return opaque == nullptr ? 0 : static_cast<HandleBase*>(opaque)->scalarBytes();
}
AOM_EXPORT void aom_reset(void* opaque) noexcept
{
    if (opaque != nullptr)
    {
        static_cast<HandleBase*>(opaque)->reset();
    }
}

// Seven packed row-major arrays: audio, MIDI, gesture, user preference, latent sound,
// context, wave feature. Widths are supplied, never inferred from a fixed model or synthesis method.
AOM_EXPORT int aom_render(void* opaque, std::size_t frames, const float* const* sources,
                          const std::size_t* widths, float* output, std::size_t outputCapacity) noexcept
{
    if (opaque == nullptr || sources == nullptr || widths == nullptr)
        return 0;
    return static_cast<HandleBase*>(opaque)->render(frames, sources, widths, output, outputCapacity);
}
