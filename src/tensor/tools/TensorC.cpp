#include "alfa/tensor/Compiler.h"
#include "alfa/tensor/Model.h"

#include <algorithm>
#include <array>
#include <memory>
#include <nlohmann/json.hpp>
#include <span>
#include <string>
#include <string_view>

#if defined(_WIN32)
#define AOM_EXPORT extern "C" __declspec(dllexport)
#else
#define AOM_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace
{
struct Handle
{
    alfa::tensor::Runtime runtime;
    std::unique_ptr<alfa::tensor::TensorModel> model;
    std::array<std::string, 1> axes{"feature"};
};

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
        auto handle = std::make_unique<Handle>();
        std::string failure;
        if (!alfa::tensor::compileJSON(document, sampleRate, handle->runtime, failure))
        {
            writeError(error, errorCapacity, failure);
            return nullptr;
        }
        const auto parsed = nlohmann::json::parse(document);
        if (parsed.contains("runtimeModel"))
        {
            alfa::tensor::Configuration configuration;
            handle->model = std::make_unique<alfa::tensor::TensorModel>();
            if (!alfa::tensor::parseJSON(document, configuration, failure) ||
                !handle->model->prepare(parsed.at("runtimeModel").dump(), configuration, failure))
            {
                writeError(error, errorCapacity, failure);
                return nullptr;
            }
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
    delete static_cast<Handle*>(opaque);
}
AOM_EXPORT std::size_t aom_output_count(void* opaque) noexcept
{
    return opaque == nullptr ? 0 : static_cast<Handle*>(opaque)->runtime.output().size();
}
AOM_EXPORT void aom_reset(void* opaque) noexcept
{
    if (opaque != nullptr)
    {
        auto& handle = *static_cast<Handle*>(opaque);
        handle.runtime.reset();
        if (handle.model)
            handle.model->reset();
    }
}

// Six packed row-major arrays: audio, MIDI, gesture, user preference, latent sound,
// context. Widths are supplied, never inferred from a fixed model or synthesis method.
AOM_EXPORT int aom_render(void* opaque, std::size_t frames, const float* const* sources,
                          const std::size_t* widths, float* output, std::size_t outputCapacity) noexcept
{
    if (opaque == nullptr || sources == nullptr || widths == nullptr)
        return 0;
    auto& handle = *static_cast<Handle*>(opaque);
    const auto outputs = handle.runtime.output().size();
    if ((outputs != 0 && frames > outputCapacity / outputs) ||
        (outputs != 0 && frames != 0 && output == nullptr))
        return 0;
    for (std::size_t category = 0; category < 6; ++category)
        if (widths[category] != 0 && sources[category] == nullptr)
            return 0;
    for (std::size_t sample = 0; sample < frames; ++sample)
    {
        std::array<alfa::tensor::TensorView, 6> views;
        std::array<std::array<std::size_t, 1>, 6> shapes;
        for (std::size_t category = 0; category < 6; ++category)
            if (widths[category] != 0)
            {
                shapes[category] = {widths[category]};
                views[category] = {
                    handle.axes,
                    shapes[category],
                    std::span<const float>(sources[category] + sample * widths[category], widths[category]),
                    {}};
            }
        const alfa::tensor::Inputs input{views[0], views[1], views[2], views[3], views[4], views[5]};
        if ((handle.model && !handle.model->apply(input, handle.runtime)) || !handle.runtime.tick(input))
            return 0;
        if (outputs != 0)
            std::copy(handle.runtime.output().begin(), handle.runtime.output().end(),
                      output + sample * outputs);
    }
    return 1;
}
