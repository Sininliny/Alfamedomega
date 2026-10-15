#include "FactoryPrograms.h"
#include "TensorProcessor.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdlib>
#include <new>
#include <nlohmann/json.hpp>
#if defined(_WIN32)
#include <malloc.h>
#endif

namespace
{
thread_local bool guarding = false;
thread_local size_t allocationCount = 0, deallocationCount = 0;
struct CallbackGuard
{
    CallbackGuard()
    {
        allocationCount = deallocationCount = 0;
        guarding = true;
    }
    ~CallbackGuard() { guarding = false; }
};
} // namespace
void* operator new(size_t size)
{
    if (guarding)
        ++allocationCount;
    if (auto* pointer = std::malloc(size ? size : 1))
        return pointer;
    throw std::bad_alloc();
}
void* operator new[](size_t size)
{
    return ::operator new(size);
}
void operator delete(void* pointer) noexcept
{
    if (guarding && pointer)
        ++deallocationCount;
    std::free(pointer);
}
void operator delete[](void* pointer) noexcept
{
    ::operator delete(pointer);
}
void operator delete(void* pointer, size_t) noexcept
{
    ::operator delete(pointer);
}
void operator delete[](void* pointer, size_t) noexcept
{
    ::operator delete(pointer);
}
void* operator new(size_t size, std::align_val_t alignment)
{
    if (guarding)
        ++allocationCount;
    const auto boundary = static_cast<size_t>(alignment);
#if defined(_WIN32)
    auto* pointer = _aligned_malloc(size ? size : 1, boundary);
#else
    const auto requested = size ? size : 1;
    if (requested > SIZE_MAX - (boundary - 1))
        throw std::bad_alloc();
    auto* pointer = std::aligned_alloc(boundary, ((requested + boundary - 1) / boundary) * boundary);
#endif
    if (pointer)
        return pointer;
    throw std::bad_alloc();
}
void* operator new[](size_t size, std::align_val_t alignment)
{
    return ::operator new(size, alignment);
}
void operator delete(void* pointer, std::align_val_t) noexcept
{
    if (guarding && pointer)
        ++deallocationCount;
#if defined(_WIN32)
    _aligned_free(pointer);
#else
    std::free(pointer);
#endif
}
void operator delete[](void* pointer, std::align_val_t alignment) noexcept
{
    ::operator delete(pointer, alignment);
}
void operator delete(void* pointer, size_t, std::align_val_t alignment) noexcept
{
    ::operator delete(pointer, alignment);
}
void operator delete[](void* pointer, size_t, std::align_val_t alignment) noexcept
{
    ::operator delete(pointer, alignment);
}

TEST_CASE("Complete native callbacks allocate and free no C++ storage across publication MIDI and bypass",
          "[tensor-host][realtime][allocation]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    // Each processor owns a different prepared program. The test intentionally
    // includes the first callback and live publication, with no warm-up exemption.
    for (const auto* tier : {"Standard", "Pro"})
        for (const auto& program : alfa::tensorhost::factoryPrograms())
        {
            INFO(program.id);
            INFO(tier);
            alfa::tensorhost::Processor processor;
            processor.prepareToPlay(48000, 128);
            auto document = nlohmann::json::parse(program.document);
            document["tier"] = tier;
            document["scalar_type"] = std::string(tier) == "Pro" ? "float64" : "float32";
            document["budget"]["boundary_conversion_bytes"] = 16384;
            document["host_latency_samples"] = 17;
            document["budget"]["host_alignment_bytes"] = 136;
            juce::String error;
            REQUIRE(processor.loadPresetJSON(document.dump(), error));
            juce::AudioBuffer<float> audio(2, 128);
            juce::MidiBuffer midi;
            midi.ensureSize(65536);
            for (int block = 0; block < 12; ++block)
            {
                if (block == 3)
                {
                    if (!document.value("control_layers", nlohmann::json::array()).empty())
                        REQUIRE(processor.editLayerValue(0, 0, .1, error));
                    else
                    {
                        document["metadata"]["callback_guard_revision"] = 1;
                        REQUIRE(processor.editConfigurationJSON(document.dump(), error));
                    }
                }
                if (block == 6)
                    processor.requestPanic();
                if (block == 9)
                    processor.setParameterValue("mix", .5f);
                midi.clear();
                for (int note = 0; note < 32; ++note)
                {
                    const auto message = block % 2 == 0
                                             ? juce::MidiMessage::noteOn(1 + note % 16, 48 + note, .7f)
                                             : juce::MidiMessage::noteOff(1 + note % 16, 48 + note);
                    midi.addEvent(message, note * 4);
                    midi.addEvent(juce::MidiMessage::pitchWheel(1 + note % 16, 8192 + note), note * 4);
                }
                for (int channel = 0; channel < 2; ++channel)
                    for (int sample = 0; sample < 128; ++sample)
                        audio.setSample(channel, sample, .05f);
                {
                    CallbackGuard guard;
                    if (block == 10)
                        processor.processBlockBypassed(audio, midi);
                    else
                        processor.processBlock(audio, midi);
                }
                CHECK(allocationCount == 0);
                CHECK(deallocationCount == 0);
                for (int channel = 0; channel < 2; ++channel)
                    for (int sample = 0; sample < 128; ++sample)
                        REQUIRE(std::isfinite(audio.getSample(channel, sample)));
            }
            CHECK(processor.failures.load() == 0);
        }
}

TEST_CASE("Independent large-state model hosts preserve bounded callbacks during edits and resets",
          "[tensor-host][realtime][allocation][model]")
{
    juce::ScopedJuceInitialiser_GUI gui;
    auto document = nlohmann::json::parse(alfa::tensorhost::Processor::referenceDocument(0).toStdString());
    document["projections"] = nlohmann::json::array();
    document["instances"][0]["primitive"] = "time.ring_delay.v1";
    document["instances"][0]["state_size"] = 384002;
    document["instances"][0]["initial_state"] = std::vector<double>(384002);
    document["tensors"]["parameters"]["values"] = {32., .5};
    document["budget"]["state_elements"] = 384002;
    document["budget"]["sparse_state_writes"] = 3;
    document["budget"]["boundary_conversion_bytes"] = 16384;
    document["control_layers"] = nlohmann::json::array(
        {{{"values", {0.}},
          {"ranges", {{-1., 1.}}},
          {"matrix", {{"axes", {"destination", "source"}}, {"shape", {2, 1}}, {"values", {0., .1}}}}}});
    document["budget"]["stored_tensor_elements"] = 8;
    auto model = nlohmann::json::parse(juce::File(juce::String(ALFA_SOURCE_DIR))
                                           .getChildFile("training/amo/testdata/runtime_gru.json")
                                           .loadFileAsString()
                                           .toStdString());
    for (auto& weights : model["weights"])
        for (auto& value : weights)
            value = 0.;
    model["weights"]["values.bias"] = {1., 32., .5, 1., .2};
    model["weights"]["presence.bias"] = {10., 10., 10., 10., 10.};
    model["output_bounds"] = {{0., 1.}, {1., 384000.}, {-.9, .9}, {-1., 1.}, {-1., 1.}};
    model["clock"] = {{"period_samples", 257}, {"phase_samples", 13}};
    model["input_widths"]["audio"] = 2;
    model["input_widths"]["midi"] = 0;
    document["runtimeModel"] = model;
    std::array<std::unique_ptr<alfa::tensorhost::Processor>, 2> processors;
    juce::String error;
    for (size_t i = 0; i < processors.size(); ++i)
    {
        processors[i] = std::make_unique<alfa::tensorhost::Processor>();
        processors[i]->prepareToPlay(48000, 128);
        processors[i]->mode.store(1);
        document["tier"] = i == 0 ? "Standard" : "Pro";
        document["scalar_type"] = i == 0 ? "float32" : "float64";
        REQUIRE(processors[i]->loadJSON(document.dump(), error));
    }
    for (int block = 0; block < 48; ++block)
    {
        const int size = block % 3 == 0 ? 32 : block % 3 == 1 ? 64 : 128;
        juce::AudioBuffer<float> audio(2, size);
        juce::MidiBuffer midi;
        for (auto& processor : processors)
        {
            if (block == 8)
                REQUIRE(processor->editLayerValue(0, 0, .2, error));
            if (block == 16)
                REQUIRE(processor->setParameterLock(1, true, error));
            if (block == 32)
                processor->requestPanic();
            audio.clear();
            audio.setSample(0, 0, .1f);
            {
                CallbackGuard guard;
                processor->processBlock(audio, midi);
            }
            CHECK(allocationCount == 0);
            CHECK(deallocationCount == 0);
            for (int channel = 0; channel < 2; ++channel)
                for (int sample = 0; sample < size; ++sample)
                    REQUIRE(std::isfinite(audio.getSample(channel, sample)));
            CHECK(processor->failures.load() == 0);
        }
    }
}
