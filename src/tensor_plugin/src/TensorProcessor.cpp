#include "TensorProcessor.h"

#include "FactoryPrograms.h"
#include "TensorEditor.h"
#include "TensorReference.h"
#include "TierConversion.h"
#include "alfa/tensor/Compiler.h"

#include <chrono>
#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace alfa::tensorhost
{
juce::String Processor::document() const
{
    std::lock_guard lock(configurationMutex_);
    try
    {
        auto data = nlohmann::json::parse(json_.toStdString());
        if (data.contains("control_layers") && !data["control_layers"].empty())
        {
            auto& values = data["control_layers"][0]["values"];
            for (size_t i = 0; i < std::min(size_t{8}, values.size()); ++i)
                values[i] = xControls[i].load();
        }
        return data.dump(2);
    }
    catch (const std::exception&)
    {
        return json_;
    }
}
namespace
{
juce::AudioProcessorValueTreeState::ParameterLayout hostParameters()
{
    juce::AudioProcessorValueTreeState::ParameterLayout layout;
    for (const auto* id : {"x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"})
        layout.add(std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{id, 1}, id, juce::NormalisableRange<float>{-1.0f, 1.0f, 0.0f}, 0.0f));
    for (const auto* id : {"input_gain", "output_gain"})
        layout.add(std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{id, 1}, id, juce::NormalisableRange<float>{-60.0f, 12.0f, 0.01f}, 0.0f));
    layout.add(
        std::make_unique<juce::AudioParameterFloat>(juce::ParameterID{"mix", 1}, "Mix", 0.0f, 1.0f, 1.0f));
    layout.add(std::make_unique<juce::AudioParameterBool>(juce::ParameterID{"bypass", 1}, "Bypass", false));
    layout.add(std::make_unique<juce::AudioParameterChoice>(
        juce::ParameterID{"mode", 1}, "Input mode", juce::StringArray{"Instrument", "FX", "Hybrid"}, 0));
    return layout;
}
} // namespace

Processor::Processor()
    : AudioProcessor(BusesProperties()
                         .withInput("Audio input", juce::AudioChannelSet::stereo(), true)
                         .withOutput("Audio output", juce::AudioChannelSet::stereo(), true))
    , parameters_(*this, &undo_, "TensorParameters", hostParameters())
{
    inputGain_ = parameters_.getRawParameterValue("input_gain");
    outputGain_ = parameters_.getRawParameterValue("output_gain");
    mix_ = parameters_.getRawParameterValue("mix");
    bypass_ = parameters_.getRawParameterValue("bypass");
    for (const auto* id : {"x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "mode"})
        parameters_.addParameterListener(id, this);
    uiSettings_.setProperty("theme", "Graphite", nullptr);
    uiSettings_.setProperty("scale", 1.0, nullptr);
    uiSettings_.setProperty("view", 0, nullptr);
    keyboard.addListener(this);
    juce::String error;
    const auto factories = factoryPrograms();
    if (!factories.empty())
        loadJSON(juce::String(factories.front().document), error);
    else
        loadJSON(referenceDocument(0), error);
    startTimerHz(4);
}
juce::String Processor::referenceDocument(int index)
{
    const std::array<juce::String, 3> names{"instrument.aom.json", "fx.aom.json", "hybrid.aom.json"};
    const auto selected = names[static_cast<size_t>(juce::jlimit(0, 2, index))];
    for (int i = 0; i < TensorReference::namedResourceListSize; ++i)
    {
        const auto* resource = TensorReference::namedResourceList[i];
        const auto* filename = TensorReference::getNamedResourceOriginalFilename(resource);
        if (filename && juce::String(filename) == selected)
        {
            int length = 0;
            auto* data = TensorReference::getNamedResource(resource, length);
            return juce::String::fromUTF8(data, length);
        }
    }
    return {};
}
Processor::~Processor()
{
    for (const auto* id : {"x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "mode"})
        parameters_.removeParameterListener(id, this);
    stopTimer();
    keyboard.removeListener(this);
}
void Processor::parameterChanged(const juce::String& id, float value)
{
    if (id == "mode")
    {
        mode.store(juce::jlimit(0, 2, juce::roundToInt(value)));
        return;
    }
    static constexpr std::array<const char*, 8> ids{"x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"};
    for (size_t i = 0; i < ids.size(); ++i)
        if (id == ids[i])
            xControls[i].store(value);
}
void Processor::setParameterValue(const juce::String& id, float actual)
{
    if (auto* parameter = parameters_.getParameter(id))
    {
        parameter->beginChangeGesture();
        parameter->setValueNotifyingHost(parameter->convertTo0to1(actual));
        parameter->endChangeGesture();
    }
}
bool Processor::editTensorParameter(size_t index, double value, juce::String& error)
{
    try
    {
        auto data = nlohmann::json::parse(document().toStdString());
        auto& values = data.at("tensors").at("parameters").at("values");
        if (index >= values.size() || !std::isfinite(value))
            throw std::invalid_argument("Invalid parameter value");
        const auto locks = data.value("parameter_locks", nlohmann::json::array());
        if (index < locks.size() && locks[index].get<int>() != 0)
            throw std::invalid_argument("Unlock this parameter before editing");
        values[index] = value;
        return editConfigurationJSON(data.dump(), error);
    }
    catch (const std::exception& failure)
    {
        error = failure.what();
        return false;
    }
}
bool Processor::setParameterLock(size_t index, bool locked, juce::String& error)
{
    try
    {
        auto data = nlohmann::json::parse(document().toStdString());
        const auto count = data.at("tensors").at("parameters").at("values").size();
        if (index >= count)
            throw std::invalid_argument("Invalid parameter index");
        if (data.value("parameter_locks", nlohmann::json::array()).empty())
            data["parameter_locks"] = std::vector<int>(count, 0);
        data["parameter_locks"][index] = locked ? 1 : 0;
        return editConfigurationJSON(data.dump(), error);
    }
    catch (const std::exception& failure)
    {
        error = failure.what();
        return false;
    }
}
bool Processor::loadPresetJSON(const juce::String& text, juce::String& error)
{
    try
    {
        const auto old = nlohmann::json::parse(document().toStdString());
        auto candidate = nlohmann::json::parse(text.toStdString());
        const auto locks = old.value("parameter_locks", nlohmann::json::array());
        if (std::any_of(locks.begin(), locks.end(),
                        [](const auto& value) { return value.template get<int>() != 0; }))
        {
            const auto& prior = old.at("tensors").at("parameters");
            auto& next = candidate.at("tensors").at("parameters");
            if (prior.at("axes") != next.at("axes") || prior.at("shape") != next.at("shape") ||
                old.at("instances") != candidate.at("instances"))
                throw std::invalid_argument("Locked parameters require matching primitive coordinates; "
                                            "unlock before changing structure");
            candidate["parameter_locks"] = locks;
            for (size_t i = 0; i < locks.size(); ++i)
                if (locks[i].get<int>() != 0)
                    next["values"][i] = prior["values"][i];
        }
        if (!loadJSON(candidate.dump(), error))
            return false;
        if (candidate.contains("preferred_mode"))
            setParameterValue(
                "mode", static_cast<float>(juce::jlimit(0, 2, candidate.at("preferred_mode").get<int>())));
        return true;
    }
    catch (const std::exception& failure)
    {
        error = failure.what();
        return false;
    }
}
void Processor::clearPerformance() noexcept
{
    notes_.fill(0);
    voiceUsed_.fill(false);
    deferredRelease_.fill(false);
    sustain_.fill(false);
    releaseSamples_.fill(0);
    noteCount_ = active_ ? active_->voiceCapacity : 0;
    gestures_.fill(0);
    context_.fill(0);
    activeNotes.store(0);
}
void Processor::requestPanic() noexcept
{
    try
    {
        juce::String error;
        if (!loadJSONImpl(document(), error, false))
            failures.fetch_add(1, std::memory_order_relaxed);
    }
    catch (...)
    {
        failures.fetch_add(1, std::memory_order_relaxed);
    }
    panic_.store(true);
}
bool Processor::editLayerValue(size_t layer, size_t index, double value, juce::String& error)
{
    try
    {
        auto data = nlohmann::json::parse(document().toStdString());
        auto& entry = data.at("control_layers").at(layer);
        if (index >= entry.at("values").size() || !std::isfinite(value))
            throw std::invalid_argument("Invalid layer coordinate");
        const auto bounds = entry.at("ranges").at(index).get<std::array<double, 2>>();
        if (value < bounds[0] || value > bounds[1])
            throw std::invalid_argument("Value exceeds this X coordinate's range");
        if (layer == 0 && index < 8)
        {
            setParameterValue("x" + juce::String(static_cast<int>(index)), static_cast<float>(value));
            return true;
        }
        entry["values"][index] = value;
        return editConfigurationJSON(data.dump(), error);
    }
    catch (const std::exception& failure)
    {
        error = failure.what();
        return false;
    }
}
bool Processor::isBusesLayoutSupported(const BusesLayout& b) const
{
    const auto out = b.getMainOutputChannelSet();
    const auto in = b.getMainInputChannelSet();
    return (out == juce::AudioChannelSet::mono() || out == juce::AudioChannelSet::stereo()) &&
           (in.isDisabled() || in == out);
}
void Processor::prepareToPlay(double rate, int)
{
    mergedMidi_.ensureSize(65536);
    rate_.store(std::isfinite(rate) && rate > 0 ? rate : 48000.0);
    clock_ = 0;
    clearPerformance();
    keyRead_.store(keyWrite_.load(std::memory_order_acquire), std::memory_order_release);
    gainCurrent_ = juce::Decibels::decibelsToGain(outputGain_->load());
    samples.store(0);
    waveformWrite_ = 0;
    waveformCursor.store(0);
    for (auto& value : waveformSnapshot)
        value.store(0.0f);
    for (auto& value : stateSnapshot)
        value.store(0.0f);
    for (auto& value : inputPeak)
        value.store(0.0f);
    for (auto& value : outputPeak)
        value.store(0.0f);
    const auto saved = document();
    if (saved.isNotEmpty())
    {
        juce::String error;
        loadJSON(saved, error);
    }
}
void Processor::queueKey(int channel, int note, float velocity, bool on) noexcept
{
    const auto write = keyWrite_.load(std::memory_order_relaxed);
    const auto next = (write + 1) % keyEvents_.size();
    if (next == keyRead_.load(std::memory_order_acquire))
    {
        failures.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    keyEvents_[write] = {channel, note, velocity, on};
    keyWrite_.store(next, std::memory_order_release);
}
void Processor::handleNoteOn(juce::MidiKeyboardState*, int channel, int note, float velocity)
{
    queueKey(channel, note, velocity, true);
}
void Processor::handleNoteOff(juce::MidiKeyboardState*, int channel, int note, float velocity)
{
    queueKey(channel, note, velocity, false);
}
void Processor::event(const juce::MidiMessage& m) noexcept
{
    if (active_ && active_->voiceCapacity > 0 && indexedEvent(m))
        return;
    const auto ch = static_cast<size_t>(juce::jlimit(1, 16, m.getChannel()) - 1);
    if (m.isNoteOnOrOff())
    {
        size_t row = 0;
        while (row < noteCount_ && (notes_[row * 10 + 8] != static_cast<float>(ch + 1) ||
                                    notes_[row * 10 + 9] != static_cast<float>(m.getNoteNumber())))
            ++row;
        if (m.isNoteOn())
        {
            if (row == noteCount_)
            {
                if (noteCount_ == 128)
                {
                    failures.fetch_add(1);
                    return;
                }
                ++noteCount_;
            }
            auto* n = notes_.data() + row * 10;
            n[0] = static_cast<float>(juce::MidiMessage::getMidiNoteInHertz(m.getNoteNumber()));
            n[1] = m.getFloatVelocity();
            n[2] = 1;
            n[3] = 0;
            n[4] = gestures_[ch * 3 + 1];
            n[5] = gestures_[ch * 3];
            n[6] = gestures_[ch * 3 + 2];
            n[7] = static_cast<float>(static_cast<double>(clock_) / rate_);
            n[8] = static_cast<float>(ch + 1);
            n[9] = static_cast<float>(m.getNoteNumber());
            context_[4] = n[7] - context_[5];
            context_[5] = n[7];
            context_[2] = n[9];
            context_[3] += 1;
        }
        else
        {
            context_[6] = static_cast<float>(m.getNoteNumber());
            context_[7] = static_cast<float>(static_cast<double>(clock_) / rate_);
            if (row < noteCount_)
            {
                --noteCount_;
                std::copy_n(notes_.data() + noteCount_ * 10, 10, notes_.data() + row * 10);
            }
        }
    }
    else if (m.isAftertouch())
    {
        for (size_t row = 0; row < noteCount_; ++row)
            if (notes_[row * 10 + 8] == static_cast<float>(ch + 1) &&
                notes_[row * 10 + 9] == static_cast<float>(m.getNoteNumber()))
                notes_[row * 10 + 4] = m.getAfterTouchValue() / 127.0f;
    }
    else if (m.isPitchWheel() || m.isChannelPressure() || (m.isController() && m.getControllerNumber() == 74))
    {
        const size_t field = m.isPitchWheel() ? 0 : m.isChannelPressure() ? 1 : 2;
        const auto value = m.isPitchWheel()        ? (m.getPitchWheelValue() - 8192) / 8192.0f
                           : m.isChannelPressure() ? m.getChannelPressureValue() / 127.0f
                                                   : m.getControllerValue() / 127.0f;
        gestures_[ch * 3 + field] = value;
        for (size_t row = 0; row < noteCount_; ++row)
            if (notes_[row * 10 + 8] == static_cast<float>(ch + 1))
            {
                notes_[row * 10 + (field == 0 ? 5 : field == 1 ? 4 : 6)] = value;
                if (field == 0 && active_ && active_->voiceCapacity > 0)
                    notes_[row * 10] = static_cast<float>(juce::MidiMessage::getMidiNoteInHertz(
                                           static_cast<int>(notes_[row * 10 + 9]))) *
                                       std::exp2(value * active_->bendSemitones / 12.0f);
            }
    }
    else if (m.isAllNotesOff() || m.isAllSoundOff())
    {
        for (size_t row = noteCount_; row-- > 0;)
            if (notes_[row * 10 + 8] == static_cast<float>(ch + 1))
            {
                --noteCount_;
                std::copy_n(notes_.data() + noteCount_ * 10, 10, notes_.data() + row * 10);
            }
    }
}
bool Processor::indexedEvent(const juce::MidiMessage& message) noexcept
{
    const auto capacity = active_->voiceCapacity;
    const auto channel = static_cast<size_t>(juce::jlimit(1, 16, message.getChannel()) - 1);
    const auto release = [&](size_t row)
    {
        notes_[row * 10 + 2] = 0;
        deferredRelease_[row] = false;
        releaseSamples_[row] = static_cast<uint64_t>(active_->releaseSeconds * rate_.load());
    };
    if (message.isController() && message.getControllerNumber() == 64)
    {
        sustain_[channel] = message.getControllerValue() >= 64;
        if (!sustain_[channel])
            for (size_t row = 0; row < capacity; ++row)
                if (deferredRelease_[row] && notes_[row * 10 + 8] == static_cast<float>(channel + 1))
                    release(row);
        return true;
    }
    if (message.isAllSoundOff())
    {
        // Arbitrary shared recurrent state has no inferred per-channel ownership.
        // Release this channel without destroying other channels or shared FX tails.
        sustain_[channel] = false;
        for (size_t row = 0; row < capacity; ++row)
            if (notes_[row * 10 + 8] == static_cast<float>(channel + 1))
                release(row);
        return true;
    }
    if (message.isAllNotesOff())
    {
        for (size_t row = 0; row < capacity; ++row)
            if (notes_[row * 10 + 8] == static_cast<float>(channel + 1))
                release(row);
        return true;
    }
    if (!message.isNoteOnOrOff())
        return false;
    size_t row = capacity;
    for (size_t i = 0; i < capacity; ++i)
        if (voiceUsed_[i] && notes_[i * 10 + 8] == static_cast<float>(channel + 1) &&
            notes_[i * 10 + 9] == static_cast<float>(message.getNoteNumber()))
        {
            row = i;
            break;
        }
    if (message.isNoteOff())
    {
        context_[6] = static_cast<float>(message.getNoteNumber());
        context_[7] = static_cast<float>(static_cast<double>(clock_) / rate_);
        if (row < capacity)
        {
            if (sustain_[channel])
                deferredRelease_[row] = true;
            else
                release(row);
        }
        return true;
    }
    if (row == capacity)
    {
        for (size_t i = 0; i < capacity; ++i)
            if (!voiceUsed_[i])
            {
                row = i;
                break;
            }
        if (row == capacity)
        {
            row = 0;
            for (size_t i = 1; i < capacity; ++i)
                if ((notes_[i * 10 + 2] == 0 && notes_[row * 10 + 2] != 0) ||
                    (notes_[i * 10 + 2] == notes_[row * 10 + 2] && notes_[i * 10 + 7] < notes_[row * 10 + 7]))
                    row = i;
        }
    }
    voiceUsed_[row] = true;
    deferredRelease_[row] = false;
    releaseSamples_[row] = 0;
    auto* note = notes_.data() + row * 10;
    note[0] = static_cast<float>(juce::MidiMessage::getMidiNoteInHertz(message.getNoteNumber())) *
              std::exp2(gestures_[channel * 3] * active_->bendSemitones / 12.0f);
    note[1] = message.getFloatVelocity();
    note[2] = 1;
    note[3] = 0;
    note[4] = gestures_[channel * 3 + 1];
    note[5] = gestures_[channel * 3];
    note[6] = gestures_[channel * 3 + 2];
    note[7] = static_cast<float>(static_cast<double>(clock_) / rate_);
    note[8] = static_cast<float>(channel + 1);
    note[9] = static_cast<float>(message.getNoteNumber());
    context_[4] = note[7] - context_[5];
    context_[5] = note[7];
    context_[2] = note[9];
    context_[3] += 1;
    return true;
}
void Processor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi)
{
    processBlockImpl(buffer, midi, false);
}
void Processor::processBlockBypassed(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi)
{
    processBlockImpl(buffer, midi, true);
}
void Processor::processBlockImpl(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi, bool forceBypass)
{
    juce::ScopedNoDenormals denormals;
    const auto began = std::chrono::steady_clock::now();
    double modelTime = 0;
    const auto write = retireWrite_.load(std::memory_order_relaxed);
    const auto nextWrite = (write + 1) % retired_.size();
    if (nextWrite != retireRead_.load(std::memory_order_acquire))
    {
        if (auto* program = pending_.exchange(nullptr, std::memory_order_acq_rel))
        {
            const auto performanceCompatible = active_ && program->preserveState &&
                                               program->voiceCapacity == active_->voiceCapacity &&
                                               program->releaseSeconds == active_->releaseSeconds &&
                                               program->bendSemitones == active_->bendSemitones;
            if (performanceCompatible)
                program->execution->adopt(*active_->execution);
            if (performanceCompatible && program->latencySamples == active_->latencySamples)
            {
                program->dryHistory.swap(active_->dryHistory);
                program->dryCursor = active_->dryCursor;
            }
            if (active_)
            {
                retired_[write] = active_;
                retireWrite_.store(nextWrite, std::memory_order_release);
            }
            active_ = program;
            scalarBytes_.store(static_cast<int>(alfa::tensor::scalarBytes(program->execution->scalarType())));
            if (!performanceCompatible)
                clearPerformance();
        }
    }
    if (panic_.exchange(false))
    {
        clearPerformance();
    }
    std::array<float, 8> controls;
    for (size_t i = 0; i < controls.size(); ++i)
        controls[i] = xControls[i].load();
    if (active_ && !active_->execution->updateControls(controls))
        failures.fetch_add(1, std::memory_order_relaxed);
    mergedMidi_.clear();
    if (midi.data.size() > 60000)
    {
        buffer.clear();
        failures.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    mergedMidi_.addEvents(midi, 0, buffer.getNumSamples(), 0);
    auto keyRead = keyRead_.load(std::memory_order_relaxed);
    while (keyRead != keyWrite_.load(std::memory_order_acquire))
    {
        const auto& key = keyEvents_[keyRead];
        mergedMidi_.addEvent(key.on ? juce::MidiMessage::noteOn(key.channel, key.note, key.velocity)
                                    : juce::MidiMessage::noteOff(key.channel, key.note),
                             0);
        keyRead = (keyRead + 1) % keyEvents_.size();
        keyRead_.store(keyRead, std::memory_order_release);
    }
    auto events = mergedMidi_.begin();
    const auto selectedMode = mode.load();
    const auto targetGain = juce::Decibels::decibelsToGain(outputGain_->load());
    const auto inputGain = juce::Decibels::decibelsToGain(inputGain_->load());
    const auto dryWet = mix_->load();
    const auto bypass = forceBypass || bypass_->load() > 0.5f;
    const auto gainSlew = static_cast<float>(1.0 - std::exp(-1.0 / (rate_.load() * 0.005)));
    std::array<float, 2> inPeak{}, outPeak{};
    for (int sample = 0; sample < buffer.getNumSamples(); ++sample)
    {
        while (events != mergedMidi_.end() && (*events).samplePosition <= sample)
        {
            if ((*events).numBytes <= 3)
                event((*events).getMessage());
            ++events;
        }
        std::array<float, 2> dry{};
        for (int channel = 0; channel < juce::jmin(2, getTotalNumInputChannels()); ++channel)
        {
            dry[static_cast<size_t>(channel)] = buffer.getSample(channel, sample);
            if (!std::isfinite(dry[static_cast<size_t>(channel)]))
            {
                dry[static_cast<size_t>(channel)] = 0;
                failures.fetch_add(1, std::memory_order_relaxed);
            }
            inPeak[static_cast<size_t>(channel)] =
                std::max(inPeak[static_cast<size_t>(channel)], std::abs(dry[static_cast<size_t>(channel)]));
        }
        for (int channel = 0; channel < buffer.getNumChannels(); ++channel)
            buffer.setSample(channel, sample, 0);
        alfa::tensor::Inputs input;
        std::array<float, 2> audio{dry[0] * inputGain, dry[1] * inputGain};
        if (selectedMode != 0)
            input.audio = {audioAxes_, stereoShape_, audio, {}};
        auto alignedDry = dry;
        if (active_ && !active_->dryHistory.empty())
        {
            alignedDry = active_->dryHistory[active_->dryCursor];
            active_->dryHistory[active_->dryCursor] = dry;
            active_->dryCursor = (active_->dryCursor + 1) % active_->dryHistory.size();
        }
        if (selectedMode != 1)
        {
            midiShape_[0] = noteCount_;
            input.midi = {midiAxes_, midiShape_, std::span<const float>(notes_.data(), noteCount_ * 10), {}};
            input.gesture = {gestureAxes_, gestureShape_, gestures_, {}};
        }
        context_[0] = static_cast<float>(static_cast<double>(clock_) / rate_);
        context_[1] = static_cast<float>(rate_);
        input.context = {contextAxes_, contextShape_, context_, {}};
        input.userPreference = {preferenceAxes_, preferenceShape_, preferences_, {}};
        if (active_)
        {
            const auto rendered = active_->execution->tick(input, modelTime);
            if (!rendered.modelAccepted)
                failures.fetch_add(1, std::memory_order_relaxed);
            if (rendered.audioRendered)
            {
                for (int channel = 0; channel < buffer.getNumChannels(); ++channel)
                    buffer.addSample(channel, sample, active_->execution->mixedOutput(channel));
            }
            else
                failures.fetch_add(1, std::memory_order_relaxed);
        }
        gainCurrent_ += (targetGain - gainCurrent_) * gainSlew;
        for (int channel = 0; channel < buffer.getNumChannels(); ++channel)
        {
            const auto wet = buffer.getSample(channel, sample);
            const auto original =
                selectedMode == 0 ? 0.0f : alignedDry[static_cast<size_t>(juce::jmin(channel, 1))];
            auto value = bypass ? original : (dryWet * wet + (1 - dryWet) * original) * gainCurrent_;
            if (!std::isfinite(value))
            {
                value = 0;
                failures.fetch_add(1, std::memory_order_relaxed);
            }
            buffer.setSample(channel, sample, value);
            if (channel < 2)
                outPeak[static_cast<size_t>(channel)] =
                    std::max(outPeak[static_cast<size_t>(channel)], std::abs(value));
        }
        waveformSnapshot[waveformWrite_++ % waveformSnapshot.size()].store(
            buffer.getNumChannels() > 0 ? buffer.getSample(0, sample) : 0);
        for (size_t n = 0; n < noteCount_; ++n)
        {
            notes_[n * 10 + 3] += static_cast<float>(1.0 / rate_);
            if (active_ && active_->voiceCapacity > 0 && voiceUsed_[n] && notes_[n * 10 + 2] == 0)
            {
                if (releaseSamples_[n] > 0)
                    --releaseSamples_[n];
                else
                {
                    voiceUsed_[n] = false;
                    std::fill_n(notes_.data() + n * 10, 10, 0.0f);
                }
            }
        }
        ++clock_;
    }
    samples.store(clock_, std::memory_order_relaxed);
    waveformCursor.store(waveformWrite_ % waveformSnapshot.size(), std::memory_order_release);
    int held = 0;
    for (size_t n = 0; n < noteCount_; ++n)
        if (notes_[n * 10 + 2] > 0)
            ++held;
    activeNotes.store(held);
    for (size_t channel = 0; channel < 2; ++channel)
    {
        inputPeak[channel].store(inPeak[channel]);
        outputPeak[channel].store(outPeak[channel]);
    }
    if (active_)
        active_->execution->snapshot(stateSnapshot);
    modelMilliseconds.store(modelTime, std::memory_order_relaxed);
    blockMilliseconds.store(
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - began).count(),
        std::memory_order_relaxed);
}
class Processor::ConfigurationEdit final : public juce::UndoableAction
{
public:
    ConfigurationEdit(Processor& owner, juce::String before, juce::String after)
        : owner_(owner)
        , before_(std::move(before))
        , after_(std::move(after))
    {
    }
    bool perform() override
    {
        if (first_)
        {
            first_ = false;
            return true;
        }
        juce::String error;
        return owner_.loadJSONImpl(after_, error, true);
    }
    bool undo() override
    {
        juce::String error;
        return owner_.loadJSONImpl(before_, error, true);
    }
    int getSizeInUnits() override
    {
        const auto characters = static_cast<int64_t>(before_.length()) + after_.length();
        return static_cast<int>(
            std::min<int64_t>(std::numeric_limits<int>::max(), std::max<int64_t>(1, characters / 256)));
    }

private:
    Processor& owner_;
    juce::String before_, after_;
    bool first_ = true;
};
bool Processor::loadJSON(const juce::String& text, juce::String& error)
{
    return loadJSONImpl(text, error, false);
}
bool Processor::editConfigurationJSON(const juce::String& text, juce::String& error)
{
    const auto before = document();
    if (!loadJSONImpl(text, error, true))
        return false;
    undo_.beginNewTransaction("Edit tensor configuration");
    return undo_.perform(new ConfigurationEdit(*this, before, text));
}
bool Processor::loadJSONImpl(const juce::String& text, juce::String& error, bool preserveState)
{
    std::lock_guard lock(configurationMutex_);
    try
    {
        const auto data = nlohmann::json::parse(text.toStdString());
        auto proposed = std::make_unique<Program>();
        std::string diagnostic;
        proposed->execution = prepareExecution(text.toStdString(), rate_.load(), diagnostic);
        if (!proposed->execution)
        {
            error = diagnostic;
            return false;
        }
        proposed->preserveState = preserveState;
        const auto policy = readNativePolicy(data);
        proposed->voiceCapacity = policy.voiceCapacity;
        proposed->releaseSeconds = policy.releaseSeconds;
        proposed->bendSemitones = policy.bendSemitones;
        const auto newTail = policy.tailSeconds;
        const auto latency = policy.latencySamples;
        proposed->latencySamples = latency;
        const auto alignmentBytes = static_cast<uint64_t>(latency) * sizeof(std::array<float, 2>);
        const auto alignmentBudget = data.at("budget").value("host_alignment_bytes", uint64_t{0});
        if (alignmentBytes > alignmentBudget)
            throw std::invalid_argument("Dry/wet alignment exceeds explicit host_alignment_bytes budget");
        proposed->dryHistory.resize(static_cast<size_t>(latency));
        const auto defaults = proposed->execution->highestDefaults();
        const auto report = proposed->execution->inspection();
        auto* publication = proposed.get();
        owned_.push_back(std::move(proposed));
        auto* superseded = pending_.exchange(publication, std::memory_order_acq_rel);
        if (superseded)
            std::erase_if(owned_, [superseded](const auto& item) { return item.get() == superseded; });
        json_ = text;
        inspection_ = report;
        if (!preserveState)
            for (size_t i = 0; i < std::min(size_t{8}, defaults.size()); ++i)
                setParameterValue("x" + juce::String(static_cast<int>(i)), defaults[i]);
        tailSeconds_.store(newTail);
        setLatencySamples(latency);
        programRevision.fetch_add(1);
        error.clear();
        return true;
    }
    catch (const std::exception& failure)
    {
        error = failure.what();
        return false;
    }
}
bool Processor::setTier(const juce::String& tier, juce::String& error)
{
    try
    {
        auto data = nlohmann::json::parse(document().toStdString());
        if (data.contains("tier_profiles"))
        {
            std::string converted, diagnostic;
            if (!convertTier(data.dump(), tier.toStdString(), rate_.load(), converted, diagnostic))
            {
                error = diagnostic;
                return false;
            }
            return editConfigurationJSON(converted, error);
        }
        data["tier"] = tier.toStdString();
        data.erase("scalar_type");
        alfa::tensor::PrecisionProfile profile;
        std::string diagnostic;
        if (!alfa::tensor::resolvePrecisionJSON(data.dump(), profile, diagnostic))
        {
            error = diagnostic;
            return false;
        }
        data["scalar_type"] = alfa::tensor::scalarTypeName(profile.scalarType);
        const auto latent = data.contains("latentSound") ? data["latentSound"]["values"].size() : size_t(0);
        data["input_capacity"] = {2, 1280, 48, 5, latent, 8};
        data["budget"]["boundary_conversion_bytes"] =
            (2 + 1280 + 48 + 5 + latent + 8) * alfa::tensor::scalarBytes(profile.scalarType);
        return editConfigurationJSON(data.dump(2), error);
    }
    catch (const std::exception& failure)
    {
        error = failure.what();
        return false;
    }
}
void Processor::getStateInformation(juce::MemoryBlock& dest)
{
    auto wrapper = nlohmann::json{{"configuration", document().toStdString()}, {"mode", mode.load()}};
    std::vector<float> values;
    for (const auto& value : xControls)
        values.push_back(value.load());
    wrapper["x_controls"] = values;
    if (auto xml = parameters_.copyState().createXml())
        wrapper["parameters"] = xml->toString().toStdString();
    if (auto xml = uiSettings_.createXml())
        wrapper["ui_settings"] = xml->toString().toStdString();
    wrapper["state_schema"] = "aom.host/2";
    const auto text = wrapper.dump();
    dest.replaceAll(text.data(), text.size());
}
void Processor::setStateInformation(const void* bytes, int length)
{
    if (bytes == nullptr || length <= 0)
        return;
    try
    {
        auto data =
            nlohmann::json::parse(static_cast<const char*>(bytes), static_cast<const char*>(bytes) + length);
        if (!data.is_object())
            return;
        const auto schema = data.value("state_schema", std::string{});
        if (!schema.empty() && schema != "aom.host/1" && schema != "aom.host/2")
            return;
        const auto configuration = data.at("configuration").get<std::string>();
        if (data.contains("mode") && !data.at("mode").is_number_integer())
            return;
        const auto restoredMode = data.value("mode", 2);
        if (restoredMode < 0 || restoredMode > 2)
            return;
        const auto values = data.contains("x_controls") ? data.at("x_controls").get<std::vector<float>>()
                                                        : data.value("intentions", std::vector<float>{});
        if (values.size() > xControls.size() ||
            std::any_of(values.begin(), values.end(),
                        [](float value) { return !std::isfinite(value) || value < -1 || value > 1; }))
            return;
        juce::ValueTree restoredParameters, restoredUI;
        if (data.contains("parameters"))
        {
            const auto xml = juce::parseXML(juce::String(data.at("parameters").get<std::string>()));
            if (!xml || !xml->hasTagName("TensorParameters"))
                return;
            restoredParameters = juce::ValueTree::fromXml(*xml);
            if (!restoredParameters.isValid())
                return;
            for (const auto& child : restoredParameters)
            {
                const auto* parameter = parameters_.getParameter(child.getProperty("id").toString());
                if (!parameter || !child.hasProperty("value"))
                    return;
                const auto text = child.getProperty("value").toString().toStdString();
                size_t consumed = 0;
                const auto value = std::stod(text, &consumed);
                const auto& range = parameter->getNormalisableRange();
                if (consumed != text.size() || !std::isfinite(value) || value < range.start ||
                    value > range.end)
                    return;
            }
        }
        if (data.contains("ui_settings"))
        {
            const auto xml = juce::parseXML(juce::String(data.at("ui_settings").get<std::string>()));
            if (!xml || !xml->hasTagName("TensorUI"))
                return;
            restoredUI = juce::ValueTree::fromXml(*xml);
            if (!restoredUI.isValid())
                return;
        }
        // Validate every wrapper field before publishing any sound or host state.
        juce::String error;
        if (!loadJSON(configuration, error))
            return;
        if (restoredParameters.isValid())
            parameters_.replaceState(restoredParameters);
        if (restoredUI.isValid())
            uiSettings_.copyPropertiesAndChildrenFrom(restoredUI, nullptr);
        mode.store(restoredMode);
        setParameterValue("mode", static_cast<float>(mode.load()));
        for (size_t i = 0; i < juce::jmin(values.size(), xControls.size()); ++i)
            if (std::isfinite(values[i]))
                setParameterValue("x" + juce::String(static_cast<int>(i)), values[i]);
    }
    catch (const std::exception&)
    {
    }
}
void Processor::timerCallback()
{
    std::lock_guard lock(configurationMutex_);
    auto read = retireRead_.load(std::memory_order_relaxed);
    while (read != retireWrite_.load(std::memory_order_acquire))
    {
        auto* retired = retired_[read];
        std::erase_if(owned_, [retired](const auto& item) { return item.get() == retired; });
        read = (read + 1) % retired_.size();
        retireRead_.store(read, std::memory_order_release);
    }
}
juce::AudioProcessorEditor* Processor::createEditor()
{
    return new Editor(*this);
}
} // namespace alfa::tensorhost
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new alfa::tensorhost::Processor();
}
